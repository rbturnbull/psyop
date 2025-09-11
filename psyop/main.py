#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Make BLAS single-threaded to avoid oversubscription / macOS crashes
import os
for _env_var in (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import re
from enum import Enum
from pathlib import Path
from typing import Optional
import xarray as xr
import typer
from rich.console import Console

from .model import build_model
from . import viz, opt

__version__ = "0.1.0"

console = Console()
app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")


class Direction(str, Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s

_num_re = re.compile(
    r"""
    ^\s*
    ([+-]?                      # sign
      (?:
        (?:\d+(?:\.\d*)?|\.\d+) # 123, 123., .123, 123.456
        (?:[eE][+-]?\d+)?       # optional exponent
      )
    )
    \s*$
    """,
    re.VERBOSE,
)

def _is_intlike_str(s: str) -> bool:
    try:
        f = float(s)
        return float(int(round(f))) == f
    except Exception:
        return False

def _to_number(s: str):
    """Return int if int-like, else float, else raise."""
    if not _num_re.match(s):
        raise ValueError
    v = float(s)
    return int(round(v)) if _is_intlike_str(s) else v

def _parse_list_like(s: str) -> list:
    # Accept comma-separated, optionally wrapped in [] or ().
    s = s.strip()
    if s.startswith(("(", "[")) and s.endswith((")", "]")):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        p = _strip_quotes(p)
        try:
            out.append(_to_number(p))
        except Exception:
            out.append(p)  # leave as raw string if not numeric
    return out

def _parse_range_call(s: str) -> tuple:
    """
    Parse 'range(a,b[,step])' inclusive on the upper bound for ints.
    Returns a tuple of ints.
    """
    m = re.fullmatch(r"\s*range\s*\(\s*([^)]*)\s*\)\s*", s)
    if not m:
        raise ValueError
    args = [t.strip() for t in m.group(1).split(",") if t.strip() != ""]
    if len(args) not in (2, 3):
        raise ValueError
    a = int(_to_number(args[0]))
    b = int(_to_number(args[1]))
    step = int(_to_number(args[2])) if len(args) == 3 else 1
    if step == 0:
        raise ValueError
    lo, hi = (a, b) if a <= b else (b, a)
    # inclusive upper bound
    seq = list(range(lo, hi + 1, abs(step)))
    return tuple(seq)


def _parse_colon_or_dots(s: str) -> slice:
    """
    Parse 'a:b', 'a..b', or 'a:b:step' → slice(start, stop, step).
    - Works for int or float endpoints (we sort so start <= stop).
    - Step is optional; if present it can be int or float.
    - Any token with ':' (or '..') yields a slice (no tuples).
    """
    s_norm = re.sub(r"\.\.+", ":", s.strip())
    parts = [p.strip() for p in s_norm.split(":")]
    if len(parts) not in (2, 3):
        raise ValueError(f"Not a range: {s!r}")

    a_str, b_str = parts[0], parts[1]
    a_num = _to_number(a_str)
    b_num = _to_number(b_str)
    if not isinstance(a_num, (int, float)) or not isinstance(b_num, (int, float)):
        raise ValueError(f"Non-numeric range endpoints: {s!r}")

    start, stop = (a_num, b_num)
    if start > stop:
        start, stop = stop, start  # normalize to ascending

    step = None
    if len(parts) == 3 and parts[2] != "":
        step_val = _to_number(parts[2])
        if not isinstance(step_val, (int, float)) or step_val == 0:
            raise ValueError(f"Invalid step in range: {s!r}")
        step = step_val

    return slice(start, stop, step)

def _parse_constraint_value(text: str):
    """
    Convert a CLI string into one of:
      - number (int/float)  -> fixed
      - slice(lo, hi)       -> float range (inclusive ends)
      - list/tuple          -> choices (finite set)
      - 'range(a,b[,s])'    -> tuple of ints
    """
    raw = _strip_quotes(str(text))

    # 1) range(...)
    try:
        return _parse_range_call(raw)
    except Exception:
        pass

    # 2) bracketed/parenthesized list or plain comma-separated list
    if (raw.startswith(("[", "(")) and raw.endswith(("]", ")"))) or ("," in raw and " " not in raw[:2]):
        items = _parse_list_like(raw)
        # coerce homogeneous int-like to tuple[int], else list[float]
        if all(isinstance(v, int) for v in items):
            return tuple(items)
        return items

    # 3) colon / dot ranges  (a:b[:step] or a..b)
    if ":" in raw or ".." in raw:
        try:
            return _parse_colon_or_dots(raw)
        except Exception:
            pass

    # 4) scalar number
    try:
        return _to_number(raw)
    except Exception:
        return raw  # fallback to string (rare; typically ignored downstream)


def _parse_unknown_cli_kv_text(args: list[str]) -> dict[str, str]:
    """
    Extract unknown --key value pairs as raw strings (no coercion here).
    Supports: --k=v  and  --k v. Repeated keys -> last wins.
    """
    out: dict[str, str] = {}
    it = iter(args)
    for tok in it:
        if not tok.startswith("--"):
            continue
        key = tok[2:]
        if "=" in key:
            k, v = key.split("=", 1)
        else:
            k = key
            try:
                nxt = next(it)
            except StopIteration:
                nxt = "true"
            if nxt.startswith("--"):
                # treat as flag without value; put it back by ignoring and storing "true"
                v = "true"
            else:
                v = nxt
        out[k.strip().replace("-", "_")] = v
    return out

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _canonicalize_feature_keys(
    model: xr.Dataset | Path, raw_map: dict[str, object]
) -> tuple[dict[str, object], dict[str, str]]:
    """
    Map user keys (any style) to either:
      - dataset *feature* names (numeric, including one-hot *member* names), OR
      - *categorical base* names (e.g., 'language') detected from one-hot blocks.

    Returns (mapped, alias). Unmatched keys are dropped with a warning.

    Notes:
      - Exact matches win.
      - Then normalized matches for full feature names.
      - Then normalized matches for categorical bases (so '--language "Linear A"' is preserved as 'language': 'Linear A').
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    features = [str(x) for x in ds["feature"].values.tolist()]

    # Indexes for feature names and categorical bases
    feature_norm_index = {_norm(f): f for f in features}
    feature_set = set(features)
    bases, base_norm_index = _categorical_bases_from_features(features)

    mapped: dict[str, object] = {}
    alias: dict[str, str] = {}

    for k, v in (raw_map or {}).items():
        # 1) Exact feature match
        if k in feature_set:
            mapped[k] = v
            alias[k] = k
            continue

        nk = _norm(k)

        # 2) Normalized feature match (full feature/member name)
        if nk in feature_norm_index:
            canonical = feature_norm_index[nk]
            mapped[canonical] = v
            alias[k] = canonical
            continue

        # 3) Categorical base match (exact or normalized)
        if (k in bases) or (nk in base_norm_index):
            base = k if k in bases else base_norm_index[nk]
            mapped[base] = v
            alias[k] = base
            continue

        raise typer.BadParameter(f"Unknown feature key: {k}")

    return mapped, alias



def parse_constraints_from_ctx(ctx: typer.Context, model: xr.Dataset | Path) -> dict[str, object]:
    """
    End-to-end: ctx.args → {key: constraint_object}.

    Values can be:
      - number (int/float)   -> fixed
      - slice(lo, hi)        -> float range (inclusive ends)
      - list/tuple           -> finite choices
      - tuple from range(...) (int choices)
      - string               -> categorical label (e.g., --language "Linear A")
    """
    raw_kv = _parse_unknown_cli_kv_text(ctx.args)
    parsed: dict[str, object] = {k: _parse_constraint_value(v) for k, v in raw_kv.items()}

    # Canonicalize keys to either feature names OR categorical bases
    constraints, _ = _canonicalize_feature_keys(model, parsed)

    # Normalize: convert range objects to tuples of ints (choices)
    for k, v in list(constraints.items()):
        if isinstance(v, range):
            constraints[k] = tuple(v)

    # Pretty print constraints
    if constraints:
        def _fmt_value(val: object) -> str:
            if isinstance(val, slice):
                # show start:stop (ignore step in preview)
                lo = getattr(val, "start", None)
                hi = getattr(val, "stop", None)
                return f"[{lo},{hi}]"
            if isinstance(val, (list, tuple, range)):
                return f"{tuple(val)}"
            if isinstance(val, str):
                # quote strings if they have spaces or special chars
                if re.search(r'\s|[,=:\.]', val):
                    return f'"{val}"'
                return val
            return str(val)

        pretty = ", ".join(f"{k}={_fmt_value(constraints[k])}" for k in constraints)
        console.print(f"[cyan]Constraints:[/] {pretty}")

    return constraints


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit.", is_eager=True),
):
    if version:
        console.print(f"[bold]psyop[/] {__version__}")
        raise typer.Exit()


@app.command(help="Fit the model on a CSV and save a single model artifact.")
def model(
    input: Path = typer.Argument(..., help="Input CSV file."),
    output: Path = typer.Argument(..., help="Path to save model artifact (.nc)."),
    target: str = typer.Option("loss", "--target", "-t", help="Target column name."),
    exclude: list[str] = typer.Option([], help="Feature columns to exclude."),
    direction: Direction = typer.Option(
        Direction.MINIMIZE, "--direction", "-d",
        help="Optimization direction for the target."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for fitting/sampling."),
    compress: bool = typer.Option(True, help="Apply compression inside the artifact."),
):
    if not input.exists():
        raise typer.BadParameter(f"Input CSV not found: {input.resolve()}")
    if input.suffix.lower() != ".csv":
        console.print(":warning: [yellow]Input does not end with .csv[/]")

    build_model(
        input=input,
        target=target,
        output=output,
        exclude=exclude,
        direction=direction.value,
        seed=seed,
        compress=compress,
    )
    console.print(f"[green]Wrote model artifact →[/] {output}")


@app.command(
    help="Suggest BO candidates (constrained EI + exploration).",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def suggest(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Where to save candidates CSV (defaults relative to model)."),
    count: int = typer.Option(1, "--count", "-k", help="Number of candidates to propose."),
    success_threshold: float = typer.Option(0.8, help="Feasibility threshold for constrained EI."),
    explore: float = typer.Option(0.34, help="Fraction of suggestions reserved for exploration."),
    seed: int = typer.Option(0, help="Random seed for proposals."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")

    model = xr.load_dataset(model)
    constraints = parse_constraints_from_ctx(ctx, model)

    opt.suggest(
        model=model,
        output=output,
        count=count,
        success_threshold=success_threshold,
        explore=explore,
        seed=seed,
        **constraints,
    )
    if output:
        console.print(f"[green]Wrote proposals →[/] {output}")


@app.command(
    help="Rank points by probability of being the best feasible minimum.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, 
)
def optimal(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Path|None = typer.Option(None, help="Where to save top candidates CSV (defaults relative to model)."),
    seed: int = typer.Option(0,  help="Random seed for MC."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")

    model = xr.load_dataset(model)
    constraints = parse_constraints_from_ctx(ctx, model)

    opt.optimal(
        model=model,
        output=output,
        seed=seed,
        **constraints,
    )
    if output:
        console.print(f"[green]Wrote top probable minima →[/] {output}")


@app.command(
    help="Create a 2D Partial Dependence of Expected Target (Pairwise Features).",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def plot2d(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Path|None = typer.Option(None, help="Output HTML (defaults relative to model)."),
    grid_size: int = typer.Option(70, help="Grid size per axis for 2D panels."),
    use_log_scale_for_target: bool = typer.Option(False, help="Log10 colors for target."),
    log_shift_epsilon: float = typer.Option(1e-9, help="Epsilon shift for log colors."),
    colorscale: str = typer.Option("RdBu", help="Colorscale name."),
    show: bool|None = typer.Option(None, help="Open the figure in a browser."),
    n_contours: int = typer.Option(12, help="Number of contour levels."),
    optimal: bool = typer.Option(True, help="Include optimal points."),
    suggest: int = typer.Option(0, help="Number of suggested points."),
    width: int = typer.Option(1000, help="Width of each panel in pixels."),
    height: int = typer.Option(1000, help="Height of each panel in pixels."),
    seed: int = typer.Option(42, help="Random seed for suggested points."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")

    model = xr.load_dataset(model)
    constraints = parse_constraints_from_ctx(ctx, model)

    show = show if show is not None else output is None  # default to True if no output file

    viz.plot2d(
        model=model,
        output=output,
        grid_size=grid_size,
        use_log_scale_for_target=use_log_scale_for_target,
        log_shift_epsilon=log_shift_epsilon,
        colorscale=colorscale,
        show=show,
        n_contours=n_contours,
        optimal=optimal,
        suggest=suggest,
        width=width,
        height=height,
        seed=seed,
        **constraints,
    )
    if output:
        console.print(f"[green]Wrote pairplot →[/] {output}")


@app.command(
    help="Create 1D Partial Dependence panels.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def plot1d(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Path|None = typer.Option(None, help="Output HTML (defaults relative to model)."),
    csv_out: Path|None = typer.Option(None, help="Optional CSV export of tidy PD data."),
    grid_size: int = typer.Option(300, help="Points along 1D sweep."),
    line_color: str = typer.Option("blue", help="Line/band color (consistent across variables)."),
    band_alpha: float = typer.Option(0.25, help="Fill alpha for ±2σ."),
    show: bool|None = typer.Option(None, help="Open the figure in a browser."),
    use_log_scale_for_target_y: bool = typer.Option(True, "--log-y/--no-log-y", help="Log scale for target (Y)."),
    log_y_epsilon: float = typer.Option(1e-9, "--log-y-eps", help="Clamp for log-Y."),
    optimal: bool = typer.Option(True, help="Include optimal points."),
    suggest: int = typer.Option(0, help="Number of suggested points."),
    width: int = typer.Option(1000, help="Width of each panel in pixels."),
    height: int = typer.Option(1000, help="Height of each panel in pixels."),
    seed: int = typer.Option(42, help="Random seed for suggested points."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")

    show = show if show is not None else output is None  # default to True if no output file

    model = xr.load_dataset(model)
    constraints = parse_constraints_from_ctx(ctx, model)

    viz.plot1d(
        model=model,
        output=output,
        csv_out=csv_out,
        grid_size=grid_size,
        line_color=line_color,
        band_alpha=band_alpha,
        show=show,
        use_log_scale_for_target_y=use_log_scale_for_target_y,
        log_y_epsilon=log_y_epsilon,
        optimal=optimal,
        suggest=suggest,
        width=width,
        height=height,
        seed=seed,
        **constraints,
    )
    if output:
        console.print(f"[green]Wrote PD HTML →[/] {output}")
    if csv_out:
        console.print(f"[green]Wrote PD CSV  →[/] {csv_out}")


def _categorical_bases_from_features(features: list[str]) -> tuple[set[str], dict[str, str]]:
    """
    Given model feature names (which may include one-hot members like 'language=Linear A'),
    return:
      - bases: a set of base names, e.g. {'language'}
      - base_norm_index: mapping from normalized base name -> canonical base string
    """
    bases: set[str] = set()
    for f in features:
        if "=" in f:
            base = f.split("=", 1)[0].strip()
            if base:
                bases.add(base)
    # normalized index for lookup
    base_norm_index = {_norm(b): b for b in bases}
    return bases, base_norm_index


if __name__ == "__main__":
    app()
