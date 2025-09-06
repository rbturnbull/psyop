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

from .model import run_model
from .viz import make_pairplot, make_partial_dependence1D
from .opt import suggest_candidates, find_optimal

__version__ = "0.1.0"

console = Console()
app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")


class Direction(str, Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"
    AUTO = "auto"


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

def _parse_colon_or_dots(s: str):
    """
    Parse 'a:b' or 'a..b' or 'a:b:step'.
    - if both ends int-like => return tuple of ints (inclusive), support int step
    - else => return slice(float(a), float(b))  (float range)
    """
    # normalize '..' to ':'
    s_norm = re.sub(r"\.\.+", ":", s.strip())
    parts = [p.strip() for p in s_norm.split(":") if p.strip() != ""]
    if len(parts) not in (2, 3):
        raise ValueError
    a_str, b_str = parts[0], parts[1]
    a_num = _to_number(a_str)
    b_num = _to_number(b_str)

    # int range (inclusive), maybe with step
    if isinstance(a_num, int) and isinstance(b_num, int):
        step = 1
        if len(parts) == 3:
            step_val = _to_number(parts[2])
            if not isinstance(step_val, int):
                raise ValueError
            step = step_val
        lo, hi = (a_num, b_num) if a_num <= b_num else (b_num, a_num)
        seq = list(range(lo, hi + 1, abs(step)))
        return tuple(seq)

    # float range → slice (inclusive semantics, but we store as slice)
    lo = float(a_num)
    hi = float(b_num)
    if lo > hi:
        lo, hi = hi, lo
    return slice(lo, hi)

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

def _canonicalize_feature_keys(model_path: Path, raw_map: dict[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    """
    Map user keys (any style) to dataset feature names by normalization.
    Returns (mapped, alias). Unmatched keys are dropped with a warning.
    """
    ds = xr.load_dataset(model_path)
    features = [str(x) for x in ds["feature"].values.tolist()]
    index = {_norm(f): f for f in features}
    feature_set = set(features)

    mapped: dict[str, object] = {}
    alias: dict[str, str] = {}

    for k, v in (raw_map or {}).items():
        if k in feature_set:
            mapped[k] = v
            alias[k] = k
            continue
        nk = _norm(k)
        if nk in index:
            canonical = index[nk]
            mapped[canonical] = v
            alias[k] = canonical
        else:
            console.print(f":warning: [yellow]Ignoring unknown feature key[/]: '{k}'")
    return mapped, alias

def parse_constraints_from_ctx(ctx: typer.Context, model_path: Path) -> dict[str, object]:
    """
    End-to-end: ctx.args → {feature: constraint_object} using the rules above.
    """
    raw_kv = _parse_unknown_cli_kv_text(ctx.args)
    parsed: dict[str, object] = {k: _parse_constraint_value(v) for k, v in raw_kv.items()}
    constraints, _ = _canonicalize_feature_keys(model_path, parsed)

    # Normalize: convert range objects to tuples of ints (choices)
    for k, v in list(constraints.items()):
        if isinstance(v, range):
            constraints[k] = tuple(v)

    if constraints:
        pretty = ", ".join(
            f"{k}="
            + (
                f"[{constraints[k].start},{constraints[k].stop}]" if isinstance(constraints[k], slice)
                else f"{tuple(constraints[k])}" if isinstance(constraints[k], (list, tuple, range))
                else f"{constraints[k]}"
            )
            for k in constraints
        )
        console.print(f"[cyan]Constraints:[/] {pretty}")

    return constraints

def _ensure_parent_dir(path: Path) -> None:
    if path.suffix and path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _force_netcdf_suffix(path: Path) -> Path:
    """Ensure path ends with .nc (normalize .netcdf to .nc)."""
    if path.suffix.lower() in {".nc", ".netcdf"}:
        return path.with_suffix(".nc")
    return path.with_suffix(".nc")


def _default_out_for(model_path: Path, stem_suffix: str, ext: str) -> Path:
    """model_path='foo.nc', stem_suffix='__pairplot', ext='.html' -> 'foo__pairplot.html'"""
    base = model_path.with_suffix("")  # strip .nc
    return base.with_name(base.name + stem_suffix).with_suffix(ext)


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
    exclude: list[str] = typer.Option([], "--exclude", help="Feature columns to exclude."),
    direction: Direction = typer.Option(
        Direction.AUTO, "--direction", "-d",
        help="Optimization direction for the target."
    ),
    success_column: Optional[str] = typer.Option(
        None, "--success-column",
        help="Optional boolean/int column for success (1) / fail (0). "
             "If omitted, success is inferred as ~isna(target)."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for fitting/sampling."),
    compress: bool = typer.Option(True, help="Apply compression inside the artifact."),
):
    if not input.exists():
        raise typer.BadParameter(f"Input CSV not found: {input.resolve()}")
    if input.suffix.lower() != ".csv":
        console.print(":warning: [yellow]Input does not end with .csv[/]")

    output = _force_netcdf_suffix(output)
    _ensure_parent_dir(output)

    run_model(
        input_path=input,
        output_path=output,
        target_column=target,
        exclude_columns=exclude,
        direction=direction.value,
        success_column=success_column,
        random_seed=seed,
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
    count: int = typer.Option(12, "--count", "-n", help="Number of candidates to propose."),
    p_success_threshold: float = typer.Option(0.8, help="Feasibility threshold for constrained EI."),
    explore_fraction: float = typer.Option(0.34, help="Fraction of suggestions reserved for exploration."),
    candidates_pool: int = typer.Option(5000, help="Random candidate pool size to score."),
    seed: int = typer.Option(0, "--seed", help="Random seed for proposals."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    model_nc = _force_netcdf_suffix(model)

    if output is None:
        output = _default_out_for(model, stem_suffix="__bo_proposals", ext=".csv")
    _ensure_parent_dir(output)

    constraints = parse_constraints_from_ctx(ctx, model_nc)

    suggest_candidates(
        model_path=model_nc,
        output_path=output,
        count=count,
        p_success_threshold=p_success_threshold,
        explore_fraction=explore_fraction,
        candidates_pool=candidates_pool,
        random_seed=seed,
        **constraints,
    )
    console.print(f"[green]Wrote proposals →[/] {output}")


@app.command(
    help="Rank points by probability of being the best feasible minimum.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, 
)
def optimal(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Where to save top candidates CSV (defaults relative to model)."),
    count: int = typer.Option(10, "--count", "-k", help="How many top rows to keep."),
    draws: int = typer.Option(2000, "--draws", help="Monte Carlo draws."),
    min_success_probability: float = typer.Option(0.0, "--min-p-success", help="Hard feasibility cutoff (0 disables)."),
    seed: int = typer.Option(0, "--seed", help="Random seed for MC."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__bo_best_probable", ext=".csv")
    _ensure_parent_dir(output)

    model_nc = _force_netcdf_suffix(model)

    constraints = parse_constraints_from_ctx(ctx, model_nc)
    breakpoint()

    find_optimal(
        model_path=_force_netcdf_suffix(model),
        output_path=output,
        count=count,
        n_draws=draws,
        min_success_probability=min_success_probability,
        random_seed=seed,
        **constraints,
    )
    console.print(f"[green]Wrote top probable minima →[/] {output}")


@app.command(
    help="Create a 2D Partial Dependence of Expected Target (Pairwise Features).",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def plot2d(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, help="Output HTML (defaults relative to model)."),
    n_points_1d: int = typer.Option(300, help="Points along 1D sweeps (diagonal)."),
    n_points_2d: int = typer.Option(70, help="Grid size per axis for 2D panels."),
    use_log_scale_for_target: bool = typer.Option(False, help="Log10 colours for target."),
    log_shift_epsilon: float = typer.Option(1e-9, help="Epsilon shift for log colours."),
    colourscale: str = typer.Option("RdBu", help="Plotly colourscale name."),
    show: bool = typer.Option(False, help="Open the figure in a browser."),
    n_contours: int = typer.Option(12, help="Number of contour levels."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__pairplot", ext=".html")
    _ensure_parent_dir(output)

    kwargs = _parse_unknown_cli_kv(ctx.args)
    kwargs, _ = _canonicalize_fixed_keys(_force_netcdf_suffix(model), kwargs)

    make_pairplot(
        model=_force_netcdf_suffix(model),
        output=output,
        n_points_1d=n_points_1d,
        n_points_2d=n_points_2d,
        use_log_scale_for_target=use_log_scale_for_target,
        log_shift_epsilon=log_shift_epsilon,
        colourscale=colourscale,
        show=show,
        n_contours=n_contours,
        **kwargs,
    )
    console.print(f"[green]Wrote pairplot →[/] {output}")


@app.command(
    help="Create 1D PD panels with shading & experimental points.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def plot1d(
    ctx: typer.Context,
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, "-o", help="Output HTML (defaults relative to model)."),
    csv_out: Optional[Path] = typer.Option(None, help="Optional CSV export of tidy PD data."),
    n_points_1d: int = typer.Option(300, help="Points along 1D sweep."),
    line_color: str = typer.Option("rgb(31,119,180)", help="Line/band color (consistent across variables)."),
    band_alpha: float = typer.Option(0.25, help="Fill alpha for ±2σ."),
    figure_height_per_row_px: int = typer.Option(320, help="Pixels per PD row."),
    show_figure: bool = typer.Option(False, "--show", help="Open the figure in a browser."),
    use_log_scale_for_target_y: bool = typer.Option(True, "--log-y/--no-log-y", help="Log scale for target (Y)."),
    log_y_epsilon: float = typer.Option(1e-9, "--log-y-eps", help="Clamp for log-Y."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__pd_1d", ext=".html")
    if csv_out is None:
        csv_out = _default_out_for(model, stem_suffix="__pd_1d", ext=".csv")
    _ensure_parent_dir(output)
    _ensure_parent_dir(csv_out)

    kwargs = _parse_unknown_cli_kv(ctx.args)
    kwargs, _ = _canonicalize_fixed_keys(_force_netcdf_suffix(model), kwargs)

    make_partial_dependence1D(
        model=_force_netcdf_suffix(model),
        output=output,
        csv_out=csv_out,
        n_points_1d=n_points_1d,
        line_color=line_color,
        band_alpha=band_alpha,
        figure_height_per_row_px=figure_height_per_row_px,
        show_figure=show_figure,
        use_log_scale_for_target_y=use_log_scale_for_target_y,
        log_y_epsilon=log_y_epsilon,
        **kwargs,
    )
    console.print(f"[green]Wrote PD HTML →[/] {output}")
    console.print(f"[green]Wrote PD CSV  →[/] {csv_out}")


if __name__ == "__main__":
    app()
