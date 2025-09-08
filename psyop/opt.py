# opt.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtr  # Φ(z), vectorized

from .model import (
    kernel_diag_m52, 
    kernel_m52_ard,
    add_jitter,
    solve_chol,
    solve_lower,
)

from rich.console import Console
from rich.table import Table

console = Console()


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table|None = None,
    show_index: bool = False,
    index_name: str|None = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = rich_table or Table(show_header=True, header_style="bold magenta")

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table



def suggest(
    model: xr.Dataset | Path | str,
    output: Path | str | None = None,
    count: int = 12,
    p_success_threshold: float = 0.8,
    explore_fraction: float = 0.34,
    candidates_pool: int = 5000,
    random_seed: int = 0,
    **kwargs,  # feature constraints: number (fixed), slice (float range), list/tuple (choices); range->tuple
) -> pd.DataFrame:
    """
    Propose candidates via constrained EI + exploration.

    kwargs semantics (ORIGINAL units):
      - number (int/float): fixed value (e.g. epochs=20).
      - slice(start, stop): inclusive float range (e.g. learning_rate=slice(1e-5, 1e-3)).
      - list/tuple: finite choices (e.g. batch_size=(16, 20, 24)).
      - range(...): converted to tuple of ints, then treated as choices.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
        
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = list(map(str, ds["feature"].values.tolist()))
    transforms    = list(map(str, ds["feature_transform"].values.tolist()))
    feat_mean = ds["feature_mean"].values.astype(float)
    feat_std  = ds["feature_std"].values.astype(float)
    Xn_train  = ds["Xn_train"].values.astype(float)

    # 1) Defaults from data
    search_specs = _infer_search_specs(ds, feature_names, transforms)

    # 2) Normalize user constraints according to the new convention
    user_fixed: dict[str, float] = {}
    user_ranges: dict[str, tuple[float, float]] = {}   # slice → (low, high) inclusive
    user_choices: dict[str, list[float | int]] = {}

    for key, raw_val in (kwargs or {}).items():
        if key not in feature_names:
            # silently ignore unknown keys
            continue

        # Convert range -> tuple of ints (then handled as choices)
        if isinstance(raw_val, range):
            raw_val = tuple(raw_val)

        # number -> fixed
        if isinstance(raw_val, (int, float, np.number)):
            val = float(raw_val)
            if np.isfinite(val):
                user_fixed[key] = val
            continue

        # slice -> float range (inclusive)
        if isinstance(raw_val, slice):
            if raw_val.start is None or raw_val.stop is None:
                continue  # require closed interval
            lo = float(raw_val.start)
            hi = float(raw_val.stop)
            if np.isfinite(lo) and np.isfinite(hi):
                if lo > hi:
                    lo, hi = hi, lo
                user_ranges[key] = (lo, hi)
            continue

        # list/tuple -> choices
        if isinstance(raw_val, (list, tuple)):
            if len(raw_val) == 0:
                continue
            # Keep ints as ints if all entries are integer-like
            if all(isinstance(v, (int, np.integer)) or abs(float(v) - round(float(v))) < 1e-12 for v in raw_val):
                user_choices[key] = [int(round(float(v))) for v in raw_val]
            else:
                user_choices[key] = [float(v) for v in raw_val]
            continue

        # anything else → ignore

    # Fixed wins over range/choices for the same key
    for k in list(user_fixed.keys()):
        user_ranges.pop(k, None)
        user_choices.pop(k, None)

    # 3) Apply user bounds/choices to search space & normalize fixed values
    _apply_user_bounds(search_specs, user_ranges, user_choices)
    fixed_norm = _normalize_fixed(user_fixed, search_specs)

    # 4) Best feasible observed target for EI baseline
    direction = str(ds.attrs.get("direction", "min"))
    best_feasible = _best_feasible_observed(ds, direction)

    # 5) Sample candidate pool (respecting bounds + fixed)
    rng = np.random.default_rng(random_seed)
    cand_df = _sample_candidates(search_specs, n=candidates_pool, rng=rng, fixed=fixed_norm)

    # 6) Predict in model space
    Xn_cands = _original_df_to_standardized(cand_df, feature_names, transforms, feat_mean, feat_std)
    p = pred_success(Xn_cands)
    mu, sd = pred_loss(Xn_cands, include_observation_noise=True)
    sd = np.maximum(sd, 1e-12)

    # 7) Acquisition: cEI + exploration + novelty
    mu_ei, best_y_ei = _maybe_flip_for_direction(mu, best_feasible, direction)
    c_ei = _constrained_EI(mu_ei, sd, p, best_y_ei, p_threshold=p_success_threshold, softness=0.05)
    expl = _exploration_score(sd, p, w_sd=1.0, w_boundary=0.5)
    nov  = _novelty_score(Xn_cands, Xn_train)

    n_explore = int(np.ceil(count * explore_fraction))
    n_exploit = max(0, count - n_explore)

    idx_exploit = np.argsort(-c_ei)[:n_exploit]
    chosen = set(idx_exploit.tolist())

    nov_norm = nov / (np.max(nov) + 1e-12)
    score_explore = expl * nov_norm
    for idx in np.argsort(-score_explore):
        if len(chosen) >= count:
            break
        if idx not in chosen:
            chosen.add(int(idx))

    chosen_idx = np.array(sorted(chosen), dtype=int)

    out = cand_df.iloc[chosen_idx].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["pred_p_success"]   = p[chosen_idx]
    out["pred_target_mean"] = mu[chosen_idx]
    out["pred_target_sd"]   = sd[chosen_idx]
    out["acq_cEI"]          = c_ei[chosen_idx]
    out["acq_explore"]      = expl[chosen_idx]
    out["novelty_norm"]     = nov_norm[chosen_idx]
    out["direction"]        = direction
    out["conditioned_on"]   = _fixed_as_string(fixed_norm)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output, index=False)

    console.print(df_to_table(out))

    return out


def optimal(
    model: xr.Dataset | Path | str,
    output: Path|None = None,
    count: int = 10,
    n_draws: int = 2000,
    min_success_probability: float = 0.5,
    random_seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """
    Rank candidates by probability of being the best feasible optimum (min/max),
    optionally conditioned on fixed variables.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    feature_names = list(map(str, ds["feature"].values.tolist()))
    transforms = list(map(str, ds["feature_transform"].values.tolist()))
    feat_mean = ds["feature_mean"].values.astype(float)
    feat_std = ds["feature_std"].values.astype(float)

    # Candidate pool with conditioning
    search_specs = _infer_search_specs(ds, feature_names, transforms)
    fixed_norm = _normalize_fixed(kwargs or {}, search_specs)
    rng = np.random.default_rng(random_seed)
    cand_df = _sample_candidates(search_specs, n=4000, rng=rng, fixed=fixed_norm)
    Xn_cands = _original_df_to_standardized(cand_df, feature_names, transforms, feat_mean, feat_std)

    # Predictions
    p = pred_success(Xn_cands)
    mu, sd = pred_loss(Xn_cands, include_observation_noise=True)
    sd = np.maximum(sd, 1e-12)

    # Optional feasibility filter
    keep = p >= float(min_success_probability)
    if not np.any(keep):
        keep = np.ones_like(p, dtype=bool)

    cand_df = cand_df.loc[keep].reset_index(drop=True)
    Xn_cands = Xn_cands[keep]
    p = p[keep]; mu = mu[keep]; sd = sd[keep]
    N = len(cand_df)

    direction = str(ds.attrs.get("direction", "min"))
    flip = -1.0 if direction == "max" else 1.0

    # Monte Carlo winner-take-all
    rng = np.random.default_rng(random_seed)
    Z = mu[:, None] + sd[:, None] * rng.standard_normal((N, n_draws))
    success_mask = rng.random((N, n_draws)) < p[:, None]
    feasible_draw = success_mask.any(axis=0)
    if not feasible_draw.any():
        result = cand_df.copy()
        result["pred_p_success"] = p
        result["pred_target_mean"] = mu
        result["pred_target_sd"] = sd
        result["prob_best_feasible"] = 0.0
        result["wins"] = 0
        result["n_draws_effective"] = 0
        result["conditioned_on"] = _fixed_as_string(fixed_norm)
        result_sorted = result.sort_values(
            ["pred_target_mean", "pred_target_sd", "pred_p_success"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)
        top = result_sorted.head(count).reset_index(drop=True)
        if output:
            top.to_csv(output, index=False)
        return top

    Z_eff = flip * Z
    Z_eff = np.where(success_mask, Z_eff, np.inf)
    Zf = Z_eff[:, feasible_draw]

    winner_idx = np.argmin(Zf, axis=0)
    counts = np.bincount(winner_idx, minlength=N)
    n_eff = int(feasible_draw.sum())
    prob_best = counts / float(n_eff)

    result = cand_df.copy()
    result["pred_p_success"] = p
    result["pred_target_mean"] = mu
    result["pred_target_sd"] = sd
    result["wins"] = counts
    result["n_draws_effective"] = n_eff
    result["prob_best_feasible"] = prob_best
    result["conditioned_on"] = _fixed_as_string(fixed_norm)

    result_sorted = result.sort_values(
        ["prob_best_feasible", "pred_p_success", "pred_target_mean", "pred_target_sd"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)

    top = result_sorted.head(count).reset_index(drop=True)
    if output:
        top.to_csv(output, index=False)

    console.print(df_to_table(top))

    return top


# =============================================================================
# Predictors reconstructed from artifact (no PyMC at runtime)
# =============================================================================

def _build_predictors(ds: xr.Dataset) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray]]
]:
    """Return (predict_success_probability, predict_conditional_target)."""
    Xn_all = ds["Xn_train"].values.astype(float)
    y_success = ds["y_success"].values.astype(float)  # not used, but handy to keep
    Xn_ok = ds["Xn_success_only"].values.astype(float)
    y_loss_centered = ds["y_loss_centered"].values.astype(float)

    # Success head MAP
    ell_s = ds["map_success_ell"].values.astype(float)
    eta_s = float(ds["map_success_eta"].values)
    sigma_s = float(ds["map_success_sigma"].values)
    beta0_s = float(ds["map_success_beta0"].values)

    # Loss head MAP
    ell_l = ds["map_loss_ell"].values.astype(float)
    eta_l = float(ds["map_loss_eta"].values)
    sigma_l = float(ds["map_loss_sigma"].values)
    mean_c = float(ds["map_loss_mean_const"].values)
    cond_mean = float(ds["conditional_loss_mean"].values)

    # Cholesky precomputations
    K_s = kernel_m52_ard(Xn_all, Xn_all, ell_s, eta_s) + (sigma_s**2) * np.eye(Xn_all.shape[0])
    L_s = np.linalg.cholesky(add_jitter(K_s))
    alpha_s = solve_chol(L_s, (y_success - beta0_s))

    K_l = kernel_m52_ard(Xn_ok, Xn_ok, ell_l, eta_l) + (sigma_l**2) * np.eye(Xn_ok.shape[0])
    L_l = np.linalg.cholesky(add_jitter(K_l))
    alpha_l = solve_chol(L_l, (y_loss_centered - mean_c))

    def predict_success_probability(Xn: np.ndarray) -> np.ndarray:
        Ks = kernel_m52_ard(Xn, Xn_all, ell_s, eta_s)
        mu = beta0_s + Ks @ alpha_s
        return np.clip(mu, 0.0, 1.0)

    def predict_conditional_target(Xn: np.ndarray, include_observation_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        Kl = kernel_m52_ard(Xn, Xn_ok, ell_l, eta_l)
        mu_c = mean_c + Kl @ alpha_l
        mu = mu_c + cond_mean
        v = solve_lower(L_l, Kl.T)
        var = kernel_diag_m52(Xn, ell_l, eta_l) - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
        if include_observation_noise:
            var = var + sigma_l**2
        sd = np.sqrt(var)
        return mu, sd

    return predict_success_probability, predict_conditional_target


# =============================================================================
# Search space, conditioning, and featurization
# =============================================================================

def _infer_search_specs(
    ds: xr.Dataset,
    feature_names: list[str],
    transforms: list[str],
    pad_frac: float = 0.10,
) -> dict[str, dict]:
    """
    Build per-feature search specs from the *original-unit* columns present in the artifact.
    Returns dict: name -> spec, where spec is one of:
      {"type":"float", "lo":float, "hi":float}
      {"type":"int",   "lo":int,   "hi":int, "step":int (optional)}
      {"type":"choice","choices": list[int|float], "dtype":"int"|"float"}
    """
    specs: dict[str, dict] = {}

    df_raw = pd.DataFrame({k: ds[k].values for k in ds.data_vars if ds[k].dims == ("row",)})
    # prefer top-level columns if present
    for name in feature_names:
        if name in df_raw.columns:
            vals = pd.to_numeric(pd.Series(df_raw[name]), errors="coerce").dropna().to_numpy()
        else:
            # fallback: reconstruct original units from standardized arrays if needed
            # (in your artifact, raw columns are stored; so this path is rarely used)
            vals = pd.to_numeric(pd.Series(ds[name].values), errors="coerce").dropna().to_numpy()

        if vals.size == 0:
            # degenerate column; fall back to [0,1]
            specs[name] = {"type": "float", "lo": 0.0, "hi": 1.0}
            continue

        # detect integer-ish
        intish = np.all(np.isfinite(vals)) and np.allclose(vals, np.round(vals))

        # robust bounds with padding
        p1, p99 = np.percentile(vals, [1, 99])
        span = max(p99 - p1, 1e-12)
        lo = p1 - pad_frac * span
        hi = p99 + pad_frac * span

        if intish:
            lo_i = int(np.floor(lo))
            hi_i = int(np.ceil(hi))
            specs[name] = {"type": "int", "lo": lo_i, "hi": hi_i}
        else:
            specs[name] = {"type": "float", "lo": float(lo), "hi": float(hi)}
    return specs


def _normalize_fixed(
    fixed_raw: dict[str, object],
    specs: dict[str, dict],
) -> dict[str, object]:
    """
    Normalize user constraints to sanitized forms within inferred bounds.
    Keeps the *shape*:
      - number (int/float)  -> fixed (clipped to [lo,hi])
      - slice(lo, hi)       -> float range (clipped to [lo,hi])
      - list/tuple          -> finite choices (filtered to within [lo,hi], cast to int for int specs)
    Returns a dict usable directly by _sample_candidates.
    """
    fixed_norm: dict[str, object] = {}

    for name, val in (fixed_raw or {}).items():
        if name not in specs:
            # unknown feature already warned upstream; skip silently here
            continue

        sp = specs[name]
        typ = sp["type"]

        # helper clamps
        def _clip_float(x: float) -> float:
            return float(np.clip(x, sp["lo"], sp["hi"]))

        def _clip_int(x: int) -> int:
            lo, hi = int(sp.get("lo", x)), int(sp.get("hi", x))
            return int(np.clip(int(round(x)), lo, hi))

        # numeric fixed
        if isinstance(val, (int, float, np.number)):
            if typ == "int":
                fixed_norm[name] = _clip_int(int(round(val)))
            elif typ == "choice" and sp.get("dtype") == "int":
                fixed_norm[name] = _clip_int(int(round(val)))
            else:
                fixed_norm[name] = _clip_float(float(val))
            continue

        # float range via slice(lo, hi)
        if isinstance(val, slice):
            lo = float(val.start)
            hi = float(val.stop)
            if lo > hi:
                lo, hi = hi, lo
            if typ in ("float", "choice") and sp.get("dtype") != "int":
                lo_c = _clip_float(lo); hi_c = _clip_float(hi)
                if lo_c > hi_c: lo_c, hi_c = hi_c, lo_c
                fixed_norm[name] = slice(lo_c, hi_c)
            else:
                # int spec: convert to inclusive integer tuple
                lo_i = _clip_int(int(np.floor(lo)))
                hi_i = _clip_int(int(np.ceil(hi)))
                choices = tuple(range(lo_i, hi_i + 1))
                fixed_norm[name] = choices
            continue

        # choices via list/tuple
        if isinstance(val, (list, tuple)):
            if typ in ("int",) or (typ == "choice" and sp.get("dtype") == "int"):
                vv = [ _clip_int(int(round(x))) for x in val ]
                # de-dup and sort
                vv = sorted(set(vv))
                if not vv:
                    # fallback to center
                    center = _clip_int(int(np.round((sp["lo"] + sp["hi"]) / 2)))
                    vv = [center]
                fixed_norm[name] = tuple(vv)
            else:
                vv = [ _clip_float(float(x)) for x in val ]
                vv = sorted(set(vv))
                if not vv:
                    center = _clip_float((sp["lo"] + sp["hi"]) / 2.0)
                    vv = [center]
                # keep list/tuple shape (tuple preferred)
                fixed_norm[name] = tuple(vv)
            continue

        # otherwise: ignore incompatible type
        # (you could raise here if you prefer a hard failure)
    return fixed_norm


def _sample_candidates(
    specs: dict[str, dict],
    n: int,
    rng: np.random.Generator,
    fixed: dict[str, object] | None = None,
) -> pd.DataFrame:
    """
    Sample n candidates in ORIGINAL units given search specs and optional fixed constraints.
    """
    fixed = fixed or {}
    cols: dict[str, np.ndarray] = {}

    for name, sp in specs.items():
        typ = sp["type"]

        # If fixed: honor numeric / slice / choices shape
        if name in fixed:
            val = fixed[name]

            # numeric: constant column
            if isinstance(val, (int, float, np.number)):
                cols[name] = np.full(n, val, dtype=float)

            # float range slice
            elif isinstance(val, slice):
                lo = float(val.start); hi = float(val.stop)
                if lo > hi: lo, hi = hi, lo
                cols[name] = rng.uniform(lo, hi, size=n)

            # choices: list/tuple -> sample from set
            elif isinstance(val, (list, tuple)):
                arr = np.array(val, dtype=float)
                if arr.size == 0:
                    # fallback to center of spec
                    if typ == "int":
                        center = int(np.round((sp["lo"] + sp["hi"]) / 2))
                        arr = np.array([center], dtype=float)
                    else:
                        center = (sp["lo"] + sp["hi"]) / 2.0
                        arr = np.array([center], dtype=float)
                idx = rng.integers(0, len(arr), size=n)
                cols[name] = arr[idx]

            else:
                # unknown fixed type; fallback to spec sampling
                if typ == "choice":
                    choices = np.asarray(sp["choices"], dtype=float)
                    idx = rng.integers(0, len(choices), size=n)
                    cols[name] = choices[idx]
                elif typ == "int":
                    cols[name] = rng.integers(int(sp["lo"]), int(sp["hi"]) + 1, size=n).astype(float)
                else:
                    cols[name] = rng.uniform(sp["lo"], sp["hi"], size=n)

        else:
            # Not fixed: sample from spec
            if typ == "choice":
                choices = np.asarray(sp["choices"], dtype=float)
                idx = rng.integers(0, len(choices), size=n)
                cols[name] = choices[idx]
            elif typ == "int":
                cols[name] = rng.integers(int(sp["lo"]), int(sp["hi"]) + 1, size=n).astype(float)
            else:
                cols[name] = rng.uniform(sp["lo"], sp["hi"], size=n)

    df = pd.DataFrame(cols)
    # ensure integer columns are ints if the spec says so (pretty output)
    for name, sp in specs.items():
        if sp["type"] == "int" or (sp["type"] == "choice" and sp.get("dtype") == "int"):
            df[name] = df[name].round().astype(int)
    return df


def _original_df_to_standardized(
    df: pd.DataFrame,
    feature_names: list[str],
    transforms: list[str],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
) -> np.ndarray:
    cols = []
    for j, name in enumerate(feature_names):
        x = df[name].to_numpy().astype(float)
        tr = transforms[j]
        if tr == "log10":
            x = np.where(x <= 0, np.nan, x)
            x = np.log10(x)
        cols.append((x - feat_mean[j]) / feat_std[j])
    return np.column_stack(cols).astype(float)


# =============================================================================
# Acquisition functions & utilities
# =============================================================================

def _expected_improvement_minimize(mu: np.ndarray, sd: np.ndarray, best_y: float) -> np.ndarray:
    sd = np.maximum(sd, 1e-12)
    z = (best_y - mu) / sd
    Phi = ndtr(z)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return sd * (z * Phi + phi)


def _constrained_EI(mu: np.ndarray, sd: np.ndarray, p_success: np.ndarray, best_y: float,
                    p_threshold: float = 0.8, softness: float = 0.05) -> np.ndarray:
    ei = _expected_improvement_minimize(mu, sd, best_y)
    s = 1.0 / (1.0 + np.exp(-(p_success - p_threshold) / max(softness, 1e-6)))
    return ei * s


def _exploration_score(sd_loss: np.ndarray, p_success: np.ndarray,
                       w_sd: float = 1.0, w_boundary: float = 0.5) -> np.ndarray:
    return w_sd * sd_loss + w_boundary * (p_success * (1.0 - p_success))


def _novelty_score(Xn_cands: np.ndarray, Xn_seen: np.ndarray) -> np.ndarray:
    m = Xn_cands.shape[0]
    batch = 1024
    out = np.empty(m, dtype=float)
    for i in range(0, m, batch):
        sl = slice(i, min(i + batch, m))
        diff = Xn_cands[sl, None, :] - Xn_seen[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        out[sl] = np.min(d, axis=1)
    return out


def _maybe_flip_for_direction(mu: np.ndarray, best_y: float, direction: str) -> tuple[np.ndarray, float]:
    if direction == "max":
        return -mu, -best_y
    return mu, best_y


def _best_feasible_observed(ds: xr.Dataset, direction: str) -> float:
    y_ok = ds["y_loss_success"].values.astype(float)
    if y_ok.size == 0:
        return np.inf if direction != "max" else -np.inf
    if direction == "max":
        return float(np.nanmax(y_ok))
    return float(np.nanmin(y_ok))


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _fmt_num(x) -> str:
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _fixed_as_string(fixed: dict) -> str:
    """
    Human-readable constraints:
      - number  -> k=12 or k=0.00123
      - slice   -> k=lo:hi   (inclusive; None shows as -inf/inf)
      - list/tuple -> k=[v1, v2, ...]
      - range   -> k=[start, stop, step]  (rare; usually normalized earlier)
      - other scalars (str/bool) -> k=value
    Keys are sorted for stability.
    """
    parts: list[str] = []
    for k in sorted(fixed.keys()):
        v = fixed[k]
        if isinstance(v, slice):
            a = "-inf" if v.start is None else _fmt_num(v.start)
            b =  "inf" if v.stop  is None else _fmt_num(v.stop)
            parts.append(f"{k}={a}:{b}")
        elif isinstance(v, range):
            parts.append(f"{k}=[{', '.join(_fmt_num(u) for u in (v.start, v.stop, v.step))}]")
        elif isinstance(v, (list, tuple, np.ndarray)):
            elems = ", ".join(_fmt_num(u) if _is_number(u) else str(u) for u in v)
            parts.append(f"{k}=[{elems}]")
        elif _is_number(v):
            parts.append(f"{k}={_fmt_num(v)}")
        else:
            # fallback for str/bool/other scalars
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _apply_user_bounds(
    specs: dict[str, dict[str, Any]],
    ranges: dict[str, tuple[float, float]],
    choices: dict[str, list[float]],
) -> None:
    """
    Mutate `specs` with user-provided bounds/choices.
    """
    for name, (lo, hi) in ranges.items():
        if name not in specs:
            continue
        sp = specs[name]
        sp["kind"] = sp.get("kind", "float")
        if sp["kind"] == "choice":
            # Convert to float/int range if user provided range for a choice var
            sp["kind"] = "float"
        sp["low"] = float(lo)
        sp["high"] = float(hi)
        sp.pop("choices", None)

    for name, opts in choices.items():
        if name not in specs:
            continue
        sp = specs[name]
        # Keep kind="choice" and store list
        sp["kind"] = "choice"
        # Cast ints if all values are close to ints
        if all(abs(v - round(v)) < 1e-12 for v in opts):
            sp["choices"] = [int(round(v)) for v in opts]
        else:
            sp["choices"] = [float(v) for v in opts]
        # Drop bounds (not used for choice)
        sp.pop("low", None)
        sp.pop("high", None)
