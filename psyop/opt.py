# opt.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtr  # Φ(z), vectorized


# ======================================================================================
# Public API
# ======================================================================================

def suggest_candidates(
    model_path: Path,
    output_path: Path,
    count: int = 12,
    p_success_threshold: float = 0.8,
    explore_fraction: float = 0.34,
    candidates_pool: int = 5000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Propose a batch of candidates using constrained Expected Improvement (cEI)
    plus exploration (uncertainty + boundary + novelty).
    """
    ds = xr.load_dataset(model_path)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = list(map(str, ds["feature"].values.tolist()))
    transforms = list(map(str, ds["feature_transform"].values.tolist()))
    feat_mean = ds["feature_mean"].values.astype(float)
    feat_std = ds["feature_std"].values.astype(float)
    Xn_train = ds["Xn_train"].values.astype(float)

    # Determine search spaces from raw columns
    search_specs = _infer_search_specs(ds, feature_names, transforms)

    # Best feasible observed target (respect direction)
    target_name = str(ds.attrs.get("target_column", "target"))
    direction = str(ds.attrs.get("direction", "min"))
    best_feasible = _best_feasible_observed(ds, direction)

    # Sample candidate pool in ORIGINAL units
    rng = np.random.default_rng(random_seed)
    cand_df = _sample_candidates(search_specs, n=candidates_pool, rng=rng)

    # Standardize to model space
    Xn_cands = _original_df_to_standardized(cand_df, feature_names, transforms, feat_mean, feat_std)

    # Model predictions
    p = pred_success(Xn_cands)  # (N,)
    mu, sd = pred_loss(Xn_cands, include_observation_noise=True)  # (N,), (N,)
    sd = np.maximum(sd, 1e-12)

    # Scores
    mu_ei, best_y_ei = _maybe_flip_for_direction(mu, best_feasible, direction)
    c_ei = _constrained_EI(mu_ei, sd, p, best_y_ei, p_threshold=p_success_threshold, softness=0.05)
    expl = _exploration_score(sd, p, w_sd=1.0, w_boundary=0.5)
    nov = _novelty_score(Xn_cands, Xn_train)

    n_explore = int(np.ceil(count * explore_fraction))
    n_exploit = max(0, count - n_explore)

    # 1) Exploit: top by cEI
    idx_exploit = np.argsort(-c_ei)[:n_exploit]
    chosen = set(idx_exploit.tolist())

    # 2) Explore: rank by exploration * normalized novelty
    nov_norm = nov / (np.max(nov) + 1e-12)
    score_explore = expl * nov_norm
    idx_explore_sorted = np.argsort(-score_explore)
    for idx in idx_explore_sorted:
        if len(chosen) >= count:
            break
        if idx not in chosen:
            chosen.add(int(idx))

    chosen_idx = np.array(sorted(chosen), dtype=int)
    out = cand_df.iloc[chosen_idx].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["pred_p_success"] = p[chosen_idx]
    out["pred_target_mean"] = mu[chosen_idx]
    out["pred_target_sd"] = sd[chosen_idx]
    out["acq_cEI"] = c_ei[chosen_idx]
    out["acq_explore"] = expl[chosen_idx]
    out["novelty_norm"] = nov_norm[chosen_idx]
    out["target_name"] = target_name
    out["direction"] = direction

    out.to_csv(output_path, index=False)
    return out


def find_optimal(
    model_path: Path,
    output_path: Path,
    top_k: int = 10,
    n_draws: int = 2000,
    min_success_probability: float = 0.0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Rank candidates by probability of being the best feasible minimum/maximum.
    Uses Monte-Carlo winner frequency with feasibility and direction handling.
    """
    ds = xr.load_dataset(model_path)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = list(map(str, ds["feature"].values.tolist()))
    transforms = list(map(str, ds["feature_transform"].values.tolist()))
    feat_mean = ds["feature_mean"].values.astype(float)
    feat_std = ds["feature_std"].values.astype(float)

    # Candidate pool (a bit larger for stability)
    search_specs = _infer_search_specs(ds, feature_names, transforms)
    rng = np.random.default_rng(random_seed)
    cand_df = _sample_candidates(search_specs, n=4000, rng=rng)

    Xn_cands = _original_df_to_standardized(cand_df, feature_names, transforms, feat_mean, feat_std)

    p = pred_success(Xn_cands)
    mu, sd = pred_loss(Xn_cands, include_observation_noise=True)
    sd = np.maximum(sd, 1e-12)

    # Optional hard feasibility filter
    keep = p >= float(min_success_probability)
    if not np.any(keep):
        keep = np.ones_like(p, dtype=bool)

    cand_df = cand_df.loc[keep].reset_index(drop=True)
    Xn_cands = Xn_cands[keep]
    p = p[keep]; mu = mu[keep]; sd = sd[keep]
    N = len(cand_df)

    direction = str(ds.attrs.get("direction", "min"))
    # For MC "winner": we minimize an effective score; flip sign if searching for maximum
    flip = -1.0 if direction == "max" else 1.0

    # Monte Carlo draws
    rng = np.random.default_rng(random_seed)
    Z = mu[:, None] + sd[:, None] * rng.standard_normal((N, n_draws))  # (N, M)
    success_mask = rng.random((N, n_draws)) < p[:, None]
    feasible_mask = success_mask.any(axis=0)
    if not feasible_mask.any():
        # Fallback: no feasible draws; rank deterministically
        result = cand_df.copy()
        result["pred_p_success"] = p
        result["pred_target_mean"] = mu
        result["pred_target_sd"] = sd
        result["prob_best_feasible"] = 0.0
        result["wins"] = 0
        result["n_draws_effective"] = 0
        result_sorted = result.sort_values(
            ["pred_target_mean", "pred_target_sd", "pred_p_success"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)
        top = result_sorted.head(top_k).reset_index(drop=True)
        top.to_csv(output_path, index=False)
        return top

    Z_eff = flip * Z  # minimize effective score
    Z_eff = np.where(success_mask, Z_eff, np.inf)
    Zf = Z_eff[:, feasible_mask]  # (N, M_eff)

    winner_idx = np.argmin(Zf, axis=0)
    counts = np.bincount(winner_idx, minlength=N)
    n_eff = int(feasible_mask.sum())
    prob_best = counts / float(n_eff)

    result = cand_df.copy()
    result["pred_p_success"] = p
    result["pred_target_mean"] = mu
    result["pred_target_sd"] = sd
    result["wins"] = counts
    result["n_draws_effective"] = n_eff
    result["prob_best_feasible"] = prob_best

    # Unique ranking with sensible tie-breakers
    result_sorted = result.sort_values(
        ["prob_best_feasible", "pred_p_success", "pred_target_mean", "pred_target_sd"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)

    top = result_sorted.head(top_k).reset_index(drop=True)
    top.to_csv(output_path, index=False)
    return top


# ======================================================================================
# Predictors reconstructed from artifact (no PyMC dependency at runtime)
# ======================================================================================

def _build_predictors(ds: xr.Dataset) -> Tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, bool], Tuple[np.ndarray, np.ndarray]]
]:
    """Return (predict_success_probability, predict_conditional_target)."""

    # Training matrices
    Xn_all = ds["Xn_train"].values.astype(float)               # (N, p)
    y_success = ds["y_success"].values.astype(float)           # (N,)
    ok_idx = ds["row_success"].values.astype(int)              # indices into row
    Xn_ok = ds["Xn_success_only"].values.astype(float)         # (N_ok, p)
    y_loss_centered = ds["y_loss_centered"].values.astype(float)

    # Success head MAP params
    ell_s = ds["map_success_ell"].values.astype(float)         # (p,)
    eta_s = float(ds["map_success_eta"].values)
    sigma_s = float(ds["map_success_sigma"].values)
    beta0_s = float(ds["map_success_beta0"].values)

    # Loss head MAP params
    ell_l = ds["map_loss_ell"].values.astype(float)            # (p,)
    eta_l = float(ds["map_loss_eta"].values)
    sigma_l = float(ds["map_loss_sigma"].values)
    mean_c = float(ds["map_loss_mean_const"].values)
    cond_mean = float(ds["conditional_loss_mean"].values)

    # Precompute Cholesky factors for both heads
    K_s = _kernel_m52_ard(Xn_all, Xn_all, ell_s, eta_s) + (sigma_s**2) * np.eye(Xn_all.shape[0])
    L_s = np.linalg.cholesky(_add_jitter(K_s))
    alpha_s = _solve_chol(L_s, (y_success - beta0_s))

    K_l = _kernel_m52_ard(Xn_ok, Xn_ok, ell_l, eta_l) + (sigma_l**2) * np.eye(Xn_ok.shape[0])
    L_l = np.linalg.cholesky(_add_jitter(K_l))
    alpha_l = _solve_chol(L_l, (y_loss_centered - mean_c))

    def predict_success_probability(Xn: np.ndarray) -> np.ndarray:
        Ks = _kernel_m52_ard(Xn, Xn_all, ell_s, eta_s)                  # (M, N)
        mu = beta0_s + Ks @ alpha_s
        # predictive variance not needed downstream for success; clip to [0,1]
        return np.clip(mu, 0.0, 1.0)

    def predict_conditional_target(Xn: np.ndarray, include_observation_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        Kl = _kernel_m52_ard(Xn, Xn_ok, ell_l, eta_l)                   # (M, N_ok)
        # mean
        mu_c = mean_c + Kl @ alpha_l                                    # centered mean
        mu = mu_c + cond_mean                                           # uncentered mean
        # variance
        v = _solve_lower(L_l, Kl.T)                                     # (N_ok, M)
        var = _kernel_diag_m52(Xn, ell_l, eta_l) - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
        if include_observation_noise:
            var = var + sigma_l**2
        sd = np.sqrt(var)
        return mu, sd

    return predict_success_probability, predict_conditional_target


# ======================================================================================
# Search space & featurization
# ======================================================================================

def _infer_search_specs(ds: xr.Dataset, feature_names: List[str], transforms: List[str]) -> List[Dict]:
    """
    Build a list of per-feature sampling specs from observed data.
    Each spec: {
        'name': str, 'transform': 'identity'|'log10',
        'kind': 'continuous'|'integer'|'discrete',
        'bounds': (lo, hi) in ORIGINAL units (for cont/int),
        'choices': np.ndarray (for discrete)
    }
    """
    specs: List[Dict] = []
    for j, name in enumerate(feature_names):
        tr = str(transforms[j])
        raw = ds[name].values
        # Drop NaNs
        raw = raw[np.isfinite(raw)]
        if raw.size == 0:
            # Fallback: use standardized stats to back-solve rough original range
            lo, hi = -2.0, 2.0
            if tr == "log10":
                lo = 1e-6
                hi = 1.0
            specs.append({"name": name, "transform": tr, "kind": "continuous", "bounds": (float(lo), float(hi))})
            continue

        # Detect discrete grid
        uniq = np.unique(raw[~np.isnan(raw)])
        is_int_like = np.all(np.isclose(uniq, np.round(uniq)))
        few_unique = uniq.size <= 20

        if is_int_like and few_unique:
            specs.append({"name": name, "transform": tr, "kind": "discrete", "choices": uniq.astype(float)})
            continue

        # Numeric continuous/integer with bounds from 1–99% + 10% padding
        p1, p99 = np.percentile(raw, [1, 99])
        span = p99 - p1
        lo = p1 - 0.10 * span
        hi = p99 + 0.10 * span

        # For log10 variables, ensure positive and keep in ORIGINAL units
        if tr == "log10":
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 1.0001)

        kind = "integer" if is_int_like else "continuous"
        specs.append({"name": name, "transform": tr, "kind": kind, "bounds": (float(lo), float(hi))})
    return specs


def _sample_candidates(specs: List[Dict], n: int, rng: np.random.Generator) -> pd.DataFrame:
    cols = {}
    for spec in specs:
        name = spec["name"]
        kind = spec["kind"]
        tr = spec["transform"]
        if kind == "discrete":
            choices = spec["choices"]
            cols[name] = rng.choice(choices, size=n)
        else:
            lo, hi = spec["bounds"]
            if tr == "log10":
                # log-uniform in ORIGINAL units
                lo, hi = max(lo, 1e-12), max(hi, lo * 1.0001)
                cols[name] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), size=n)
            else:
                if kind == "integer":
                    cols[name] = rng.integers(int(np.floor(lo)), int(np.ceil(hi)) + 1, size=n)
                else:
                    cols[name] = rng.uniform(lo, hi, size=n)

    return pd.DataFrame(cols)


def _original_df_to_standardized(
    df: pd.DataFrame,
    feature_names: List[str],
    transforms: List[str],
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
        # identity otherwise
        cols.append((x - feat_mean[j]) / feat_std[j])
    return np.column_stack(cols).astype(float)


# ======================================================================================
# Acquisition functions & utilities
# ======================================================================================

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
    # min L2 distance to any seen point (normalized space)
    # simple blocking to reduce peak memory if needed
    m = Xn_cands.shape[0]
    batch = 1024
    out = np.empty(m, dtype=float)
    for i in range(0, m, batch):
        sl = slice(i, min(i + batch, m))
        diff = Xn_cands[sl, None, :] - Xn_seen[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        out[sl] = np.min(d, axis=1)
    return out


def _maybe_flip_for_direction(mu: np.ndarray, best_y: float, direction: str) -> Tuple[np.ndarray, float]:
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


# ======================================================================================
# GP kernel & linear algebra
# ======================================================================================

def _kernel_m52_ard(XA: np.ndarray, XB: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
    """
    Matern 5/2 with ARD lengthscales. XA: (N, p), XB: (M, p)
    k(r) = η^2 (1 + √5 r + 5 r^2 / 3) exp(-√5 r), r = sqrt(sum(((x_a - x_b)/ls)^2))
    """
    XA = np.asarray(XA, float)
    XB = np.asarray(XB, float)
    ls = np.asarray(ls, float).reshape(1, 1, -1)
    diff = (XA[:, None, :] - XB[None, :, :]) / ls
    r2 = np.sum(diff * diff, axis=2)
    r = np.sqrt(np.maximum(r2, 0.0))
    sqrt5_r = np.sqrt(5.0) * r
    k = (eta ** 2) * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * np.exp(-sqrt5_r)
    return k


def _kernel_diag_m52(XA: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
    return np.full(XA.shape[0], eta ** 2, dtype=float)


def _add_jitter(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # add tiny jitter proportional to average diagonal
    jitter = eps * float(np.mean(np.diag(K)) + 1.0)
    return K + jitter * np.eye(K.shape[0], dtype=K.dtype)


def _solve_chol(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Solve (L L^T) x = b
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Solve L V = B for V (where L lower triangular), return V
    return np.linalg.solve(L, B)
