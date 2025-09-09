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

from pathlib import Path
import base64
import pickle
import json

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from rich.console import Console
from rich.table import Table

from .util import get_rng, df_to_table

def _safe_vec(ds: xr.Dataset, name: str, nF: int) -> np.ndarray:
    if name not in ds:
        return np.full(nF, np.nan)
    arr = np.asarray(ds[name].values)
    if arr.size == nF:
        return arr
    out = np.full(nF, np.nan)
    out[:min(nF, arr.size)] = arr.ravel()[:min(nF, arr.size)]
    return out

def _safe_scalar(ds: xr.Dataset, name: str) -> float:
    if name in ds:
        try: return float(np.asarray(ds[name].values).item())
        except Exception: return float(np.nan)
    if name in ds.attrs:
        try: return float(ds.attrs[name])
        except Exception: return float(np.nan)
    return float(np.nan)

def _diagnostic_feature_dataframe(ds: xr.Dataset, top_k: int = 20) -> pd.DataFrame:
    """Top-K features by |corr_loss_success| without any repeated globals."""
    f = [str(x) for x in ds["feature"].values]
    nF = len(f)

    is_oh = ds["feature_is_onehot_member"].values.astype(int) if "feature_is_onehot_member" in ds else np.zeros(nF, int)
    base  = ds["feature_onehot_base"].values if "feature_onehot_base" in ds else np.array([""]*nF, object)

    df = pd.DataFrame({
        "feature": f,
        "type": np.where(is_oh == 1, "categorical(one-hot)", "numeric"),
        "onehot_base": np.where(is_oh == 1, base, ""),
        "n_unique_raw": _safe_vec(ds, "n_unique_raw", nF),
        "raw_min": _safe_vec(ds, "raw_min", nF),
        "raw_max": _safe_vec(ds, "raw_max", nF),
        "Xn_span": _safe_vec(ds, "Xn_span", nF),
        "ell_s": _safe_vec(ds, "map_success_ell", nF),
        "ell_l": _safe_vec(ds, "map_loss_ell", nF),
        "ell/span_s": _safe_vec(ds, "ell_over_span_success", nF),
        "ell/span_l": _safe_vec(ds, "ell_over_span_loss", nF),
        "corr_success": _safe_vec(ds, "corr_success", nF),
        "corr_loss_success": _safe_vec(ds, "corr_loss_success", nF),
    })
    df["|corr_loss_success|"] = np.abs(df["corr_loss_success"].astype(float))
    return df.sort_values("|corr_loss_success|", ascending=False, kind="mergesort").head(top_k).reset_index(drop=True)

def _diagnostic_global_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Single-row DataFrame of global/model-level diagnostics."""
    # Scalars from attrs
    target          = ds.attrs.get("target", "")
    direction       = ds.attrs.get("direction", "")
    n_rows          = int(ds.attrs.get("n_rows", np.nan))
    n_success_rows  = int(ds.attrs.get("n_success_rows", np.nan)) if "n_success_rows" in ds.attrs else int(np.sum(np.asarray(ds["success_mask"].values)) if "success_mask" in ds else np.nan)
    success_rate    = float(ds.attrs.get("success_rate", np.nan))
    rng_bitgen      = ds.attrs.get("rng_bitgen", "")
    numpy_version   = ds.attrs.get("numpy_version", "")
    pymc_version    = ds.attrs.get("pymc_version", "")

    # Scalars from data_vars (with attr fallback)
    conditional_loss_mean = _safe_scalar(ds, "conditional_loss_mean")

    # GP MAP scalars
    map_success_eta   = _safe_scalar(ds, "map_success_eta")
    map_success_sigma = _safe_scalar(ds, "map_success_sigma")
    map_success_beta0 = _safe_scalar(ds, "map_success_beta0")

    map_loss_eta      = _safe_scalar(ds, "map_loss_eta")
    map_loss_sigma    = _safe_scalar(ds, "map_loss_sigma")
    map_loss_mean_c   = _safe_scalar(ds, "map_loss_mean_const")

    # Data dispersion on successes
    y_ls_std = float(np.nanstd(ds["y_loss_success"].values)) if "y_loss_success" in ds else np.nan

    # Handy amplitude/noise ratio for loss head
    eta_l_over_sigma_l = (map_loss_eta / map_loss_sigma) if (np.isfinite(map_loss_eta) and np.isfinite(map_loss_sigma) and map_loss_sigma != 0) else np.nan

    rows = [{
        "target": target,
        "direction": direction,
        "n_rows": n_rows,
        "n_success_rows": n_success_rows,
        "success_rate": success_rate,
        "conditional_loss_mean": conditional_loss_mean,
        "map_success_eta": map_success_eta,
        "map_success_sigma": map_success_sigma,
        "map_success_beta0": map_success_beta0,
        "map_loss_eta": map_loss_eta,
        "map_loss_sigma": map_loss_sigma,
        "map_loss_mean_const": map_loss_mean_c,
        "y_loss_success_std": y_ls_std,
        "eta_l/sigma_l": eta_l_over_sigma_l,
        "rng_bitgen": rng_bitgen,
        "numpy_version": numpy_version,
        "pymc_version": pymc_version,
    }]
    return pd.DataFrame(rows)



def _print_diagnostics_table(ds: xr.Dataset, top_k: int = 20) -> None:
    """3-sigfig Rich table with key diagnostics."""
    feat_df  = _diagnostic_feature_dataframe(ds, top_k=20)
    global_df = _diagnostic_global_dataframe(ds)

    console = Console()

    console.print("\n[bold]Model diagnostics (top by |corr_loss_success|):[/]")
    console.print(df_to_table(feat_df, transpose=False, show_index=False))  # regular table

    console.print("\n[bold]Model globals:[/]")
    # transpose for key/value look, with magenta header column
    console.print(df_to_table(global_df, transpose=True))


def build_model(
    input: pd.DataFrame|Path|str,
    target: str,
    output: Path | str | None = None,
    exclude: list[str] | str | None = None,
    direction: str = "auto",
    seed: int | np.random.Generator | None = 42,
    compress: bool = True,
) -> xr.Dataset:
    """
    Fit two-head GP (success prob + conditional loss) and save a single NetCDF artifact.
    Also stores rich diagnostics to help debug flat PD curves and model wiring.
    """
    # ---------- Load ----------
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        input_path = Path(input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")
        df = pd.read_csv(input_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV.")

    # Keep a raw copy for artifact (strings as pandas 'string' so NetCDF can handle them)
    df_raw = df.copy()
    for c in df_raw.columns:
        if df_raw[c].dtype == object:
            df_raw[c] = df_raw[c].astype("string")

    # ---------- Success inference ----------
    success = (~df[target].isna()).to_numpy().astype(int)
    has_success = bool(np.any(success == 1))
    if not has_success:
        raise RuntimeError("No successful rows detected (cannot fit conditional-loss GP).")

    # ---------- Feature selection ----------
    reserved_internals = {"__success__", "__fail__", "__status__"}
    exclude = [exclude] if isinstance(exclude, str) else (exclude or [])
    excluded = set(exclude) | {target} | reserved_internals

    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [
        c for c in df.columns
        if c not in excluded
        and (pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))
    ]

    feature_names: list[str] = []
    transforms: list[str] = []
    X_raw_cols: list[np.ndarray] = []

    # Diagnostics to fill
    onehot_base_per_feature: list[str] = []  # "" for numeric / non-onehot
    is_onehot_member: list[int] = []
    categorical_groups: dict[str, dict] = {}  # base -> {"labels":[...], "members":[...]}

    # numeric features
    for name in numeric_cols:
        col = df[name].to_numpy(dtype=float)
        tr = _choose_transform(name, col)
        transforms.append(tr)
        feature_names.append(name)
        X_raw_cols.append(_apply_transform(tr, col))
        onehot_base_per_feature.append("")
        is_onehot_member.append(0)

    # categoricals → one-hot
    for base in cat_cols:
        s_cat = pd.Categorical(df[base].astype("string").fillna("<NA>"))
        H = pd.get_dummies(s_cat, prefix=base, prefix_sep="=", dtype=float)  # e.g., language=Linear A
        members = []
        labels = []
        for new_col in H.columns:
            feature_names.append(new_col)
            transforms.append("identity")
            X_raw_cols.append(H[new_col].to_numpy(dtype=float))
            onehot_base_per_feature.append(base)
            is_onehot_member.append(1)
            members.append(new_col)
            # label is the part after "base="
            labels.append(str(new_col.split("=", 1)[1]) if "=" in new_col else str(new_col))
        categorical_groups[base] = {"labels": labels, "members": members}

    X_raw = np.column_stack(X_raw_cols).astype(float)
    n, p = X_raw.shape

    # ---------- Standardize ----------
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0)
    X_std  = np.where(X_std == 0.0, 1.0, X_std)  # keep inert dims harmless
    Xn = (X_raw - X_mean) / X_std

    # Targets
    y_success = success.astype(float)
    ok_mask = (success == 1)
    y_loss_success = df.loc[ok_mask, target].to_numpy(dtype=float)

    conditional_loss_mean = float(np.nanmean(y_loss_success)) if len(y_loss_success) else 0.0
    y_loss_centered = y_loss_success - conditional_loss_mean
    Xn_success_only = Xn[ok_mask, :]

    rng = get_rng(seed)
    state_bytes = pickle.dumps(rng.bit_generator.state)
    rng_state_b64 = base64.b64encode(state_bytes).decode("ascii")

    base_success_rate = float(y_success.mean())

    # ---------- Fit Head A: success ----------
    with pm.Model() as model_s:
        ell_s   = pm.HalfNormal("ell",   sigma=2.0, shape=p)
        eta_s   = pm.HalfNormal("eta",   sigma=2.0)
        sigma_s = pm.HalfNormal("sigma", sigma=0.3)
        beta0_s = pm.Normal("beta0", mu=base_success_rate, sigma=0.15)

        K_s = eta_s**2 * pm.gp.cov.Matern52(input_dim=p, ls=ell_s)
        m_s = pm.gp.mean.Constant(beta0_s)
        gp_s = pm.gp.Marginal(mean_func=m_s, cov_func=K_s)

        _ = gp_s.marginal_likelihood("y_obs_s", X=Xn, y=y_success, sigma=sigma_s)
        map_s = pm.find_MAP()

    with model_s:
        mu_s, var_s = gp_s.predict(Xn, point=map_s, diag=True, pred_noise=True)
    mu_s = np.clip(mu_s, 0.0, 1.0)

    # ---------- Fit Head B: conditional loss (success-only) ----------
    if Xn_success_only.shape[0] == 0:
        raise RuntimeError("No successful rows to fit the conditional-loss GP.")

    with pm.Model() as model_l:
        ell_l = pm.TruncatedNormal("ell", mu=1.0, sigma=0.5, lower=0.1, shape=p)
        eta_l = pm.HalfNormal("eta", sigma=1.0)
        sigma_l = pm.HalfNormal("sigma", sigma=1.0)
        mean_c = pm.Normal("mean_const", mu=0.0, sigma=10.0)

        K_l = eta_l**2 * pm.gp.cov.Matern52(input_dim=p, ls=ell_l)
        m_l = pm.gp.mean.Constant(mean_c)
        gp_l = pm.gp.Marginal(mean_func=m_l, cov_func=K_l)

        _ = gp_l.marginal_likelihood("y_obs", X=Xn_success_only, y=y_loss_centered, sigma=sigma_l)
        map_l = pm.find_MAP()

    with model_l:
        mu_l_c, var_l = gp_l.predict(Xn_success_only, point=map_l, diag=True, pred_noise=True)
    mu_l = mu_l_c + conditional_loss_mean
    sd_l = np.sqrt(var_l)

    # ---------- Diagnostics (per-feature) ----------
    # raw stats in ORIGINAL units (before any transform)
    raw_stats = {
        "raw_min":  [], "raw_max":  [], "raw_mean": [], "raw_std":  [], "n_unique_raw": [],
    }
    for k, name in enumerate(feature_names):
        # Try to recover a raw column if present; otherwise invert transform on X_raw[:,k]
        if name in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[name]):
            raw_col = df_raw[name].to_numpy(dtype=float)
        else:
            # Inverse transform of "internal" values only if it's a simple one:
            internal = X_raw[:, k]
            tr = transforms[k]
            if tr == "log10":
                raw_col = np.power(10.0, internal)
            else:
                raw_col = internal
        x = np.asarray(raw_col, dtype=float)
        x_finite = x[np.isfinite(x)]
        if x_finite.size == 0:
            x_finite = np.array([np.nan])
        raw_stats["raw_min"].append(float(np.nanmin(x_finite)))
        raw_stats["raw_max"].append(float(np.nanmax(x_finite)))
        raw_stats["raw_mean"].append(float(np.nanmean(x_finite)))
        raw_stats["raw_std"].append(float(np.nanstd(x_finite)))
        raw_stats["n_unique_raw"].append(int(np.unique(np.round(x_finite, 12)).size))

    # internal (transformed) stats PRIOR to standardization
    internal_min = np.nanmin(X_raw, axis=0)
    internal_max = np.nanmax(X_raw, axis=0)
    internal_mean = np.nanmean(X_raw, axis=0)
    internal_std  = np.nanstd(X_raw, axis=0)

    # standardized 1–99% span (what PD uses)
    Xn_p01 = np.percentile(Xn, 1, axis=0)
    Xn_p99 = np.percentile(Xn, 99, axis=0)
    Xn_span = Xn_p99 - Xn_p01
    Xn_span = np.where(np.isfinite(Xn_span), Xn_span, np.nan)

    # lengthscale-to-span ratios (big → likely flat)
    ell_s_arr = _np1d(map_s["ell"], p)
    ell_l_arr = _np1d(map_l["ell"], p)
    with np.errstate(divide="ignore", invalid="ignore"):
        ell_over_span_success = np.where(Xn_span > 0, ell_s_arr / Xn_span, np.nan)
        ell_over_span_loss    = np.where(Xn_span > 0, ell_l_arr / Xn_span, np.nan)

    # simple correlations (training)
    def _safe_corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return np.nan
        va = np.var(a[m]); vb = np.var(b[m])
        if va == 0 or vb == 0:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0,1])

    corr_success = np.array([_safe_corr(X_raw[:, j], y_success) for j in range(p)], dtype=float)
    corr_loss_success = np.array(
        [_safe_corr(X_raw[ok_mask, j], y_loss_success) if ok_mask.any() else np.nan for j in range(p)],
        dtype=float
    )

    # ---------- Build xarray Dataset ----------
    feature_coord = xr.DataArray(np.array(feature_names, dtype=object), dims=("feature",), name="feature")
    row_coord = xr.DataArray(np.arange(n, dtype=np.int64), dims=("row",), name="row")
    row_ok_coord = xr.DataArray(np.where(ok_mask)[0].astype(np.int64), dims=("row_success",), name="row_success")

    # Raw columns (strings and numerics) — from df_raw
    raw_vars = {}
    for col in df_raw.columns:
        s = df_raw[col]
        if (
            pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_float_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            raw_vars[col] = (("row",), s.to_numpy())
        else:
            s_str = s.astype("string").fillna("<NA>")
            vals = s_str.to_numpy(dtype="U")
            raw_vars[col] = (("row",), vals)

    ds = xr.Dataset(
        data_vars={
            # Design matrices (standardized)
            "Xn_train": (("row", "feature"), Xn),
            "Xn_success_only": (("row_success", "feature"), Xn_success_only),

            # Targets
            "y_success": (("row",), y_success),
            "y_loss_success": (("row_success",), y_loss_success),
            "y_loss_centered": (("row_success",), y_loss_centered),

            # Standardization + transforms
            "feature_mean": (("feature",), X_mean),
            "feature_std":  (("feature",), X_std),
            "feature_transform": (("feature",), np.array(transforms, dtype=object)),

            # Masks / indexing
            "success_mask": (("row",), ok_mask.astype(np.int8)),

            # Head A (success) MAP params
            "map_success_ell":   (("feature",), ell_s_arr),
            "map_success_eta":   ((), float(np.asarray(map_s["eta"]))),
            "map_success_sigma": ((), float(np.asarray(map_s["sigma"]))),
            "map_success_beta0": ((), float(np.asarray(map_s["beta0"]))),

            # Head B (loss|success) MAP params
            "map_loss_ell":      (("feature",), ell_l_arr),
            "map_loss_eta":      ((), float(np.asarray(map_l["eta"]))),
            "map_loss_sigma":    ((), float(np.asarray(map_l["sigma"]))),
            "map_loss_mean_const": ((), float(np.asarray(map_l["mean_const"]))),

            # Convenience predictions on training data
            "pred_success_mu_train":       (("row",), mu_s),
            "pred_success_var_train":      (("row",), var_s),
            "pred_loss_mu_success_train":  (("row_success",), mu_l),
            "pred_loss_sd_success_train":  (("row_success",), sd_l),

            # Useful scalars for predictors
            "conditional_loss_mean": ((), float(conditional_loss_mean)),

            # ------- Diagnostics (per-feature) -------
            "raw_min":   (("feature",), np.array(raw_stats["raw_min"],  dtype=float)),
            "raw_max":   (("feature",), np.array(raw_stats["raw_max"],  dtype=float)),
            "raw_mean":  (("feature",), np.array(raw_stats["raw_mean"], dtype=float)),
            "raw_std":   (("feature",), np.array(raw_stats["raw_std"],  dtype=float)),
            "n_unique_raw": (("feature",), np.array(raw_stats["n_unique_raw"], dtype=np.int32)),

            "internal_min":  (("feature",), internal_min.astype(float)),
            "internal_max":  (("feature",), internal_max.astype(float)),
            "internal_mean": (("feature",), internal_mean.astype(float)),
            "internal_std":  (("feature",), internal_std.astype(float)),

            "Xn_p01": (("feature",), Xn_p01.astype(float)),
            "Xn_p99": (("feature",), Xn_p99.astype(float)),
            "Xn_span": (("feature",), Xn_span.astype(float)),

            "ell_over_span_success": (("feature",), ell_over_span_success.astype(float)),
            "ell_over_span_loss":    (("feature",), ell_over_span_loss.astype(float)),

            "corr_success":       (("feature",), corr_success),
            "corr_loss_success":  (("feature",), corr_loss_success),

            "feature_is_onehot_member": (("feature",), np.array(is_onehot_member, dtype=np.int8)),
            "feature_onehot_base":      (("feature",), np.array(onehot_base_per_feature, dtype=object)),
        },
        coords={
            "feature": feature_coord,
            "row": row_coord,
            "row_success": row_ok_coord,
        },
        attrs={
            "artifact_version": "0.1.2",   # bumped for diagnostics
            "target": target,
            "direction": direction,
            "rng_state": rng_state_b64,
            "rng_bitgen": rng.bit_generator.__class__.__name__,
            "numpy_version": np.__version__,
            "pymc_version": pm.__version__,
            "n_rows": int(n),
            "n_success_rows": int(int(ok_mask.sum())),
            "success_rate": float(base_success_rate),
            "categorical_groups_json": json.dumps(categorical_groups),  # base -> {labels, members}
        },
    )

    # Attach raw columns
    for k, v in raw_vars.items():
        ds[k] = v

    # ---------- Save ----------
    encoding = None
    if compress:
        encoding = {}
        for name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                encoding[name] = {"zlib": True, "complevel": 4}

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        engine, encoding = _select_netcdf_engine_and_encoding(ds, compress=compress)
        ds.to_netcdf(output, engine=engine, encoding=encoding)

    _print_diagnostics_table(ds, top_k=20)

    return ds



def kernel_diag_m52(XA: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
    return np.full(XA.shape[0], eta ** 2, dtype=float)


def kernel_m52_ard(XA: np.ndarray, XB: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
    XA = np.asarray(XA, float)
    XB = np.asarray(XB, float)
    ls = np.asarray(ls, float).reshape(1, 1, -1)
    diff = (XA[:, None, :] - XB[None, :, :]) / ls
    r2 = np.sum(diff * diff, axis=2)
    r = np.sqrt(np.maximum(r2, 0.0))
    sqrt5_r = np.sqrt(5.0) * r
    k = (eta ** 2) * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * np.exp(-sqrt5_r)
    return k


def add_jitter(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    jitter = eps * float(np.mean(np.diag(K)) + 1.0)
    return K + jitter * np.eye(K.shape[0], dtype=K.dtype)


def solve_chol(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L, B)


def feature_raw_from_artifact_or_reconstruct(
    ds: xr.Dataset,
    j: int,
    name: str,
    transform: str,
) -> np.ndarray:
    """
    Return the feature values in ORIGINAL units for each training row.
    Prefer a stored raw column (ds[name]) if present; otherwise reconstruct
    from Xn_train using feature_mean/std and the recorded transform.
    """
    # 1) Use stored raw per-row column if present
    if name in ds.data_vars:
        da = ds[name]
        if "row" in da.dims and da.sizes.get("row", None) == ds.sizes.get("row", None):
            vals = np.asarray(da.values, dtype=float)
            return vals

    # 2) Reconstruct from standardized training matrix
    Xn = ds["Xn_train"].values.astype(float)            # (N, p)
    mu = ds["feature_mean"].values.astype(float)[j]
    sd = ds["feature_std"].values.astype(float)[j]
    x_internal = Xn[:, j] * sd + mu                    # internal model space
    if transform == "log10":
        raw = 10.0 ** x_internal
    else:
        raw = x_internal
    return raw


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_bool01(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary truthy/falsy values to {0,1} int array."""
    if arr.dtype == bool:
        return arr.astype(np.int32)
    if np.issubdtype(arr.dtype, np.number):
        return (arr != 0).astype(np.int32)
    truthy = {"1", "true", "yes", "y", "ok", "success"}
    return np.array([1 if (str(x).strip().lower() in truthy) else 0 for x in arr], dtype=np.int32)


def _choose_transform(name: str, col: np.ndarray) -> str:
    """
    Choose a simple per-feature transform: 'identity' or 'log10'.
    Heuristics:
      - if column name looks like a learning rate (lr/learning_rate) AND >0 => log10
      - else if strictly positive and p99/p1 >= 1e3 => log10
      - else identity
    """
    name_l = name.lower()
    strictly_pos = np.all(np.isfinite(col)) and np.nanmin(col) > 0.0
    looks_lr = ("learning_rate" in name_l) or (name_l == "lr")
    if strictly_pos and (looks_lr or _large_dynamic_range(col)):
        return "log10"
    return "identity"


def _large_dynamic_range(col: np.ndarray) -> bool:
    x = col[np.isfinite(col)]
    if x.size == 0:
        return False
    p1, p99 = np.percentile(x, [1, 99])
    p1 = max(p1, 1e-12)
    return (p99 / p1) >= 1e3


def _apply_transform(tr: str, col: np.ndarray) -> np.ndarray:
    if tr == "log10":
        x = np.asarray(col, dtype=float)
        x = np.where(x <= 0, np.nan, x)
        return np.log10(x)
    return np.asarray(col, dtype=float)


def _np1d(x: np.ndarray, p: int) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size != p:
        a = np.full((p,), float(a.item()) if a.size == 1 else np.nan, dtype=float)
    return a


# ---- choose engine + encoding safely across backends ----
def _select_netcdf_engine_and_encoding(ds: xr.Dataset, compress: bool):
    # Prefer netcdf4
    try:
        import netCDF4  # noqa: F401
        engine = "netcdf4"
        if not compress:
            return engine, None
        enc = {}
        for name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                enc[name] = {"zlib": True, "complevel": 4}
        return engine, enc
    except Exception:
        pass

    # Then h5netcdf
    try:
        import h5netcdf  # noqa: F401
        engine = "h5netcdf"
        if not compress:
            return engine, None
        enc = {}
        for name, da in ds.data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                enc[name] = {"compression": "gzip", "compression_opts": 4}
        return engine, enc
    except Exception:
        pass

    # Finally scipy (no compression supported)
    engine = "scipy"
    return engine, None
