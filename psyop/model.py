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

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr


def run_model(
    input_path: Path,
    output_path: Path,               # must be .nc (handled in CLI)
    target_column: str,
    exclude_columns: list[str],
    direction: str,                  # "min" | "max" | "auto"
    success_column: str | None,
    random_seed: int,
    compress: bool,
) -> None:
    """
    Fit two-head GP (success prob + conditional loss) and save a single NetCDF artifact.

    The artifact contains:
      - raw columns (for plotting),
      - feature list + transforms + standardization stats,
      - training design matrices (standardized) and targets for both heads,
      - MAP hyperparameters for both heads,
      - convenience training predictions,
      - metadata (direction, schema, seed, etc.).
    """
    # -----------------------
    # Load CSV & basic checks
    # -----------------------
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {Path(input_path).resolve()}")

    df: pd.DataFrame = pd.read_csv(input_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    # Normalize dtypes: keep a copy of raw for artifact
    df_raw = df.copy()
    # Convert object to string to be NetCDF-safe
    for c in df_raw.columns:
        if df_raw[c].dtype == object:
            df_raw[c] = df_raw[c].astype("string")

    # -----------------------
    # Success inference/usage
    # -----------------------
    if success_column is not None:
        if success_column not in df.columns:
            raise ValueError(f"success_column '{success_column}' not found in CSV.")
        success = _to_bool01(df[success_column].to_numpy())
    else:
        # Infer success as ~isna(target)
        success = (~df[target_column].isna()).to_numpy().astype(int)

    df["__success__"] = success
    has_success = bool(np.any(success == 1))
    if not has_success:
        raise RuntimeError("No successful rows detected (cannot fit conditional-loss GP).")

    # -----------------------
    # Feature selection (generic)
    # -----------------------
    excluded = set(exclude_columns) | {target_column}
    if success_column:
        excluded.add(success_column)

    # Use numeric columns as candidates for features (float/int/boolean)
    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found after exclusions.")

    # -----------------------
    # Per-feature transforms (auto)
    # -----------------------
    feature_names = list(numeric_cols)
    transforms = []
    X_raw_cols = []
    for name in feature_names:
        col = df[name].to_numpy().astype(float)
        tr = _choose_transform(name, col)
        transforms.append(tr)
        X_raw_cols.append(_apply_transform(tr, col))

    X_raw = np.column_stack(X_raw_cols).astype(float)
    n, p = X_raw.shape

    # -----------------------
    # Standardize features
    # -----------------------
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std = np.where(X_std == 0.0, 1.0, X_std)
    Xn = (X_raw - X_mean) / X_std

    # Targets
    y_success = success.astype(float)
    ok_mask = (success == 1)
    y_loss_success = df.loc[ok_mask, target_column].to_numpy().astype(float)

    conditional_loss_mean = float(np.nanmean(y_loss_success)) if len(y_loss_success) else 0.0
    y_loss_centered = y_loss_success - conditional_loss_mean

    Xn_success_only = Xn[ok_mask, :]

    # -----------------------
    # Fit Head A: success LS-GP
    # -----------------------
    rng = np.random.default_rng(random_seed)
    base_success_rate = float(y_success.mean())

    with pm.Model() as model_s:
        ell_s = pm.HalfNormal("ell", sigma=2.0, shape=p)
        eta_s = pm.HalfNormal("eta", sigma=2.0)
        sigma_s = pm.HalfNormal("sigma", sigma=0.3)
        beta0_s = pm.Normal("beta0", mu=base_success_rate, sigma=0.15)

        K_s = eta_s**2 * pm.gp.cov.Matern52(input_dim=p, ls=ell_s)
        m_s = pm.gp.mean.Constant(beta0_s)
        gp_s = pm.gp.Marginal(mean_func=m_s, cov_func=K_s)

        _ = gp_s.marginal_likelihood("y_obs_s", X=Xn, y=y_success, noise=sigma_s)
        map_s = pm.find_MAP()

    # Training preds for success (mean only used downstream)
    with model_s:
        mu_s, var_s = gp_s.predict(Xn, point=map_s, diag=True, pred_noise=True)
    mu_s = np.clip(mu_s, 0.0, 1.0)

    # -----------------------
    # Fit Head B: conditional loss GP (success rows only)
    # -----------------------
    if Xn_success_only.shape[0] == 0:
        raise RuntimeError("No successful rows to fit the conditional-loss GP.")

    with pm.Model() as model_l:
        ell_l = pm.HalfNormal("ell", sigma=1.0, shape=p)
        eta_l = pm.HalfNormal("eta", sigma=1.0)
        sigma_l = pm.HalfNormal("sigma", sigma=1.0)
        mean_c = pm.Normal("mean_const", mu=0.0, sigma=10.0)

        K_l = eta_l**2 * pm.gp.cov.Matern52(input_dim=p, ls=ell_l)
        m_l = pm.gp.mean.Constant(mean_c)
        gp_l = pm.gp.Marginal(mean_func=m_l, cov_func=K_l)

        _ = gp_l.marginal_likelihood("y_obs", X=Xn_success_only, y=y_loss_centered, noise=sigma_l)
        map_l = pm.find_MAP()

    with model_l:
        mu_l_c, var_l = gp_l.predict(Xn_success_only, point=map_l, diag=True, pred_noise=True)
    mu_l = mu_l_c + conditional_loss_mean
    sd_l = np.sqrt(var_l)

    # -----------------------
    # Build xarray Dataset artifact
    # -----------------------
    # Coords
    feature_coord = xr.DataArray(feature_names, dims=("feature",), name="feature")
    row_coord = xr.DataArray(np.arange(n, dtype=np.int64), dims=("row",), name="row")
    row_ok_coord = xr.DataArray(np.where(ok_mask)[0].astype(np.int64), dims=("row_success",), name="row_success")

    # Raw columns (strings and numerics)
    raw_vars = {}
    for col in df_raw.columns:
        vals = df_raw[col].to_numpy()
        if pd.api.types.is_integer_dtype(vals) or pd.api.types.is_float_dtype(vals) or pd.api.types.is_bool_dtype(vals):
            raw_vars[col] = (("row",), vals)
        else:
            raw_vars[col] = (("row",), vals.astype("string"))

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
            "feature_std": (("feature",), X_std),
            "feature_transform": (("feature",), np.array(transforms, dtype="object")),

            # Masks / indexing
            "success_mask": (("row",), ok_mask.astype(np.int8)),

            # Head A (success) MAP params
            "map_success_ell": (("feature",), _np1d(map_s["ell"], p)),
            "map_success_eta": ((), float(np.asarray(map_s["eta"]))),
            "map_success_sigma": ((), float(np.asarray(map_s["sigma"]))),
            "map_success_beta0": ((), float(np.asarray(map_s["beta0"]))),

            # Head B (loss|success) MAP params
            "map_loss_ell": (("feature",), _np1d(map_l["ell"], p)),
            "map_loss_eta": ((), float(np.asarray(map_l["eta"]))),
            "map_loss_sigma": ((), float(np.asarray(map_l["sigma"]))),
            "map_loss_mean_const": ((), float(np.asarray(map_l["mean_const"]))),

            # Convenience predictions on training data
            "pred_success_mu_train": (("row",), mu_s),
            "pred_success_var_train": (("row",), var_s),
            "pred_loss_mu_success_train": (("row_success",), mu_l),
            "pred_loss_sd_success_train": (("row_success",), sd_l),

            # Useful scalars
            "conditional_loss_mean": ((), float(conditional_loss_mean)),
            "base_success_rate": ((), float(base_success_rate)),
        },
        coords={
            "feature": feature_coord,
            "row": row_coord,
            "row_success": row_ok_coord,
        },
        attrs={
            "artifact_version": "0.1.0",
            "target_column": target_column,
            "direction": direction,
            "success_column": success_column if success_column is not None else "__inferred__",
            "random_seed": int(random_seed),
            "n_rows": int(n),
            "n_features": int(p),
            "input_csv": str(Path(input_path)),
        },
    )

    # Attach raw columns under top-level dataset
    for k, v in raw_vars.items():
        ds[k] = v

    # Compression
    encoding = None
    if compress:
        encoding = {}
        for name, da in ds.data_vars.items():
            # zlib only for numeric arrays
            if np.issubdtype(da.dtype, np.number):
                encoding[name] = {"zlib": True, "complevel": 4}

    # Write artifact
    output_path = Path(output_path)
    ds.to_netcdf(output_path, encoding=encoding)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_bool01(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary truthy/falsy values to {0,1} int array."""
    if arr.dtype == bool:
        return arr.astype(np.int32)
    if np.issubdtype(arr.dtype, np.number):
        return (arr != 0).astype(np.int32)
    # strings/objects
    truthy = {"1", "true", "yes", "y", "ok", "success"}
    return np.array([1 if (str(x).strip().lower() in truthy) else 0 for x in arr], dtype=np.int32)


def _choose_transform(name: str, col: np.ndarray) -> str:
    """
    Choose a simple per-feature transform: 'identity' or 'log10'.
    Heuristics:
      - if column name looks like a learning rate (lr/learning_rate) AND >0 => log10
      - else if strictly positive and has large dynamic range (p99/p1 >= 1e3) => log10
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
        # Guard against non-positive just in case
        x = np.asarray(col, dtype=float)
        x = np.where(x <= 0, np.nan, x)
        return np.log10(x)
    return np.asarray(col, dtype=float)


def _np1d(x: np.ndarray, p: int) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size != p:
        # broadcast scalar to length p (shouldn't happen for ell)
        a = np.full((p,), float(a.item()) if a.size == 1 else np.nan, dtype=float)
    return a
