#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Make BLAS single-threaded to avoid oversubscription / macOS crashes
# ---------------------------------------------------------------------
import os
for _env_var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import pymc as pm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtr  # vectorized Φ(z)
from pathlib import Path

# ---------------------------------------------------------------------
# Load & validate data
# ---------------------------------------------------------------------
input_csv_path = Path("trials.csv")
if not input_csv_path.exists():
    raise FileNotFoundError(f"Input CSV not found: {input_csv_path.resolve()}")

dataframe_input: pd.DataFrame = pd.read_csv(input_csv_path)

required_columns = {
    "trial_id", "learning_rate", "batch_size", "r_drop_alpha",
    "dropout", "epochs", "loss"
}
missing = required_columns - set(dataframe_input.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

# ---------------------------------------------------------------------
# Infer success and build feature matrix
# ---------------------------------------------------------------------
# Success: present loss => success; NaN loss => failure
success_indicator_array: np.ndarray = (~dataframe_input["loss"].isna()).to_numpy().astype(int)
dataframe_input["success"] = success_indicator_array

# Feature engineering:
# - Use log10 for learning_rate (GP distances more sensible)
# - Cast batch_size and epochs to float (treat as continuous for this GP)
feature_matrix_raw: np.ndarray = np.column_stack([
    np.log10(dataframe_input["learning_rate"].to_numpy()),
    dataframe_input["r_drop_alpha"].to_numpy(),
    dataframe_input["batch_size"].astype(float).to_numpy(),
    dataframe_input["dropout"].to_numpy(),
    dataframe_input["epochs"].astype(float).to_numpy(),
])
feature_names_list = ["learning_rate", "r_drop_alpha", "batch_size", "dropout", "epochs"]
num_features = feature_matrix_raw.shape[1]

# Standardize features (helps GP lengthscales)
feature_means_array: np.ndarray = feature_matrix_raw.mean(axis=0)
feature_stds_array: np.ndarray = feature_matrix_raw.std(axis=0)
# Guard against zero std (unlikely but safe)
feature_stds_array = np.where(feature_stds_array == 0, 1.0, feature_stds_array)

feature_matrix_standardized: np.ndarray = (feature_matrix_raw - feature_means_array) / feature_stds_array

# Targets
observed_success_float_array: np.ndarray = success_indicator_array.astype(float)
successful_rows_mask: np.ndarray = (success_indicator_array == 1)
observed_loss_success_only_array: np.ndarray = dataframe_input.loc[successful_rows_mask, "loss"].to_numpy()

# Center the conditional loss target for numerics
conditional_loss_mean: float = float(np.nanmean(observed_loss_success_only_array)) if len(observed_loss_success_only_array) else 0.0
centered_loss_success_only_array: np.ndarray = observed_loss_success_only_array - conditional_loss_mean

feature_matrix_success_only_standardized: np.ndarray = feature_matrix_standardized[successful_rows_mask]

# ---------------------------------------------------------------------
# Head A: FAST success model (LS-GP on 0/1 labels)
# ---------------------------------------------------------------------
base_success_rate: float = float(observed_success_float_array.mean())

with pm.Model() as success_model:
    # Priors (slightly wider on ell/eta to allow flexibility)
    lengthscales_success = pm.HalfNormal("ell", sigma=2.0, shape=num_features)
    signal_std_success = pm.HalfNormal("eta", sigma=2.0)
    noise_std_success = pm.HalfNormal("sigma", sigma=0.3)     # label noise on [0,1]
    intercept_success = pm.Normal("beta0", mu=base_success_rate, sigma=0.15)

    covariance_success = signal_std_success**2 * pm.gp.cov.Matern52(
        input_dim=num_features, ls=lengthscales_success
    )
    mean_function_success = pm.gp.mean.Constant(intercept_success)

    gp_success = pm.gp.Marginal(mean_func=mean_function_success, cov_func=covariance_success)
    _ = gp_success.marginal_likelihood(
        "y_obs_s", X=feature_matrix_standardized, y=observed_success_float_array, noise=noise_std_success
    )

    map_estimates_success = pm.find_MAP()

def predict_success_probability(feature_matrix_standardized_grid: np.ndarray) -> np.ndarray:
    """Return predicted mean success probability (clipped to [0,1]) at grid points."""
    with success_model:
        mu_success, _var_success = gp_success.predict(
            feature_matrix_standardized_grid, point=map_estimates_success, diag=True, pred_noise=True
        )
    return np.clip(mu_success, 0.0, 1.0)

# ---------------------------------------------------------------------
# Head B: Conditional loss model (GP on successful rows only)
# ---------------------------------------------------------------------
if len(feature_matrix_success_only_standardized) == 0:
    raise RuntimeError("No successful rows to fit the conditional-loss GP.")

with pm.Model() as conditional_loss_model:
    lengthscales_loss = pm.HalfNormal("ell", sigma=1.0, shape=num_features)
    signal_std_loss = pm.HalfNormal("eta", sigma=1.0)
    noise_std_loss = pm.HalfNormal("sigma", sigma=1.0)
    intercept_loss = pm.Normal("mean_const", mu=0.0, sigma=10.0)

    covariance_loss = signal_std_loss**2 * pm.gp.cov.Matern52(
        input_dim=num_features, ls=lengthscales_loss
    )
    mean_function_loss = pm.gp.mean.Constant(intercept_loss)

    gp_conditional_loss = pm.gp.Marginal(mean_func=mean_function_loss, cov_func=covariance_loss)
    _ = gp_conditional_loss.marginal_likelihood(
        "y_obs", X=feature_matrix_success_only_standardized,
        y=centered_loss_success_only_array, noise=noise_std_loss
    )

    map_estimates_loss = pm.find_MAP()

def predict_conditional_loss(feature_matrix_standardized_grid: np.ndarray, include_observation_noise: bool = True):
    """
    Return (mean, std) of E[loss | success, x] at grid points.
    If include_observation_noise=True, the returned std includes noise.
    """
    with conditional_loss_model:
        mu_loss_centered, var_loss = gp_conditional_loss.predict(
            feature_matrix_standardized_grid, point=map_estimates_loss,
            diag=True, pred_noise=include_observation_noise
        )
    mu_loss = mu_loss_centered + conditional_loss_mean
    sd_loss = np.sqrt(var_loss)
    return mu_loss, sd_loss

# ---------------------------------------------------------------------
# Training-point diagnostics (optional CSV)
# ---------------------------------------------------------------------
predicted_success_prob_training = predict_success_probability(feature_matrix_standardized)
predicted_loss_mean_ok, predicted_loss_sd_ok = predict_conditional_loss(feature_matrix_success_only_standardized)

diagnostic_rows = []
idx_ok = 0
for row_idx in range(len(dataframe_input)):
    row_dict = {
        "trial_id": int(dataframe_input.loc[row_idx, "trial_id"]),
        "success_observed": int(success_indicator_array[row_idx]),
        "success_probability_predicted": float(predicted_success_prob_training[row_idx]),
        "loss_observed": (None if not successful_rows_mask[row_idx] else float(dataframe_input.loc[row_idx, "loss"]))
    }
    if successful_rows_mask[row_idx]:
        row_dict.update({
            "loss_conditional_predicted_mean": float(predicted_loss_mean_ok[idx_ok]),
            "loss_conditional_predicted_sd": float(predicted_loss_sd_ok[idx_ok]),
        })
        idx_ok += 1
    else:
        row_dict.update({
            "loss_conditional_predicted_mean": None,
            "loss_conditional_predicted_sd": None,
        })
    diagnostic_rows.append(row_dict)

pd.DataFrame(diagnostic_rows).to_csv("gp_training_predictions.csv", index=False)
print("Wrote gp_training_predictions.csv")