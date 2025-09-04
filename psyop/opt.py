import pandas as pd
import numpy as np

rng = np.random.default_rng(0)

# ---- Utilities: featurize <-> original ----
def original_to_raw_row(candidate: dict[str, float]) -> np.ndarray:
    """Map a candidate dict in ORIGINAL units to the raw feature row used before standardization."""
    return np.array([
        np.log10(candidate["learning_rate"]),
        candidate["r_drop_alpha"],
        float(candidate["batch_size"]),
        candidate["dropout"],
        float(candidate["epochs"]),
    ], dtype=float)

def raw_to_standardized_row(x_raw: np.ndarray) -> np.ndarray:
    return (x_raw - feature_means_array) / feature_stds_array

def original_to_standardized_row(candidate: dict[str, float]) -> np.ndarray:
    return raw_to_standardized_row(original_to_raw_row(candidate))

def batch_original_to_standardized(df: pd.DataFrame) -> np.ndarray:
    X_raw = np.column_stack([
        np.log10(df["learning_rate"].to_numpy()),
        df["r_drop_alpha"].to_numpy(),
        df["batch_size"].astype(float).to_numpy(),
        df["dropout"].to_numpy(),
        df["epochs"].astype(float).to_numpy(),
    ])
    return (X_raw - feature_means_array) / feature_stds_array


# --- search space inferred from data (adjust if you want fixed bounds) ---
def _pct_bounds(series: pd.Series, pad_frac=0.10):
    lo, hi = np.percentile(series.dropna().to_numpy(), [1, 99])
    span = hi - lo
    return lo - pad_frac*span, hi + pad_frac*span


_lr_min, _lr_max = np.percentile(dataframe_input["learning_rate"].to_numpy(), [1, 99])
_LR_MIN = max(_lr_min * 0.5, 1e-6)
_LR_MAX = min(_lr_max * 2.0, 1.0)

_RDA_MIN, _RDA_MAX = _pct_bounds(dataframe_input["r_drop_alpha"])
_RDA_MIN, _RDA_MAX = max(0.0, _RDA_MIN), min(4.0, _RDA_MAX)

_DROPOUT_MIN, _DROPOUT_MAX = 0.0, 1.0

_EP_MIN, _EP_MAX = _pct_bounds(dataframe_input["epochs"])
_EP_MIN = max(1, int(np.floor(_EP_MIN)))
_EP_MAX = int(np.ceil(_EP_MAX))

_BATCH_CHOICES = sorted(np.unique(dataframe_input["batch_size"].astype(int).to_numpy()).tolist())
_EPOCH_IS_DISCRETE = (len(np.unique(dataframe_input["epochs"])) < 30)

def _sample_candidates(n_candidates: int, rng: np.random.Generator) -> pd.DataFrame:
    lr = 10 ** rng.uniform(np.log10(_LR_MIN), np.log10(_LR_MAX), size=n_candidates)
    rda = rng.uniform(_RDA_MIN, _RDA_MAX, size=n_candidates)
    batch = rng.choice(_BATCH_CHOICES, size=n_candidates)
    dropout = rng.uniform(_DROPOUT_MIN, _DROPOUT_MAX, size=n_candidates)
    if _EPOCH_IS_DISCRETE:
        epochs = rng.integers(_EP_MIN, _EP_MAX + 1, size=n_candidates)
    else:
        epochs = rng.uniform(_EP_MIN, _EP_MAX, size=n_candidates).round().astype(int)
    return pd.DataFrame({
        "learning_rate": lr,
        "r_drop_alpha": rda,
        "batch_size": batch.astype(int),
        "dropout": dropout,
        "epochs": epochs.astype(int),
    })

def find_most_probable_minimum(
    n_candidates: int = 4000,
    n_draws: int = 2000,                 # <- 2000 by default
    top_k: int = 10,
    min_success_probability: float = 0.0,
    include_observation_noise: bool = True,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Monte Carlo estimate of P(candidate is the best feasible minimum), with unique ranking.
    - Skips draws where all candidates fail.
    - Tie-breakers: prob_best_feasible ↓, pred_p_success ↓, pred_loss_mean ↑, pred_loss_sd ↑.
    """
    rng = np.random.default_rng(random_seed)

    # 1) sample candidate hyperparams (original units)
    candidates_df = _sample_candidates(n_candidates, rng)
    Xn_cands = batch_original_to_standardized(candidates_df)

    # 2) model predictions
    p = predict_success_probability(Xn_cands)                                     # (N,)
    mu, sd = predict_conditional_loss(Xn_cands, include_observation_noise=include_observation_noise)

    # Optional hard feasibility filter
    keep = p >= float(min_success_probability)
    if not np.any(keep):
        keep = np.ones_like(p, dtype=bool)

    candidates_df = candidates_df.loc[keep].reset_index(drop=True)
    Xn_cands = Xn_cands[keep]
    p = p[keep]; mu = mu[keep]; sd = sd[keep]
    N = len(candidates_df)
    sd = np.maximum(sd, 1e-12)

    # 3) Monte Carlo winner-take-all over *feasible* draws only
    Z = mu[:, None] + sd[:, None] * rng.standard_normal((N, n_draws))
    success_mask = rng.random((N, n_draws)) < p[:, None]
    feasible_any_draw = success_mask.any(axis=0)          # skip all-fail draws
    if not feasible_any_draw.any():
        # Fallback: no feasible draws at all — rank by mean loss then success prob
        result = candidates_df.copy()
        result["pred_p_success"] = p
        result["pred_loss_mean"] = mu
        result["pred_loss_sd"]   = sd
        result["prob_best_feasible"] = 0.0
        result["wins"] = 0
        result["n_draws_effective"] = 0
        result_sorted = result.sort_values(
            ["pred_loss_mean", "pred_loss_sd", "pred_p_success"],
            ascending=[True, True, False],
            kind="mergesort"
        ).reset_index(drop=True)
        result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)
        top = result_sorted.head(top_k).reset_index(drop=True)
        top.to_csv("bo_best_probable_minima.csv", index=False)
        print("Wrote bo_best_probable_minima.csv (no feasible draws; fallback ranking).")
        return top

    Z_feasible = np.where(success_mask, Z, np.inf)
    Zf = Z_feasible[:, feasible_any_draw]
    winner_idx = np.argmin(Zf, axis=0)                     # (M_effective,)
    counts = np.bincount(winner_idx, minlength=N)
    n_effective = int(feasible_any_draw.sum())
    prob_best = counts / float(n_effective)

    # 4) Assemble + unique ranking with tie-breakers
    result = candidates_df.copy()
    result["pred_p_success"] = p
    result["pred_loss_mean"] = mu
    result["pred_loss_sd"]   = sd
    result["prob_best_feasible"] = prob_best
    result["wins"] = counts
    result["n_draws_effective"] = n_effective

    # Deterministic tie-breakers → unique rank
    result_sorted = result.sort_values(
        ["prob_best_feasible", "pred_p_success", "pred_loss_mean", "pred_loss_sd"],
        ascending=[False, False, True, True],
        kind="mergesort"  # stable
    ).reset_index(drop=True)
    result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)

    top = result_sorted.head(top_k).reset_index(drop=True)
    top.to_csv("bo_best_probable_minima.csv", index=False)
    print("Wrote bo_best_probable_minima.csv")
    return top