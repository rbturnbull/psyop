# opt.py
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Any
import re

import numpy as np
import pandas as pd
import xarray as xr
import hashlib
from scipy.special import ndtr  # Φ(z), vectorized

from .util import get_rng, df_to_table
from .model import (
    kernel_diag_m52, 
    kernel_m52_ard,
    add_jitter,
    solve_chol,
    solve_lower,
)
from .model import feature_raw_from_artifact_or_reconstruct

from rich.console import Console
from rich.table import Table

console = Console()

_ONEHOT_RE = re.compile(r"^(?P<base>[^=]+)=(?P<label>.+)$")


def _pretty_conditioned_on(
    fixed_norm_numeric: dict | None = None,
    cat_fixed_label: dict | None = None,
) -> str:
    """
    Combine numeric fixed constraints (already normalized to model space)
    with categorical fixed choices into a single human-readable string.

    Examples:
      - fixed_norm_numeric = {"epochs": 12.0, "batch_size": 32}
      - cat_fixed_label    = {"language": "Linear B"}

    Returns:
      "epochs=12, batch_size=32, language=Linear B"
      (ordering is deterministic: keys sorted within each group)
    """
    fixed_norm_numeric = fixed_norm_numeric or {}
    cat_fixed_label = cat_fixed_label or {}

    parts = []

    # Prefer the project-standard formatter if present.
    try:
        if fixed_norm_numeric:
            txt = _fixed_as_string(fixed_norm_numeric)  # e.g. "epochs=12, batch_size=32"
            if txt:
                parts.append(txt)
    except Exception:
        # Fallback: simple k=v with general formatting.
        if fixed_norm_numeric:
            items = []
            for k, v in sorted(fixed_norm_numeric.items()):
                try:
                    items.append(f"{k}={float(v):.6g}")
                except Exception:
                    items.append(f"{k}={v}")
            parts.append(", ".join(items))

    # Append categorical fixed choices as "base=Label"
    if cat_fixed_label:
        cat_txt = ", ".join(f"{b}={lab}" for b, lab in sorted(cat_fixed_label.items()))
        if cat_txt:
            parts.append(cat_txt)

    return ", ".join(p for p in parts if p)


def _split_constraints_for_numeric_and_categorical(
    feature_names: list[str],
    kwargs: dict[str, object],
):
    """
    Split user constraints into:
      - numeric: user_fixed, user_ranges, user_choices_num (by feature name)
      - categorical: cat_fixed_label (base->label), cat_allowed (base->set(labels))
      - and return one-hot groups

    Interp rules:
      * For a categorical base key (e.g. 'language'):
          - str  -> fixed single label
          - list/tuple of str -> allowed label set
      * For a numeric feature key (non one-hot member):
          - number -> fixed
          - slice(lo,hi[,step]) -> range (lo,hi) inclusive on ends in post-filter
          - list/tuple of numbers -> finite choices
          - range(...) (python range) -> tuple of ints (choices)
    """
    groups = _onehot_groups(feature_names)
    bases = set(groups.keys())
    feature_set = set(feature_names)

    user_fixed: dict[str, float] = {}
    user_ranges: dict[str, tuple[float, float]] = {}
    user_choices_num: dict[str, list[int | float]] = {}

    cat_fixed_label: dict[str, str] = {}
    cat_allowed: dict[str, set[str]] = {}

    # helper
    def _is_intlike(x) -> bool:
        try:
            return float(int(round(float(x)))) == float(x)
        except Exception:
            return False

    for key, raw in (kwargs or {}).items():
        # --- CATEGORICAL (by base key, not member name) ---
        if key in bases:
            labels = groups[key]["labels"]
            # fixed single label
            if isinstance(raw, str):
                if raw not in labels:
                    raise ValueError(f"Unknown category for {key!r}: {raw!r}. Choices: {labels}")
                cat_fixed_label[key] = raw
                cat_allowed[key] = {raw}
                continue
            # list/tuple of labels (choices restriction)
            if isinstance(raw, (list, tuple, set)):
                chosen = [v for v in raw if isinstance(v, str) and (v in labels)]
                if not chosen:
                    raise ValueError(f"No valid categories for {key!r} in {raw!r}. Choices: {labels}")
                cat_allowed[key] = set(chosen)
                continue
            # anything else -> ignore for cats
            continue

        # --- NUMERIC (by feature name; skip one-hot member names) ---
        # If user accidentally passes member name 'language=Linear A', ignore here
        if key not in feature_set or _ONEHOT_RE.match(key):
            # Unknown or member-level keys are ignored at this stage
            continue

        # python range -> tuple of ints
        if isinstance(raw, range):
            raw = tuple(raw)

        # number -> fixed
        if isinstance(raw, (int, float, np.number)):
            val = float(raw)
            if np.isfinite(val):
                user_fixed[key] = val
            continue

        # slice -> float range
        if isinstance(raw, slice):
            if raw.start is None or raw.stop is None:
                continue
            lo = float(raw.start); hi = float(raw.stop)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            if lo > hi:
                lo, hi = hi, lo
            user_ranges[key] = (lo, hi)
            continue

        # list/tuple -> numeric choices
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                continue
            # preserve ints if all int-like, else floats
            if all(_is_intlike(v) for v in raw):
                user_choices_num[key] = [int(round(float(v))) for v in raw]
            else:
                user_choices_num[key] = [float(v) for v in raw]
            continue

        # otherwise: ignore

    # Numeric fixed wins over its own range/choices
    for k in list(user_fixed.keys()):
        user_ranges.pop(k, None)
        user_choices_num.pop(k, None)

    return groups, user_fixed, user_ranges, user_choices_num, cat_fixed_label, cat_allowed


def _detect_categorical_groups(feature_names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """
    Detect one-hot groups: {"language": [("language=Linear A","Linear A"), ("language=Linear B","Linear B"), ...]}
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for name in feature_names:
        m = _ONEHOT_RE.match(name)
        if not m:
            continue
        base = m.group("base")
        lab  = m.group("label")
        groups.setdefault(base, []).append((name, lab))
    # deterministic order
    for base in groups:
        groups[base].sort(key=lambda t: t[1])
    return groups

def _project_categoricals_to_valid_onehot(df: pd.DataFrame, groups: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    """
    For each categorical group ensure exactly one column is 1 and the rest 0 (argmax projection).
    Works whether columns are 0/1 already or arbitrary scores in [0,1].
    """
    for base, pairs in groups.items():
        cols = [name for name, _ in pairs if name in df.columns]
        if len(cols) <= 1:
            continue
        sub = df[cols].to_numpy(dtype=float)
        # treat NaNs as -inf so they never win
        sub = np.where(np.isfinite(sub), sub, -np.inf)
        if sub.size == 0:
            continue
        idx = np.argmax(sub, axis=1)
        new = np.zeros_like(sub)
        new[np.arange(sub.shape[0]), idx] = 1.0
        df.loc[:, cols] = new
    return df


def _apply_categorical_constraints(df: pd.DataFrame,
                                   groups: dict[str, list[tuple[str, str]]],
                                   fixed_str: dict[str, str],
                                   allowed_strs: dict[str, list[str]]) -> pd.DataFrame:
    """
    Filter rows by categorical constraints expressed on the base names, e.g.
      fixed_str = {"language": "Linear B"}
      allowed_strs = {"language": ["Linear A", "Linear B"]}
    Operates on one-hot columns, so call BEFORE collapsing to string columns.
    """
    mask = np.ones(len(df), dtype=bool)
    for base, val in (fixed_str or {}).items():
        if base not in groups:
            continue
        cols = {label: name for name, label in groups[base] if name in df.columns}
        want = cols.get(val)
        if want is None:
            # no matching one-hot column — drop all rows
            mask &= False
        else:
            mask &= (df[want] >= 0.5)  # after projection, exactly 1 column is 1
    for base, vals in (allowed_strs or {}).items():
        if base not in groups:
            continue
        cols = {label: name for name, label in groups[base] if name in df.columns}
        want_cols = [cols[v] for v in vals if v in cols]
        if want_cols:
            mask &= (df[want_cols].sum(axis=1) >= 0.5)
        else:
            mask &= False
    return df.loc[mask].reset_index(drop=True)


def _onehot_groups(feature_names: list[str]) -> dict[str, dict]:
    """
    Detect one-hot groups among feature names like 'language=Linear A'.
    Returns:
      {
        base: {
          "labels": [label1, ...],
          "members": [(feat_name, label), ...],
          "name_by_label": {label: feat_name}
        },
        ...
      }
    """
    groups: dict[str, dict] = {}
    for name in feature_names:
        m = _ONEHOT_RE.match(name)
        if not m:
            continue
        base = m.group("base")
        label = m.group("label")
        g = groups.setdefault(base, {"labels": [], "members": [], "name_by_label": {}})
        g["labels"].append(label)
        g["members"].append((name, label))
        g["name_by_label"][label] = name
    # stable order for labels
    for g in groups.values():
        # keep insertion order from feature_names, but ensure uniqueness
        seen = set()
        uniq = []
        for lab in g["labels"]:
            if lab not in seen:
                uniq.append(lab); seen.add(lab)
        g["labels"] = uniq
    return groups



def _numeric_specs_only(search_specs: dict, groups: dict) -> dict:
    """
    Return a copy of search_specs with one-hot member feature names removed.
    `groups` is the output of _onehot_groups(feature_names).
    """
    if not groups:
        return dict(search_specs)

    onehot_member_names = set()
    for g in groups.values():
        onehot_member_names.update(g["name_by_label"].values())

    return {k: v for k, v in search_specs.items() if k not in onehot_member_names}


def _assert_valid_onehot(df: pd.DataFrame, groups: dict[str, dict], where: str = "") -> None:
    """
    Assert every one-hot block has exactly one '1' per row (no NaNs).
    Prints a small diagnostic if not.
    """
    for base, g in groups.items():
        member_cols = [g["name_by_label"][lab] for lab in g["labels"] if g["name_by_label"][lab] in df.columns]
        if not member_cols:
            print(f"[onehot] {where}: base={base} has no member columns present")
            continue

        block = df[member_cols].to_numpy()
        nonfinite_mask = ~np.isfinite(block)
        sums = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1)

        bad = np.where(nonfinite_mask.any(axis=1) | (sums != 1))[0]
        if bad.size:
            print(f"[BUG onehot] {where}: base={base}, rows with invalid one-hot: {bad[:20].tolist()} (showing first 20)")
            print("member_cols:", member_cols)
            print(df.iloc[bad[:5]][member_cols])  # show a few bad rows
            raise RuntimeError(f"Invalid one-hot block for base={base} at {where}")

def _get_float_attr(obj, names, default=0.0):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            # skip boolean flags like mean_only
            if isinstance(v, (bool, np.bool_)):
                continue
            try:
                return float(v)
            except Exception:
                pass
    return float(default)



import itertools
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- small utils (reuse in this file) ----------
def _orig_to_std(j: int, x, transforms, mu, sd):
    arr = np.asarray(x, dtype=float)
    if transforms[j] == "log10":
        arr = np.where(arr <= 0, np.nan, arr)
        arr = np.log10(arr)
    return (arr - mu[j]) / sd[j]

def _std_to_orig(j: int, arr, transforms, mu, sd):
    x = np.asarray(arr, float) * sd[j] + mu[j]
    if transforms[j] == "log10":
        x = np.power(10.0, x)
    return x

def _groups_from_feature_names(feature_names: list[str]) -> dict:
    # same grouping logic you already use elsewhere
    groups = {}
    for nm in feature_names:
        if "=" in nm:
            base, lab = nm.split("=", 1)
            g = groups.setdefault(base, {"labels": [], "name_by_label": {}, "members": []})
            g["labels"].append(lab)
            g["name_by_label"][lab] = nm
            g["members"].append(nm)
    # Stable order
    for b in groups:
        labs = list(dict.fromkeys(groups[b]["labels"]))
        groups[b]["labels"] = labs
        groups[b]["members"] = [groups[b]["name_by_label"][lab] for lab in labs]
    return groups

def _pick_attr(obj, names, allow_none=False):
    """Return the first present attribute in names without truth-testing arrays."""
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if (v is not None) or allow_none:
                return v
    return None

# ---------- analytic GP (mean + grad) ----------
class _GPMarginal:
    def __init__(self, Xtr, ytr, ell, eta, sigma, mean_const):
        self.X = np.asarray(Xtr, float)              # (N,p)
        self.y = np.asarray(ytr, float)              # (N,)
        self.ell = np.asarray(ell, float)            # (p,)
        self.eta = float(eta)
        self.sigma = float(sigma)
        self.m = float(mean_const)
        K = kernel_m52_ard(self.X, self.X, self.ell, self.eta)
        K[np.diag_indices_from(K)] += self.sigma**2
        L = np.linalg.cholesky(add_jitter(K))
        self.L = L
        self.alpha = solve_chol(L, (self.y - self.m))
        self.X_train = getattr(self, "X_train", getattr(self, "Xtr", getattr(self, "X", None)))
        self.ell     = getattr(self, "ell", getattr(self, "ls", None))
        # Back-compat aliases:
        self.Xtr = self.X_train
        self.ls  = self.ell
        self.Xtr = self.X_train
        self.ell = ell; self.ls = self.ell
        self.m0 = float(mean_const); self.mean_const = self.m0

    def sd_at(self, x: np.ndarray, include_observation_noise: bool = True) -> float:
        """
        Predictive standard deviation at a single standardized point x.
        """
        x = np.asarray(x, float).reshape(1, -1)  # (1, p)
        Ks = kernel_m52_ard(x, self.Xtr, self.ls, self.eta)  # (1, N)
        v  = solve_lower(self.L, Ks.T)                      # (N, 1)
        kss = kernel_diag_m52(x, self.ls, self.eta)[0]      # scalar diag K(x,x) = eta^2
        var = float(kss - np.sum(v * v))
        if include_observation_noise:
            var += float(self.sigma ** 2)
        var = max(var, 1e-12)
        return float(np.sqrt(var))

    def _k_and_grad(self, x):
        """k(x, X), ∂k/∂x (p-dimensional gradient aggregated over train points)."""
        x = np.asarray(x, float).reshape(1, -1)                     # (1,p)
        X = self.X
        ell = self.ell
        eta = self.eta

        # distances in lengthscale space
        D = (x[:, None, :] - X[None, :, :]) / ell[None, None, :]    # (1,N,p)
        r2 = np.sum(D*D, axis=2)                                    # (1,N)
        r  = np.sqrt(np.maximum(r2, 0.0))                           # (1,N)
        sqrt5_r = np.sqrt(5.0) * r
        # kernel
        k = (eta**2) * (1.0 + sqrt5_r + (5.0/3.0)*r2) * np.exp(-sqrt5_r)  # (1,N)

        # grad wrt x:  -(5η^2/3) e^{-√5 r} (1 + √5 r) * (x - xi)/ell^2
        # handle r=0 safely -> derivative is 0
        coef = -(5.0 * (eta**2) / 3.0) * np.exp(-sqrt5_r) * (1.0 + sqrt5_r)  # (1,N)
        S = (x[:, None, :] - X[None, :, :]) / (ell[None, None, :]**2)        # (1,N,p)
        grad = np.sum(coef[:, :, None] * S, axis=1)                          # (1,p)

        return k.ravel(), grad.ravel()

    def mean_and_grad(self, x: np.ndarray):
        # --- resolve training matrix
        Xtr = _pick_attr(self, ["X_train", "Xtr", "X"])
        if Xtr is None:
            raise AttributeError("GPMarginal: training inputs not found (tried X_train, Xtr, X).")
        Xtr = np.asarray(Xtr, float)

        # --- resolve hyperparams / vectors
        ell = _pick_attr(self, ["ell", "ls"])
        if ell is None:
            raise AttributeError("GPMarginal: lengthscales not found (tried ell, ls).")
        ell = np.asarray(ell, float)

        eta = _get_float_attr(self, ["eta"])
        alpha = _pick_attr(self, ["alpha", "alpha_vec"])
        if alpha is None:
            raise AttributeError("GPMarginal: alpha not found (tried alpha, alpha_vec).")
        alpha = np.asarray(alpha, float).ravel()

        # mean constant (name differs across versions)
        m0 = _get_float_attr(self, ["mean_const", "m0", "beta0", "mean_c", "mean"], default=0.0)

        # --- mean
        Ks = kernel_m52_ard(x[None, :], Xtr, ell, eta).ravel()   # (N_train,)
        mu = float(m0 + Ks @ alpha)

        # --- gradient wrt x (shape (p,))
        grad_k  = _grad_k_m52_ard_wrt_x(x, Xtr, ell, eta)        # (N_train, p)
        grad_mu = grad_k.T @ alpha                               # (p,)

        return mu, grad_mu


    def mean_only(self, X):
        Ks = kernel_m52_ard(X, self.X, self.ell, self.eta)
        return self.m + Ks @ self.alpha


def _grad_k_m52_ard_wrt_x(x: np.ndarray, Xtr: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
    """
    ∂k(x, Xtr_i)/∂x for Matérn 5/2 ARD.
    Returns (N_train, p) — one row per training point.
    """
    x   = np.asarray(x, float).reshape(1, -1)     # (1, p)
    Xtr = np.asarray(Xtr, float)                  # (N, p)
    ls  = np.asarray(ls, float).reshape(1, -1)    # (1, p)

    diff = x - Xtr                                # (N, p)
    z    = diff / ls                              # (N, p)
    r    = np.sqrt(np.sum(z*z, axis=1))           # (N,)

    sr5  = np.sqrt(5.0)
    coef = -(5.0 * (eta**2) / 3.0) * np.exp(-sr5 * r) * (1.0 + sr5 * r)  # (N,)
    grad = coef[:, None] * (diff / (ls*ls))       # (N, p)
    return grad


def rng_for_dataset(ds, seed=None):
    if isinstance(seed, np.random.Generator):
        return seed

    # Hash something stable from the dataset; Xn_train is fine.
    x = np.ascontiguousarray(ds["Xn_train"].values.astype(np.float64))
    digest64 = int.from_bytes(hashlib.sha256(x.tobytes()).digest()[:8], "big")  # 64 bits

    if seed is None:
        mixed = np.uint64(digest64)                # dataset-deterministic
    else:
        mixed = np.uint64(seed) ^ np.uint64(digest64)  # mix user seed with dataset hash

    return np.random.default_rng(int(mixed))

def suggest(
    model: xr.Dataset | Path | str,
    count: int = 10,
    output: Path | None = None,
    repulsion: float = 0.34,              # repulsion radius & weight
    explore: float = 0.5,                 # probability to optimize EI (explore)
    success_threshold: float = 0.8,
    softmax_temp: float | None = 0.2,     # τ for EI softmax; None/0 => greedy EI
    n_starts: int = 32,
    max_iters: int = 200,
    penalty_lambda: float = 1.0,
    penalty_beta: float = 10.0,
    direction: str | None = None,         # defaults to model's
    seed: int | np.random.Generator | None = 42,
    **kwargs,                              # constraints in ORIGINAL units
) -> pd.DataFrame:
    import itertools, math
    import numpy as np, pandas as pd

    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    rng = rng_for_dataset(ds, seed)  # dataset-aware determinism

    # --- metadata
    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    mu_f = ds["feature_mean"].values.astype(float)
    sd_f = ds["feature_std"].values.astype(float)
    p = len(feature_names)
    name_to_idx = {nm: j for j, nm in enumerate(feature_names)}
    groups = _groups_from_feature_names(feature_names)

    # --- GP heads
    gp_s = _GPMarginal(
        Xtr=ds["Xn_train"].values.astype(float),
        ytr=ds["y_success"].values.astype(float),
        ell=ds["map_success_ell"].values.astype(float),
        eta=float(ds["map_success_eta"].values),
        sigma=float(ds["map_success_sigma"].values),
        mean_const=float(ds["map_success_beta0"].values),
    )
    cond_mean = float(ds["conditional_loss_mean"].values) if "conditional_loss_mean" in ds else 0.0
    gp_l = _GPMarginal(
        Xtr=ds["Xn_success_only"].values.astype(float),
        ytr=ds["y_loss_centered"].values.astype(float),
        ell=ds["map_loss_ell"].values.astype(float),
        eta=float(ds["map_loss_eta"].values),
        sigma=float(ds["map_loss_sigma"].values),
        mean_const=float(ds["map_loss_mean_const"].values),
    )

    # --- direction & EI baseline
    if direction is None:
        direction = str(ds.attrs.get("direction", "min"))
    flip = -1.0 if direction == "max" else 1.0
    best_feasible = _best_feasible_observed(ds, direction)

    # --- constraints -> bounds (std space)
    cat_allowed: dict[str, list[str]] = {b: list(g["labels"]) for b, g in groups.items()}
    cat_fixed: dict[str, str] = {}
    fixed_num_std: dict[int, float] = {}
    range_num_std: dict[int, tuple[float, float]] = {}
    choice_num: dict[int, np.ndarray] = {}

    def canon_key(k: str) -> str:
        import re as _re
        raw = str(k)
        stripped = _re.sub(r"[^a-z0-9]+", "", raw.lower())
        if raw in name_to_idx: return raw
        for base in groups.keys():
            if stripped == _re.sub(r"[^a-z0-9]+", "", base.lower()):
                return base
        return raw

    for k, v in (kwargs or {}).items():
        ck = canon_key(k)
        if ck in groups:  # categorical base
            labels = groups[ck]["labels"]
            if isinstance(v, str):
                if v not in labels:
                    raise ValueError(f"Unknown category for {ck}: {v}. Choices: {labels}")
                cat_fixed[ck] = v
            else:
                L = [x for x in (list(v) if isinstance(v, (list, tuple, set)) else [v]) if isinstance(x, str) and x in labels]
                if not L:
                    raise ValueError(f"No valid categories for {ck} in {v}. Choices: {labels}")
                if len(L) == 1:
                    cat_fixed[ck] = L[0]
                else:
                    cat_allowed[ck] = L
        elif ck in name_to_idx:  # numeric
            j = name_to_idx[ck]
            if isinstance(v, range):
                v = tuple(v)
            if isinstance(v, slice):
                lo = _orig_to_std(j, v.start, transforms, mu_f, sd_f)
                hi = _orig_to_std(j, v.stop,  transforms, mu_f, sd_f)
                lo, hi = float(np.nanmin([lo, hi])), float(np.nanmax([lo, hi]))
                range_num_std[j] = (lo, hi)
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = _orig_to_std(j, np.asarray(v, float), transforms, mu_f, sd_f)
                choice_num[j] = np.asarray(arr, float)
            else:
                fixed_num_std[j] = float(_orig_to_std(j, float(v), transforms, mu_f, sd_f))
        else:
            raise ValueError(f"Unknown constraint key: {k!r}")

    Xn = ds["Xn_train"].values.astype(float)
    p01 = np.percentile(Xn, 1, axis=0)
    p99 = np.percentile(Xn, 99, axis=0)
    wide_lo = np.minimum(p01 - 1.0, -3.0)   # allow outside training range
    wide_hi = np.maximum(p99 + 1.0,  3.0)

    bounds: list[tuple[float, float] | None] = [None]*p
    for j in range(p):
        if j in fixed_num_std:
            val = fixed_num_std[j]
            bounds[j] = (val, val)
        elif j in range_num_std:
            bounds[j] = range_num_std[j]
        elif j in choice_num:
            lo = float(np.nanmin(choice_num[j])); hi = float(np.nanmax(choice_num[j]))
            bounds[j] = (lo, hi)
        else:
            bounds[j] = (float(wide_lo[j]), float(wide_hi[j]))

    # --- helpers
    onehot_members = {m for g in groups.values() for m in g["members"]}
    numeric_idx = [j for j, nm in enumerate(feature_names) if nm not in onehot_members]
    num_bounds = [bounds[j] for j in numeric_idx]

    def apply_onehot(vec_std: np.ndarray, base: str, label: str):
        for lab in groups[base]["labels"]:
            member_name = groups[base]["name_by_label"][lab]
            j = name_to_idx[member_name]
            raw = 1.0 if lab == label else 0.0
            vec_std[j] = _orig_to_std(j, raw, transforms, mu_f, sd_f)

    # exploitation objective (μ + soft penalty), with gradient
    def obj_grad_exploit(x_full: np.ndarray):
        mu_l, g_l = gp_l.mean_and_grad(x_full)
        mu = mu_l + cond_mean
        mu_p, g_p = gp_s.mean_and_grad(x_full)
        z = success_threshold - mu_p
        sig = 1.0 / (1.0 + np.exp(-penalty_beta * z))
        penalty = penalty_lambda * (np.log1p(np.exp(penalty_beta * z)) / penalty_beta)
        grad_pen = - penalty_lambda * sig * g_p
        J = flip * mu + penalty
        gJ = flip * g_l + grad_pen
        return float(J), gJ.astype(float)

    # exploration objective: -EI with feasibility gate (no analytic grad)
    def obj_scalar_explore(x_full: np.ndarray) -> float:
        mu_l = gp_l.mean_only(x_full[None, :])[0]
        mu   = float(mu_l + cond_mean)
        sd   = float(gp_l.sd_at(x_full, include_observation_noise=True))
        ps   = float(gp_s.mean_only(x_full[None, :])[0])
        mu_signed, best_signed = _maybe_flip_for_direction(np.array([mu]), float(best_feasible), direction)
        gate = 1.0 / (1.0 + np.exp(-penalty_beta * (ps - success_threshold)))
        ei = float(_expected_improvement_minimize(mu_signed, np.array([sd]), best_signed)[0]) * gate
        return -ei  # minimize

    # numeric grad by central differences
    def _numeric_grad(fun_scalar, x_num: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        g = np.zeros_like(x_num, dtype=float)
        for i in range(x_num.size):
            e = np.zeros_like(x_num); e[i] = eps
            g[i] = (fun_scalar(x_num + e) - fun_scalar(x_num - e)) / (2.0 * eps)
        return g

    def sample_start():
        x = np.zeros(p, float)
        for j in numeric_idx:
            lo, hi = num_bounds[numeric_idx.index(j)]
            x[j] = lo if lo == hi else rng.uniform(lo, hi)
        for j, choices in choice_num.items():
            x[j] = choices[np.argmin(np.abs(choices - x[j]))]
        for j, v in fixed_num_std.items():
            x[j] = v
        return x

    # categorical combos
    cat_bases = list(groups.keys())
    combo_space = []
    for b in cat_bases:
        combo_space.append([cat_fixed[b]] if b in cat_fixed else cat_allowed[b])
    all_label_combos = list(itertools.product(*combo_space)) if combo_space else [()]

    # repulsion
    rep_sigma2 = float(repulsion) ** 2
    rep_weight = float(repulsion)

    def _is_dup(xa: np.ndarray, xb: np.ndarray, tol=1e-3) -> bool:
        # allow >1 per combo when there are no numeric free dims
        if xa.size == 0 and xb.size == 0:
            return False
        return bool(np.linalg.norm(xa - xb) < tol)

    def _accept_row(template, best_xnum, labels, labels_t, accepted_combo, accepted_global, rows):
        # compose full point
        x_full = template.copy()
        x_full[numeric_idx] = best_xnum
        for j, choices in choice_num.items():
            x_full[j] = float(choices[np.argmin(np.abs(choices - x_full[j]))])
        for idx in numeric_idx:
            lo, hi = num_bounds[numeric_idx.index(idx)]
            x_full[idx] = float(np.clip(x_full[idx], lo, hi))

        x_num_std = x_full[numeric_idx].copy()
        # dedupe (combo + global)
        if any(_is_dup(x_num_std, prev) for prev in accepted_combo):
            return False
        if any((labels_t == labt) and _is_dup(x_num_std, prev) for labt, prev in accepted_global):
            return False

        # accept
        accepted_combo.append(x_num_std)
        accepted_global.append((labels_t, x_num_std))

        mu_l, _ = gp_l.mean_and_grad(x_full); mu = float(mu_l + cond_mean)
        ps, _  = gp_s.mean_and_grad(x_full);  ps = float(np.clip(ps, 0.0, 1.0))
        sd_opt = float(gp_l.sd_at(x_full, include_observation_noise=True))

        row = {
            "pred_p_success": ps,
            "pred_target_mean": mu,
            "pred_target_sd": sd_opt,
        }
        onehot_members_local = {m for g in groups.values() for m in g["members"]}
        for j, nm in enumerate(feature_names):
            if nm in onehot_members_local:
                continue
            row[nm] = float(_std_to_orig(j, x_full[j], transforms, mu_f, sd_f))
        for b, lab in zip(cat_bases, labels):
            row[b] = lab

        rows.append(row)
        return True

    def _optimize_take(template, accepted_combo, use_explore):
        # inner objective in numeric subspace
        def f_g_only_num(x_num: np.ndarray):
            x_full = template.copy()
            x_full[numeric_idx] = x_num

            def add_repulsion(J: float, g_num: np.ndarray | None):
                nonlocal accepted_combo
                if accepted_combo and rep_sigma2 > 0.0 and rep_weight > 0.0:
                    for xk in accepted_combo:
                        d = x_num - xk
                        r2 = float(d @ d)
                        w = math.exp(-0.5 * r2 / rep_sigma2)
                        J += rep_weight * w
                        if g_num is not None:
                            g_num += rep_weight * w * (-d / rep_sigma2)
                return J, g_num

            if not use_explore:
                J, g = obj_grad_exploit(x_full)
                J, g_num = add_repulsion(J, g[numeric_idx])
                return float(J), g_num

            # exploration branch: -EI, numerical grad
            def scalar_for_grad(xn: np.ndarray) -> float:
                x_tmp = template.copy()
                x_tmp[numeric_idx] = xn
                J = obj_scalar_explore(x_tmp)
                # include repulsion inside scalar for finite-diff consistency
                if accepted_combo and rep_sigma2 > 0.0 and rep_weight > 0.0:
                    for xk in accepted_combo:
                        d = xn - xk
                        r2 = float(d @ d)
                        w = math.exp(-0.5 * r2 / rep_sigma2)
                        J += rep_weight * w
                return float(J)

            J = scalar_for_grad(x_num)
            g_num = _numeric_grad(scalar_for_grad, x_num, eps=1e-4)
            return float(J), g_num

        # collect best from multistarts
        from scipy.optimize import fmin_l_bfgs_b
        starts = [sample_start()[numeric_idx] for _ in range(n_starts)]
        if starts and starts[0].size == 0:
            # no numeric dims free → just return a zero-length vector
            return np.zeros((0,), float)

        best_val = None
        best_xnum = None
        explore_candidates: list[tuple[np.ndarray, float]] = []
        for x0 in starts:
            xopt, fval, _ = fmin_l_bfgs_b(
                func=lambda x: f_g_only_num(x),
                x0=x0,
                fprime=None,
                bounds=num_bounds,
                maxiter=max_iters,
            )
            fval = float(fval)
            if not use_explore:
                if (best_val is None) or (fval < best_val):
                    best_val = fval
                    best_xnum = xopt
            else:
                # candidate scored by gated EI (no repulsion in score)
                x_tmp = template.copy()
                x_tmp[numeric_idx] = xopt
                gated_ei = -obj_scalar_explore(x_tmp)
                if not any(np.linalg.norm(xopt - c[0]) < 1e-3 for c in explore_candidates):
                    explore_candidates.append((xopt, float(gated_ei)))

        if use_explore:
            if not explore_candidates:
                return None
            if softmax_temp and softmax_temp > 0.0:
                eis = np.array([ei for _, ei in explore_candidates], dtype=float)
                z = eis - np.max(eis)
                probs = np.exp(z / float(softmax_temp))
                probs = probs / probs.sum()
                idx = rng.choice(len(explore_candidates), p=probs)
                return explore_candidates[idx][0]
            # greedy EI
            idx = int(np.argmax([ei for _, ei in explore_candidates]))
            return explore_candidates[idx][0]
        return best_xnum

    # ---------------- Core loop with dynamic allocation ----------------
    rows: list[dict] = []
    accepted_global: list[tuple[tuple[str, ...], np.ndarray]] = []
    all_label_combos = all_label_combos or [()]

    n_combos = max(1, len(all_label_combos))
    for combo_idx, labels in enumerate(all_label_combos):
        if len(rows) >= count:
            break
        labels_t = tuple(labels) if labels else tuple()

        template = np.zeros(p, float)
        for b, lab in zip(cat_bases, labels):
            apply_onehot(template, b, lab)

        accepted_combo: list[np.ndarray] = []

        remain_total = count - len(rows)
        remain_combos = max(1, n_combos - combo_idx)
        k_each = max(1, math.ceil(remain_total / remain_combos))

        takes = 0
        while (takes < k_each) and (len(rows) < count):
            use_explore = (rng.random() < float(explore))
            best_xnum = _optimize_take(template, accepted_combo, use_explore)
            if best_xnum is None:
                # try the opposite mode once
                best_xnum = _optimize_take(template, accepted_combo, not use_explore)
                if best_xnum is None:
                    break  # give up this take for this combo
            ok = _accept_row(template, best_xnum, labels, labels_t, accepted_combo, accepted_global, rows)
            if ok:
                takes += 1

    # ---------------- Refill loop if still short ----------------
    if len(rows) < count:
        # relax repulsion and try a few refill rounds with fresh starts
        for _refill in range(3):
            if len(rows) >= count:
                break
            rep_sigma2 *= 0.7
            rep_weight *= 0.7
            for combo_idx, labels in enumerate(all_label_combos):
                if len(rows) >= count:
                    break
                labels_t = tuple(labels) if labels else tuple()
                template = np.zeros(p, float)
                for b, lab in zip(cat_bases, labels):
                    apply_onehot(template, b, lab)
                # start with an empty combo-accepted set to avoid over-repelling
                accepted_combo = []
                remain_total = count - len(rows)
                remain_combos = max(1, n_combos - combo_idx)
                k_each = max(1, math.ceil(remain_total / remain_combos))
                takes = 0
                while (takes < k_each) and (len(rows) < count):
                    use_explore = (rng.random() < float(explore))
                    best_xnum = _optimize_take(template, accepted_combo, use_explore)
                    if best_xnum is None:
                        best_xnum = _optimize_take(template, accepted_combo, not use_explore)
                        if best_xnum is None:
                            break
                    ok = _accept_row(template, best_xnum, labels, labels_t, accepted_combo, accepted_global, rows)
                    if ok:
                        takes += 1

    # ---------------- Last-resort fill with random projections ----------------
    # Only used if optimization couldn't find enough unique points but space likely allows more.
    if len(rows) < count:
        tries = 0
        max_tries = max(200, 20 * (count - len(rows)))
        while (len(rows) < count) and (tries < max_tries):
            tries += 1
            # random labels
            labels = []
            for b in cat_bases:
                pool = [cat_fixed[b]] if b in cat_fixed else cat_allowed[b]
                labels.append(pool[int(rng.integers(0, len(pool)))])
            labels_t = tuple(labels) if labels else tuple()
            # template
            template = np.zeros(p, float)
            for b, lab in zip(cat_bases, labels):
                apply_onehot(template, b, lab)
            # random numeric in bounds
            x_num = np.zeros(len(numeric_idx), float)
            for ii, j in enumerate(numeric_idx):
                lo, hi = num_bounds[ii]
                x_num[ii] = lo if lo == hi else rng.uniform(lo, hi)
            # accept (weak dedupe by numeric subspace)
            accepted_combo = []  # local (empty) so only global dedupe applies
            _accept_row(template, x_num, labels, labels_t, accepted_combo, accepted_global, rows)

    # ---------------- Assemble & rank ----------------
    if not rows:
        raise ValueError("No solutions produced; check constraints.")

    df = pd.DataFrame(rows)
    asc_mu = (direction != "max")
    df = df.sort_values(["pred_p_success", "pred_target_mean"],
                        ascending=[False, asc_mu],
                        kind="mergesort").reset_index(drop=True)
    # trim or pad (should be exact now, but keep the guard)
    if len(df) > count:
        df = df.head(count)
    df["rank"] = np.arange(1, len(df) + 1)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    try:
        console.print(f"\n[bold]Top {len(df)} suggested candidates:[/]")
        console.print(df_to_table(df))  # type: ignore[arg-type]
    except Exception:
        pass
    return df


def _collapse_onehot_to_categorical(df: pd.DataFrame, groups: dict[str, dict]) -> pd.DataFrame:
    """
    Collapse one-hot blocks (e.g. language=Linear A, language=Linear B) into a single
    categorical column 'language'. Leaves <NA> only if a row is ambiguous (sum!=1).
    """
    out = df.copy()

    for base, g in groups.items():
        # column order must match label order
        labels = list(g["labels"])
        member_cols = [g["name_by_label"][lab] for lab in labels if g["name_by_label"][lab] in out.columns]
        if not member_cols:
            continue

        # robust numeric block: NaN→0, float for safe sums/argmax
        block = out[member_cols].to_numpy(dtype=float)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

        row_sums = block.sum(axis=1)
        argmax   = np.argmax(block, axis=1)

        # exactly-one-hot per row (tolerant to tiny fp wiggle)
        valid = np.isfinite(row_sums) & (np.abs(row_sums - 1.0) <= 1e-9)

        chosen = np.full(len(out), None, dtype=object)
        if valid.any():
            lab_arr = np.array(labels, dtype=object)
            chosen[valid] = lab_arr[argmax[valid]]

        # write the categorical column with proper alignment
        out[base] = pd.Series(chosen, index=out.index, dtype="string")

        # drop the one-hot members
        out.drop(columns=[c for c in member_cols if c in out.columns], inplace=True)

    return out


def _inject_onehot_groups(
    cand_df: pd.DataFrame,
    groups: dict[str, dict],
    rng: np.random.Generator,
    cat_fixed_label: dict[str, str],
    cat_allowed: dict[str, set[str]],
) -> pd.DataFrame:
    """
    Ensure each one-hot block has exactly one '1' per row (or a fixed label),
    by initializing member columns to 0 then writing the chosen label as 1.
    """
    out = cand_df.copy()
    n = len(out)

    for base, g in groups.items():
        labels = g["labels"]
        member_cols = [g["name_by_label"][lab] for lab in labels]

        # Create/overwrite member columns with zeros to avoid NaNs
        for col in member_cols:
            out[col] = 0

        # Allowed labels for this base
        allowed = list(cat_allowed.get(base, set(labels)))
        if not allowed:
            allowed = labels

        # Choose a label per row
        if base in cat_fixed_label:
            chosen = np.full(n, cat_fixed_label[base], dtype=object)
        else:
            idx = rng.integers(0, len(allowed), size=n)
            chosen = np.array([allowed[i] for i in idx], dtype=object)

        # Set one-hot = 1 for the chosen label, keep others at 0
        for lab, col in zip(labels, member_cols):
            out.loc[chosen == lab, col] = 1

        # Enforce integer dtype (clean)
        out[member_cols] = out[member_cols].astype(int)

    return out


def _postfilter_numeric_constraints(
    df: pd.DataFrame,
    user_fixed_num: dict,
    user_ranges_num: dict,
    user_choices_num: dict,
) -> pd.DataFrame:
    """
    Keep rows satisfying numeric constraints (fixed / ranges / choices).
    Nonexistent columns are ignored.
    """
    if df.empty:
        return df

    mask = np.ones(len(df), dtype=bool)

    # ranges: inclusive
    for k, (lo, hi) in user_ranges_num.items():
        if k in df.columns:
            mask &= (df[k] >= lo) & (df[k] <= hi)

    # finite numeric choices
    for k, vals in user_choices_num.items():
        if k in df.columns:
            mask &= df[k].isin(vals)

    # fixed values (tolerate tiny float error)
    for k, val in user_fixed_num.items():
        if k in df.columns:
            col = df[k]
            if pd.api.types.is_integer_dtype(col.dtype):
                mask &= (col == int(round(val)))
            else:
                mask &= np.isfinite(col) & (np.abs(col - float(val)) <= 1e-12)

    return df.loc[mask].reset_index(drop=True)


def optimal(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    count: int = 10,                 # ignored (we always return 1)
    n_draws: int = 0,                # ignored (mean-only optimizer)
    success_threshold: float = 0.8,
    seed: int | np.random.Generator | None = 42,
    **kwargs,                        # constraints in ORIGINAL units
) -> pd.DataFrame:
    """
    Best single candidate by optimizing the GP *mean* posterior under constraints,
    using L-BFGS-B on standardized features. Like `suggest` but returns 1 row.

    Objective (we minimize):
        J(x) = flip * μ_loss(x) + λ * softplus( threshold - p_success(x) )

    Notes:
      • Categorical bases handled by enumeration over allowed labels.
      • Numeric choices are projected to nearest allowed value after optimize.
      • `count` and `n_draws` are ignored (kept for API compatibility).
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)

    rng = rng_for_dataset(ds, seed)

    # --- metadata
    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    mu_f = ds["feature_mean"].values.astype(float)
    sd_f = ds["feature_std"].values.astype(float)
    p    = len(feature_names)
    name_to_idx = {nm: j for j, nm in enumerate(feature_names)}
    groups = _groups_from_feature_names(feature_names)  # {base:{labels, name_by_label, members}}

    # --- GP heads (shared helper class)
    gp_s = _GPMarginal(
        Xtr=ds["Xn_train"].values.astype(float),
        ytr=ds["y_success"].values.astype(float),
        ell=ds["map_success_ell"].values.astype(float),
        eta=float(ds["map_success_eta"].values),
        sigma=float(ds["map_success_sigma"].values),
        mean_const=float(ds["map_success_beta0"].values),
    )
    cond_mean = float(ds["conditional_loss_mean"].values) if "conditional_loss_mean" in ds else 0.0
    gp_l = _GPMarginal(
        Xtr=ds["Xn_success_only"].values.astype(float),
        ytr=ds["y_loss_centered"].values.astype(float),
        ell=ds["map_loss_ell"].values.astype(float),
        eta=float(ds["map_loss_eta"].values),
        sigma=float(ds["map_loss_sigma"].values),
        mean_const=float(ds["map_loss_mean_const"].values),
    )

    # --- direction
    direction = str(ds.attrs.get("direction", "min"))
    flip = -1.0 if direction == "max" else 1.0

    # --- parse constraints (numeric vs categorical)
    cat_allowed: dict[str, list[str]] = {}
    cat_fixed: dict[str, str] = {}
    fixed_num_std: dict[int, float] = {}
    range_num_std: dict[int, tuple[float, float]] = {}
    choice_num: dict[int, np.ndarray] = {}

    # default allowed = all labels for each base
    for b, g in groups.items():
        cat_allowed[b] = list(g["labels"])

    import re
    def canon_key(k: str) -> str:
        raw = str(k)
        stripped = re.sub(r"[^a-z0-9]+", "", raw.lower())
        if raw in name_to_idx:
            return raw
        for base in groups.keys():
            if stripped == re.sub(r"[^a-z0-9]+", "", base.lower()):
                return base
        return raw

    for k, v in (kwargs or {}).items():
        ck = canon_key(k)
        if ck in groups:
            labels = groups[ck]["labels"]
            if isinstance(v, str):
                if v not in labels:
                    raise ValueError(f"Unknown category for {ck}: {v}. Choices: {labels}")
                cat_fixed[ck] = v
            else:
                L = [x for x in (list(v) if isinstance(v, (list, tuple, set)) else [v])
                     if isinstance(x, str) and x in labels]
                if not L:
                    raise ValueError(f"No valid categories for {ck} in {v}. Choices: {labels}")
                if len(L) == 1:
                    cat_fixed[ck] = L[0]
                else:
                    cat_allowed[ck] = L
        elif ck in name_to_idx:
            j = name_to_idx[ck]
            if isinstance(v, slice):
                lo = _orig_to_std(j, v.start, transforms, mu_f, sd_f)
                hi = _orig_to_std(j, v.stop,  transforms, mu_f, sd_f)
                lo, hi = float(np.nanmin([lo, hi])), float(np.nanmax([lo, hi]))
                range_num_std[j] = (lo, hi)
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = _orig_to_std(j, np.asarray(v, float), transforms, mu_f, sd_f)
                choice_num[j] = np.asarray(arr, float)
            else:
                fixed_num_std[j] = float(_orig_to_std(j, float(v), transforms, mu_f, sd_f))
        else:
            raise ValueError(f"Unknown constraint key: {k!r}")

    # --- numeric bounds (std space); allow outside training range
    Xn = ds["Xn_train"].values.astype(float)
    p01 = np.percentile(Xn, 1, axis=0)
    p99 = np.percentile(Xn, 99, axis=0)
    wide_lo = np.minimum(p01 - 1.0, -3.0)
    wide_hi = np.maximum(p99 + 1.0,  3.0)

    bounds: list[tuple[float, float] | None] = [None]*p
    for j in range(p):
        if j in fixed_num_std:
            v = fixed_num_std[j]
            bounds[j] = (v, v)
        elif j in range_num_std:
            bounds[j] = range_num_std[j]
        elif j in choice_num:
            lo = float(np.nanmin(choice_num[j])); hi = float(np.nanmax(choice_num[j]))
            bounds[j] = (lo, hi)
        else:
            bounds[j] = (float(wide_lo[j]), float(wide_hi[j]))

    # --- helpers
    def apply_onehot(vec_std: np.ndarray, base: str, label: str):
        for lab in groups[base]["labels"]:
            member_name = groups[base]["name_by_label"][lab]
            j = name_to_idx[member_name]
            raw = 1.0 if lab == label else 0.0
            vec_std[j] = _orig_to_std(j, raw, transforms, mu_f, sd_f)

    penalty_lambda = 1.0
    penalty_beta   = 10.0

    def obj_grad(x_std_full: np.ndarray) -> tuple[float, np.ndarray]:
        # mean + grad for loss head (centered); add back cond_mean
        mu_l, g_l = gp_l.mean_and_grad(x_std_full)
        mu = mu_l + cond_mean
        # success head (smooth, not clipped)
        mu_p, g_p = gp_s.mean_and_grad(x_std_full)
        # softplus penalty for p<thr
        z   = success_threshold - mu_p
        sig = 1.0 / (1.0 + np.exp(-penalty_beta * z))
        penalty = penalty_lambda * (np.log1p(np.exp(penalty_beta * z)) / penalty_beta)
        grad_pen = - penalty_lambda * sig * g_p
        J  = float(flip * mu + penalty)
        gJ = (flip * g_l + grad_pen).astype(float)
        return J, gJ

    # exclude one-hot members from numeric optimization
    onehot_members = {m for g in groups.values() for m in g["members"]}
    numeric_idx = [j for j, nm in enumerate(feature_names) if nm not in onehot_members]
    num_bounds = [bounds[j] for j in numeric_idx]

    # uniform starts inside numeric bounds (std space)
    def sample_start() -> np.ndarray:
        x = np.zeros(p, float)
        for j in numeric_idx:
            lo, hi = num_bounds[numeric_idx.index(j)]
            x[j] = lo if lo == hi else rng.uniform(lo, hi)
        for j, choices in choice_num.items():
            x[j] = choices[np.argmin(np.abs(choices - x[j]))]
        for j, v in fixed_num_std.items():
            x[j] = v
        return x

    # enumerate categorical combos (fixed → single)
    cat_bases = list(groups.keys())
    combo_space = []
    for b in cat_bases:
        combo_space.append([cat_fixed[b]] if b in cat_fixed else cat_allowed[b])
    all_label_combos = list(itertools.product(*combo_space)) if combo_space else [()]

    # --- optimize (multi-start) and keep the single best over all combos
    from scipy.optimize import fmin_l_bfgs_b
    n_starts   = 32
    max_iters  = 200

    best_global_val: float | None = None
    best_global_x:   np.ndarray | None = None
    best_global_labels: tuple[str, ...] | tuple = tuple()

    for labels in all_label_combos:
        template = np.zeros(p, float)
        for b, lab in zip(cat_bases, labels):
            apply_onehot(template, b, lab)

        # numeric-only wrapper
        def f_g_only_num(x_num: np.ndarray):
            x_full = template.copy()
            x_full[numeric_idx] = x_num
            J, g = obj_grad(x_full)
            return J, g[numeric_idx]

        # multi-starts
        starts = []
        for _ in range(n_starts):
            s = sample_start()
            for b, lab in zip(cat_bases, labels):
                apply_onehot(s, b, lab)
            starts.append(s[numeric_idx])

        # pick best for this combo
        best_val = None
        best_xnum = None
        for x0 in starts:
            xopt, fval, _ = fmin_l_bfgs_b(
                func=lambda x: f_g_only_num(x),
                x0=x0,
                fprime=None,
                bounds=num_bounds,
                maxiter=max_iters,
            )
            fval = float(fval)
            if (best_val is None) or (fval < best_val):
                best_val = fval
                best_xnum = xopt

        if best_xnum is None:
            continue

        # assemble full point, project choices, clip to bounds
        x_full = template.copy()
        x_full[numeric_idx] = best_xnum
        for j, choices in choice_num.items():
            x_full[j] = float(choices[np.argmin(np.abs(choices - x_full[j]))])
        for idx in numeric_idx:
            lo, hi = num_bounds[numeric_idx.index(idx)]
            x_full[idx] = float(np.clip(x_full[idx], lo, hi))

        if (best_global_val is None) or (best_val < best_global_val):
            best_global_val = float(best_val)
            best_global_x = x_full.copy()
            best_global_labels = tuple(labels) if labels else tuple()

    if best_global_x is None:
        raise ValueError("No feasible optimum produced; check/relax constraints.")

    # --- build single-row DataFrame in ORIGINAL units
    x_opt = best_global_x
    mu_l_opt, _ = gp_l.mean_and_grad(x_opt)
    mu_opt = float(mu_l_opt + cond_mean)
    p_opt, _  = gp_s.mean_and_grad(x_opt)
    p_opt = float(np.clip(p_opt, 0.0, 1.0))

    sd_opt = gp_l.sd_at(x_opt, include_observation_noise=True)

    onehot_members = {m for g in groups.values() for m in g["members"]}

    row: dict[str, object] = {
        "pred_p_success": p_opt,
        "pred_target_mean": mu_opt,
        "pred_target_sd": float(sd_opt),
        "rank": 1,
    }
    # numerics in original units (drop one-hot members)
    for j, nm in enumerate(feature_names):
        if nm in onehot_members:
            continue
        row[nm] = float(_std_to_orig(j, x_opt[j], transforms, mu_f, sd_f))
    # categorical base columns
    for b, lab in zip(cat_bases, best_global_labels):
        row[b] = lab

    df = pd.DataFrame([row])

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

    try:
        console.print(f"\n[bold]Optimal candidate (mean posterior):[/]")
        console.print(df_to_table(df))  # type: ignore[arg-type]
    except Exception:
        pass
    return df


def optimal_old(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    count: int = 10,
    n_draws: int = 0,
    success_threshold: float = 0.8,
    seed: int | np.random.Generator | None = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Rank candidates by probability of being the best feasible optimum (min/max),
    honoring numeric *and* categorical constraints.

    Constraints (original units):
      - number (int/float): fixed value, e.g. epochs=20
      - slice(lo, hi): inclusive float range, e.g. learning_rate=slice(1e-5, 1e-3)
      - list/tuple: finite numeric choices, e.g. batch_size=(16, 32, 64)
      - range(...): converted to tuple of ints (choices)
      - categorical base, e.g. language="Linear B" or language=("Linear A","Linear B")
        (use the *base* name; model stores one-hot members internally)
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    # --- model metadata
    feature_names = list(map(str, ds["feature"].values.tolist()))
    transforms    = list(map(str, ds["feature_transform"].values.tolist()))
    feat_mean     = ds["feature_mean"].values.astype(float)
    feat_std      = ds["feature_std"].values.astype(float)

    # --- detect categorical one-hot groups from feature names
    groups = _onehot_groups(feature_names)  # { base: {"labels":[...], "name_by_label":{label->member}, "members":[...]} }

    # --- infer numeric search specs from data (includes one-hot members but we’ll drop them below)
    specs_full = _infer_search_specs(ds, feature_names, transforms)

    # --- split user kwargs into numeric vs categorical constraints
    (groups,              # same structure as above (returned for convenience)
     user_fixed_num,      # {numeric_feature: value}
     user_ranges_num,     # {numeric_feature: (lo, hi)}
     user_choices_num,    # {numeric_feature: [choices]}
     cat_fixed_label,     # {base: "Label"}  (fixed single label)
     cat_allowed) = _split_constraints_for_numeric_and_categorical(feature_names, kwargs)

    # numeric fixed beats numeric ranges/choices
    for k in list(user_fixed_num.keys()):
        user_ranges_num.pop(k, None)
        user_choices_num.pop(k, None)

    # --- keep only *numeric* specs (drop one-hot members)
    numeric_specs = _numeric_specs_only(specs_full, groups)

    # apply numeric bounds/choices, normalize numeric fixed
    _apply_user_bounds(numeric_specs, user_ranges_num, user_choices_num)
    fixed_norm_num = _normalize_fixed(user_fixed_num, numeric_specs)

    # --- EI baseline: best feasible observed target
    direction = str(ds.attrs.get("direction", "min"))
    best_feasible = _best_feasible_observed(ds, direction)
    flip = -1.0 if direction == "max" else 1.0

    # --- sample candidate pool
    rng = get_rng(seed)
    target_pool = max(4000, count * 200)  # make sure MC has enough variety

    def _sample_pool(n: int) -> pd.DataFrame:
        # sample numerics
        base_num = _sample_candidates(numeric_specs, n=n, rng=rng, fixed=fixed_norm_num)
        # inject legal one-hot blocks for categoricals
        with_cats = _inject_onehot_groups(base_num, groups, rng, cat_fixed_label, cat_allowed)
        # hard filter numerics (ranges/choices/fixed)
        filtered = _postfilter_numeric_constraints(with_cats, user_fixed_num, user_ranges_num, user_choices_num)
        return filtered

    cand_df = _sample_pool(target_pool)
    # if tight constraints reduce pool too much, try a few refills
    attempts = 0
    while len(cand_df) < max(count * 50, 1000) and attempts < 6:
        extra = _sample_pool(target_pool)
        if not extra.empty:
            cand_df = pd.concat([cand_df, extra], ignore_index=True).drop_duplicates()
        attempts += 1

    if cand_df.empty:
        raise ValueError("No candidates satisfy the provided constraints; relax the ranges or choices.")

    # --- predictions in model space (use full feature order incl. one-hot members)
    Xn_cands = _original_df_to_standardized(cand_df[feature_names], feature_names, transforms, feat_mean, feat_std)
    p  = pred_success(Xn_cands)
    mu, sd = pred_loss(Xn_cands, include_observation_noise=True)
    sd = np.maximum(sd, 1e-12)

    # --- optional feasibility filter
    keep = p >= float(success_threshold)
    if not np.any(keep):
        keep = np.ones_like(p, dtype=bool)

    cand_df = cand_df.loc[keep].reset_index(drop=True)
    Xn_cands = Xn_cands[keep]
    p = p[keep]; mu = mu[keep]; sd = sd[keep]
    N = len(cand_df)
    if N == 0:
        raise ValueError("All sampled candidates were filtered out by success_threshold.")

    # --- mean-only mode when n_draws == 0
    if int(n_draws) <= 0:
        result = cand_df.copy()
        result["pred_p_success"]   = p
        result["pred_target_mean"] = mu
        result["pred_target_sd"]   = sd
        # keep columns for API parity
        result["wins"] = 0
        result["n_draws_effective"] = 0
        result["prob_best_feasible"] = 0.0
        result["conditioned_on"] = _pretty_conditioned_on(
            fixed_norm_numeric=fixed_norm_num,
            cat_fixed_label=cat_fixed_label,
        )

        # Direction-aware sort by μ, then lower σ, then higher p
        if str(ds.attrs.get("direction", "min")) == "max":
            sort_cols = ["pred_target_mean", "pred_target_sd", "pred_p_success"]
            ascending = [False, True, False]
        else:  # "min"
            sort_cols = ["pred_target_mean", "pred_target_sd", "pred_p_success"]
            ascending = [True,  True, False]

        result_sorted = result.sort_values(
            sort_cols, ascending=ascending, kind="mergesort"
        ).reset_index(drop=True)
        result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)

        top = result_sorted.head(count).reset_index(drop=True)
        # collapse one-hot → single categorical columns (e.g., 'language')
        top_view = _collapse_onehot_to_categorical(top, groups)

        if output:
            top_view.to_csv(output, index=False)

        console.print(f"\n[bold]Top {len(top_view)} optimal solutions (mean-only, n_draws=0):[/]")
        console.print(df_to_table(top_view))
        return top_view

    # --- Monte Carlo winner-take-all over feasible draws
    Z = mu[:, None] + sd[:, None] * rng.standard_normal((N, n_draws))
    success_mask = rng.random((N, n_draws)) < p[:, None]
    feasible_draw = success_mask.any(axis=0)
    if not feasible_draw.any():
        # fallback: deterministic sort (rare)
        result = cand_df.copy()
        result["pred_p_success"] = p
        result["pred_target_mean"] = mu
        result["pred_target_sd"] = sd
        result["prob_best_feasible"] = 0.0
        result["wins"] = 0
        result["n_draws_effective"] = 0
        # prettify conditioning (numeric fixed + categorical fixed)
        result["conditioned_on"] = _pretty_conditioned_on(
            fixed_norm_numeric=fixed_norm_num,
            cat_fixed_label=cat_fixed_label,
        )
        result_sorted = result.sort_values(
            ["pred_target_mean", "pred_target_sd", "pred_p_success"],
            ascending=[True, True, False],
            kind="mergesort",
        ).reset_index(drop=True)
        result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)
        top = result_sorted.head(count).reset_index(drop=True)
        # collapse one-hot → single categorical columns for output
        top_view = _collapse_onehot_to_categorical(top, groups)
        if output:
            top_view.to_csv(output, index=False)
        console.print(f"\n[bold]Top {len(top_view)} optimal solutions:[/]")
        console.print(df_to_table(top_view))
        return top_view

    Z_eff = flip * np.where(success_mask, Z, np.inf)
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
    result["conditioned_on"] = _pretty_conditioned_on(
        fixed_norm_numeric=fixed_norm_num,
        cat_fixed_label=cat_fixed_label,
    )

    result_sorted = result.sort_values(
        ["prob_best_feasible", "pred_p_success", "pred_target_mean", "pred_target_sd"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result_sorted["rank_prob_best"] = np.arange(1, len(result_sorted) + 1)

    top = result_sorted.head(count).reset_index(drop=True)
    # collapse one-hot → single categorical columns (e.g. 'language')
    top_view = _collapse_onehot_to_categorical(top, groups)

    if output:
        top_view.to_csv(output, index=False)

    console.print(f"\n[bold]Top {len(top_view)} optimal solutions:[/]")
    console.print(df_to_table(top_view))
    return top_view



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
    for j, name in enumerate(feature_names):
        if name in df_raw.columns:
            vals = pd.to_numeric(pd.Series(df_raw[name]), errors="coerce").dropna().to_numpy()
        else:
            # fallback: reconstruct original units from standardized arrays if needed
            # (in your artifact, raw columns are stored; so this path is rarely used)
            try:
                base_vals = ds[name].values  # raw per-row column, if present
            except KeyError:
                # Not stored as a data_var (e.g., one-hot feature); reconstruct from Xn_train
                # j is the feature index in feature_names; transforms[j] is 'identity' or 'log10'
                base_vals = feature_raw_from_artifact_or_reconstruct(ds, j, name, transforms[j])

            vals = pd.to_numeric(pd.Series(base_vals), errors="coerce").dropna().to_numpy()
            

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
