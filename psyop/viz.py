# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import get_colorscale, sample_colorscale

from .model import (
    kernel_diag_m52, 
    kernel_m52_ard,
    add_jitter,
    solve_chol,
    solve_lower,
    feature_raw_from_artifact_or_reconstruct,
)
from . import opt


def _canon_key_set(ds) -> dict[str, str]:
    feats = [str(x) for x in ds["feature"].values.tolist()]
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    return {**{f: f for f in feats}, **{_norm(f): f for f in feats}}


def _edges_from_centers(vals: np.ndarray, is_log: bool) -> tuple[float, float]:
    """Return (min_edge, max_edge) that tightly bound a heatmap with given center coords."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)
    if v.size == 1:
        # tiny pad
        if is_log:
            lo = max(v[0] / 1.5, 1e-12)
            hi = v[0] * 1.5
        else:
            span = max(abs(v[0]) * 0.5, 1e-9)
            lo, hi = v[0] - span, v[0] + span
        return float(lo), float(hi)

    if is_log:
        lv = np.log10(v)
        l0 = lv[0] - 0.5 * (lv[1] - lv[0])
        lN = lv[-1] + 0.5 * (lv[-1] - lv[-2])
        lo = 10.0 ** l0
        hi = 10.0 ** lN
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.0000001)
        return float(lo), float(hi)
    else:
        d0 = v[1] - v[0]
        dN = v[-1] - v[-2]
        lo = v[0] - 0.5 * d0
        hi = v[-1] + 0.5 * dN
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        return float(lo), float(hi)

def _update_axis_type_and_range(
    fig, *, row: int, col: int, axis: str, centers: np.ndarray, is_log: bool
):
    """Set axis type and range to heatmap edges so tiles meet the axes exactly."""
    lo, hi = _edges_from_centers(centers, is_log)
    if axis == "x":
        if is_log:
            fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], row=row, col=col)
        else:
            fig.update_xaxes(range=[lo, hi], row=row, col=col)
    else:
        if is_log:
            fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], row=row, col=col)
        else:
            fig.update_yaxes(range=[lo, hi], row=row, col=col)


def plot2d(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    grid_size: int = 70,
    use_log_scale_for_target: bool = False,   # log10 colors for heatmaps
    log_shift_epsilon: float = 1e-9,
    colorscale: str = "RdBu",
    show: bool = False,
    n_contours: int = 12,
    optimal: bool = True,
    suggest: int = 0,
    seed: int|None = 42,
    width:int|None = None,
    height:int|None = None,
    **kwargs,
) -> go.Figure:
    """
    2D Partial Dependence of E[target|success] (pairwise features), including
    categorical variables as single axes (one row/column per base).

    Conditioning via kwargs (original units):
      - numeric: fixed scalar, slice(lo,hi), list/tuple choices
      - categorical base (e.g. language="Linear A" or language=("Linear A","Linear B")):
        * single string: fixed to that label (axis removed and clamped)
        * list/tuple of labels: restrict the categorical axis to those labels

    Notes:
      * one-hot member features (e.g. language=Linear A) never appear as axes.
      * when a categorical axis is present, we render a heatmap over category index
        with tick labels set to the category names; data overlays use jitter.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    # --- features & transforms
    feature_names = [str(x) for x in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std  = ds["feature_std"].values.astype(float)
    name_to_idx = {nm: i for i, nm in enumerate(feature_names)}

    # one-hot groups
    groups = opt._onehot_groups(feature_names)  # { base: {"labels":[...], "name_by_label":{label->member}, "members":[...] } }
    bases  = set(groups.keys())
    onehot_member_names = {m for g in groups.values() for m in g["members"]}

    # raw df + train design
    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    n_rows   = Xn_train.shape[0]

    # --- split kwargs into numeric vs categorical (keys are canonical already when coming from CLI)
    kw_num: dict[str, object] = {}
    kw_cat: dict[str, object] = {}
    for k, v in (kwargs or {}).items():
        if k in bases:
            kw_cat[k] = v
        elif k in name_to_idx:
            kw_num[k] = v
        else:
            # unknown/ignored
            pass

    # --- resolve categorical constraints:
    #     - cat_fixed[base] -> fixed single label (axis removed and clamped)
    #     - cat_allowed[base] -> labels that are allowed on that axis (if not fixed)
    cat_fixed: dict[str, str] = {}
    cat_allowed: dict[str, list[str]] = {}
    for base in bases:
        labels = list(groups[base]["labels"])
        if base not in kw_cat:
            cat_allowed[base] = labels  # unrestricted axis (if not fixed numerically later)
            continue
        val = kw_cat[base]
        if isinstance(val, str):
            if val not in labels:
                raise ValueError(f"Unknown category for {base!r}: {val!r}. Choices: {labels}")
            cat_fixed[base] = val
        elif isinstance(val, (list, tuple, set)):
            chosen = [x for x in val if isinstance(x, str) and x in labels]
            if not chosen:
                raise ValueError(f"No valid categories for {base!r} in {val!r}. Choices: {labels}")
            # if only one remains, treat as fixed; else allowed list
            if len(chosen) == 1:
                cat_fixed[base] = chosen[0]
            else:
                cat_allowed[base] = chosen
        else:
            raise ValueError(f"Categorical constraint for {base!r} must be a string or list/tuple of strings.")

    # --- filter rows to categorical *fixed* selections for medians/percentiles & overlays
    row_mask = np.ones(n_rows, dtype=bool)
    for base, label in cat_fixed.items():
        if base in df_raw.columns:
            row_mask &= (df_raw[base].astype("string") == pd.Series([label]*len(df_raw), dtype="string")).to_numpy()
        else:
            member = groups[base]["name_by_label"][label]
            j = name_to_idx[member]
            raw_j = feature_raw_from_artifact_or_reconstruct(ds, j, member, transforms[j]).astype(float)
            row_mask &= (raw_j >= 0.5)

    df_raw_f = df_raw.loc[row_mask].reset_index(drop=True) if cat_fixed else df_raw
    Xn_train_f = Xn_train[row_mask, :] if cat_fixed else Xn_train

    # --- numeric constraints (standardized)
    def _orig_to_std(j: int, x, transforms, mu, sd):
        x = np.asarray(x, dtype=float)
        if transforms[j] == "log10":
            x = np.where(x <= 0, np.nan, x)
            x = np.log10(x)
        return (x - mu[j]) / sd[j]

    fixed_scalars_std: dict[int, float] = {}
    range_windows_std: dict[int, tuple[float, float]] = {}
    choice_values_std: dict[int, np.ndarray] = {}

    for name, val in kw_num.items():
        j = name_to_idx[name]
        if isinstance(val, slice):
            lo = _orig_to_std(j, float(val.start), transforms, X_mean, X_std)
            hi = _orig_to_std(j, float(val.stop),  transforms, X_mean, X_std)
            lo, hi = float(min(lo, hi)), float(max(lo, hi))
            range_windows_std[j] = (lo, hi)
        elif isinstance(val, (list, tuple, np.ndarray)):
            arr = _orig_to_std(j, np.asarray(val, dtype=float), transforms, X_mean, X_std)
            choice_values_std[j] = np.asarray(arr, dtype=float)
        else:
            fixed_scalars_std[j] = float(_orig_to_std(j, float(val), transforms, X_mean, X_std))

    # --- apply categorical *fixed* selections as standardized 0/1 on their member features
    for base, label in cat_fixed.items():
        labels = groups[base]["labels"]
        for lab in labels:
            member = groups[base]["name_by_label"][lab]
            j = name_to_idx[member]
            raw_val = 1.0 if (lab == label) else 0.0
            fixed_scalars_std[j] = float(_orig_to_std(j, raw_val, transforms, X_mean, X_std))

    # --- free axes = numeric features not scalar-fixed & not one-hot members, plus categorical bases not fixed
    free_numeric_idx = [
        j for j, nm in enumerate(feature_names)
        if (j not in fixed_scalars_std) and (nm not in onehot_member_names)
    ]
    free_cat_bases = [b for b in bases if b not in cat_fixed]  # we already filtered by allowed above

    panels: list[tuple[str, object]] = [("num", j) for j in free_numeric_idx] + [("cat", b) for b in free_cat_bases]
    if not panels:
        raise ValueError("All features are fixed (or only single-category categoricals remain); nothing to plot.")

    # --- base point (median in standardized space of filtered rows), then apply scalar fixes
    base_std = np.median(Xn_train_f, axis=0)
    for j, vstd in fixed_scalars_std.items():
        base_std[j] = vstd

    # --- per-feature grids (numeric) over filtered 1–99% + respecting ranges/choices
    p01p99 = [np.percentile(Xn_train_f[:, j], [1, 99]) for j in range(len(feature_names))]
    def _grid_std_num(j: int) -> np.ndarray:
        p01, p99 = p01p99[j]
        if j in choice_values_std:
            vals = np.asarray(choice_values_std[j], dtype=float)
            vals = vals[(vals >= p01) & (vals <= p99)]
            return np.unique(np.sort(vals)) if vals.size else np.array([np.median(Xn_train_f[:, j])])
        lo, hi = p01, p99
        if j in range_windows_std:
            rlo, rhi = range_windows_std[j]
            lo, hi = max(lo, rlo), min(hi, rhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-9
        return np.linspace(lo, hi, grid_size)

    grids_std_num = {j: _grid_std_num(j) for j in free_numeric_idx}

    # --- helpers for categorical evaluation ---------------------------------
    def _std_for_member(member_name: str, raw01: float) -> float:
        j = name_to_idx[member_name]
        return float(_orig_to_std(j, raw01, transforms, X_mean, X_std))

    def _apply_onehot_for_base(Xn_block: np.ndarray, base: str, label: str) -> None:
        # set the whole block's rows to the 0/1 standardized values for this label
        for lab in groups[base]["labels"]:
            member = groups[base]["name_by_label"][lab]
            j = name_to_idx[member]
            Xn_block[:, j] = _std_for_member(member, 1.0 if lab == label else 0.0)

    def _denorm_inv(j: int, std_vals: np.ndarray) -> np.ndarray:
        internal = std_vals * X_std[j] + X_mean[j]
        return _inverse_transform(transforms[j], internal)

    # 1) Robustly detect one-hot member columns.
    #    Use both the detector output AND a fallback "base=" prefix scan,
    #    so any columns like "language=Linear A" are guaranteed to be excluded.
    onehot_member_names: set[str] = set()
    for base, g in groups.items():
        # detector-known members
        onehot_member_names.update(g["members"])
        # prefix fallback
        prefix = f"{base}="
        onehot_member_names.update([nm for nm in feature_names if nm.startswith(prefix)])

    # 2) Build panel list: keep numeric features that are not scalar-fixed AND
    #    are not one-hot members; plus categorical bases that are not fixed.
    free_numeric_idx = [
        j for j, nm in enumerate(feature_names)
        if (j not in fixed_scalars_std) and (nm not in onehot_member_names)
    ]
    free_cat_bases = [b for b in bases if b not in cat_fixed]

    panels: list[tuple[str, object]] = [("num", j) for j in free_numeric_idx] + [("cat", b) for b in free_cat_bases]
    if not panels:
        raise ValueError("All features are fixed (or only single-category categoricals remain); nothing to plot.")

    # 3) Sanity check: no one-hot member should survive as a numeric panel.
    assert all(
        (feature_names[key] not in onehot_member_names) if kind == "num" else True
        for kind, key in panels
    ), "internal: one-hot member leaked into numeric panels"

    # 4) Subplot scaffold (matrix layout k x k) with clear titles.
    def _panel_title(kind: str, key: object) -> str:
        return feature_names[int(key)] if kind == "num" else str(key)

    k = len(panels)
    fig = make_subplots(
        rows=k,
        cols=k,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
        subplot_titles=[_panel_title(kind, key) for kind, key in panels],
    )

    # (Keep the rest of your cell-evaluation and rendering logic unchanged.
    #  Because we filtered `onehot_member_names`, rows/columns like
    #  "language=Linear A" / "language=Linear B" will no longer appear.
    #  Categorical bases (e.g., "language") will show as a single axis.)


    # overlays prepared under the SAME constraints (pass original kwargs straight through)
    optimal_df = opt.optimal(ds, count=1, seed=seed, **kwargs) if optimal else None
    suggest_df = opt.suggest(ds, count=suggest, seed=seed, **kwargs) if (suggest and suggest > 0) else None

    # masks for data overlays (already filtered if cat_fixed)
    tgt_col = str(ds.attrs["target"])
    success_mask = ~pd.isna(df_raw_f[tgt_col]).to_numpy()
    fail_mask    = ~success_mask

    # collect Z blocks for global color bounds
    all_blocks: list[np.ndarray] = []
    cell_payload: dict[tuple[int,int], dict] = {}

    # --- build each cell payload (numeric/num, cat/num, num/cat, cat/cat)
    for r, (kind_r, key_r) in enumerate(panels):
        for c, (kind_c, key_c) in enumerate(panels):
            # X axis = column; Y axis = row
            if kind_r == "num" and kind_c == "num":
                i = int(key_r); j = int(key_c)
                xg = grids_std_num[j]; yg = grids_std_num[i]
                if i == j:
                    grid = grids_std_num[j]
                    Xn_1d = np.repeat(base_std[None, :], len(grid), axis=0)
                    Xn_1d[:, j] = grid
                    mu_1d, _ = pred_loss(Xn_1d, include_observation_noise=True)
                    p_1d     = pred_success(Xn_1d)
                    Zmu = 0.5 * (mu_1d[:, None] + mu_1d[None, :])
                    Zp  = np.minimum(p_1d[:, None], p_1d[None, :])
                    x_orig = _denorm_inv(j, grid)
                    y_orig = x_orig
                else:
                    XX, YY = np.meshgrid(xg, yg)
                    Xn_grid = np.repeat(base_std[None, :], XX.size, axis=0)
                    Xn_grid[:, j] = XX.ravel()
                    Xn_grid[:, i] = YY.ravel()
                    mu_flat, _ = pred_loss(Xn_grid, include_observation_noise=True)
                    p_flat     = pred_success(Xn_grid)
                    Zmu = mu_flat.reshape(YY.shape)
                    Zp  = p_flat.reshape(YY.shape)
                    x_orig = _denorm_inv(j, xg)
                    y_orig = _denorm_inv(i, yg)
                cell_payload[(r, c)] = dict(kind=("num","num"), i=i, j=j, x=x_orig, y=y_orig, Zmu=Zmu, Zp=Zp)

            elif kind_r == "cat" and kind_c == "num":
                base = str(key_r); j = int(key_c)
                labels = list(cat_allowed.get(base, groups[base]["labels"]))
                xg = grids_std_num[j]
                # build rows per label
                Zmu_rows = []; Zp_rows = []
                for lab in labels:
                    Xn_grid = np.repeat(base_std[None, :], len(xg), axis=0)
                    Xn_grid[:, j] = xg
                    _apply_onehot_for_base(Xn_grid, base, lab)
                    mu_row, _ = pred_loss(Xn_grid, include_observation_noise=True)
                    p_row     = pred_success(Xn_grid)
                    Zmu_rows.append(mu_row[None, :])
                    Zp_rows.append(p_row[None, :])
                Zmu = np.concatenate(Zmu_rows, axis=0)  # (n_labels, n_x)
                Zp  = np.concatenate(Zp_rows,  axis=0)
                x_orig = _denorm_inv(j, xg)
                y_cats = labels  # categorical ticks
                cell_payload[(r,c)] = dict(kind=("cat","num"), base=base, j=j, x=x_orig, y=y_cats, Zmu=Zmu, Zp=Zp)

            elif kind_r == "num" and kind_c == "cat":
                i = int(key_r); base = str(key_c)
                labels = list(cat_allowed.get(base, groups[base]["labels"]))
                yg = grids_std_num[i]
                # columns per label
                Zmu_cols = []; Zp_cols = []
                for lab in labels:
                    Xn_grid = np.repeat(base_std[None, :], len(yg), axis=0)
                    Xn_grid[:, i] = yg
                    _apply_onehot_for_base(Xn_grid, base, lab)
                    mu_col, _ = pred_loss(Xn_grid, include_observation_noise=True)
                    p_col     = pred_success(Xn_grid)
                    Zmu_cols.append(mu_col[:, None])
                    Zp_cols.append(p_col[:, None])
                Zmu = np.concatenate(Zmu_cols, axis=1)  # (n_y, n_labels)
                Zp  = np.concatenate(Zp_cols,  axis=1)
                x_cats = labels
                y_orig = _denorm_inv(i, yg)
                cell_payload[(r,c)] = dict(kind=("num","cat"), i=i, base=base, x=x_cats, y=y_orig, Zmu=Zmu, Zp=Zp)

            else:  # kind_r == "cat" and kind_c == "cat"
                base_r = str(key_r); base_c = str(key_c)
                labels_r = list(cat_allowed.get(base_r, groups[base_r]["labels"]))
                labels_c = list(cat_allowed.get(base_c, groups[base_c]["labels"]))
                Z = np.zeros((len(labels_r), len(labels_c)), dtype=float)
                P = np.zeros_like(Z)
                # evaluate each pair
                for rr, lab_r in enumerate(labels_r):
                    for cc, lab_c in enumerate(labels_c):
                        Xn_grid = base_std[None, :].copy()
                        _apply_onehot_for_base(Xn_grid, base_r, lab_r)
                        _apply_onehot_for_base(Xn_grid, base_c, lab_c)
                        mu_val, _ = pred_loss(Xn_grid, include_observation_noise=True)
                        p_val     = pred_success(Xn_grid)
                        Z[rr, cc] = float(mu_val[0])
                        P[rr, cc] = float(p_val[0])
                cell_payload[(r,c)] = dict(kind=("cat","cat"), x=labels_c, y=labels_r, Zmu=Z, Zp=P)

            all_blocks.append(cell_payload[(r,c)]["Zmu"].ravel())

    # --- color transform bounds
    def _color_xform(z_raw: np.ndarray) -> tuple[np.ndarray, float]:
        if not use_log_scale_for_target:
            return z_raw, 0.0
        zmin = float(np.nanmin(z_raw))
        shift = 0.0 if zmin > 0 else -zmin + float(log_shift_epsilon)
        return np.log10(np.maximum(z_raw + shift, log_shift_epsilon)), shift

    z_all = np.concatenate(all_blocks) if all_blocks else np.array([0.0, 1.0])
    z_all_t, global_shift = _color_xform(z_all)
    cmin_t = float(np.nanmin(z_all_t))
    cmax_t = float(np.nanmax(z_all_t))
    cs = get_colorscale(colorscale)

    def _contour_line_color(level_raw: float) -> str:
        zt = np.log10(max(level_raw + global_shift, log_shift_epsilon)) if use_log_scale_for_target else level_raw
        t = 0.5 if cmax_t == cmin_t else (zt - cmin_t) / (cmax_t - cmin_t)
        rgb = sample_colorscale(cs, [float(np.clip(t, 0.0, 1.0))])[0]
        r, g, b = _rgb_string_to_tuple(rgb)
        lum = (0.2126*r + 0.7152*g + 0.0722*b)/255.0
        grey = int(round((1.0 - lum) * 255))
        return f"rgba({grey},{grey},{grey},0.9)"

    # --- render cells
    def _is_log_feature(j: int) -> bool: return (transforms[j] == "log10")

    for (r, c), PAY in cell_payload.items():
        kind = PAY["kind"]; Zmu_raw = PAY["Zmu"]; Zp = PAY["Zp"]
        Z_t, _ = _color_xform(Zmu_raw)

        # axes values (numeric arrays or category indices)
        if kind == ("num","num"):
            x_vals = PAY["x"]; y_vals = PAY["y"]
            fig.add_trace(go.Heatmap(
                x=x_vals, y=y_vals, z=Z_t,
                coloraxis="coloraxis", zsmooth=False, showscale=False,
                hovertemplate=(f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                               f"{feature_names[PAY['i']]}: %{{y:.6g}}"
                               "<br>E[target|success]: %{customdata:.3f}<extra></extra>"),
                customdata=Zmu_raw
            ), row=r+1, col=c+1)

            # p(success) shading + contours
            for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
                mask = np.where(Zp < thr, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    x=x_vals, y=y_vals, z=mask, zmin=0, zmax=1,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                    showscale=False, hoverinfo="skip"
                ), row=r+1, col=c+1)

            # contour lines
            zmin_r, zmax_r = float(np.nanmin(Zmu_raw)), float(np.nanmax(Zmu_raw))
            levels = np.linspace(zmin_r, zmax_r, max(n_contours, 2))
            for lev in levels:
                color = _contour_line_color(lev)
                fig.add_trace(go.Contour(
                    x=x_vals, y=y_vals, z=Zmu_raw,
                    autocontour=False,
                    contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                    line=dict(width=1),
                    colorscale=[[0, color], [1, color]],
                    showscale=False, hoverinfo="skip"
                ), row=r+1, col=c+1)

            # data overlays (success/fail)
            def _data_vals_for_feature(j_full: int) -> np.ndarray:
                nm = feature_names[j_full]
                if nm in df_raw_f.columns:
                    return df_raw_f[nm].to_numpy().astype(float)
                return feature_raw_from_artifact_or_reconstruct(ds, j_full, nm, transforms[j_full]).astype(float)[row_mask] \
                       if cat_fixed else \
                       feature_raw_from_artifact_or_reconstruct(ds, j_full, nm, transforms[j_full]).astype(float)

            xd = _data_vals_for_feature(PAY["j"])
            yd = _data_vals_for_feature(PAY["i"])
            show_leg = (r == 0 and c == 0)
            fig.add_trace(go.Scattergl(
                x=xd[success_mask], y=yd[success_mask], mode="markers",
                marker=dict(size=4, color="black", line=dict(width=0)),
                name="data (success)", legendgroup="data_succ", showlegend=show_leg,
                hovertemplate=("trial_id: %{customdata[0]}<br>"
                               f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                               f"{feature_names[PAY['i']]}: %{{y:.6g}}<br>"
                               f"{tgt_col}: %{{customdata[1]:.4f}}<extra></extra>"),
                customdata=np.column_stack([
                    df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask],
                    df_raw_f[tgt_col].to_numpy()[success_mask],
                ])
            ), row=r+1, col=c+1)
            fig.add_trace(go.Scattergl(
                x=xd[fail_mask], y=yd[fail_mask], mode="markers",
                marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
                name="data (failed)", legendgroup="data_fail", showlegend=show_leg,
                hovertemplate=("trial_id: %{customdata}<br>"
                               f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                               f"{feature_names[PAY['i']]}: %{{y:.6g}}<br>"
                               "status: failed (NaN target)<extra></extra>"),
                customdata=df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[fail_mask]
            ), row=r+1, col=c+1)

            # overlays (optimal/suggest) on numeric axes only
            if optimal and (optimal_df is not None):
                if feature_names[PAY["j"]] in optimal_df.columns and feature_names[PAY["i"]] in optimal_df.columns:
                    ox = np.asarray(optimal_df[feature_names[PAY["j"]]].values, dtype=float)
                    oy = np.asarray(optimal_df[feature_names[PAY["i"]]].values, dtype=float)
                    if np.isfinite(ox).all() and np.isfinite(oy).all():
                        pmu = float(optimal_df["pred_target_mean"].values[0])
                        psd = float(optimal_df["pred_target_sd"].values[0])
                        fig.add_trace(go.Scattergl(
                            x=ox, y=oy, mode="markers",
                            marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                            name="optimal", legendgroup="optimal", showlegend=(r == 0 and c == 0),
                            hovertemplate=(f"predicted: {pmu:.2g} ± {psd:.2g}<br>"
                                           f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                                           f"{feature_names[PAY['i']]}: %{{y:.6g}}<extra></extra>")
                        ), row=r+1, col=c+1)
            if suggest and (suggest_df is not None):
                have = (feature_names[PAY["j"]] in suggest_df.columns) and (feature_names[PAY["i"]] in suggest_df.columns)
                if have:
                    sx = np.asarray(suggest_df[feature_names[PAY["j"]]].values, dtype=float)
                    sy = np.asarray(suggest_df[feature_names[PAY["i"]]].values, dtype=float)
                    keep_s = np.isfinite(sx) & np.isfinite(sy)
                    if keep_s.any():
                        sx, sy = sx[keep_s], sy[keep_s]
                        mu_s = suggest_df.loc[keep_s, "pred_target_mean"].values if "pred_target_mean" in suggest_df else None
                        sd_s = suggest_df.loc[keep_s, "pred_target_sd"].values   if "pred_target_sd"   in suggest_df else None
                        ps_s = suggest_df.loc[keep_s, "pred_p_success"].values   if "pred_p_success"   in suggest_df else None
                        if (mu_s is not None) and (sd_s is not None) and (ps_s is not None):
                            custom_s = np.column_stack([mu_s, sd_s, ps_s])
                            hover_s = (
                                f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                                f"{feature_names[PAY['i']]}: %{{y:.6g}}<br>"
                                "pred: %{customdata[0]:.3g} ± %{customdata[1]:.3g}<br>"
                                "p(success): %{customdata[2]:.2f}<extra>suggested</extra>"
                            )
                        else:
                            custom_s = None
                            hover_s = (
                                f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                                f"{feature_names[PAY['i']]}: %{{y:.6g}}<extra>suggested</extra>"
                            )
                        fig.add_trace(go.Scattergl(
                            x=sx, y=sy, mode="markers",
                            marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                            name="suggested", legendgroup="suggested",
                            showlegend=(r == 0 and c == 0),
                            customdata=custom_s, hovertemplate=hover_s
                        ), row=r+1, col=c+1)

            # axis types/ranges
            _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="x", centers=x_vals, is_log=_is_log_feature(PAY["j"]))
            _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="y", centers=y_vals, is_log=_is_log_feature(PAY["i"]))

        elif kind == ("cat","num"):
            base = PAY["base"]; x_vals = PAY["x"]; labels = PAY["y"]
            nlab = len(labels)
            # heatmap (categories on Y)
            fig.add_trace(go.Heatmap(
                x=x_vals, y=np.arange(nlab), z=Z_t,
                coloraxis="coloraxis", zsmooth=False, showscale=False,
                hovertemplate=(f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                               f"{base}: %{{text}}"
                               "<br>E[target|success]: %{customdata:.3f}<extra></extra>"),
                text=np.array(labels)[:, None].repeat(len(x_vals), axis=1),
                customdata=Zmu_raw
            ), row=r+1, col=c+1)
            # p(success) shading
            for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
                mask = np.where(Zp < thr, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    x=x_vals, y=np.arange(nlab), z=mask, zmin=0, zmax=1,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                    showscale=False, hoverinfo="skip"
                ), row=r+1, col=c+1)
            # categorical ticks
            fig.update_yaxes(tickmode="array", tickvals=list(range(nlab)), ticktext=labels, row=r+1, col=c+1)
            # data overlays: numeric vs categorical with jitter on Y
            if base in df_raw_f.columns and feature_names[PAY["j"]] in df_raw_f.columns:
                cat_series = df_raw_f[base].astype("string")
                cat_to_idx = {lab: i for i, lab in enumerate(labels)}
                y_map = cat_series.map(cat_to_idx)
                ok = y_map.notna().to_numpy()
                y_idx = y_map.to_numpy(dtype=float)
                jitter = 0.10 * (np.random.default_rng(0).standard_normal(size=len(y_idx)))
                yj = y_idx + jitter
                xd = df_raw_f[feature_names[PAY["j"]]].to_numpy(dtype=float)
                show_leg = (r == 0 and c == 0)
                fig.add_trace(go.Scattergl(
                    x=xd[success_mask & ok], y=yj[success_mask & ok], mode="markers",
                    marker=dict(size=4, color="black", line=dict(width=0)),
                    name="data (success)", legendgroup="data_succ", showlegend=show_leg,
                    hovertemplate=("trial_id: %{customdata[0]}<br>"
                                   f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                                   f"{base}: %{{customdata[1]}}<br>"
                                   f"{tgt_col}: %{{customdata[2]:.4f}}<extra></extra>"),
                    customdata=np.column_stack([
                        df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask & ok],
                        cat_series.to_numpy()[success_mask & ok],
                        df_raw_f[tgt_col].to_numpy()[success_mask & ok],
                    ])
                ), row=r+1, col=c+1)
                fig.add_trace(go.Scattergl(
                    x=xd[fail_mask & ok], y=yj[fail_mask & ok], mode="markers",
                    marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
                    name="data (failed)", legendgroup="data_fail", showlegend=show_leg,
                    hovertemplate=("trial_id: %{customdata[0]}<br>"
                                   f"{feature_names[PAY['j']]}: %{{x:.6g}}<br>"
                                   f"{base}: %{{customdata[1]}}<br>"
                                   "status: failed (NaN target)<extra></extra>"),
                    customdata=np.column_stack([
                        df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[fail_mask & ok],
                        cat_series.to_numpy()[fail_mask & ok],
                    ])
                ), row=r+1, col=c+1)
            # axes: x numeric; y categorical range
            _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="x", centers=x_vals, is_log=_is_log_feature(PAY["j"]))
            fig.update_yaxes(range=[-0.5, nlab - 0.5], row=r+1, col=c+1)

        elif kind == ("num","cat"):
            base = PAY["base"]; y_vals = PAY["y"]; labels = PAY["x"]
            nlab = len(labels)
            # heatmap (categories on X)
            fig.add_trace(go.Heatmap(
                x=np.arange(nlab), y=y_vals, z=Z_t,
                coloraxis="coloraxis", zsmooth=False, showscale=False,
                hovertemplate=(f"{base}: %{{text}}<br>"
                               f"{feature_names[PAY['i']]}: %{{y:.6g}}"
                               "<br>E[target|success]: %{customdata:.3f}<extra></extra>"),
                text=np.array(labels)[None, :].repeat(len(y_vals), axis=0),
                customdata=Zmu_raw
            ), row=r+1, col=c+1)
            for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
                mask = np.where(Zp < thr, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    x=np.arange(nlab), y=y_vals, z=mask, zmin=0, zmax=1,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                    showscale=False, hoverinfo="skip"
                ), row=r+1, col=c+1)
            fig.update_xaxes(tickmode="array", tickvals=list(range(nlab)), ticktext=labels, row=r+1, col=c+1)
            # data overlays with jitter on X
            if base in df_raw_f.columns and feature_names[PAY["i"]] in df_raw_f.columns:
                cat_series = df_raw_f[base].astype("string")
                cat_to_idx = {lab: i for i, lab in enumerate(labels)}
                x_map = cat_series.map(cat_to_idx)
                ok = x_map.notna().to_numpy()
                x_idx = x_map.to_numpy(dtype=float)
                jitter = 0.10 * (np.random.default_rng(0).standard_normal(size=len(x_idx)))
                xj = x_idx + jitter
                yd = df_raw_f[feature_names[PAY["i"]]].to_numpy(dtype=float)
                show_leg = (r == 0 and c == 0)
                fig.add_trace(go.Scattergl(
                    x=xj[success_mask & ok], y=yd[success_mask & ok], mode="markers",
                    marker=dict(size=4, color="black", line=dict(width=0)),
                    name="data (success)", legendgroup="data_succ", showlegend=show_leg,
                    hovertemplate=("trial_id: %{customdata[0]}<br>"
                                   f"{base}: %{{customdata[1]}}<br>"
                                   f"{feature_names[PAY['i']]}: %{{y:.6g}}<br>"
                                   f"{tgt_col}: %{{customdata[2]:.4f}}<extra></extra>"),
                    customdata=np.column_stack([
                        df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask & ok],
                        cat_series.to_numpy()[success_mask & ok],
                        df_raw_f[tgt_col].to_numpy()[success_mask & ok],
                    ])
                ), row=r+1, col=c+1)
                fig.add_trace(go.Scattergl(
                    x=xj[fail_mask & ok], y=yd[fail_mask & ok], mode="markers",
                    marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
                    name="data (failed)", legendgroup="data_fail", showlegend=show_leg,
                    hovertemplate=("trial_id: %{customdata[0]}<br>"
                                   f"{base}: %{{customdata[1]}}<br>"
                                   f"{feature_names[PAY['i']]}: %{{y:.6g}}<br>"
                                   "status: failed (NaN target)<extra></extra>"),
                    customdata=np.column_stack([
                        df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[fail_mask & ok],
                        cat_series.to_numpy()[fail_mask & ok],
                    ])
                ), row=r+1, col=c+1)
            # axes: x categorical; y numeric
            fig.update_xaxes(range=[-0.5, nlab - 0.5], row=r+1, col=c+1)
            _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="y", centers=y_vals, is_log=_is_log_feature(PAY["i"]))

        elif kind == ("cat","cat"):
            labels_y = PAY["y"]
            labels_x = PAY["x"]
            ny, nx = len(labels_y), len(labels_x)

            # Build customdata carrying (row_label, col_label) for hovertemplate.
            custom = np.dstack((
                np.array(labels_y, dtype=object)[:, None].repeat(nx, axis=1),
                np.array(labels_x, dtype=object)[None, :].repeat(ny, axis=0),
            ))

            # Heatmap over categorical indices
            fig.add_trace(go.Heatmap(
                x=np.arange(nx),
                y=np.arange(ny),
                z=Z_t,
                coloraxis="coloraxis",
                zsmooth=False,
                showscale=False,
                hovertemplate=(
                    "row: %{customdata[0]}<br>"
                    "col: %{customdata[1]}<br>"
                    "E[target|success]: %{z:.3f}<extra></extra>"
                ),
                customdata=custom,
            ), row=r+1, col=c+1)

            # p(success) shading overlays
            for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
                mask = np.where(Zp < thr, 1.0, np.nan)
                fig.add_trace(go.Heatmap(
                    x=np.arange(nx),
                    y=np.arange(ny),
                    z=mask,
                    zmin=0,
                    zmax=1,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                    showscale=False,
                    hoverinfo="skip",
                ), row=r+1, col=c+1)

            # Categorical tick labels on both axes
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(nx)),
                ticktext=labels_x,
                range=[-0.5, nx - 0.5],
                row=r+1,
                col=c+1,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(ny)),
                ticktext=labels_y,
                range=[-0.5, ny - 0.5],
                row=r+1,
                col=c+1,
            )

    # --- outer axis labels
    def _panel_title(kind: str, key: object) -> str:
        return feature_names[int(key)] if kind == "num" else str(key)

    for c, (_, key_c) in enumerate(panels):
        fig.update_xaxes(title_text=_panel_title(panels[c][0], key_c), row=k, col=c+1)
    for r, (kind_r, key_r) in enumerate(panels):
        fig.update_yaxes(title_text=_panel_title(kind_r, key_r), row=r+1, col=1)

    # --- title
    def _fmt_c(v):
        if isinstance(v, slice):
            a = f"{v.start:g}" if v.start is not None else ""
            b = f"{v.stop:g}"  if v.stop  is not None else ""
            return f"[{a},{b}]"
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                return "[" + ",".join(f"{float(x):g}" for x in np.asarray(v).tolist()) + "]"
            except Exception:
                return "[" + ",".join(map(str, v)) + "]"
        return str(v)

    title_parts = [f"2D partial dependence of expected {tgt_col}"]

    # numeric constraints shown
    for name, val in kw_num.items():
        title_parts.append(f"{name}={_fmt_c(val)}")
    # categorical constraints: fixed shown as base=Label; allowed ranges omitted in title
    for base, lab in cat_fixed.items():
        title_parts.append(f"{base}={lab}")
    title = " — ".join([title_parts[0], ", ".join(title_parts[1:])]) if len(title_parts) > 1 else title_parts[0]

    # --- layout
    cell = 250
    z_title = "E[target|success]" + (" (log10)" if use_log_scale_for_target else "")
    if use_log_scale_for_target and global_shift > 0:
        z_title += f" (shift Δ={global_shift:.3g})"

    width = width if (width and width > 0) else cell * k
    width = max(width, 400)
    height = height if (height and height > 0) else cell * k
    height = max(height, 400)

    fig.update_layout(
        template="simple_white",
        width=width,
        height=height,
        title=title,
        legend_title_text="",
        coloraxis=dict(
            colorscale=colorscale,
            cmin=cmin_t, cmax=cmax_t,
            colorbar=dict(
                title=z_title,
                thickness=10,          # thinner bar
                len=0.55,              # shorter bar (fraction of plot height)
                lenmode="fraction",
                x=1.02, y=0.5,         # just right of plot, vertically centered
                xanchor="left", yanchor="middle",
            ),
        ),
        legend=dict(
            orientation="v",
            x=1.02, xanchor="left",   # to the right of the colorbar
            y=1.0,  yanchor="top",
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(t=90, r=100),       # room for title + legend + colorbar
    )

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn")
    if show:
        fig.show("browser")
    return fig


def plot1d(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    csv_out: Path | None = None,
    grid_size: int = 300,
    line_color: str = "rgb(31,119,180)",
    band_alpha: float = 0.25,
    show: bool = False,
    use_log_scale_for_target_y: bool = True,   # log-y for target
    log_y_epsilon: float = 1e-9,
    optimal: bool = True,
    suggest: int = 0,
    width:int|None = None,
    height:int|None = None,
    seed: int|None = 42,
    **kwargs,
) -> go.Figure:
    """
    Vertical 1D PD panels of E[target|success] vs each *free* feature.
    Scalars (fix & hide), slices (restrict sweep & x-range), lists/tuples (discrete grids).
    Categorical bases (e.g. language) are plotted as a single categorical subplot
    when not fixed; passing --language "Linear A" fixes that base and removes it
    from the plotted axes.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std  = ds["feature_std"].values.astype(float)

    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    n_rows, p = Xn_train.shape

    # --- one-hot categorical groups ---
    groups = opt._onehot_groups(feature_names)   # { base: {"labels":[...], "name_by_label":{label:member}, "members":[...]} }
    bases  = set(groups.keys())
    name_to_idx = {name: j for j, name in enumerate(feature_names)}

    # --- canonicalize kwargs: numeric vs categorical (base) ---
    idx_map = _canon_key_set(ds)
    kw_num_raw: dict[str, object] = {}
    kw_cat_raw: dict[str, object] = {}
    for k, v in kwargs.items():
        if k in bases:
            kw_cat_raw[k] = v
            continue
        if k in idx_map:
            kw_num_raw[idx_map[k]] = v
            continue
        import re as _re
        nk = _re.sub(r"[^a-z0-9]+", "", str(k).lower())
        if nk in idx_map:
            kw_num_raw[idx_map[nk]] = v

    # --- resolve categorical constraints: fixed (single) vs allowed (multiple) ---
    cat_fixed: dict[str, str] = {}
    cat_allowed: dict[str, list[str]] = {}
    for base, val in kw_cat_raw.items():
        labels = groups[base]["labels"]
        if isinstance(val, str):
            if val not in labels:
                raise ValueError(f"Unknown category for {base!r}: {val!r}. Choices: {labels}")
            cat_fixed[base] = val
        elif isinstance(val, (list, tuple, set)):
            chosen = [x for x in val if isinstance(x, str) and x in labels]
            if not chosen:
                raise ValueError(f"No valid categories for {base!r} in {val!r}. Choices: {labels}")
            # multiple -> treat as allowed subset (NOT fixed)
            cat_allowed[base] = list(dict.fromkeys(chosen))
        else:
            raise ValueError(f"Categorical constraint for {base!r} must be a string or list/tuple of strings.")

    # --- filter rows by fixed categoricals (affects medians/percentiles & overlays) ---
    row_mask = np.ones(n_rows, dtype=bool)
    for base, label in cat_fixed.items():
        if base in df_raw.columns:
            row_mask &= (df_raw[base].astype("string") == pd.Series([label]*len(df_raw), dtype="string")).to_numpy()
        else:
            member_name = groups[base]["name_by_label"][label]
            j = name_to_idx[member_name]
            raw_j = feature_raw_from_artifact_or_reconstruct(ds, j, member_name, transforms[j]).astype(float)
            row_mask &= (raw_j >= 0.5)

    df_raw_f = df_raw.loc[row_mask].reset_index(drop=True)
    Xn_train_f = Xn_train[row_mask, :]

    # --- helpers to transform original <-> standardized for feature j ---
    def _orig_to_std(j: int, x, transforms, mu, sd):
        x = np.asarray(x, dtype=float)
        if transforms[j] == "log10":
            x = np.where(x <= 0, np.nan, x)
            x = np.log10(x)
        return (x - mu[j]) / sd[j]

    # --- numeric constraint split (STANDARDIZED) ---
    fixed_scalars: dict[int, float] = {}
    range_windows: dict[int, tuple[float, float]] = {}
    choice_values: dict[int, np.ndarray] = {}
    for name, val in kw_num_raw.items():
        if name not in name_to_idx:
            continue
        j = name_to_idx[name]
        if isinstance(val, slice):
            lo = _orig_to_std(j, float(val.start), transforms, X_mean, X_std)
            hi = _orig_to_std(j, float(val.stop),  transforms, X_mean, X_std)
            lo, hi = float(min(lo, hi)), float(max(lo, hi))
            range_windows[j] = (lo, hi)
        elif isinstance(val, (list, tuple, np.ndarray)):
            arr = _orig_to_std(j, np.asarray(val, dtype=float), transforms, X_mean, X_std)
            choice_values[j] = np.asarray(arr, dtype=float)
        else:
            fixed_scalars[j] = float(_orig_to_std(j, float(val), transforms, X_mean, X_std))

    # --- apply categorical fixed as standardized scalar fixes on each one-hot member ---
    for base, label in cat_fixed.items():
        labels = groups[base]["labels"]
        for lab in labels:
            member_name = groups[base]["name_by_label"][lab]
            j = name_to_idx[member_name]
            raw_val = 1.0 if lab == label else 0.0
            fixed_scalars[j] = float(_orig_to_std(j, raw_val, transforms, X_mean, X_std))

    # --- overlays conditioned on the same kwargs (numeric + categorical) ---
    optimal_df  = opt.optimal(model, count=1, seed=seed, **kwargs) if optimal else None
    suggest_df  = opt.suggest(model, count=suggest, seed=seed, **kwargs) if (suggest and suggest > 0) else None

    # --- base standardized point (median over filtered rows), then apply scalar fixes ---
    base_std = np.median(Xn_train_f, axis=0)
    for j, vstd in fixed_scalars.items():
        base_std[j] = vstd

    # --- plotted panels: numeric free features + categorical bases not fixed ---
    onehot_members = set()
    for base, g in groups.items():
        onehot_members.update(g["members"])
    free_numeric_idx = [j for j in range(p) if (j not in fixed_scalars) and (feature_names[j] not in onehot_members)]
    free_cat_bases   = [b for b in bases if b not in cat_fixed]  # optional: filtered by cat_allowed later

    panels: list[tuple[str, object]] = [("num", j) for j in free_numeric_idx] + [("cat", b) for b in free_cat_bases]
    if not panels:
        raise ValueError("All features are fixed (or categorical only with single category chosen); nothing to plot.")

    # --- empirical 1–99% from filtered rows for numeric bounds ---
    p01p99 = [np.percentile(Xn_train_f[:, j], [1, 99]) for j in range(p)]
    def _grid_1d(j: int, n: int) -> np.ndarray:
        p01, p99 = p01p99[j]
        if j in choice_values:
            vals = np.asarray(choice_values[j], dtype=float)
            vals = vals[(vals >= p01) & (vals <= p99)]
            return np.unique(np.sort(vals)) if vals.size else np.array([np.median(Xn_train_f[:, j])], dtype=float)
        lo, hi = p01, p99
        if j in range_windows:
            rlo, rhi = range_windows[j]
            lo, hi = max(lo, rlo), min(hi, rhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = p01, max(p01 + 1e-9, p99)
        return np.linspace(lo, hi, n)

    # --- one-hot member names (robust) ---
    onehot_member_names: set[str] = set()
    for base, g in groups.items():
        # names recorded by the detector
        onehot_member_names.update(g["members"])
        # fallback pattern match in case detector missed anything
        prefix = f"{base}="
        onehot_member_names.update([nm for nm in feature_names if nm.startswith(prefix)])

    # --- build panel list: numeric free features + categorical bases (not fixed) ---
    free_numeric_idx = [
        j for j, nm in enumerate(feature_names)
        if (j not in fixed_scalars) and (nm not in onehot_member_names)
    ]
    free_cat_bases = [b for b in bases if b not in cat_fixed]

    panels: list[tuple[str, object]] = [("num", j) for j in free_numeric_idx] + [("cat", b) for b in free_cat_bases]
    if not panels:
        raise ValueError("All features are fixed (or only single-category categoricals remain); nothing to plot.")

    # sanity: ensure we didn't accidentally keep any one-hot member columns
    assert all(
        (feature_names[key] not in onehot_member_names) if kind == "num" else True
        for kind, key in panels
    ), "internal: one-hot member leaked into numeric panels"

    # --- figure scaffold with clean titles ---
    def _panel_title(kind: str, key: object) -> str:
        return feature_names[int(key)] if kind == "num" else str(key)

    subplot_titles = [_panel_title(kind, key) for kind, key in panels]
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
    )

    # --- masks/data from filtered rows ---
    tgt_col = str(ds.attrs["target"])
    success_mask = ~pd.isna(df_raw_f[tgt_col]).to_numpy()
    fail_mask    = ~success_mask
    losses_success = df_raw_f.loc[success_mask, tgt_col].to_numpy().astype(float)
    trial_ids_success = df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask]
    trial_ids_fail    = df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[fail_mask]
    band_fill_rgba = _rgb_to_rgba(line_color, band_alpha)

    tidy_rows: list[dict] = []

    row_pos = 0
    for kind, key in panels:
        row_pos += 1

        if kind == "num":
            j = key
            grid = _grid_1d(j, grid_size)
            Xn_grid = np.repeat(base_std[None, :], len(grid), axis=0)
            Xn_grid[:, j] = grid

            # # --- DEBUG: confirm the feature is actually changing in standardized space ---
            # print(f"[{feature_names[j]}] std grid head: {grid[:6]}")
            # print(f"[{feature_names[j]}] std grid ptp (range): {np.ptp(grid)}")
            # print(f"[{feature_names[j]}] Xn_grid[:2, j]: {Xn_grid[:2, j]}")
            # print(f"[{feature_names[j]}] Xn 1–99%: {p01p99[j]}")

            p_grid = pred_success(Xn_grid)
            mu_grid, sd_grid = pred_loss(Xn_grid, include_observation_noise=True)
            # print(feature_names[j], "mu range:", float(np.ptp(mu_grid)))

            x_internal = grid * X_std[j] + X_mean[j]
            x_display  = _inverse_transform(transforms[j], x_internal)

            # print(f"[{feature_names[j]}] orig head: {x_display[:6]}")
            # print(f"[{feature_names[j]}] orig ptp (range): {np.ptp(x_display)}")

            if use_log_scale_for_target_y:
                mu_plot = np.maximum(mu_grid, log_y_epsilon)
                lo_plot = np.maximum(mu_grid - 2.0 * sd_grid, log_y_epsilon)
                hi_plot = np.maximum(mu_grid + 2.0 * sd_grid, log_y_epsilon)
                losses_s_plot = np.maximum(losses_success, log_y_epsilon) if losses_success.size else losses_success
            else:
                mu_plot = mu_grid
                lo_plot = mu_grid - 2.0 * sd_grid
                hi_plot = mu_grid + 2.0 * sd_grid
                losses_s_plot = losses_success

            y_arrays = [lo_plot, hi_plot] + ([losses_s_plot] if losses_s_plot.size else [])
            y_low  = float(np.nanmin([np.nanmin(a) for a in y_arrays]))
            y_high = float(np.nanmax([np.nanmax(a) for a in y_arrays]))
            pad = 0.05 * (y_high - y_low + 1e-12)
            y0_plot = (y_low - pad) if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)
            y1_tmp  = (y_high + pad) if not use_log_scale_for_target_y else y_high * 1.2
            y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
            if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
                y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
            y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

            _add_low_success_shading_1d(fig, row_pos, x_display, p_grid, y0_plot, y1_plot)

            show_legend = (row_pos == 1)
            fig.add_trace(go.Scatter(x=x_display, y=lo_plot, mode="lines",
                                     line=dict(width=0, color=line_color),
                                     name="±2σ", legendgroup="band", showlegend=False, hoverinfo="skip"),
                          row=row_pos, col=1)
            fig.add_trace(go.Scatter(x=x_display, y=hi_plot, mode="lines", fill="tonexty",
                                     line=dict(width=0, color=line_color), fillcolor=band_fill_rgba,
                                     name="±2σ", legendgroup="band", showlegend=show_legend,
                                     hovertemplate="E[target|success]: %{y:.3f}<extra>±2σ</extra>"),
                          row=row_pos, col=1)
            fig.add_trace(go.Scatter(x=x_display, y=mu_plot, mode="lines",
                                     line=dict(width=2, color=line_color),
                                     name="E[target|success]", legendgroup="mean", showlegend=show_legend,
                                     hovertemplate=f"{feature_names[j]}: %{{x:.6g}}<br>E[target|success]: %{{y:.3f}}<extra></extra>"),
                          row=row_pos, col=1)

            # experimental points
            if feature_names[j] in df_raw_f.columns:
                x_data_all = df_raw_f[feature_names[j]].to_numpy().astype(float)
            else:
                full_vals = feature_raw_from_artifact_or_reconstruct(ds, j, feature_names[j], transforms[j]).astype(float)
                x_data_all = full_vals[row_mask]

            x_succ = x_data_all[success_mask]
            if x_succ.size:
                fig.add_trace(go.Scattergl(
                    x=x_succ, y=losses_s_plot, mode="markers",
                    marker=dict(size=5, color="black", line=dict(width=0)),
                    name="data (success)", legendgroup="data_s", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<br>"
                                   f"{tgt_col}: %{{y:.4f}}<extra></extra>"),
                    customdata=trial_ids_success
                ), row=row_pos, col=1)

            x_fail = x_data_all[fail_mask]
            if x_fail.size:
                y_fail_plot = np.full_like(x_fail, y_failed_band, dtype=float)
                fig.add_trace(go.Scattergl(
                    x=x_fail, y=y_fail_plot, mode="markers",
                    marker=dict(size=6, color="red", line=dict(color="black", width=0.8)),
                    name="data (failed)", legendgroup="data_f", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<br>"
                                   "status: failed (NaN target)<extra></extra>"),
                    customdata=trial_ids_fail
                ), row=row_pos, col=1)

            # overlays
            if optimal_df is not None and feature_names[j] in optimal_df.columns:
                x_opt = optimal_df[feature_names[j]].values
                y_opt = optimal_df["pred_target_mean"].values
                y_sd  = optimal_df["pred_target_sd"].values
                fig.add_trace(go.Scattergl(
                    x=x_opt, y=y_opt, mode="markers",
                    marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                    name="optimal", legendgroup="optimal", showlegend=show_legend,
                    hovertemplate=(f"predicted: %{{y:.3g}} ± {y_sd[0]:.3g}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<extra></extra>")
                ), row=row_pos, col=1)

            if suggest_df is not None and feature_names[j] in suggest_df.columns:
                x_sug = suggest_df[feature_names[j]].values
                y_sug = suggest_df["pred_target_mean"].values
                y_sd  = suggest_df["pred_target_sd"].values
                fig.add_trace(go.Scattergl(
                    x=x_sug, y=y_sug, mode="markers",
                    marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                    name="suggested", legendgroup="suggested", showlegend=show_legend,
                    hovertemplate=(f"predicted: %{{y:.3g}} ± {{y_sd:.3g}}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<extra></extra>")
                ), row=row_pos, col=1)

            # axes
            _maybe_log_axis(fig, row_pos, 1, feature_names[j], axis="x", transforms=transforms, j=j)
            fig.update_yaxes(title_text=f"{tgt_col}", row=row_pos, col=1)
            _set_yaxis_range(fig, row=row_pos, col=1,
                             y0=y0_plot, y1=y1_plot,
                             log=use_log_scale_for_target_y, eps=log_y_epsilon)

            # restrict x-range if constrained
            is_log_x = (transforms[j] == "log10")
            def _std_to_orig(val_std: float) -> float:
                vi = val_std * X_std[j] + X_mean[j]
                return float(_inverse_transform(transforms[j], np.array([vi]))[0])

            x_min_override = x_max_override = None
            if j in range_windows:
                lo_std, hi_std = range_windows[j]
                x_min_override = min(_std_to_orig(lo_std), _std_to_orig(hi_std))
                x_max_override = max(_std_to_orig(lo_std), _std_to_orig(hi_std))
            elif j in choice_values:
                ints = choice_values[j] * X_std[j] + X_mean[j]
                origs = _inverse_transform(transforms[j], ints)
                x_min_override = float(np.min(origs))
                x_max_override = float(np.max(origs))

            if (x_min_override is not None) and (x_max_override is not None):
                if is_log_x:
                    x0 = max(x_min_override, 1e-12)
                    x1 = max(x_max_override, x0 * (1 + 1e-9))
                    pad = (x1 / x0) ** 0.03
                    fig.update_xaxes(type="log",
                                     range=[np.log10(x0 / pad), np.log10(x1 * pad)],
                                     row=row_pos, col=1)
                else:
                    span = (x_max_override - x_min_override) or 1.0
                    pad = 0.02 * span
                    fig.update_xaxes(range=[x_min_override - pad, x_max_override + pad],
                                     row=row_pos, col=1)

            fig.update_xaxes(title_text=feature_names[j], row=row_pos, col=1)

            # tidy rows
            for xd, xi, mu_i, sd_i, p_i in zip(x_display, x_internal, mu_grid, sd_grid, p_grid):
                tidy_rows.append({
                    "feature": feature_names[j],
                    "x_display": float(xd),
                    "x_internal": float(xi),
                    "target_conditional_mean": float(mu_i),
                    "target_conditional_sd": float(sd_i),
                    "success_probability": float(p_i),
                })

        else:
            base = key  # categorical base
            labels_all = groups[base]["labels"]
            labels = cat_allowed.get(base, labels_all)

            # Build standardized design for each label at the base point
            Xn_grid = np.repeat(base_std[None, :], len(labels), axis=0)
            for r, lab in enumerate(labels):
                for lab2 in labels_all:
                    member_name = groups[base]["name_by_label"][lab2]
                    j2 = name_to_idx[member_name]
                    raw_val = 1.0 if (lab2 == lab) else 0.0
                    # standardized set:
                    Xi = (raw_val - X_mean[j2]) / X_std[j2]
                    Xn_grid[r, j2] = Xi

            p_vec = pred_success(Xn_grid)
            mu_vec, sd_vec = pred_loss(Xn_grid, include_observation_noise=True)
            print(feature_names[j], "mu range:", float(np.ptp(mu_grid)))
            
            # y transform
            if use_log_scale_for_target_y:
                mu_plot = np.maximum(mu_vec, log_y_epsilon)
                lo_plot = np.maximum(mu_vec - 2.0 * sd_vec, log_y_epsilon)
                hi_plot = np.maximum(mu_vec + 2.0 * sd_vec, log_y_epsilon)
                losses_s_plot = np.maximum(df_raw_f.loc[success_mask, tgt_col].to_numpy().astype(float), log_y_epsilon) if success_mask.any() else np.array([])
            else:
                mu_plot = mu_vec
                lo_plot = mu_vec - 2.0 * sd_vec
                hi_plot = mu_vec + 2.0 * sd_vec
                losses_s_plot = df_raw_f.loc[success_mask, tgt_col].to_numpy().astype(float) if success_mask.any() else np.array([])

            # y-range
            y_arrays = [lo_plot, hi_plot] + ([losses_s_plot] if losses_s_plot.size else [])
            y_low  = float(np.nanmin([np.nanmin(a) for a in y_arrays])) if y_arrays else 0.0
            y_high = float(np.nanmax([np.nanmax(a) for a in y_arrays])) if y_arrays else 1.0
            pad = 0.05 * (y_high - y_low + 1e-12)
            y0_plot = (y_low - pad) if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)
            y1_tmp  = (y_high + pad) if not use_log_scale_for_target_y else y_high * 1.2
            y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
            if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
                y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
            y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

            # x positions are 0..K-1 with tick labels = category names
            x_pos = np.arange(len(labels), dtype=float)

            # shading per-category threshold regions using shapes
            def _shade_for_thresh(thr: float, alpha: float):
                for k_i, p_i in enumerate(p_vec):
                    if p_i < thr:
                        fig.add_shape(
                            type="rect",
                            xref=f"x{'' if row_pos==1 else row_pos}",
                            yref=f"y{'' if row_pos==1 else row_pos}",
                            x0=k_i - 0.5, x1=k_i + 0.5,
                            y0=y0_plot, y1=y1_plot,
                            line=dict(width=0),
                            fillcolor=f"rgba(128,128,128,{alpha})",
                            layer="below",
                            row=row_pos, col=1
                        )
            _shade_for_thresh(0.8, 0.40)
            _shade_for_thresh(0.5, 0.25)

            show_legend = (row_pos == 1)

            # mean with error bars (±2σ)
            fig.add_trace(go.Scatter(
                x=x_pos, y=mu_plot, mode="lines+markers",
                line=dict(width=2, color=line_color),
                marker=dict(size=7, color=line_color),
                error_y=dict(type="data", array=(hi_plot - mu_plot), arrayminus=(mu_plot - lo_plot), visible=True),
                name="E[target|success]", legendgroup="mean", showlegend=show_legend,
                hovertemplate=(f"{base}: %{{text}}<br>E[target|success]: %{{y:.3f}}"
                               "<br>±2σ shown as error bar<extra></extra>"),
                text=labels
            ), row=row_pos, col=1)

            # experimental points: map each row's label to index
            if base in df_raw_f.columns:
                lab_series = df_raw_f[base].astype("string")
            else:
                # reconstruct from one-hot members
                member_cols = [groups[base]["name_by_label"][lab] for lab in labels_all]
                idx_max = df_raw_f[member_cols].to_numpy().argmax(axis=1)
                lab_series = pd.Series([labels_all[i] for i in idx_max], dtype="string")

            label_to_idx = {lab: i for i, lab in enumerate(labels)}
            x_idx_all = lab_series.map(lambda s: label_to_idx.get(str(s), np.nan)).to_numpy(dtype=float)
            x_idx_succ = x_idx_all[success_mask]
            x_idx_fail = x_idx_all[fail_mask]

            # jitter for visibility
            rng = np.random.default_rng(0)
            jitter = lambda n: (rng.random(n) - 0.5) * 0.15

            if x_idx_succ.size:
                fig.add_trace(go.Scattergl(
                    x=x_idx_succ + jitter(x_idx_succ.size),
                    y=losses_s_plot,
                    mode="markers",
                    marker=dict(size=5, color="black", line=dict(width=0)),
                    name="data (success)", legendgroup="data_s", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{base}: %{{text}}<br>"
                                   f"{tgt_col}: %{{y:.4f}}<extra></extra>"),
                    text=[labels[int(i)] if np.isfinite(i) and int(i) < len(labels) else "?" for i in x_idx_succ],
                    customdata=trial_ids_success
                ), row=row_pos, col=1)

            if x_idx_fail.size:
                y_fail_plot = np.full_like(x_idx_fail, y_failed_band, dtype=float)
                fig.add_trace(go.Scattergl(
                    x=x_idx_fail + jitter(x_idx_fail.size), y=y_fail_plot, mode="markers",
                    marker=dict(size=6, color="red", line=dict(color="black", width=0.8)),
                    name="data (failed)", legendgroup="data_f", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{base}: %{{text}}<br>"
                                   "status: failed (NaN target)<extra></extra>"),
                    text=[labels[int(i)] if np.isfinite(i) and int(i) < len(labels) else "?" for i in x_idx_fail],
                    customdata=trial_ids_fail
                ), row=row_pos, col=1)

            # overlays for categorical base: map label to x index
            if optimal_df is not None and (base in optimal_df.columns):
                lab_opt = str(optimal_df[base].values[0])
                if lab_opt in label_to_idx:
                    x_opt = [float(label_to_idx[lab_opt])]
                    y_opt = optimal_df["pred_target_mean"].values
                    y_sd  = optimal_df["pred_target_sd"].values
                    fig.add_trace(go.Scattergl(
                        x=x_opt, y=y_opt, mode="markers",
                        marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                        name="optimal", legendgroup="optimal", showlegend=show_legend,
                        hovertemplate=(f"predicted: %{{y:.3g}} ± {y_sd[0]:.3g}<br>"
                                       f"{base}: {lab_opt}<extra></extra>")
                    ), row=row_pos, col=1)

            if suggest_df is not None and (base in suggest_df.columns):
                labs_sug = suggest_df[base].astype(str).tolist()
                xs = [label_to_idx[l] for l in labs_sug if l in label_to_idx]
                if xs:
                    keep_mask = [l in label_to_idx for l in labs_sug]
                    y_sug = suggest_df.loc[keep_mask, "pred_target_mean"].values
                    fig.add_trace(go.Scattergl(
                        x=np.array(xs, dtype=float), y=y_sug, mode="markers",
                        marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                        name="suggested", legendgroup="suggested", showlegend=show_legend,
                        hovertemplate=(f"{base}: %{{text}}<br>"
                                       "predicted: %{{y:.3g}}<extra>suggested</extra>"),
                        text=[labels[int(i)] for i in xs]
                    ), row=row_pos, col=1)

            # axes: categorical ticks
            fig.update_xaxes(
                tickmode="array",
                tickvals=x_pos.tolist(),
                ticktext=labels,
                row=row_pos, col=1
            )
            fig.update_yaxes(title_text=f"{tgt_col}", row=row_pos, col=1)
            _set_yaxis_range(fig, row=row_pos, col=1,
                             y0=y0_plot, y1=y1_plot,
                             log=use_log_scale_for_target_y, eps=log_y_epsilon)
            fig.update_xaxes(title_text=base, row=row_pos, col=1)

            # tidy rows
            for lab, mu_i, sd_i, p_i in zip(labels, mu_vec, sd_vec, p_vec):
                tidy_rows.append({
                    "feature": base,
                    "x_display": str(lab),
                    "x_internal": float("nan"),
                    "target_conditional_mean": float(mu_i),
                    "target_conditional_sd": float(sd_i),
                    "success_probability": float(p_i),
                })

    # title w/ constraints summary
    def _fmt_c(v):
        if isinstance(v, slice):
            a = "" if v.start is None else f"{v.start:g}"
            b = "" if v.stop  is None else f"{v.stop:g}"
            return f"[{a},{b}]"
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                return "[" + ",".join(f"{float(x):g}" for x in np.asarray(v).tolist()) + "]"
            except Exception:
                return "[" + ",".join(map(str, v)) + "]"
        try:
            return f"{float(v):g}"
        except Exception:
            return str(v)

    parts = [f"1D partial dependence of expected {tgt_col}"]
    if kw_num_raw:
        parts.append(", ".join(f"{k}={_fmt_c(v)}" for k, v in kw_num_raw.items()))
    if cat_fixed:
        parts.append(", ".join(f"{b}={lab}" for b, lab in cat_fixed.items()))
    if cat_allowed:
        parts.append(", ".join(f"{b}∈{{{', '.join(v)}}}" for b, v in cat_allowed.items()))
    title = " — ".join(parts) if len(parts) > 1 else parts[0]

    width = width if (width and width > 0) else 1200
    height = height if (height and height > 0) else 1200

    fig.update_layout(
        height=height,
        width=width,
        template="simple_white",
        title=title,
        legend_title_text=""
    )

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn")
    if csv_out:
        csv_out = Path(csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tidy_rows).to_csv(str(csv_out), index=False)
    if show:
        fig.show("browser")

    return fig


# =============================================================================
# Helpers: dataset → predictors & featurization
# =============================================================================
def _build_predictors(ds: xr.Dataset):
    """Reconstruct fast GP predictors from the artifact using shared helpers."""
    # Training matrices / targets
    Xn_all = ds["Xn_train"].values.astype(float)               # (N, p)
    y_success = ds["y_success"].values.astype(float)           # (N,)
    Xn_ok = ds["Xn_success_only"].values.astype(float)         # (Ns, p)
    y_loss_centered = ds["y_loss_centered"].values.astype(float)

    # Compatibility: conditional_loss_mean may be a var or an attr
    cond_mean = (
        float(ds["conditional_loss_mean"].values)
        if "conditional_loss_mean" in ds
        else float(ds.attrs.get("conditional_loss_mean", 0.0))
    )

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

    # --- Cholesky precomputations (success) ---
    K_s = kernel_m52_ard(Xn_all, Xn_all, ell_s, eta_s) + (sigma_s**2) * np.eye(Xn_all.shape[0])
    L_s = np.linalg.cholesky(add_jitter(K_s))
    alpha_s = solve_chol(L_s, (y_success - beta0_s))

    # --- Cholesky precomputations (loss | success) ---
    K_l = kernel_m52_ard(Xn_ok, Xn_ok, ell_l, eta_l) + (sigma_l**2) * np.eye(Xn_ok.shape[0])
    L_l = np.linalg.cholesky(add_jitter(K_l))
    alpha_l = solve_chol(L_l, (y_loss_centered - mean_c))

    def predict_success_probability(Xn: np.ndarray) -> np.ndarray:
        Ks = kernel_m52_ard(Xn, Xn_all, ell_s, eta_s)
        mu = beta0_s + Ks @ alpha_s
        return np.clip(mu, 0.0, 1.0)

    def predict_conditional_target(
        Xn: np.ndarray,
        include_observation_noise: bool = True
    ):
        Kl = kernel_m52_ard(Xn, Xn_ok, ell_l, eta_l)
        mu_centered = mean_c + Kl @ alpha_l
        mu = mu_centered + cond_mean

        # diag predictive variance
        v = solve_lower(L_l, Kl.T)  # (Ns, Nt)
        var = kernel_diag_m52(Xn, ell_l, eta_l) - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
        if include_observation_noise:
            var = var + sigma_l**2
        sd = np.sqrt(var)
        return mu, sd

    return predict_success_probability, predict_conditional_target


def _raw_dataframe_from_dataset(ds: xr.Dataset) -> pd.DataFrame:
    """Collect raw columns from the artifact into a DataFrame for plotting."""
    cols = {}
    for name in ds.data_vars:
        # include only row-aligned arrays
        da = ds[name]
        if "row" in da.dims and len(da.dims) == 1 and da.sizes["row"] == ds.sizes["row"]:
            cols[name] = da.values
    # Ensure trial_id exists for hover
    if "trial_id" not in cols:
        cols["trial_id"] = np.arange(ds.sizes["row"], dtype=int)
    return pd.DataFrame(cols)


def _apply_fixed_to_base(
    base_std: np.ndarray,
    fixed: dict[str, float],
    feature_names: list[str],
    transforms: list[str],
    X_mean: np.ndarray,
    X_std: np.ndarray,
) -> np.ndarray:
    """Override base point in standardized space with fixed ORIGINAL values."""
    out = base_std.copy()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for k, v in fixed.items():
        if k not in name_to_idx:
            raise KeyError(f"Fixed variable '{k}' is not a model feature.")
        j = name_to_idx[k]
        x_raw = _forward_transform(transforms[j], float(v))
        out[j] = (x_raw - X_mean[j]) / X_std[j]
    return out


def _denormalize_then_inverse_transform(j: int, x_std: np.ndarray, transforms, X_mean, X_std) -> np.ndarray:
    x_raw = x_std * X_std[j] + X_mean[j]
    return _inverse_transform(transforms[j], x_raw)


def _forward_transform(tr: str, x: float | np.ndarray) -> np.ndarray:
    if tr == "log10":
        x = np.asarray(x, dtype=float)
        return np.log10(np.maximum(x, 1e-12))
    return np.asarray(x, dtype=float)


def _inverse_transform(tr: str, x: np.ndarray) -> np.ndarray:
    if tr == "log10":
        return 10.0 ** x
    return x


def _maybe_log_axis(fig: go.Figure, row: int, col: int, name: str, axis: str = "x", transforms: list[str] | None = None, j: int | None = None):
    """Use log axis for features that were log10-transformed."""
    use_log = False
    if transforms is not None and j is not None:
        use_log = (transforms[j] == "log10")
    else:
        use_log = ("learning_rate" in name.lower() or name.lower() == "lr")
    if use_log:
        if axis == "x":
            fig.update_xaxes(type="log", row=row, col=col)
        else:
            fig.update_yaxes(type="log", row=row, col=col)


def _rgb_string_to_tuple(s: str) -> tuple[int, int, int]:
    vals = s[s.find("(") + 1 : s.find(")")].split(",")
    r, g, b = [int(float(v)) for v in vals[:3]]
    return r, g, b


def _rgb_to_rgba(rgb: str, alpha: float) -> str:
    # expects "rgb(r,g,b)" or "rgba(r,g,b,a)"
    try:
        r, g, b = _rgb_string_to_tuple(rgb)
    except Exception:
        r, g, b = (31, 119, 180)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _add_low_success_shading_1d(fig: go.Figure, row_idx: int, x_vals: np.ndarray, p: np.ndarray, y0: float, y1: float):
    xref = "x" if row_idx == 1 else f"x{row_idx}"
    yref = "y" if row_idx == 1 else f"y{row_idx}"

    def _spans(vals: np.ndarray, mask: np.ndarray):
        m = mask.astype(int)
        diff = np.diff(np.concatenate([[0], m, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        return [(vals[s], vals[e]) for s, e in zip(starts, ends)]

    for x0, x1 in _spans(x_vals, p < 0.5):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref=xref, yref=yref,
                      line=dict(width=0), fillcolor="rgba(128,128,128,0.25)", layer="below")
    for x0, x1 in _spans(x_vals, p < 0.8):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref=xref, yref=yref,
                      line=dict(width=0), fillcolor="rgba(128,128,128,0.40)", layer="below")


def _set_yaxis_range(fig, *, row: int, col: int, y0: float, y1: float, log: bool, eps: float = 1e-12):
    """Update a subplot's Y axis to [y0, y1]. For log axes, the range is given in log10 units."""
    if log:
        y0 = max(y0, eps)
        y1 = max(y1, y0 * (1.0 + 1e-6))
        fig.update_yaxes(type="log", range=[np.log10(y0), np.log10(y1)], row=row, col=col)
    else:
        fig.update_yaxes(type="-", range=[y0, y1], row=row, col=col)


def plot1d_at_optimum(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    csv_out: Path | None = None,
    grid_size: int = 300,
    line_color: str = "rgb(31,119,180)",
    band_alpha: float = 0.25,
    show: bool = False,
    use_log_scale_for_target_y: bool = True,
    log_y_epsilon: float = 1e-9,
    suggest: int = 0,                 # optional overlay
    width: int | None = None,
    height: int | None = None,
    seed: int | None = 42,
    **kwargs,                         # constraints in ORIGINAL units (as in your plot1d)
) -> go.Figure:
    """
    1D partial-dependence panels anchored at the *optimal* hyperparameter setting:
    - Compute x* = argmin/argmax mean posterior from opt.optimal(...)
    - For each feature, sweep that feature; keep all *other* features fixed at x*.
    Supports numeric constraints (scalars/slices/choices) and categorical bases.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    # --- metadata ---
    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean        = ds["feature_mean"].values.astype(float)
    X_std         = ds["feature_std"].values.astype(float)

    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    n_rows, p = Xn_train.shape

    # --- one-hot categorical groups ---
    groups = opt._onehot_groups(feature_names)
    bases  = set(groups.keys())
    name_to_idx = {name: j for j, name in enumerate(feature_names)}

    # --- canonicalize kwargs: numeric vs categorical (base) ---
    idx_map = _canon_key_set(ds)  # your helper: maps normalized names -> exact feature column
    kw_num_raw: dict[str, object] = {}
    kw_cat_raw: dict[str, object] = {}
    for k, v in kwargs.items():
        if k in bases:
            kw_cat_raw[k] = v
        elif k in idx_map:
            kw_num_raw[idx_map[k]] = v
        else:
            import re as _re
            nk = _re.sub(r"[^a-z0-9]+", "", str(k).lower())
            if nk in idx_map:
                kw_num_raw[idx_map[nk]] = v

    # --- resolve categorical constraints: fixed vs allowed subset ---
    cat_fixed: dict[str, str] = {}
    cat_allowed: dict[str, list[str]] = {}
    for base, val in kw_cat_raw.items():
        labels = groups[base]["labels"]
        if isinstance(val, str):
            if val not in labels:
                raise ValueError(f"Unknown category for {base!r}: {val!r}. Choices: {labels}")
            cat_fixed[base] = val
        elif isinstance(val, (list, tuple, set)):
            chosen = [x for x in val if isinstance(x, str) and x in labels]
            if not chosen:
                raise ValueError(f"No valid categories for {base!r} in {val!r}. Choices: {labels}")
            cat_allowed[base] = list(dict.fromkeys(chosen))
        else:
            raise ValueError(f"Categorical constraint for {base!r} must be a string or list/tuple of strings.")

    # ---------- 1) Find the *optimal* base point (original units) ----------
    opt_df = opt.optimal(model, count=1, seed=seed, **kwargs)  # uses your gradient-based optimal()
    # We’ll use this row both for overlays and as the anchor point.
    # Expect numeric feature columns and categorical base columns present.
    x_opt_std = np.zeros(p, dtype=float)

    # Fill numerics from optimal row (orig -> internal -> std)
    def _to_std_single(j: int, x_orig: float) -> float:
        xi = x_orig
        if transforms[j] == "log10":
            xi = np.log10(np.maximum(x_orig, 1e-300))
        return float((xi - X_mean[j]) / X_std[j])

    # Mark one-hot member names
    onehot_members: set[str] = set()
    for base, g in groups.items():
        onehot_members.update(g["members"])

    # numeric features (skip one-hot members)
    for j, nm in enumerate(feature_names):
        if nm in onehot_members:
            continue
        if nm in opt_df.columns:
            x_opt_std[j] = _to_std_single(j, float(opt_df.iloc[0][nm]))
        else:
            # Fall back to dataset median if not present (rare)
            x_opt_std[j] = float(np.median(Xn_train[:, j]))

    # Categorical bases: set one-hot block to the optimal label (or fixed)
    for base, g in groups.items():
        # priority: fixed in kwargs → else from optimal row → else keep current (median/std)
        if base in cat_fixed:
            label = cat_fixed[base]
        elif base in opt_df.columns:
            label = str(opt_df.iloc[0][base])
        else:
            # fallback: most frequent label in data
            if base in df_raw.columns:
                label = str(df_raw[base].astype("string").mode(dropna=True).iloc[0])
            else:
                label = g["labels"][0]

        for lab in g["labels"]:
            member_name = g["name_by_label"][lab]
            j2 = name_to_idx[member_name]
            raw = 1.0 if lab == label else 0.0
            # raw (0/1) → standardized using the artifact stats
            x_opt_std[j2] = (raw - X_mean[j2]) / X_std[j2]

    # ---------- 2) Numeric constraints in STANDARDIZED space ----------
    def _orig_to_std(j: int, x, transforms, mu, sd):
        x = np.asarray(x, dtype=float)
        if transforms[j] == "log10":
            x = np.where(x <= 0, np.nan, x)
            x = np.log10(x)
        return (x - mu[j]) / sd[j]

    fixed_scalars: dict[int, float] = {}
    range_windows: dict[int, tuple[float, float]] = {}
    choice_values: dict[int, np.ndarray] = {}

    for name, val in kw_num_raw.items():
        if name not in name_to_idx:
            continue
        j = name_to_idx[name]
        if isinstance(val, slice):
            lo = _orig_to_std(j, float(val.start), transforms, X_mean, X_std)
            hi = _orig_to_std(j, float(val.stop),  transforms, X_mean, X_std)
            lo, hi = float(min(lo, hi)), float(max(lo, hi))
            range_windows[j] = (lo, hi)
        elif isinstance(val, (list, tuple, np.ndarray)):
            arr = _orig_to_std(j, np.asarray(val, dtype=float), transforms, X_mean, X_std)
            choice_values[j] = np.asarray(arr, dtype=float)
        else:
            fixed_scalars[j] = float(_orig_to_std(j, float(val), transforms, X_mean, X_std))

    # apply numeric fixed overrides on the base point
    for j, vstd in fixed_scalars.items():
        x_opt_std[j] = vstd

    # ---------- 3) Panels: sweep ONE var at a time around x* ----------
    # numeric free = not one-hot member and not fixed via kwargs
    free_numeric_idx = [
        j for j, nm in enumerate(feature_names)
        if (nm not in onehot_members) and (j not in fixed_scalars)
    ]
    # categorical bases: sweep if not fixed; otherwise not shown
    free_cat_bases = [b for b in bases if b not in cat_fixed]

    panels: list[tuple[str, object]] = [("num", j) for j in free_numeric_idx] + [("cat", b) for b in free_cat_bases]
    if not panels:
        raise ValueError("All features are fixed at the optimum (or categoricals fixed); nothing to plot.")

    # empirical 1–99% per feature (for default sweep range)
    Xn_p01 = np.percentile(Xn_train, 1, axis=0)
    Xn_p99 = np.percentile(Xn_train, 99, axis=0)

    def _grid_1d(j: int, n: int) -> np.ndarray:
        # default range in std space
        lo, hi = float(Xn_p01[j]), float(Xn_p99[j])
        if j in range_windows:
            lo = max(lo, range_windows[j][0])
            hi = min(hi, range_windows[j][1])
        if j in choice_values:
            vals = np.asarray(choice_values[j], dtype=float)
            vals = vals[(vals >= lo) & (vals <= hi)]
            return np.unique(np.sort(vals)) if vals.size else np.array([x_opt_std[j]], dtype=float)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = x_opt_std[j] - 1.0, x_opt_std[j] + 1.0
        return np.linspace(lo, hi, n)

    # figure scaffold
    subplot_titles = [feature_names[int(k)] if t == "num" else str(k) for t, k in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=False, subplot_titles=subplot_titles)

    tgt_col = str(ds.attrs["target"])
    success_mask = ~pd.isna(df_raw[tgt_col]).to_numpy()
    fail_mask    = ~success_mask
    losses_success = df_raw.loc[success_mask, tgt_col].to_numpy().astype(float)
    trial_ids_success = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask]
    trial_ids_fail    = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]
    band_fill_rgba = _rgb_to_rgba(line_color, band_alpha)

    # optional overlay
    suggest_df = opt.suggest(model, count=suggest, seed=seed, **kwargs) if (suggest and suggest > 0) else None

    tidy_rows: list[dict] = []
    row_pos = 0
    for kind, key in panels:
        row_pos += 1

        if kind == "num":
            j = int(key)
            grid = _grid_1d(j, grid_size)
            Xn_grid = np.repeat(x_opt_std[None, :], len(grid), axis=0)
            Xn_grid[:, j] = grid

            p_grid = pred_success(Xn_grid)
            mu_grid, sd_grid = pred_loss(Xn_grid, include_observation_noise=True)

            x_internal = grid * X_std[j] + X_mean[j]
            x_display  = _inverse_transform(transforms[j], x_internal)

            # y transform
            if use_log_scale_for_target_y:
                mu_plot = np.maximum(mu_grid, log_y_epsilon)
                lo_plot = np.maximum(mu_grid - 2.0 * sd_grid, log_y_epsilon)
                hi_plot = np.maximum(mu_grid + 2.0 * sd_grid, log_y_epsilon)
                losses_s_plot = np.maximum(losses_success, log_y_epsilon) if losses_success.size else losses_success
            else:
                mu_plot = mu_grid
                lo_plot = mu_grid - 2.0 * sd_grid
                hi_plot = mu_grid + 2.0 * sd_grid
                losses_s_plot = losses_success

            y_arrays = [lo_plot, hi_plot] + ([losses_s_plot] if losses_s_plot.size else [])
            y_low  = float(np.nanmin([np.nanmin(a) for a in y_arrays]))
            y_high = float(np.nanmax([np.nanmax(a) for a in y_arrays]))
            pad = 0.05 * (y_high - y_low + 1e-12)
            y0_plot = (y_low - pad) if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)
            y1_tmp  = (y_high + pad) if not use_log_scale_for_target_y else y_high * 1.2
            y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
            if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
                y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
            y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

            _add_low_success_shading_1d(fig, row_pos, x_display, p_grid, y0_plot, y1_plot)

            show_legend = (row_pos == 1)
            # ±2σ band
            fig.add_trace(go.Scatter(x=x_display, y=lo_plot, mode="lines",
                                     line=dict(width=0, color=line_color),
                                     name="±2σ", legendgroup="band", showlegend=False, hoverinfo="skip"),
                          row=row_pos, col=1)
            fig.add_trace(go.Scatter(x=x_display, y=hi_plot, mode="lines", fill="tonexty",
                                     line=dict(width=0, color=line_color), fillcolor=band_fill_rgba,
                                     name="±2σ", legendgroup="band", showlegend=show_legend,
                                     hovertemplate="E[target|success]: %{y:.3f}<extra>±2σ</extra>"),
                          row=row_pos, col=1)
            # mean
            fig.add_trace(go.Scatter(x=x_display, y=mu_plot, mode="lines",
                                     line=dict(width=2, color=line_color),
                                     name="E[target|success]", legendgroup="mean", showlegend=show_legend,
                                     hovertemplate=f"{feature_names[j]}: %{{x:.6g}}<br>E[target|success]: %{{y:.3f}}<extra></extra>"),
                          row=row_pos, col=1)

            # experimental points at y
            if feature_names[j] in df_raw.columns:
                x_data_all = df_raw[feature_names[j]].to_numpy().astype(float)
            else:
                full_vals = feature_raw_from_artifact_or_reconstruct(ds, j, feature_names[j], transforms[j]).astype(float)
                x_data_all = full_vals

            x_succ = x_data_all[success_mask]
            if x_succ.size:
                fig.add_trace(go.Scattergl(
                    x=x_succ, y=losses_s_plot, mode="markers",
                    marker=dict(size=5, color="black", line=dict(width=0)),
                    name="data (success)", legendgroup="data_s", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<br>"
                                   f"{tgt_col}: %{{y:.4f}}<extra></extra>"),
                    customdata=trial_ids_success
                ), row=row_pos, col=1)

            x_fail = x_data_all[fail_mask]
            if x_fail.size:
                y_fail_plot = np.full_like(x_fail, y_failed_band, dtype=float)
                fig.add_trace(go.Scattergl(
                    x=x_fail, y=y_fail_plot, mode="markers",
                    marker=dict(size=6, color="red", line=dict(color="black", width=0.8)),
                    name="data (failed)", legendgroup="data_f", showlegend=show_legend,
                    hovertemplate=("trial_id: %{customdata}<br>"
                                   f"{feature_names[j]}: %{{x:.6g}}<br>"
                                   "status: failed (NaN target)<extra></extra>"),
                    customdata=trial_ids_fail
                ), row=row_pos, col=1)

            # overlays: optimal (single point) and suggested (optional many)
            x_opt_disp = None
            if feature_names[j] in opt_df.columns:
                x_opt_disp = float(opt_df.iloc[0][feature_names[j]])
                y_opt      = float(opt_df.iloc[0]["pred_target_mean"])
                y_opt_sd   = float(opt_df.iloc[0].get("pred_target_sd", np.nan))
                fig.add_trace(go.Scattergl(
                    x=[x_opt_disp], y=[y_opt], mode="markers",
                    marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                    name="optimal", legendgroup="optimal", showlegend=show_legend,
                    hovertemplate=(f"predicted: %{{y:.3g}}"
                                   + ("" if np.isnan(y_opt_sd) else f" ± {y_opt_sd:.3g}")
                                   + f"<br>{feature_names[j]}: %{{x:.6g}}<extra></extra>")
                ), row=row_pos, col=1)

            if suggest and (suggest_df is not None) and (feature_names[j] in suggest_df.columns):
                xs = suggest_df[feature_names[j]].values.astype(float)
                ys = suggest_df["pred_target_mean"].values.astype(float)
                ysd = suggest_df.get("pred_target_sd", pd.Series([np.nan]*len(suggest_df))).values
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                    name="suggested", legendgroup="suggested", showlegend=show_legend,
                    hovertemplate=("predicted: %{y:.3g}"
                                   + (" ± %{customdata:.3g}" if not np.isnan(ysd).all() else "")
                                   + f"<br>{feature_names[j]}: %{{x:.6g}}<extra>suggested</extra>"),
                    customdata=ysd
                ), row=row_pos, col=1)

            # axes + ranges
            _maybe_log_axis(fig, row_pos, 1, feature_names[j], axis="x", transforms=transforms, j=j)
            fig.update_yaxes(title_text=f"{tgt_col}", row=row_pos, col=1)
            _set_yaxis_range(fig, row=row_pos, col=1,
                             y0=y0_plot, y1=y1_plot,
                             log=use_log_scale_for_target_y, eps=log_y_epsilon)
            fig.update_xaxes(title_text=feature_names[j], row=row_pos, col=1)

            # If a constraint limited the sweep, respect it on the displayed axis
            def _std_to_orig(val_std: float) -> float:
                vi = val_std * X_std[j] + X_mean[j]
                return float(_inverse_transform(transforms[j], np.array([vi]))[0])

            if j in range_windows:
                lo_std, hi_std = range_windows[j]
                x_min_override = min(_std_to_orig(lo_std), _std_to_orig(hi_std))
                x_max_override = max(_std_to_orig(lo_std), _std_to_orig(hi_std))
                span = (x_max_override - x_min_override) or 1.0
                pad = 0.02 * span
                fig.update_xaxes(range=[x_min_override - pad, x_max_override + pad], row=row_pos, col=1)
            elif j in choice_values and choice_values[j].size:
                ints  = choice_values[j] * X_std[j] + X_mean[j]
                origs = _inverse_transform(transforms[j], ints)
                span = float(np.max(origs) - np.min(origs)) or 1.0
                pad = 0.05 * span
                fig.update_xaxes(range=[float(np.min(origs) - pad), float(np.max(origs) + pad)], row=row_pos, col=1)

            # tidy rows
            for xd, xi, mu_i, sd_i, p_i in zip(x_display, x_internal, mu_grid, sd_grid, p_grid):
                tidy_rows.append({
                    "feature": feature_names[j],
                    "x_display": float(xd),
                    "x_internal": float(xi),
                    "target_conditional_mean": float(mu_i),
                    "target_conditional_sd": float(sd_i),
                    "success_probability": float(p_i),
                })

        else:
            base = str(key)
            labels_all = groups[base]["labels"]
            labels = cat_allowed.get(base, labels_all)

            # Evaluate each label with numerics and other bases fixed at x_opt_std
            Xn_grid = np.repeat(x_opt_std[None, :], len(labels), axis=0)
            for r, lab in enumerate(labels):
                for lab2 in labels_all:
                    member_name = groups[base]["name_by_label"][lab2]
                    j2 = name_to_idx[member_name]
                    raw_val = 1.0 if (lab2 == lab) else 0.0
                    Xn_grid[r, j2] = (raw_val - X_mean[j2]) / X_std[j2]

            p_vec = pred_success(Xn_grid)
            mu_vec, sd_vec = pred_loss(Xn_grid, include_observation_noise=True)

            # y transform
            if use_log_scale_for_target_y:
                mu_plot = np.maximum(mu_vec, log_y_epsilon)
                lo_plot = np.maximum(mu_vec - 2.0 * sd_vec, log_y_epsilon)
                hi_plot = np.maximum(mu_vec + 2.0 * sd_vec, log_y_epsilon)
                losses_s_plot = np.maximum(df_raw.loc[success_mask, tgt_col].to_numpy().astype(float), log_y_epsilon) if success_mask.any() else np.array([])
            else:
                mu_plot = mu_vec
                lo_plot = mu_vec - 2.0 * sd_vec
                hi_plot = mu_vec + 2.0 * sd_vec
                losses_s_plot = df_raw.loc[success_mask, tgt_col].to_numpy().astype(float) if success_mask.any() else np.array([])

            y_arrays = [lo_plot, hi_plot] + ([losses_s_plot] if losses_s_plot.size else [])
            y_low  = float(np.nanmin([np.nanmin(a) for a in y_arrays])) if y_arrays else 0.0
            y_high = float(np.nanmax([np.nanmax(a) for a in y_arrays])) if y_arrays else 1.0
            pad = 0.05 * (y_high - y_low + 1e-12)
            y0_plot = (y_low - pad) if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)
            y1_tmp  = (y_high + pad) if not use_log_scale_for_target_y else y_high * 1.2
            y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
            if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
                y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
            y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

            # x = 0..K-1 with tick labels
            x_pos = np.arange(len(labels), dtype=float)

            # grey out infeasible (p<thr)
            def _shade_for_thresh(thr: float, alpha: float):
                for k_i, p_i in enumerate(p_vec):
                    if p_i < thr:
                        fig.add_shape(
                            type="rect",
                            xref=f"x{'' if row_pos==1 else row_pos}",
                            yref=f"y{'' if row_pos==1 else row_pos}",
                            x0=k_i - 0.5, x1=k_i + 0.5,
                            y0=y0_plot, y1=y1_plot,
                            line=dict(width=0),
                            fillcolor=f"rgba(128,128,128,{alpha})",
                            layer="below",
                            row=row_pos, col=1
                        )
            _shade_for_thresh(0.8, 0.40)
            _shade_for_thresh(0.5, 0.25)

            show_legend = (row_pos == 1)
            fig.add_trace(go.Scatter(
                x=x_pos, y=mu_plot, mode="lines+markers",
                line=dict(width=2, color=line_color),
                marker=dict(size=7, color=line_color),
                error_y=dict(type="data", array=(hi_plot - mu_plot), arrayminus=(mu_plot - lo_plot), visible=True),
                name="E[target|success]", legendgroup="mean", showlegend=show_legend,
                hovertemplate=(f"{base}: %{{text}}<br>E[target|success]: %{{y:.3f}}<extra></extra>"),
                text=labels
            ), row=row_pos, col=1)

            # overlay optimal point for this base (single label at x*=opt)
            if base in opt_df.columns:
                lab_opt = str(opt_df.iloc[0][base])
                if lab_opt in labels:
                    xi = float(labels.index(lab_opt))
                    y_opt = float(opt_df.iloc[0]["pred_target_mean"])
                    y_opt_sd = float(opt_df.iloc[0].get("pred_target_sd", np.nan))
                    fig.add_trace(go.Scattergl(
                        x=[xi], y=[y_opt], mode="markers",
                        marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                        name="optimal", legendgroup="optimal", showlegend=show_legend,
                        hovertemplate=(f"predicted: %{{y:.3g}}"
                                       + ("" if np.isnan(y_opt_sd) else f" ± {y_opt_sd:.3g}")
                                       + f"<br>{base}: {lab_opt}<extra></extra>")
                    ), row=row_pos, col=1)

            # overlay suggestions (optional)
            if suggest and (suggest_df is not None) and (base in suggest_df.columns):
                labs_sug = suggest_df[base].astype(str).tolist()
                xs = [labels.index(l) for l in labs_sug if l in labels]
                if xs:
                    keep_mask = [l in labels for l in labs_sug]
                    y_sug = suggest_df.loc[keep_mask, "pred_target_mean"].values
                    fig.add_trace(go.Scattergl(
                        x=np.array(xs, dtype=float), y=y_sug, mode="markers",
                        marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                        name="suggested", legendgroup="suggested", showlegend=show_legend,
                        hovertemplate=(f"{base}: %{{text}}<br>"
                                       "predicted: %{{y:.3g}}<extra>suggested</extra>"),
                        text=[labels[int(i)] for i in xs]
                    ), row=row_pos, col=1)

            fig.update_xaxes(
                tickmode="array",
                tickvals=x_pos.tolist(),
                ticktext=labels,
                title_text=base,
                row=row_pos, col=1
            )
            fig.update_yaxes(title_text=f"{tgt_col}", row=row_pos, col=1)
            _set_yaxis_range(fig, row=row_pos, col=1,
                             y0=y0_plot, y1=y1_plot,
                             log=use_log_scale_for_target_y, eps=log_y_epsilon)

            # tidy rows
            for lab, mu_i, sd_i, p_i in zip(labels, mu_vec, sd_vec, p_vec):
                tidy_rows.append({
                    "feature": base,
                    "x_display": str(lab),
                    "x_internal": float("nan"),
                    "target_conditional_mean": float(mu_i),
                    "target_conditional_sd": float(sd_i),
                    "success_probability": float(p_i),
                })

    # ---- layout & IO ----
    parts = [f"1D PD at optimal setting of all other hyperparameters ({ds.attrs.get('target', 'target')})"]
    if kw_num_raw:
        def _fmt_c(v):
            if isinstance(v, slice):
                a = "" if v.start is None else f"{v.start:g}"
                b = "" if v.stop  is None else f"{v.stop:g}"
                return f"[{a},{b}]"
            if isinstance(v, (list, tuple, np.ndarray)):
                try:
                    return "[" + ",".join(f"{float(x):g}" for x in np.asarray(v).tolist()) + "]"
                except Exception:
                    return "[" + ",".join(map(str, v)) + "]"
            try:
                return f"{float(v):g}"
            except Exception:
                return str(v)
        parts.append(", ".join(f"{k}={_fmt_c(v)}" for k, v in kw_num_raw.items()))
    if cat_fixed:
        parts.append(", ".join(f"{b}={lab}" for b, lab in cat_fixed.items()))
    title = " — ".join(parts)

    width = width if (width and width > 0) else 1200
    height = height if (height and height > 0) else 1200
    fig.update_layout(height=height, width=width, template="simple_white", title=title, legend_title_text="")

    if output:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn")
    if csv_out:
        csv_out = Path(csv_out); csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tidy_rows).to_csv(str(csv_out), index=False)
    if show:
        fig.show("browser")
    return fig
