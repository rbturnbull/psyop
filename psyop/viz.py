# -*- coding: utf-8 -*-
from pathlib import Path
import re
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
from .opt import (
    suggest_candidates,
    find_optimal,
)


def _canon_key_set(ds) -> dict[str, str]:
    feats = [str(x) for x in ds["feature"].values.tolist()]
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    return {**{f: f for f in feats}, **{_norm(f): f for f in feats}}

def _apply_transform_name(tr: str, x):
    x = np.asarray(x, dtype=float)
    if tr == "log10":
        return np.log10(np.maximum(x, 1e-300))
    return x

def _orig_to_std(j: int, vals, transforms, X_mean, X_std):
    raw = _apply_transform_name(transforms[j], np.asarray(vals, dtype=float))
    return (raw - X_mean[j]) / X_std[j]


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


def make_pairplot(
    model: xr.Dataset | Path | str,
    output: Path| None = None,
    n_points_1d: int = 300,
    n_points_2d: int = 70,
    use_log_scale_for_target: bool = False,   # log10 colors for heatmaps
    log_shift_epsilon: float = 1e-9,
    colorscale: str = "RdBu",
    show: bool = False,
    n_contours: int = 12,
    optimal:bool = True,
    **kwargs,
) -> go.Figure:
    """
    2D Partial Dependence of E[target|success] (pairwise features), optionally
    conditioned on fixed variables passed as kwargs, e.g. --epochs 20.
    Fixed variables are clamped in the slice and **not plotted** as axes.
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)

    optimal_df = find_optimal(ds, count=1, **kwargs) if optimal else None

    pred_success, pred_loss = _build_predictors(ds)

    # --- features & transforms
    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std = ds["feature_std"].values.astype(float)

    # canonicalize kwargs keys to exact feature names
    # (simple case-sensitive match; you can reuse your CLI canonicalizer if needed)
    kw_fixed = {k: v for k, v in kwargs.items() if k in feature_names}

    # raw DF for experimental points
    df_raw = _raw_dataframe_from_dataset(ds)

    # standardized training X for ranges/medians
    Xn_train = ds["Xn_train"].values.astype(float)
    p = Xn_train.shape[1]

    # --- map kwargs keys to real feature names (accept hyphens/underscores/case)
    idx = _canon_key_set(ds)
    kw_fixed_raw = {}
    for k, v in kwargs.items():
        if k in idx: kw_fixed_raw[idx[k]] = v
        else:
            nk = re.sub(r"[^a-z0-9]+", "", k.lower())
            if nk in idx: kw_fixed_raw[idx[nk]] = v

    # --- split constraints:
    # scalars -> fix & hide axis; slice -> [lo,hi] window; list/tuple -> discrete grid
    fixed_scalars: dict[int, float] = {}
    range_windows: dict[int, tuple[float, float]] = {}
    choice_values: dict[int, np.ndarray] = {}

    for j, name in enumerate(feature_names):
        if name not in kw_fixed_raw: 
            continue
        val = kw_fixed_raw[name]
        if isinstance(val, slice):
            # endpoints are in ORIGINAL units → convert to standardized
            lo = _orig_to_std(j, val.start, transforms, X_mean, X_std)
            hi = _orig_to_std(j, val.stop,  transforms, X_mean, X_std)
            lo, hi = (float(min(lo, hi)), float(max(lo, hi)))
            range_windows[j] = (lo, hi)
        elif isinstance(val, (list, tuple)):
            # choices in ORIGINAL units → convert each to standardized
            choice_values[j] = _orig_to_std(j, np.array(val, dtype=float), transforms, X_mean, X_std)
        else:
            # scalar in ORIGINAL units → fix & hide axis
            fixed_scalars[j] = float(_orig_to_std(j, float(val), transforms, X_mean, X_std))

    # --- base point in standardized space; apply fixed slice there
    base_std = np.median(Xn_train, axis=0)
    for j, vstd in fixed_scalars.items():
        base_std[j] = vstd

    free_idx = [j for j in range(len(feature_names)) if j not in fixed_scalars]
    if not free_idx:
        raise ValueError("All features are fixed; nothing to plot.")

    # helper to test if a feature is log on axis
    def _is_log_feature(j: int) -> bool:
        return (transforms[j] == "log10")

    # 1D / 2D sweep ranges for *all* features (we'll index by free_idx)
    p01p99 = [np.percentile(Xn_train[:, j], [1, 99]) for j in range(len(feature_names))]

    def _grid_1d(j: int, n: int) -> np.ndarray:
        p01, p99 = p01p99[j]
        if j in choice_values:
            # keep only within empirical bounds
            vals = np.asarray(choice_values[j], dtype=float)
            vals = vals[(vals >= p01) & (vals <= p99)]
            return np.unique(np.sort(vals))
        lo, hi = (p01, p99)
        if j in range_windows:
            cw_lo, cw_hi = range_windows[j]
            lo, hi = max(lo, cw_lo), min(hi, cw_hi)
        if hi <= lo:
            hi = lo + 1e-9
        return np.linspace(lo, hi, n)

    ranges_1d_std = [_grid_1d(j, n_points_1d) for j in range(len(feature_names))]
    ranges_2d_std = [_grid_1d(j, n_points_2d) for j in range(len(feature_names))]

    # ---------- diagonal payload over FREE features ----------
    diag_payload: dict[int, dict] = {}
    for pos, j in enumerate(free_idx):
        grid = ranges_1d_std[j]
        Xn_grid = np.repeat(base_std[None, :], len(grid), axis=0)
        Xn_grid[:, j] = grid

        p_1d = pred_success(Xn_grid)
        mu_1d, _ = pred_loss(Xn_grid, include_observation_noise=True)

        x_orig = _denormalize_then_inverse_transform(j, grid, transforms, X_mean, X_std)
        diag_payload[pos] = {
            "j": j,              # real feature index
            "x": x_orig,         # original units
            "Zmu": 0.5 * (mu_1d[:, None] + mu_1d[None, :]),
            "Zp":  np.minimum(p_1d[:, None], p_1d[None, :]),
        }

    # ---------- off-diagonal payload over FREE×FREE ----------
    off_payload: dict[tuple[int, int], dict] = {}
    for r, i in enumerate(free_idx):
        for c, j in enumerate(free_idx):
            if i == j:
                continue
            xg = ranges_2d_std[j]
            yg = ranges_2d_std[i]
            XX, YY = np.meshgrid(xg, yg)

            Xn_grid = np.repeat(base_std[None, :], XX.size, axis=0)
            Xn_grid[:, j] = XX.ravel()
            Xn_grid[:, i] = YY.ravel()

            mu_flat, _ = pred_loss(Xn_grid, include_observation_noise=True)
            p_flat = pred_success(Xn_grid)

            off_payload[(r, c)] = {
                "i": i, "j": j,
                "x": _denormalize_then_inverse_transform(j, xg, transforms, X_mean, X_std),
                "y": _denormalize_then_inverse_transform(i, yg, transforms, X_mean, X_std),
                "Zmu": mu_flat.reshape(YY.shape),
                "Zp":  p_flat.reshape(YY.shape),
            }

    # ---------- color transform (global bounds over plotted cells only) ----------
    def color_transform(z_raw: np.ndarray) -> tuple[np.ndarray, float]:
        if not use_log_scale_for_target:
            return z_raw, 0.0
        zmin = float(np.nanmin(z_raw))
        shift = 0.0 if zmin > 0 else -zmin + float(log_shift_epsilon)
        return np.log10(np.maximum(z_raw + shift, log_shift_epsilon)), shift

    blocks = [d["Zmu"].ravel() for d in diag_payload.values()] + \
             [d["Zmu"].ravel() for d in off_payload.values()]
    z_all = np.concatenate(blocks) if blocks else np.array([0.0, 1.0])
    z_all_t, global_shift = color_transform(z_all)
    cmin_t = float(np.nanmin(z_all_t))
    cmax_t = float(np.nanmax(z_all_t))

    cs = get_colorscale(colorscale)

    def contour_line_color(level_raw: float) -> str:
        zt = np.log10(max(level_raw + global_shift, log_shift_epsilon)) if use_log_scale_for_target else level_raw
        t = 0.5 if cmax_t == cmin_t else (zt - cmin_t) / (cmax_t - cmin_t)
        from_colorscale = sample_colorscale(cs, [float(np.clip(t, 0.0, 1.0))])[0]
        r, g, b = _rgb_string_to_tuple(from_colorscale)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        inv = 1.0 - lum
        grey = int(round(inv * 255))
        return f"rgba({grey},{grey},{grey},0.9)"

    # ---------- figure ----------
    k = len(free_idx)
    fig = make_subplots(
        rows=k, cols=k,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.03, vertical_spacing=0.03,
    )

    # success/fail masks for experimental points
    tgt_col = str(ds.attrs["target_column"])
    success_mask = ~pd.isna(df_raw[tgt_col]).to_numpy()
    fail_mask = ~success_mask

    def data_vals_for_feature(j_full: int) -> np.ndarray:
        name = feature_names[j_full]
        if name in df_raw.columns:
            return df_raw[name].to_numpy().astype(float)
        return feature_raw_from_artifact_or_reconstruct(ds, j_full, name, transforms[j_full]).astype(float)

    # ---- off-diagonals (r,c) over FREE features only
    for (r, c), PAY in off_payload.items():
        i, j = PAY["i"], PAY["j"]
        x_vals = PAY["x"]; y_vals = PAY["y"]; Zmu_raw = PAY["Zmu"]; Zp = PAY["Zp"]
        Z_t, _ = color_transform(Zmu_raw)

        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=Z_t,
            coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names[j]}: %{{x:.6g}}<br>{feature_names[i]}: %{{y:.6g}}"
                "<br>E[target|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Zmu_raw
        ), row=r+1, col=c+1)

        for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
            mask = np.where(Zp < thr, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                x=x_vals, y=y_vals, z=mask, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                showscale=False, hoverinfo="skip"
            ), row=r+1, col=c+1)

        zmin_r, zmax_r = float(np.nanmin(Zmu_raw)), float(np.nanmax(Zmu_raw))
        levels = np.linspace(zmin_r, zmax_r, max(n_contours, 2))
        for lev in levels:
            fig.add_trace(go.Contour(
                x=x_vals, y=y_vals, z=Zmu_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=contour_line_color(lev)),
                showscale=False, hoverinfo="skip"
            ), row=r+1, col=c+1)

        # experimental points for (i,j)
        xd = data_vals_for_feature(j); yd = data_vals_for_feature(i)
        fig.add_trace(go.Scattergl(
            x=xd[success_mask], y=yd[success_mask], mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_succ",
            showlegend=(r == 0 and c == 0),
            hovertemplate=(
                "trial_id: %{customdata[0]}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{feature_names[i]}: %{{y:.6g}}<br>"
                f"{tgt_col}: %{{customdata[1]:.4f}}<extra></extra>"
            ),
            customdata=np.column_stack([
                df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask],
                df_raw[tgt_col].to_numpy()[success_mask],
            ])
        ), row=r+1, col=c+1)
        fig.add_trace(go.Scattergl(
            x=xd[fail_mask], y=yd[fail_mask], mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_fail",
            showlegend=(r == 0 and c == 0),
            hovertemplate=(
                "trial_id: %{customdata}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{feature_names[i]}: %{{y:.6g}}<br>"
                "status: failed (NaN target)<extra></extra>"
            ),
            customdata=df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]
        ), row=r+1, col=c+1)

        if optimal:
            opt_x = optimal_df[feature_names[j]].values
            opt_y = optimal_df[feature_names[i]].values
            pred_target_mean = optimal_df["pred_target_mean"].values[0]
            fig.add_trace(go.Scattergl(
                x=opt_x, y=opt_y, mode="markers",
                marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                name="optimal", legendgroup="optimal",
                showlegend=(r == 0 and c == 0),
                hovertemplate=(
                    f"predicted: {pred_target_mean:.2g} ± {optimal_df['pred_target_sd'].values[0]:.2g}<br>"
                    f"{feature_names[j]}: %{{x:.6g}}<br>"
                    f"{feature_names[i]}: %{{y:.6g}}<extra></extra>"
                ),
            ), row=r+1, col=c+1)

        _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="x", centers=x_vals, is_log=_is_log_feature(j))
        _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="y", centers=y_vals, is_log=_is_log_feature(i))

    # ---- diagonals over FREE features only
    for pos, PAY in diag_payload.items():
        j = PAY["j"]
        axis = PAY["x"]; Zmu_raw = PAY["Zmu"]; Zp = PAY["Zp"]
        Z_t, _ = color_transform(Zmu_raw)

        fig.add_trace(go.Heatmap(
            x=axis, y=axis, z=Z_t,
            coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names[j]}: %{{x:.6g}} / %{{y:.6g}}"
                "<br>E[target|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Zmu_raw
        ), row=pos+1, col=pos+1)

        for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
            mask = np.where(Zp < thr, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                x=axis, y=axis, z=mask, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                showscale=False, hoverinfo="skip"
            ), row=pos+1, col=pos+1)

        zmin_r, zmax_r = float(np.nanmin(Zmu_raw)), float(np.nanmax(Zmu_raw))
        levels = np.linspace(zmin_r, zmax_r, max(n_contours, 2))
        for lev in levels:
            fig.add_trace(go.Contour(
                x=axis, y=axis, z=Zmu_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=contour_line_color(lev)),
                showscale=False, hoverinfo="skip"
            ), row=pos+1, col=pos+1)

        # experimental points (x=y) for this feature
        xd = data_vals_for_feature(j)
        fig.add_trace(go.Scattergl(
            x=xd[success_mask], y=xd[success_mask], mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_succ",
            showlegend=False,
            hovertemplate=(
                "trial_id: %{customdata[0]}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{tgt_col}: %{{customdata[1]:.4f}}<extra></extra>"
            ),
            customdata=np.column_stack([
                df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask],
                df_raw[tgt_col].to_numpy()[success_mask],
            ])
        ), row=pos+1, col=pos+1)
        fig.add_trace(go.Scattergl(
            x=xd[fail_mask], y=xd[fail_mask], mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_fail",
            showlegend=False,
            hovertemplate=(
                "trial_id: %{customdata}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                "status: failed (NaN target)<extra></extra>"
            ),
            customdata=df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]
        ), row=pos+1, col=pos+1)

        _update_axis_type_and_range(fig, row=pos+1, col=pos+1, axis="x", centers=axis, is_log=_is_log_feature(j))
        _update_axis_type_and_range(fig, row=pos+1, col=pos+1, axis="y", centers=axis, is_log=_is_log_feature(j))

    # axis titles (free only)
    free_names = [feature_names[j] for j in free_idx]
    for c, name in enumerate(free_names):
        fig.update_xaxes(title_text=name, row=len(free_idx), col=c+1)
    for r, name in enumerate(free_names):
        fig.update_yaxes(title_text=name, row=r+1, col=1)


    def _fmt_c(v):
        if isinstance(v, slice):
            a = f"{v.start:g}" if v.start is not None else ""
            b = f"{v.stop:g}"  if v.stop  is not None else ""
            return f"[{a},{b}]"
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ",".join(f"{x:g}" for x in np.asarray(v).tolist()) + "]"
        return f"{v:g}"

    title = "2D Partial Dependence of Expected Target"
    if kwargs:
        # show original kwargs (as passed) in title
        # re-use kw_fixed_raw so we show feature names canonically
        parts = [f"{k}={_fmt_c(kw_fixed_raw[k])}" for k in kw_fixed_raw]
        title += " — " + ", ".join(parts)

    # layout
    cell = 250
    colorbar_title = "E[target|success]" + (" (log10)" if use_log_scale_for_target else "")
    if use_log_scale_for_target and global_shift > 0:
        colorbar_title += f" (shift Δ={global_shift:.3g})"
    fig.update_layout(
        coloraxis=dict(colorscale=colorscale, cmin=cmin_t, cmax=cmax_t,
                       colorbar=dict(title=colorbar_title)),
        template="simple_white",
        width=cell * len(free_idx),
        height=cell * len(free_idx),
        title=title,
        legend_title_text=""
    )

    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs="cdn")
    if show:
        fig.show("browser")

    return fig


def make_partial_dependence1D(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    csv_out: Path | None = None,
    n_points_1d: int = 300,
    line_color: str = "rgb(31,119,180)",
    band_alpha: float = 0.25,
    figure_height_per_row_px: int = 320,
    show_figure: bool = False,
    use_log_scale_for_target_y: bool = True,   # log-y for target
    log_y_epsilon: float = 1e-9,
    optimal: bool = True,
    suggest: int = 0,
    **kwargs,
) -> go.Figure:
    """
    Vertical 1D PD panels of E[target|success] vs each *free* feature.
    Scalars (fix & hide), slices (restrict sweep & x-range), lists/tuples (discrete grids).
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std  = ds["feature_std"].values.astype(float)

    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    p = Xn_train.shape[1]

    # --- canonicalize kwargs -> real feature names (same as plot2d) ---
    idx_map = _canon_key_set(ds)
    kw_fixed_raw: dict[str, object] = {}
    for k, v in kwargs.items():
        if k in idx_map:
            kw_fixed_raw[idx_map[k]] = v
        else:
            import re
            nk = re.sub(r"[^a-z0-9]+", "", str(k).lower())
            if nk in idx_map:
                kw_fixed_raw[idx_map[nk]] = v

    # --- split constraints into scalar fixes, ranges (slices), choices (lists/tuples) in STANDARDIZED space ---
    fixed_scalars: dict[int, float] = {}
    range_windows: dict[int, tuple[float, float]] = {}
    choice_values: dict[int, np.ndarray] = {}

    for j, name in enumerate(feature_names):
        if name not in kw_fixed_raw:
            continue
        val = kw_fixed_raw[name]
        if isinstance(val, slice):
            lo = _orig_to_std(j, val.start, transforms, X_mean, X_std)
            hi = _orig_to_std(j, val.stop,  transforms, X_mean, X_std)
            lo, hi = float(min(lo, hi)), float(max(lo, hi))
            range_windows[j] = (lo, hi)
        elif isinstance(val, (list, tuple)):
            choice_values[j] = _orig_to_std(j, np.asarray(val, dtype=float), transforms, X_mean, X_std)
        else:
            fixed_scalars[j] = float(_orig_to_std(j, float(val), transforms, X_mean, X_std))

    # --- optional overlays (use the same canonical constraints) ---
    optimal_df  = find_optimal(model, count=1, **kw_fixed_raw) if optimal else None
    suggest_df  = suggest_candidates(model, count=suggest, **kw_fixed_raw) if (suggest and suggest > 0) else None

    # --- base point (median) then apply scalar fixes ---
    base_std = np.median(Xn_train, axis=0)
    for j, vstd in fixed_scalars.items():
        base_std[j] = vstd

    # --- features to plot (exclude only scalar-fixed) ---
    free_idx = [j for j in range(p) if j not in fixed_scalars]
    if not free_idx:
        raise ValueError("All features are fixed; nothing to plot. Fix fewer variables.")

    # empirical 1–99% for sane bounds
    p01p99 = [np.percentile(Xn_train[:, j], [1, 99]) for j in range(p)]

    def _grid_1d(j: int, n: int) -> np.ndarray:
        p01, p99 = p01p99[j]
        if j in choice_values:
            vals = np.asarray(choice_values[j], dtype=float)
            vals = vals[(vals >= p01) & (vals <= p99)]
            return np.unique(np.sort(vals)) if vals.size else np.array([np.median(Xn_train[:, j])], dtype=float)
        lo, hi = p01, p99
        if j in range_windows:
            rlo, rhi = range_windows[j]
            lo, hi = max(lo, rlo), min(hi, rhi)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = p01, max(p01 + 1e-9, p99)
        return np.linspace(lo, hi, n)

    # sweeps (standardized)
    sweeps_std = {j: _grid_1d(j, n_points_1d) for j in free_idx}

    # masks/data
    tgt_col = str(ds.attrs["target_column"])
    success_mask = ~pd.isna(df_raw[tgt_col]).to_numpy()
    fail_mask    = ~success_mask
    losses_success = df_raw.loc[success_mask, tgt_col].to_numpy().astype(float)
    trial_ids_success = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask]
    trial_ids_fail    = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]

    band_fill_rgba = _rgb_to_rgba(line_color, band_alpha)

    free_names = [feature_names[j] for j in free_idx]
    fig = make_subplots(
        rows=len(free_idx), cols=1, shared_xaxes=False,
        subplot_titles=[f"E[target|success] — {name}" for name in free_names]
    )

    tidy_rows: list[dict] = []

    for row_pos, j in enumerate(free_idx, start=1):
        grid = sweeps_std[j]
        Xn_grid = np.repeat(base_std[None, :], len(grid), axis=0)
        Xn_grid[:, j] = grid

        # predictions
        p_grid = pred_success(Xn_grid)
        mu_grid, sd_grid = pred_loss(Xn_grid, include_observation_noise=True)

        # x (original units)
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

        # y-range
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

        # shading + PD band + mean
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
        if feature_names[j] in df_raw.columns:
            x_data_all = df_raw[feature_names[j]].to_numpy().astype(float)
        else:
            x_data_all = feature_raw_from_artifact_or_reconstruct(ds, j, feature_names[j], transforms[j]).astype(float)

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
            fig.add_trace(go.Scattergl(
                x=x_sug, y=y_sug, mode="markers",
                marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                name="suggested", legendgroup="suggested", showlegend=show_legend,
                hovertemplate=(f"predicted: %{{y:.3g}}<br>"
                               f"{feature_names[j]}: %{{x:.6g}}<extra></extra>")
            ), row=row_pos, col=1)

        # axes
        _maybe_log_axis(fig, row_pos, 1, feature_names[j], axis="x", transforms=transforms, j=j)
        fig.update_yaxes(title_text=f"{tgt_col} (E[target|success])", row=row_pos, col=1)
        _set_yaxis_range(fig, row=row_pos, col=1,
                         y0=y0_plot, y1=y1_plot,
                         log=use_log_scale_for_target_y, eps=log_y_epsilon)

        # --- HARD-RESTRICT X-RANGE TO CONSTRAINT WINDOW (original units) ---
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
                # small multiplicative pad
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

    # title w/ constraints summary
    def _fmt_c(v):
        if isinstance(v, slice):
            a = "" if v.start is None else f"{v.start:g}"
            b = "" if v.stop  is None else f"{v.stop:g}"
            return f"[{a},{b}]"
        if isinstance(v, (list, tuple, np.ndarray)):
            return "[" + ",".join(f"{x:g}" for x in np.asarray(v).tolist()) + "]"
        try:
            return f"{float(v):g}"
        except Exception:
            return str(v)

    title = "1D Partial Dependence of Expected Target (conditioned slice)"
    if kw_fixed_raw:
        title += " — " + ", ".join(f"{k}={_fmt_c(kw_fixed_raw[k])}" for k in kw_fixed_raw)

    fig.update_layout(
        height=figure_height_per_row_px * len(free_idx),
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
    if show_figure:
        fig.show("browser")

    return fig

# =============================================================================
# Helpers: dataset → predictors & featurization
# =============================================================================

def _build_predictors(ds: xr.Dataset):
    """Reconstruct fast GP predictors from the artifact (no PyMC required)."""
    # Training matrices / targets
    Xn_all = ds["Xn_train"].values.astype(float)               # (N, p)
    y_success = ds["y_success"].values.astype(float)           # (N,)
    Xn_ok = ds["Xn_success_only"].values.astype(float)         # (N_ok, p)
    y_loss_centered = ds["y_loss_centered"].values.astype(float)
    cond_mean = float(ds["conditional_loss_mean"].values)

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

    def predict_conditional_target(Xn: np.ndarray, include_observation_noise: bool = True):
        Kl = kernel_m52_ard(Xn, Xn_ok, ell_l, eta_l)
        mu_centered = mean_c + Kl @ alpha_l
        mu = mu_centered + cond_mean
        v = solve_lower(L_l, Kl.T)
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
