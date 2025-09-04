# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import get_colorscale, sample_colorscale
from plotly.colors import get_colorscale, sample_colorscale  # make sure you have these imports

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

def _is_log_feature(j: int, transforms: list[str]) -> bool:
    return str(transforms[j]).lower() == "log10"


def make_pairplot(
    model: Path,
    output: Path,
    n_points_1d: int = 300,
    n_points_2d: int = 70,
    use_log_scale_for_target: bool = False,   # log10 colours for heatmaps
    log_shift_epsilon: float = 1e-9,
    colourscale: str = "RdBu",
    show: bool = False,
    n_contours: int = 12,
    fixed: Optional[dict[str, float]] = None,  # <-- condition on variables in ORIGINAL units
) -> None:
    """
    Pairplot of E[target | success] under the two-head GP:
      - diagonal: symmetric 2D heatmap from 1D PD (Z = 0.5*(mu(x_i)+mu(x_j)))
      - off-diagonals: true 2D heatmaps over (x_j, x_i)
      - shading: grey where P(success)<0.8 (darker) and <0.5 (lighter)
      - contours: line colour is inverse greyscale of colourscale luminance at same level
      - experimental points: successes (black), failures (red with black outline)

    `fixed` lets you condition the slice: all other dims are set to their **fixed**
    values (standardized internally), otherwise to the median in standardized space.
    """
    ds = xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    # Features & transforms
    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std = ds["feature_std"].values.astype(float)

    # Raw dataframe for experimental points (columns are stored directly in ds)
    df_raw = _raw_dataframe_from_dataset(ds)

    # Standardized training X for ranges & medians
    Xn_train = ds["Xn_train"].values.astype(float)
    num_features = Xn_train.shape[1]

    # Base point in standardized space (median), conditioned by 'fixed'
    base_std = np.median(Xn_train, axis=0)
    if fixed:
        base_std = _apply_fixed_to_base(base_std, fixed, feature_names, transforms, X_mean, X_std)

    # 1D and 2D sweep ranges (in standardized space) from empirical 1–99% quantiles
    ranges_1d_std = [
        np.linspace(*np.percentile(Xn_train[:, j], [1, 99]), n_points_1d)
        for j in range(num_features)
    ]
    ranges_2d_std = [
        np.linspace(*np.percentile(Xn_train[:, j], [1, 99]), n_points_2d)
        for j in range(num_features)
    ]

    # ---------- diagonal (1D -> symmetric 2D) ----------
    diag_payload: dict[int, dict] = {}
    for j in range(num_features):
        grid = ranges_1d_std[j]
        Xn_grid = np.repeat(base_std[None, :], len(grid), axis=0)
        Xn_grid[:, j] = grid

        # predictions
        p_1d = pred_success(Xn_grid)
        mu_1d, _ = pred_loss(Xn_grid, include_observation_noise=True)

        # original display axis (inverse transform)
        x_orig = _denormalize_then_inverse_transform(j, grid, transforms, X_mean, X_std)
        x_display = x_orig  # original units
        # store symmetric Z for diagonal
        Z_mu_diag = 0.5 * (mu_1d[:, None] + mu_1d[None, :])
        Z_p_diag = np.minimum(p_1d[:, None], p_1d[None, :])
        diag_payload[j] = {"x": x_display, "Zmu": Z_mu_diag, "Zp": Z_p_diag}

    # ---------- off-diagonals (true 2D) ----------
    off_payload: dict[tuple[int, int], dict] = {}
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                continue
            xg = ranges_2d_std[j]
            yg = ranges_2d_std[i]
            XX, YY = np.meshgrid(xg, yg)  # (Ny, Nx)

            Xn_grid = np.repeat(base_std[None, :], XX.size, axis=0)
            Xn_grid[:, j] = XX.ravel()
            Xn_grid[:, i] = YY.ravel()

            mu_flat, _ = pred_loss(Xn_grid, include_observation_noise=True)
            p_flat = pred_success(Xn_grid)

            Zmu = mu_flat.reshape(YY.shape)
            Zp = p_flat.reshape(YY.shape)

            x_orig = _denormalize_then_inverse_transform(j, xg, transforms, X_mean, X_std)
            y_orig = _denormalize_then_inverse_transform(i, yg, transforms, X_mean, X_std)
            off_payload[(i, j)] = {"x": x_orig, "y": y_orig, "Zmu": Zmu, "Zp": Zp}

    # ---------- colour transform globals ----------
    def color_transform(z_raw: np.ndarray) -> tuple[np.ndarray, float]:
        """Return transformed Z and global shift used."""
        if not use_log_scale_for_target:
            return z_raw, 0.0
        zmin = float(np.nanmin(z_raw))
        shift = 0.0
        if zmin <= 0:
            shift = -zmin + float(log_shift_epsilon)
        return np.log10(np.maximum(z_raw + shift, log_shift_epsilon)), shift

    all_blocks = [d["Zmu"].ravel() for d in diag_payload.values()] + [d["Zmu"].ravel() for d in off_payload.values()]
    if len(all_blocks) == 0:
        z_all = np.array([0.0, 1.0])
    else:
        z_all = np.concatenate(all_blocks)
    z_all_t, global_shift = color_transform(z_all)
    cmin_t = float(np.nanmin(z_all_t))
    cmax_t = float(np.nanmax(z_all_t))

    # For contour line greys: invert luminance of colourscale at normalized transformed level
    cs = get_colorscale(colourscale)

    def contour_line_color(level_raw: float) -> str:
        zt = np.log10(max(level_raw + global_shift, log_shift_epsilon)) if use_log_scale_for_target else level_raw
        t = 0.5 if cmax_t == cmin_t else (zt - cmin_t) / (cmax_t - cmin_t)
        t = float(np.clip(t, 0.0, 1.0))
        rgb = sample_colorscale(cs, [t])[0]  # "rgb(r,g,b)"
        r, g, b = _rgb_string_to_tuple(rgb)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        inv = 1.0 - lum
        grey = int(round(inv * 255))
        return f"rgba({grey},{grey},{grey},0.9)"

    # ---------- figure ----------
    fig = make_subplots(
        rows=num_features, cols=num_features,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.06, vertical_spacing=0.06,
    )

    # Experimental points setup
    success_mask = ~pd.isna(df_raw[ds.attrs["target_column"]]).to_numpy()
    fail_mask = ~success_mask

    def data_vals_for_feature(index: int) -> np.ndarray:
        name = feature_names[index]
        # use raw column if available
        if name in df_raw.columns:
            return df_raw[name].to_numpy().astype(float)
        # fallback: reconstruct from Xn_train + (mean, std, transform)
        return _feature_raw_from_artifact_or_reconstruct(ds, index, name, transforms[index]).astype(float)

    # Off-diagonals
    for (i, j), PAY in off_payload.items():
        row, col = i + 1, j + 1
        x_vals = PAY["x"]; y_vals = PAY["y"]; Zmu_raw = PAY["Zmu"]; Zp = PAY["Zp"]
        Z_t, _ = color_transform(Zmu_raw)

        # Heatmap
        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=Z_t, coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names[j]}: %{{x:.6g}}<br>{feature_names[i]}: %{{y:.6g}}"
                "<br>E[target|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Zmu_raw
        ), row=row, col=col)

        # Probability shading
        for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
            mask = np.where(Zp < thr, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                x=x_vals, y=y_vals, z=mask, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                showscale=False, hoverinfo="skip"
            ), row=row, col=col)

        # Contours
        zmin_r, zmax_r = float(np.nanmin(Zmu_raw)), float(np.nanmax(Zmu_raw))
        levels = np.linspace(zmin_r, zmax_r, max(n_contours, 2))
        for lev in levels:
            fig.add_trace(go.Contour(
                x=x_vals, y=y_vals, z=Zmu_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=contour_line_color(lev)),
                showscale=False, hoverinfo="skip"
            ), row=row, col=col)

        # Experimental points
        xd = data_vals_for_feature(j)
        yd = data_vals_for_feature(i)

        fig.add_trace(go.Scattergl(
            x=xd[success_mask], y=yd[success_mask], mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_succ",
            showlegend=(row == 1 and col == 1),
            hovertemplate=(
                "trial_id: %{customdata[0]}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{feature_names[i]}: %{{y:.6g}}<br>"
                f"{ds.attrs['target_column']}: %{{customdata[1]:.4f}}<extra></extra>"
            ),
            customdata=np.column_stack([
                df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask],
                df_raw[ds.attrs["target_column"]].to_numpy()[success_mask],
            ])
        ), row=row, col=col)

        fig.add_trace(go.Scattergl(
            x=xd[fail_mask], y=yd[fail_mask], mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_fail",
            showlegend=(row == 1 and col == 1),
            hovertemplate=(
                "trial_id: %{customdata}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{feature_names[i]}: %{{y:.6g}}<br>"
                "status: failed (NaN target)<extra></extra>"
            ),
            customdata=df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]
        ), row=row, col=col)

        is_log_x = _is_log_feature(j, transforms)
        is_log_y = _is_log_feature(i, transforms)
        _update_axis_type_and_range(fig, row=row, col=col, axis="x", centers=x_vals, is_log=is_log_x)
        _update_axis_type_and_range(fig, row=row, col=col, axis="y", centers=y_vals, is_log=is_log_y)

    # Diagonals
    for j, PAY in diag_payload.items():
        row = col = j + 1
        axis = PAY["x"]
        Zmu_raw = PAY["Zmu"]; Zp = PAY["Zp"]
        Z_t, _ = color_transform(Zmu_raw)

        fig.add_trace(go.Heatmap(
            x=axis, y=axis, z=Z_t, coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names[j]}: %{{x:.6g}} / %{{y:.6g}}"
                "<br>E[target|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Zmu_raw
        ), row=row, col=col)

        for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
            mask = np.where(Zp < thr, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                x=axis, y=axis, z=mask, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                showscale=False, hoverinfo="skip"
            ), row=row, col=col)

        zmin_r, zmax_r = float(np.nanmin(Zmu_raw)), float(np.nanmax(Zmu_raw))
        levels = np.linspace(zmin_r, zmax_r, max(n_contours, 2))
        for lev in levels:
            fig.add_trace(go.Contour(
                x=axis, y=axis, z=Zmu_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=contour_line_color(lev)),
                showscale=False, hoverinfo="skip"
            ), row=row, col=col)

        # Experimental points (x=y)
        xd = data_vals_for_feature(j)
        fig.add_trace(go.Scattergl(
            x=xd[success_mask], y=xd[success_mask], mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_succ",
            showlegend=False,
            hovertemplate=(
                "trial_id: %{customdata[0]}<br>"
                f"{feature_names[j]}: %{{x:.6g}}<br>"
                f"{ds.attrs['target_column']}: %{{customdata[1]:.4f}}<extra></extra>"
            ),
            customdata=np.column_stack([
                df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask],
                df_raw[ds.attrs["target_column"]].to_numpy()[success_mask],
            ])
        ), row=row, col=col)

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
        ), row=row, col=col)

        is_log = _is_log_feature(j, transforms)
        _update_axis_type_and_range(fig, row=row, col=col, axis="x", centers=axis, is_log=is_log)
        _update_axis_type_and_range(fig, row=row, col=col, axis="y", centers=axis, is_log=is_log)

    # Axis titles
    for j in range(num_features):
        fig.update_xaxes(title_text=feature_names[j], row=num_features, col=j + 1)
    for i in range(num_features):
        fig.update_yaxes(title_text=feature_names[i], row=i + 1, col=1)

    # Layout & export
    cell = 250
    colorbar_title = "E[target|success]" + (" (log10)" if use_log_scale_for_target else "")
    if use_log_scale_for_target and global_shift > 0:
        colorbar_title += f" (shift Δ={global_shift:.3g})"
    fig.update_layout(
        coloraxis=dict(colorscale=colourscale, cmin=cmin_t, cmax=cmax_t,
                       colorbar=dict(title=colorbar_title)),
        template="simple_white",
        width=cell * num_features,
        height=cell * num_features,
        title="Pairplot — E[target|success] with inverted-luminance contours & data points",
        legend_title_text=""
    )

    fig.write_html(str(output), include_plotlyjs="cdn")
    if show:
        fig.show("browser")


def make_partial_dependence1D(
    model: Path,
    output: Path,
    csv_out: Path,
    n_points_1d: int = 300,
    line_color: str = "rgb(31,119,180)",
    band_alpha: float = 0.25,
    figure_height_per_row_px: int = 320,
    show_figure: bool = False,
    use_log_scale_for_target_y: bool = True,   # log-y for target
    log_y_epsilon: float = 1e-9,
    fixed: Optional[dict[str, float]] = None,  # <-- condition on variables in ORIGINAL units
) -> None:
    """
    Vertical 1D PD panels of E[target|success] vs each feature:
      - single colour & legend across variables
      - ±2σ band, mean curve
      - shading where P(success)<0.8 (darker) and <0.5 (lighter)
      - experimental points: successes (black); failures (red w/ black outline, drawn above)
    """
    ds = xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std = ds["feature_std"].values.astype(float)

    df_raw = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    num_features = Xn_train.shape[1]

    # Base standardized point (median), then apply 'fixed'
    base_std = np.median(Xn_train, axis=0)
    if fixed:
        base_std = _apply_fixed_to_base(base_std, fixed, feature_names, transforms, X_mean, X_std)

    # Build standardized sweeps per feature
    sweeps_std = [
        np.linspace(*np.percentile(Xn_train[:, j], [1, 99]), n_points_1d)
        for j in range(num_features)
    ]

    # Masks and data for experimental points
    tgt_col = str(ds.attrs["target_column"])
    success_mask = ~pd.isna(df_raw[tgt_col]).to_numpy()
    fail_mask = ~success_mask
    losses_success = df_raw.loc[success_mask, tgt_col].to_numpy().astype(float)
    trial_ids_success = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[success_mask]
    trial_ids_fail = df_raw.get("trial_id", pd.Series(np.arange(len(df_raw)))).to_numpy()[fail_mask]

    # Helpers for colour fill
    band_fill_rgba = _rgb_to_rgba(line_color, band_alpha)

    # Figure
    fig = make_subplots(
        rows=num_features, cols=1, shared_xaxes=False, vertical_spacing=0.08,
        subplot_titles=[f"E[target|success] — {name}" for name in feature_names]
    )

    tidy_rows: list[dict] = []

    for j in range(num_features):
        grid = sweeps_std[j]
        Xn_grid = np.repeat(base_std[None, :], len(grid), axis=0)
        Xn_grid[:, j] = grid

        p_grid = pred_success(Xn_grid)
        mu_grid, sd_grid = pred_loss(Xn_grid, include_observation_noise=True)

        x_internal = grid * X_std[j] + X_mean[j]
        x_display = _inverse_transform(transforms[j], x_internal)

        # Target transforms for plotting
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

        # Y-range pad; include observed successes
        y_arrays = [lo_plot, hi_plot] + ([losses_s_plot] if losses_s_plot.size else [])
        y_low  = float(np.nanmin([np.nanmin(arr) for arr in y_arrays]))
        y_high = float(np.nanmax([np.nanmax(arr) for arr in y_arrays]))

        pad = 0.05 * (y_high - y_low + 1e-12)
        y0_plot = (y_low - pad) if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)
        y1_tmp = (y_high + pad) if not use_log_scale_for_target_y else y_high * 1.2
        y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
        if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
            y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
        y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

        # Shading (below the PD)
        _add_low_success_shading_1d(fig, j + 1, x_display, p_grid, y0_plot, y1_plot)

        # ±2σ band (legend shown only once)
        show_legend = (j == 0)
        fig.add_trace(go.Scatter(
            x=x_display, y=lo_plot, mode="lines",
            line=dict(width=0, color=line_color),
            name="±2σ", legendgroup="band", showlegend=False, hoverinfo="skip"
        ), row=j + 1, col=1)
        fig.add_trace(go.Scatter(
            x=x_display, y=hi_plot, mode="lines", fill="tonexty",
            line=dict(width=0, color=line_color), fillcolor=band_fill_rgba,
            name="±2σ", legendgroup="band", showlegend=show_legend,
            hovertemplate="E[target|success]: %{y:.3f}<extra>±2σ</extra>"
        ), row=j + 1, col=1)

        # Mean curve
        fig.add_trace(go.Scatter(
            x=x_display, y=mu_plot, mode="lines",
            line=dict(width=2, color=line_color),
            name="E[target|success]", legendgroup="mean", showlegend=show_legend,
            hovertemplate=f"{feature_names[j]}: %{{x:.6g}}<br>E[target|success]: %{{y:.3f}}<extra></extra>"
        ), row=j + 1, col=1)

        # Experimental successes
        name_j = feature_names[j]
        if name_j in df_raw.columns:
            x_data_all = df_raw[name_j].to_numpy().astype(float)
        else:
            x_data_all = _feature_raw_from_artifact_or_reconstruct(ds, j, name_j, transforms[j]).astype(float)

        x_succ = x_data_all[success_mask]
        if x_succ.size:
            fig.add_trace(go.Scattergl(
                x=x_succ, y=losses_s_plot,
                mode="markers",
                marker=dict(size=5, color="black", line=dict(width=0)),
                name="data (success)", legendgroup="data_s", showlegend=show_legend,
                hovertemplate=(
                    "trial_id: %{customdata}<br>"
                    f"{feature_names[j]}: %{{x:.6g}}<br>"
                    f"{tgt_col}: %{{y:.4f}}<extra></extra>"
                ),
                customdata=trial_ids_success
            ), row=j + 1, col=1)

        # Experimental failures (hover shows failure status)
        x_fail = x_data_all[fail_mask]
        if x_fail.size:
            y_fail_plot = np.full_like(x_fail, y_failed_band, dtype=float)
            fig.add_trace(go.Scattergl(
                x=x_fail, y=y_fail_plot, mode="markers",
                marker=dict(size=6, color="red", line=dict(color="black", width=0.8)),
                name="data (failed)", legendgroup="data_f", showlegend=show_legend,
                hovertemplate=(
                    "trial_id: %{customdata}<br>"
                    f"{feature_names[j]}: %{{x:.6g}}<br>"
                    "status: failed (NaN target)<extra></extra>"
                ),
                customdata=trial_ids_fail
            ), row=j + 1, col=1)

        # Axes
        _maybe_log_axis(fig, j + 1, 1, feature_names[j], axis="x", transforms=transforms, j=j)
        fig.update_yaxes(title_text=f"{tgt_col} (E[target|success])", row=j + 1, col=1)
        _set_yaxis_range(
            fig, row=j + 1, col=1,
            y0=y0_plot, y1=y1_plot,
            log=use_log_scale_for_target_y, eps=log_y_epsilon
        )
        fig.update_xaxes(title_text=feature_names[j], row=j + 1, col=1)

        # tidy lines
        for xd, xi, mu_i, sd_i, p_i in zip(x_display, x_internal, mu_grid, sd_grid, p_grid):
            tidy_rows.append({
                "feature": feature_names[j],
                "x_display": float(xd),
                "x_internal": float(xi),
                "target_conditional_mean": float(mu_i),
                "target_conditional_sd": float(sd_i),
                "success_probability": float(p_i),
            })

    fig.update_layout(
        height=figure_height_per_row_px * num_features,
        template="simple_white",
        title="1D Partial Dependence: E[target|success] with experimental points & low-success shading",
        legend_title_text=""
    )

    fig.write_html(str(output), include_plotlyjs="cdn")
    pd.DataFrame(tidy_rows).to_csv(str(csv_out), index=False)
    if show_figure:
        fig.show("browser")


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
    K_s = _kernel_m52_ard(Xn_all, Xn_all, ell_s, eta_s) + (sigma_s**2) * np.eye(Xn_all.shape[0])
    L_s = np.linalg.cholesky(_add_jitter(K_s))
    alpha_s = _solve_chol(L_s, (y_success - beta0_s))

    K_l = _kernel_m52_ard(Xn_ok, Xn_ok, ell_l, eta_l) + (sigma_l**2) * np.eye(Xn_ok.shape[0])
    L_l = np.linalg.cholesky(_add_jitter(K_l))
    alpha_l = _solve_chol(L_l, (y_loss_centered - mean_c))

    def predict_success_probability(Xn: np.ndarray) -> np.ndarray:
        Ks = _kernel_m52_ard(Xn, Xn_all, ell_s, eta_s)
        mu = beta0_s + Ks @ alpha_s
        return np.clip(mu, 0.0, 1.0)

    def predict_conditional_target(Xn: np.ndarray, include_observation_noise: bool = True):
        Kl = _kernel_m52_ard(Xn, Xn_ok, ell_l, eta_l)
        mu_centered = mean_c + Kl @ alpha_l
        mu = mu_centered + cond_mean
        v = _solve_lower(L_l, Kl.T)
        var = _kernel_diag_m52(Xn, ell_l, eta_l) - np.sum(v * v, axis=0)
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


# =============================================================================
# GP kernel & linear algebra (same as in opt.py)
# =============================================================================

def _kernel_m52_ard(XA: np.ndarray, XB: np.ndarray, ls: np.ndarray, eta: float) -> np.ndarray:
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
    jitter = eps * float(np.mean(np.diag(K)) + 1.0)
    return K + jitter * np.eye(K.shape[0], dtype=K.dtype)


def _solve_chol(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L, B)


def _feature_raw_from_artifact_or_reconstruct(
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


def _set_yaxis_range(fig, *, row: int, col: int, y0: float, y1: float, log: bool, eps: float = 1e-12):
    """Update a subplot's Y axis to [y0, y1]. For log axes, the range is given in log10 units."""
    if log:
        y0 = max(y0, eps)
        y1 = max(y1, y0 * (1.0 + 1e-6))
        fig.update_yaxes(type="log", range=[np.log10(y0), np.log10(y1)], row=row, col=col)
    else:
        fig.update_yaxes(type="-", range=[y0, y1], row=row, col=col)
