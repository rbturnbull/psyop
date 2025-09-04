import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def percentiles_1_99(array_1d: np.ndarray):
    return np.percentile(array_1d, [1, 99])

def denormalize_feature_value(feature_index: int, value_in_standardized_space: np.ndarray) -> np.ndarray:
    """Map standardized feature back to the model's original feature space (log10 for lr)."""
    return value_in_standardized_space * feature_stds_array[feature_index] + feature_means_array[feature_index]

def contiguous_spans(x_values_sorted: np.ndarray, boolean_mask: np.ndarray):
    """Return [(x_start, x_end), ...] for contiguous True segments in boolean_mask over sorted x_values."""
    spans = []
    if len(x_values_sorted) == 0:
        return spans
    mask_int = boolean_mask.astype(int)
    padded_diff = np.diff(np.concatenate([[0], mask_int, [0]]))
    starts = np.where(padded_diff == 1)[0]
    ends = np.where(padded_diff == -1)[0] - 1
    for s, e in zip(starts, ends):
        spans.append((x_values_sorted[s], x_values_sorted[e]))
    return spans

def add_low_success_shading(
    figure: go.Figure, subplot_row_index: int,
    x_axis_values_display: np.ndarray, success_probabilities: np.ndarray,
    y_min: float, y_max: float
):
    """
    Add shading rectangles:
      - Light grey where P(success) < 0.5
      - Darker grey where P(success) < 0.8
    Note: The <0.8 shading draws on top (so all p<0.8 regions appear darker).
    """
    xref = "x" if subplot_row_index == 1 else f"x{subplot_row_index}"
    yref = "y" if subplot_row_index == 1 else f"y{subplot_row_index}"

    spans_lt_05 = contiguous_spans(x_axis_values_display, success_probabilities < 0.5)
    for x0, x1 in spans_lt_05:
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y_min, y1=y_max,
            xref=xref, yref=yref, line=dict(width=0),
            fillcolor="rgba(128,128,128,0.25)", layer="below"
        )

    spans_lt_08 = contiguous_spans(x_axis_values_display, success_probabilities < 0.8)
    for x0, x1 in spans_lt_08:
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y_min, y1=y_max,
            xref=xref, yref=yref, line=dict(width=0),
            fillcolor="rgba(128,128,128,0.40)", layer="below"
        )


def make_1d_partial_dependence_plots(
    feature_matrix_standardized: np.ndarray,
    feature_names_list: list[str],
    predict_conditional_loss,           # fn: Xn -> (mu, sd)
    predict_success_probability,        # fn: Xn -> p
    feature_stds_array: np.ndarray,
    feature_means_array: np.ndarray,
    dataframe_input: pd.DataFrame,      # <-- needed for experimental points
    outfile_html: str = "pd_loss_conditional.html",
    outfile_csv: str = "gp_partial_dependence_loss_only.csv",
    n_points_1d: int = 300,
    line_color: str = "rgb(31,119,180)",     # same color for all variables
    band_alpha: float = 0.25,                # fill alpha for ±2σ
    figure_height_per_row_px: int = 320,
    show_figure: bool = True,
    use_log_scale_for_target_y: bool = False,
    log_y_epsilon: float = 1e-9,             # clamp for log-y safety
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Vertical 1D PD plots (one per feature) of E[loss|success] vs feature value,
    with low-success shading and experimental points overlaid.
    - successful trials: black dots at (x=value, y=loss)
    - failed trials (NaN loss): red dots with black outline at a thin band above the plot
    - optional log scale on Y (target/loss)
    """

    num_features = feature_matrix_standardized.shape[1]

    # ---- experimental data masks and columns ----
    success_mask = ~dataframe_input["loss"].isna().to_numpy()
    fail_mask = ~success_mask
    loss_success = dataframe_input.loc[success_mask, "loss"].to_numpy().astype(float)
    trial_ids_success = dataframe_input.loc[success_mask, "trial_id"].to_numpy()
    trial_ids_fail = dataframe_input.loc[fail_mask, "trial_id"].to_numpy()

    def data_vals_for_feature(j: int) -> np.ndarray:
        name = feature_names_list[j]
        if name not in dataframe_input.columns:
            raise KeyError(f"Column '{name}' not found in dataframe_input.")
        vals = dataframe_input[name].to_numpy().astype(float)
        # learning_rate already in original units; others are raw
        return vals

    # ---- color helpers ----
    def rgb_to_rgba(rgb: str, alpha: float) -> str:
        m = re.match(r"\s*rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", rgb)
        if not m:
            return f"rgba(31,119,180,{alpha:.3f})"
        r, g, b = map(float, m.groups())
        return f"rgba({int(r)},{int(g)},{int(b)},{alpha:.3f})"

    band_fill_color = rgb_to_rgba(line_color, band_alpha)

    # ---- base point & sweeps ----
    base_point_standardized = np.median(feature_matrix_standardized, axis=0)
    standardized_sweep_values_per_feature = [
        np.linspace(*percentiles_1_99(feature_matrix_standardized[:, j]), n_points_1d)
        for j in range(num_features)
    ]

    # ---- figure ----
    fig = make_subplots(
        rows=num_features, cols=1, shared_xaxes=False, vertical_spacing=0.08,
        subplot_titles=[f"E[loss|success] — {name}" for name in feature_names_list]
    )

    tidy_rows = []

    for j in range(num_features):
        sweep_std = standardized_sweep_values_per_feature[j]

        # Design matrix: vary one feature at a time
        Xn_grid = np.repeat(base_point_standardized[None, :], len(sweep_std), axis=0)
        Xn_grid[:, j] = sweep_std

        # Predictions
        p_grid = predict_success_probability(Xn_grid)
        mu_grid, sd_grid = predict_conditional_loss(Xn_grid)

        # X display values (original units; log axis for learning_rate on X)
        x_internal = denormalize_feature_value(j, sweep_std)
        if j == 0:  # learning_rate
            x_display = 10.0 ** x_internal
            x_title = "learning_rate"
            log_x = True
        else:
            x_display = x_internal
            x_title = feature_names_list[j]
            log_x = False

        # Clamp for log-y if requested
        if use_log_scale_for_target_y:
            mu_plot = np.maximum(mu_grid, log_y_epsilon)
            lo_plot = np.maximum(mu_grid - 2.0 * sd_grid, log_y_epsilon)
            hi_plot = np.maximum(mu_grid + 2.0 * sd_grid, log_y_epsilon)
            # Successful observed losses must also be >0
            loss_success_plot = np.maximum(loss_success, log_y_epsilon) if loss_success.size else loss_success
        else:
            mu_plot = mu_grid
            lo_plot = mu_grid - 2.0 * sd_grid
            hi_plot = mu_grid + 2.0 * sd_grid
            loss_success_plot = loss_success

        # Y-range from PD bands and observed successful losses
        y_parts = [lo_plot, hi_plot]
        if loss_success_plot.size:
            y_parts.append(loss_success_plot)
        y_low = float(np.min([np.min(arr) for arr in y_parts]))
        y_high = float(np.max([np.max(arr) for arr in y_parts]))
        pad = 0.05 * (y_high - y_low + 1e-12)
        y0_plot = y_low - pad if not use_log_scale_for_target_y else max(y_low / 1.5, log_y_epsilon)

        # Temporary high bound to compute a band for failed points
        y1_tmp = y_high + pad if not use_log_scale_for_target_y else y_high * 1.2
        # Failed points band just above the main range
        y_failed_band = y1_tmp + (y_high - y_low + 1e-12) * (0.08 if not use_log_scale_for_target_y else 0.3)
        if use_log_scale_for_target_y and y_failed_band <= log_y_epsilon:
            y_failed_band = max(10.0 * log_y_epsilon, y_high * 2.0)
        y1_plot = y_failed_band + (0.02 if not use_log_scale_for_target_y else 0.05) * (y_high - y_low + 1e-12)

        # --- low-success shading ---
        add_low_success_shading(
            fig, j + 1,
            x_axis_values_display=x_display,
            success_probabilities=p_grid,
            y_min=y0_plot, y_max=y1_plot
        )

        # --- ±2σ band (one legend entry total) ---
        show_leg = (j == 0)
        fig.add_trace(go.Scatter(
            x=x_display, y=lo_plot,
            mode="lines", line=dict(width=0, color=line_color),
            name="±2σ", legendgroup="pd1d_band", showlegend=False, hoverinfo="skip"
        ), row=j + 1, col=1)
        fig.add_trace(go.Scatter(
            x=x_display, y=hi_plot,
            mode="lines", fill="tonexty",
            line=dict(width=0, color=line_color),
            fillcolor=band_fill_color,
            name="±2σ", legendgroup="pd1d_band", showlegend=show_leg,
            hovertemplate="E[loss|success]: %{y:.3f}<extra>±2σ</extra>"
        ), row=j + 1, col=1)

        # --- mean curve (one legend entry total) ---
        fig.add_trace(go.Scatter(
            x=x_display, y=mu_plot, mode="lines",
            line=dict(width=2, color=line_color),
            name="E[loss|success]", legendgroup="pd1d_mean", showlegend=show_leg,
            hovertemplate=f"{x_title}: %{{x:.6g}}<br>E[loss|success]: %{{y:.3f}}<extra></extra>"
        ), row=j + 1, col=1)

        # --- experimental points: success (black) ---
        x_data_all = data_vals_for_feature(j)
        x_data_success = x_data_all[success_mask]
        if x_data_success.size:
            fig.add_trace(go.Scattergl(
                x=x_data_success, y=loss_success_plot,
                mode="markers",
                marker=dict(size=5, color="black", line=dict(width=0)),
                name="data (success)", legendgroup="data_success",
                showlegend=show_leg,
                hovertemplate=(
                    f"trial_id: %{{customdata[0]}}<br>"
                    f"{x_title}: %{{x:.6g}}<br>"
                    "loss: %{y:.4f}<extra></extra>"
                ),
                customdata=np.column_stack([trial_ids_success])
            ), row=j + 1, col=1)

        # --- experimental points: failed (red with black outline) ---
        x_data_fail = x_data_all[fail_mask]
        if x_data_fail.size:
            y_fail_plot = np.full_like(x_data_fail, y_failed_band, dtype=float)
            fig.add_trace(go.Scattergl(
                x=x_data_fail, y=y_fail_plot,
                mode="markers",
                marker=dict(size=6, color="red", line=dict(color="black", width=0.8)),
                name="data (failed)", legendgroup="data_failed",
                showlegend=show_leg,
                hovertemplate=(
                    f"trial_id: %{{customdata}}<br>"
                    f"{x_title}: %{{x:.6g}}<br>"
                    "status: failed (NaN loss)<extra></extra>"
                ),
                customdata=trial_ids_fail
            ), row=j + 1, col=1)

        # Axes
        if log_x:
            fig.update_xaxes(type="log", row=j + 1, col=1)
        fig.update_yaxes(
            title_text="loss (E[loss|success])",
            type="log" if use_log_scale_for_target_y else "-",
            range=None,  # let Plotly compute in log/linear, we already padded
            row=j + 1, col=1
        )
        fig.update_yaxes(range=[y0_plot, y1_plot], row=j + 1, col=1)
        fig.update_xaxes(title_text=x_title, row=j + 1, col=1)

        # tidy rows
        for xd, xi, mu_i, sd_i, p_i in zip(x_display, x_internal, mu_grid, sd_grid, p_grid):
            tidy_rows.append({
                "feature": x_title,
                "x_display": float(xd),
                "x_internal": float(xi),
                "loss_conditional_mean": float(mu_i),
                "loss_conditional_sd": float(sd_i),
                "success_probability": float(p_i),
            })

    # Layout & export
    fig.update_layout(
        height=figure_height_per_row_px * num_features,
        template="simple_white",
        title="1D Partial Dependence: E[loss | success] with experimental points & low-success shading",
        legend_title_text=""
    )

    fig.write_html(outfile_html, include_plotlyjs="cdn")
    pd.DataFrame(tidy_rows).to_csv(outfile_csv, index=False)
    print(f"Wrote {Path(outfile_html).name}")
    print(f"Wrote {Path(outfile_csv).name}")

    if show_figure:
        fig.show()

    return fig, pd.DataFrame(tidy_rows)



def make_pairplot(
    feature_matrix_standardized: np.ndarray,
    feature_means_array: np.ndarray,
    feature_stds_array: np.ndarray,
    feature_names_list: list[str],
    predict_loss_given_success,          # fn: X -> (mu, sd)
    predict_success_probability,         # fn: X -> p
    dataframe_input: pd.DataFrame,       # to plot experimental points
    outfile: str = "pairplot_pd.html",
    n_points_1d: int = 300,
    n_points_2d: int = 70,
    use_log_scale_for_target: bool = False,
    log_shift_epsilon: float = 1e-9,
    colourscale: str = "RdBu",
    show_figure: bool = False,
    n_contours: int = 12,
):
    """
    Pairplot of E[loss|success]:
      - diagonal: symmetric 2D heatmap built from 1D PD (Z = 0.5*(mu(x_i)+mu(x_j)))
      - off-diagonal: true 2D heatmaps
      - probability shading: grey for P(success)<0.8 (darker) and <0.5 (lighter)
      - experimental points: success=black dots; failed=red dots with black outline
      - contours: one trace per level with line grey = 1 - luminance(colour_at_level)
    Colour mapping can be linear or log10 (shifted if needed).
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import get_colorscale, sample_colorscale

    num_features: int = feature_matrix_standardized.shape[1]

    # ---------- helpers ----------
    def _p01_p99(arr: np.ndarray) -> tuple[float, float]:
        return np.percentile(arr, [1, 99])

    def denormalize_feature(j: int, xnorm: np.ndarray) -> np.ndarray:
        return xnorm * feature_stds_array[j] + feature_means_array[j]

    def maybe_set_log_x(fig_obj, row_idx: int, col_idx: int, feature_index: int):
        if feature_names_list[feature_index] == "learning_rate":
            fig_obj.update_xaxes(type="log", row=row_idx, col=col_idx)

    def maybe_set_log_y(fig_obj, row_idx: int, col_idx: int, feature_index: int):
        if feature_names_list[feature_index] == "learning_rate":
            fig_obj.update_yaxes(type="log", row=row_idx, col=col_idx)

    def rgb_string_to_tuple(s: str) -> tuple[int, int, int]:
        # expects "rgb(r,g,b)" or "rgba(r,g,b,a)"
        vals = s[s.find("(")+1:s.find(")")].split(",")
        r, g, b = [int(float(v)) for v in vals[:3]]
        return r, g, b

    def luminance_from_colorstring(s: str) -> float:
        r, g, b = rgb_string_to_tuple(s)
        # sRGB luma
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    # Given a raw level value, find line grey by inverting colourscale luminance at that level
    cs = get_colorscale(colourscale)

    def contour_line_color_for_level(level_raw: float, cmin_t: float, cmax_t: float) -> str:
        # map raw level -> transformed (for colour) -> normalize to [0,1] -> sample colourscale
        zt = level_raw
        if use_log_scale_for_target:
            # apply same transform used for heatmap
            shift = 0.0
            # cmin_t/cmax_t are already on transformed scale, so we only need the transformed level
            # we don't know the global shift used, but transform_for_colour below returns it when
            # called on arrays; for scalars we re-run it locally for t.
            # We'll inline a tiny version here:
            # compute shift so that level is > 0 if needed (approx; very rare if data>0)
            # but since normalization uses transformed bounds, we'd better compute
            # transformed value via the same function outside. We approximate here:
            # ensure positive
            if level_raw <= 0:
                shift = -level_raw + float(log_shift_epsilon)
            zt = np.log10(max(level_raw + shift, log_shift_epsilon))
        # normalize transformed level
        t = 0.5 if cmax_t == cmin_t else (zt - cmin_t) / (cmax_t - cmin_t)
        t = float(np.clip(t, 0.0, 1.0))
        rgb = sample_colorscale(cs, [t])[0]          # "rgb(r,g,b)"
        lum = luminance_from_colorstring(rgb)        # 0..1
        inv = 1.0 - lum
        g = int(round(inv * 255))
        return f"rgba({g},{g},{g},0.85)"

    # ---------- data points ----------
    success_mask = ~dataframe_input["loss"].isna().to_numpy()
    fail_mask = ~success_mask

    data_feature_columns = {}
    for name in feature_names_list:
        if name not in dataframe_input.columns:
            raise KeyError(f"Column '{name}' not found in dataframe_input.")
        col = dataframe_input[name].to_numpy().astype(float)
        data_feature_columns[name] = col  # learning_rate is already real LR

    def data_vals_for_feature(index: int) -> np.ndarray:
        return data_feature_columns[feature_names_list[index]]

    # ---------- base point & ranges ----------
    feature_medians_normalized: np.ndarray = np.median(feature_matrix_standardized, axis=0)
    one_d_ranges_normalized = [
        np.linspace(*_p01_p99(feature_matrix_standardized[:, j]), n_points_1d)
        for j in range(num_features)
    ]
    two_d_ranges_normalized = [
        np.linspace(*_p01_p99(feature_matrix_standardized[:, j]), n_points_2d)
        for j in range(num_features)
    ]

    # ---------- diagonal (1D -> symmetric 2D) ----------
    diagonal_payload: dict[int, dict] = {}
    for j in range(num_features):
        grid_norm = one_d_ranges_normalized[j]
        X_grid = np.repeat(feature_medians_normalized[None, :], len(grid_norm), axis=0)
        X_grid[:, j] = grid_norm

        p_1d = predict_success_probability(X_grid)
        mu_1d, _ = predict_loss_given_success(X_grid)

        x_original = denormalize_feature(j, grid_norm)
        x_display = 10 ** x_original if feature_names_list[j] == "learning_rate" else x_original

        Z_mu_diag = 0.5 * (mu_1d[:, None] + mu_1d[None, :])
        Z_p_diag = np.minimum(p_1d[:, None], p_1d[None, :])

        diagonal_payload[j] = {"x_display": x_display, "Z_mu": Z_mu_diag, "Z_p": Z_p_diag}

    # ---------- off-diagonals (true 2D) ----------
    offdiag_payload: dict[tuple[int, int], dict] = {}
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                continue
            xg = two_d_ranges_normalized[j]
            yg = two_d_ranges_normalized[i]
            XX, YY = np.meshgrid(xg, yg)

            X_grid = np.repeat(feature_medians_normalized[None, :], XX.size, axis=0)
            X_grid[:, j] = XX.ravel()
            X_grid[:, i] = YY.ravel()

            mu_flat, _ = predict_loss_given_success(X_grid)
            p_flat = predict_success_probability(X_grid)

            Z_mu = mu_flat.reshape(YY.shape)
            Z_p = p_flat.reshape(YY.shape)

            x_original = denormalize_feature(j, xg)
            y_original = denormalize_feature(i, yg)
            x_display = 10 ** x_original if feature_names_list[j] == "learning_rate" else x_original
            y_display = 10 ** y_original if feature_names_list[i] == "learning_rate" else y_original

            offdiag_payload[(i, j)] = {"x_display": x_display, "y_display": y_display, "Z_mu": Z_mu, "Z_p": Z_p}

    # ---------- colour transform & bounds ----------
    def transform_for_colour(z_values: np.ndarray) -> tuple[np.ndarray, float, str]:
        if not use_log_scale_for_target:
            return z_values, 0.0, ""
        z_min = float(np.nanmin(z_values))
        shift = 0.0
        if z_min <= 0:
            shift = -(z_min) + float(log_shift_epsilon)
        z_shifted = np.maximum(z_values + shift, log_shift_epsilon)
        z_log = np.log10(z_shifted)
        suffix = " (log10)" if shift == 0.0 else f" (log10, shifted by Δ={shift:.3g})"
        return z_log, shift, suffix

    all_mu_blocks = [d["Z_mu"].ravel() for d in diagonal_payload.values()] + \
                    [d["Z_mu"].ravel() for d in offdiag_payload.values()]
    if len(all_mu_blocks):
        all_mu_concat = np.concatenate(all_mu_blocks)
        Z_all_for_colour, _global_shift, colorbar_suffix = transform_for_colour(all_mu_concat)
        cmin_t = float(np.nanmin(Z_all_for_colour))
        cmax_t = float(np.nanmax(Z_all_for_colour))
    else:
        cmin_t, cmax_t, colorbar_suffix = 0.0, 1.0, ""

    # ---------- figure ----------
    fig = make_subplots(
        rows=num_features, cols=num_features,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.06, vertical_spacing=0.06
    )

    # ---- off-diagonals: heatmap, shading, contours, data ----
    for (row_feat_idx, col_feat_idx), payload in offdiag_payload.items():
        row_idx, col_idx = row_feat_idx + 1, col_feat_idx + 1
        x_vals, y_vals = payload["x_display"], payload["y_display"]
        Z_mu_raw, Z_p = payload["Z_mu"], payload["Z_p"]
        Z_for_colour, _, _ = transform_for_colour(Z_mu_raw)

        # heatmap
        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=Z_for_colour,
            coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names_list[col_feat_idx]}: %{{x:.6g}}<br>"
                f"{feature_names_list[row_feat_idx]}: %{{y:.6g}}"
                "<br>E[loss|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Z_mu_raw
        ), row=row_idx, col=col_idx)

        # shading overlays (draw BEFORE contours so contours stay visible)
        mask_lt_05 = np.where(Z_p < 0.5, 1.0, np.nan)
        mask_lt_08 = np.where(Z_p < 0.8, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=mask_lt_05, zmin=0, zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(128,128,128,0.25)"]],
            showscale=False, hoverinfo="skip"
        ), row=row_idx, col=col_idx)
        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=mask_lt_08, zmin=0, zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(128,128,128,0.40)"]],
            showscale=False, hoverinfo="skip"
        ), row=row_idx, col=col_idx)

        # contours: one trace per level, greyscale inverted from colourscale luminance
        zmin_raw = float(np.nanmin(Z_mu_raw))
        zmax_raw = float(np.nanmax(Z_mu_raw))
        levels = np.linspace(zmin_raw, zmax_raw, max(n_contours, 2))
        for lev in levels:
            line_color = contour_line_color_for_level(lev, cmin_t, cmax_t)
            fig.add_trace(go.Contour(
                x=x_vals, y=y_vals, z=Z_mu_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=line_color),
                showscale=False, hoverinfo="skip"
            ), row=row_idx, col=col_idx)

        # experimental points on top
        x_data = data_vals_for_feature(col_feat_idx)
        y_data = data_vals_for_feature(row_feat_idx)
        fig.add_trace(go.Scattergl(
            x=x_data[success_mask], y=y_data[success_mask],
            mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_success",
            showlegend=(row_idx == 1 and col_idx == 1),
            hovertemplate=(
                f"trial_id: %{{customdata[0]}}<br>"
                f"{feature_names_list[col_feat_idx]}: %{{x:.6g}}<br>"
                f"{feature_names_list[row_feat_idx]}: %{{y:.6g}}<br>"
                "loss: %{customdata[1]:.4f}<extra></extra>"
            ),
            customdata=np.column_stack([
                dataframe_input["trial_id"].to_numpy()[success_mask],
                dataframe_input["loss"].to_numpy()[success_mask],
            ])
        ), row=row_idx, col=col_idx)
        fig.add_trace(go.Scattergl(
            x=x_data[fail_mask], y=y_data[fail_mask],
            mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_failed",
            showlegend=(row_idx == 1 and col_idx == 1),
            hovertemplate=(
                f"trial_id: %{{customdata}}<br>"
                f"{feature_names_list[col_feat_idx]}: %{{x:.6g}}<br>"
                f"{feature_names_list[row_feat_idx]}: %{{y:.6g}}<br>"
                "status: failed (NaN loss)<extra></extra>"
            ),
            customdata=dataframe_input["trial_id"].to_numpy()[fail_mask]
        ), row=row_idx, col=col_idx)

        maybe_set_log_x(fig, row_idx, col_idx, col_feat_idx)
        maybe_set_log_y(fig, row_idx, col_idx, row_feat_idx)

    # ---- diagonal: symmetric heatmap, shading, contours, points ----
    for j, payload in diagonal_payload.items():
        row_idx = col_idx = j + 1
        axis_x = payload["x_display"]
        axis_y = payload["x_display"]
        Z_mu_diag_raw = payload["Z_mu"]
        Z_p_diag = payload["Z_p"]
        Z_for_colour_diag, _, _ = transform_for_colour(Z_mu_diag_raw)

        fig.add_trace(go.Heatmap(
            x=axis_x, y=axis_y, z=Z_for_colour_diag,
            coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(
                f"{feature_names_list[j]}: %{{x:.6g}} / %{{y:.6g}}"
                "<br>E[loss|success]: %{customdata:.3f}<extra></extra>"
            ),
            customdata=Z_mu_diag_raw
        ), row=row_idx, col=col_idx)

        # shading first
        mask_lt_05 = np.where(Z_p_diag < 0.5, 1.0, np.nan)
        mask_lt_08 = np.where(Z_p_diag < 0.8, 1.0, np.nan)
        fig.add_trace(go.Heatmap(
            x=axis_x, y=axis_y, z=mask_lt_05, zmin=0, zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(128,128,128,0.25)"]],
            showscale=False, hoverinfo="skip"
        ), row=row_idx, col=col_idx)
        fig.add_trace(go.Heatmap(
            x=axis_x, y=axis_y, z=mask_lt_08, zmin=0, zmax=1,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(128,128,128,0.40)"]],
            showscale=False, hoverinfo="skip"
        ), row=row_idx, col=col_idx)

        # contours per level
        zmin_raw = float(np.nanmin(Z_mu_diag_raw))
        zmax_raw = float(np.nanmax(Z_mu_diag_raw))
        levels = np.linspace(zmin_raw, zmax_raw, max(n_contours, 2))
        for lev in levels:
            line_color = contour_line_color_for_level(lev, cmin_t, cmax_t)
            fig.add_trace(go.Contour(
                x=axis_x, y=axis_y, z=Z_mu_diag_raw,
                autocontour=False,
                contours=dict(coloring="lines", showlabels=False, start=lev, end=lev, size=1e-9),
                line=dict(width=1, color=line_color),
                showscale=False, hoverinfo="skip"
            ), row=row_idx, col=col_idx)

        # points (x=y)
        x_data = data_vals_for_feature(j); y_data = x_data
        fig.add_trace(go.Scattergl(
            x=x_data[success_mask], y=y_data[success_mask],
            mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_success",
            showlegend=False,
            hovertemplate=(
                f"trial_id: %{{customdata[0]}}<br>"
                f"{feature_names_list[j]}: %{{x:.6g}}<br>"
                "loss: %{customdata[1]:.4f}<extra></extra>"
            ),
            customdata=np.column_stack([
                dataframe_input["trial_id"].to_numpy()[success_mask],
                dataframe_input["loss"].to_numpy()[success_mask],
            ])
        ), row=row_idx, col=col_idx)
        fig.add_trace(go.Scattergl(
            x=x_data[fail_mask], y=y_data[fail_mask],
            mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_failed",
            showlegend=False,
            hovertemplate=(
                f"trial_id: %{{customdata}}<br>"
                f"{feature_names_list[j]}: %{{x:.6g}}<br>"
                "status: failed (NaN loss)<extra></extra>"
            ),
            customdata=dataframe_input["trial_id"].to_numpy()[fail_mask]
        ), row=row_idx, col=col_idx)

        if feature_names_list[j] == "learning_rate":
            fig.update_xaxes(type="log", row=row_idx, col=col_idx)
            fig.update_yaxes(type="log", row=row_idx, col=col_idx)

    # axis labels
    for j in range(num_features):
        fig.update_xaxes(title_text=feature_names_list[j], row=num_features, col=j + 1)
    for i in range(num_features):
        fig.update_yaxes(title_text=feature_names_list[i], row=i + 1, col=1)

    # layout
    cell_px = 250
    colorbar_title = f"E[loss|success]{colorbar_suffix}"
    fig.update_layout(
        coloraxis=dict(colorscale=colourscale, cmin=cmin_t, cmax=cmax_t,
                       colorbar=dict(title=colorbar_title)),
        template="simple_white",
        width=cell_px * num_features,
        height=cell_px * num_features,
        title="Pairplot — E[loss|success] with inverted-luminance contours & data points",
        legend_title_text=""
    )

    fig.write_html(outfile, include_plotlyjs="cdn")
    if show_figure:
        fig.show()
    print(f"Wrote {outfile}")
