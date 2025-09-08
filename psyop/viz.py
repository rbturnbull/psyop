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
from . import opt


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
    width:int|None = None,
    height:int|None = None,
    **kwargs,
) -> go.Figure:
    """
    2D Partial Dependence of E[target|success] (pairwise *numeric* features),
    optionally conditioned on:
      - numeric constraints via kwargs (fixed / slice / list/tuple)
      - categorical base constraints via kwargs, e.g. language="Linear A"

    Categorical constraints are enforced by:
      (a) filtering training rows to the chosen label(s) for medians/percentiles
      (b) fixing the corresponding one-hot features in standardized space
      (c) excluding one-hot member features from plotted axes
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    # --- features & transforms
    feature_names = [str(x) for x in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std  = ds["feature_std"].values.astype(float)

    # One-hot categorical groups (reuse opt helper)
    groups = opt._onehot_groups(feature_names)  # { base: {"labels":[...], "name_by_label":{label->member}, "members":[...]} }
    bases  = set(groups.keys())
    name_to_idx = {name: j for j, name in enumerate(feature_names)}

    # Raw DF (for overlay) and standardized training X (for medians/percentiles)
    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    n_rows   = Xn_train.shape[0]

    # --- split kwargs into numeric vs categorical (keys are already canonical from main.py)
    kw_num: dict[str, object] = {}
    kw_cat: dict[str, object] = {}
    for k, v in (kwargs or {}).items():
        if k in bases:
            kw_cat[k] = v
        elif k in name_to_idx:
            kw_num[k] = v
        else:
            # unknown key; silently ignore (main.py already warned if needed)
            pass

    # --- resolve categorical constraints to a fixed single label per base (warn if multiple; pick first)
    cat_fixed: dict[str, str] = {}
    for base, val in kw_cat.items():
        labels = groups[base]["labels"]
        if isinstance(val, str):
            if val not in labels:
                raise ValueError(f"Unknown category for {base!r}: {val!r}. Choices: {labels}")
            cat_fixed[base] = val
        elif isinstance(val, (list, tuple, set)):
            chosen = [x for x in val if isinstance(x, str) and x in labels]
            if not chosen:
                raise ValueError(f"No valid categories for {base!r} in {val!r}. Choices: {labels}")
            if len(chosen) > 1:
                console.print(f"[yellow]Warning: multiple categories for {base!r} supplied; using {chosen[0]!r}[/yellow]")
            cat_fixed[base] = chosen[0]
        else:
            raise ValueError(f"Categorical constraint for {base!r} must be a string or list/tuple of strings.")

    # --- filter rows to categorical selection (for medians/percentiles and data overlays)
    row_mask = np.ones(n_rows, dtype=bool)
    for base, label in cat_fixed.items():
        # Prefer raw string column if present; else fall back to the member one-hot feature
        if base in df_raw.columns:
            row_mask &= (df_raw[base].astype("string") == pd.Series([label]*len(df_raw), dtype="string")).to_numpy()
        else:
            member_name = groups[base]["name_by_label"][label]
            j = name_to_idx[member_name]
            # reconstruct per-row raw (0/1) for the member feature
            raw_j = feature_raw_from_artifact_or_reconstruct(ds, j, member_name, transforms[j]).astype(float)
            row_mask &= (raw_j >= 0.5)

    # Apply filtering (if any categorical fixed present)
    if cat_fixed:
        df_raw_f = df_raw.loc[row_mask].reset_index(drop=True)
        Xn_train_f = Xn_train[row_mask, :]
    else:
        df_raw_f = df_raw
        Xn_train_f = Xn_train

    # --- numeric constraints (fixed scalars, ranges, choices) → standardized
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

    # --- apply categorical fixed as standardized scalar fixes on their member features
    for base, label in cat_fixed.items():
        labels = groups[base]["labels"]
        for lab in labels:
            member_name = groups[base]["name_by_label"][lab]
            j = name_to_idx[member_name]
            raw_val = 1.0 if lab == label else 0.0
            # convert to standardized
            x_std = _orig_to_std(j, raw_val, transforms, X_mean, X_std)
            fixed_scalars_std[j] = float(x_std)

    # --- choose FREE (plotted) feature indices: exclude scalar-fixed AND all one-hot members
    onehot_members = set()
    for base, g in groups.items():
        onehot_members.update(g["members"])
    free_idx = [
        j for j in range(len(feature_names))
        if (j not in fixed_scalars_std) and (feature_names[j] not in onehot_members)
    ]
    if not free_idx:
        raise ValueError("All features are fixed (or categorical only); nothing to plot.")

    # --- base point (median in standardized space of the filtered rows), then apply scalar fixes
    base_std = np.median(Xn_train_f, axis=0)
    for j, vstd in fixed_scalars_std.items():
        base_std[j] = vstd

    # --- per-feature grids (respect numeric slices/choices + filtered 1–99% range)
    p01p99 = [np.percentile(Xn_train_f[:, j], [1, 99]) for j in range(len(feature_names))]
    def _grid_std_for(j: int) -> np.ndarray:
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

    grids_std = {j: _grid_std_for(j) for j in range(len(feature_names))}

    # --- candidate optimal/suggest under the same constraints (pass original kwargs through)
    optimal_df = opt.optimal(ds, count=1, **kwargs) if optimal else None
    suggest_df = opt.suggest(ds, count=suggest, **kwargs) if (suggest and suggest > 0) else None

    # --- figure scaffolding
    k = len(free_idx)
    fig = make_subplots(
        rows=k, cols=k, shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.03, vertical_spacing=0.03,
    )

    tgt_col = str(ds.attrs["target_column"])
    success_mask = ~pd.isna(df_raw_f[tgt_col]).to_numpy()
    fail_mask    = ~success_mask

    def data_vals_for_feature(j_full: int) -> np.ndarray:
        name = feature_names[j_full]
        if name in df_raw_f.columns:
            return df_raw_f[name].to_numpy().astype(float)
        return feature_raw_from_artifact_or_reconstruct(ds, j_full, name, transforms[j_full]).astype(float)[row_mask] \
               if cat_fixed else \
               feature_raw_from_artifact_or_reconstruct(ds, j_full, name, transforms[j_full]).astype(float)

    # --- evaluate all cells
    all_blocks: list[np.ndarray] = []
    cell_payload: dict[tuple[int,int], dict] = {}

    for r, i in enumerate(free_idx):
        for c, j in enumerate(free_idx):
            xg = grids_std[j]; yg = grids_std[i]
            if i == j:
                # diagonal: synthesize symmetric 2D from 1D sweep
                grid = grids_std[j]
                Xn_1d = np.repeat(base_std[None, :], len(grid), axis=0)
                Xn_1d[:, j] = grid
                mu_1d, _ = pred_loss(Xn_1d, include_observation_noise=True)
                p_1d     = pred_success(Xn_1d)
                Zmu = 0.5 * (mu_1d[:, None] + mu_1d[None, :])
                Zp  = np.minimum(p_1d[:, None], p_1d[None, :])
                x_orig = _denormalize_then_inverse_transform(j, grid, transforms, X_mean, X_std)
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
                x_orig = _denormalize_then_inverse_transform(j, xg, transforms, X_mean, X_std)
                y_orig = _denormalize_then_inverse_transform(i, yg, transforms, X_mean, X_std)

            cell_payload[(r, c)] = {"i": i, "j": j, "x": x_orig, "y": y_orig, "Zmu": Zmu, "Zp": Zp}
            all_blocks.append(Zmu.ravel())

    # --- color transform bounds
    def color_transform(z_raw: np.ndarray) -> tuple[np.ndarray, float]:
        if not use_log_scale_for_target:
            return z_raw, 0.0
        zmin = float(np.nanmin(z_raw))
        shift = 0.0 if zmin > 0 else -zmin + float(log_shift_epsilon)
        return np.log10(np.maximum(z_raw + shift, log_shift_epsilon)), shift

    z_all = np.concatenate(all_blocks) if all_blocks else np.array([0.0, 1.0])
    z_all_t, global_shift = color_transform(z_all)
    cmin_t = float(np.nanmin(z_all_t))
    cmax_t = float(np.nanmax(z_all_t))
    cs = get_colorscale(colorscale)

    def contour_line_color(level_raw: float) -> str:
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
        i, j = PAY["i"], PAY["j"]
        x_vals, y_vals, Zmu_raw, Zp = PAY["x"], PAY["y"], PAY["Zmu"], PAY["Zp"]
        Z_t, _ = color_transform(Zmu_raw)

        fig.add_trace(go.Heatmap(
            x=x_vals, y=y_vals, z=Z_t,
            coloraxis="coloraxis", zsmooth=False, showscale=False,
            hovertemplate=(f"{feature_names[j]}: %{{x:.6g}}<br>"
                           f"{feature_names[i]}: %{{y:.6g}}"
                           "<br>E[target|success]: %{customdata:.3f}<extra></extra>"),
            customdata=Zmu_raw
        ), row=r+1, col=c+1)

        for thr, alpha in ((0.5, 0.25), (0.8, 0.40)):
            mask = np.where(Zp < thr, 1.0, np.nan)
            fig.add_trace(go.Heatmap(
                x=x_vals, y=y_vals, z=mask, zmin=0, zmax=1,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(128,128,128,{alpha})"]],
                showscale=False, hoverinfo="skip"
            ), row=r+1, col=c+1)

        # contours
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

        # overlay data (filtered)
        xd = data_vals_for_feature(j)
        yd = data_vals_for_feature(i)
        show_leg = (r == 0 and c == 0)
        tgt_col = str(ds.attrs["target_column"])
        fig.add_trace(go.Scattergl(
            x=xd[success_mask], y=yd[success_mask], mode="markers",
            marker=dict(size=4, color="black", line=dict(width=0)),
            name="data (success)", legendgroup="data_succ", showlegend=show_leg,
            hovertemplate=("trial_id: %{customdata[0]}<br>"
                           f"{feature_names[j]}: %{{x:.6g}}<br>"
                           f"{feature_names[i]}: %{{y:.6g}}<br>"
                           f"{tgt_col}: %{{customdata[1]:.4f}}<extra></extra>"),
            customdata=np.column_stack([
                df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask],
                df_raw_f[tgt_col].to_numpy()[success_mask],
            ])
        ), row=r+1, col=c+1)
        fig.add_trace(go.Scattergl(
            x=xd[~success_mask], y=yd[~success_mask], mode="markers",
            marker=dict(size=5, color="red", line=dict(color="black", width=0.8)),
            name="data (failed)", legendgroup="data_fail", showlegend=show_leg,
            hovertemplate=("trial_id: %{customdata}<br>"
                           f"{feature_names[j]}: %{{x:.6g}}<br>"
                           f"{feature_names[i]}: %{{y:.6g}}<br>"
                           "status: failed (NaN target)<extra></extra>"),
            customdata=df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[~success_mask]
        ), row=r+1, col=c+1)

        # optional overlays (optimal/suggest) — only numeric axes are plotted
        if optimal and (optimal_df is not None):
            if feature_names[j] in optimal_df.columns and feature_names[i] in optimal_df.columns:
                opt_x = np.asarray(optimal_df[feature_names[j]].values, dtype=float)
                opt_y = np.asarray(optimal_df[feature_names[i]].values, dtype=float)
                if opt_x.size and opt_y.size and np.all(np.isfinite(opt_x)) and np.all(np.isfinite(opt_y)):
                    pred_m  = float(optimal_df["pred_target_mean"].values[0])
                    pred_sd = float(optimal_df["pred_target_sd"].values[0])
                    fig.add_trace(go.Scattergl(
                        x=opt_x, y=opt_y, mode="markers",
                        marker=dict(size=10, color="yellow", line=dict(color="black", width=1.5), symbol="x"),
                        name="optimal", legendgroup="optimal",
                        showlegend=(r == 0 and c == 0),
                        hovertemplate=(
                            f"predicted: {pred_m:.2g} ± {pred_sd:.2g}<br>"
                            f"{feature_names[j]}: %{{x:.6g}}<br>"
                            f"{feature_names[i]}: %{{y:.6g}}<extra></extra>"
                        ),
                    ), row=r+1, col=c+1)

        if suggest and (suggest_df is not None):
            have = (
                (feature_names[j] in suggest_df.columns)
                and (feature_names[i] in suggest_df.columns)
            )
            if have:
                x_sug = np.asarray(suggest_df[feature_names[j]].values, dtype=float)
                y_sug = np.asarray(suggest_df[feature_names[i]].values, dtype=float)
                keep_s = np.isfinite(x_sug) & np.isfinite(y_sug)
                x_sug = x_sug[keep_s]
                y_sug = y_sug[keep_s]
                if x_sug.size:
                    mu_s = suggest_df.loc[keep_s, "pred_target_mean"].values if "pred_target_mean" in suggest_df else None
                    sd_s = suggest_df.loc[keep_s, "pred_target_sd"].values   if "pred_target_sd"   in suggest_df else None
                    ps_s = suggest_df.loc[keep_s, "pred_p_success"].values   if "pred_p_success"   in suggest_df else None
                    if (mu_s is not None) and (sd_s is not None) and (ps_s is not None):
                        custom_s = np.column_stack([mu_s, sd_s, ps_s])
                        hover_s = (
                            f"{feature_names[j]}: %{{x:.6g}}<br>"
                            f"{feature_names[i]}: %{{y:.6g}}<br>"
                            "pred: %{customdata[0]:.3g} ± %{customdata[1]:.3g}<br>"
                            "p(success): %{customdata[2]:.2f}<extra>suggested</extra>"
                        )
                    else:
                        custom_s = None
                        hover_s = (
                            f"{feature_names[j]}: %{{x:.6g}}<br>"
                            f"{feature_names[i]}: %{{y:.6g}}<extra>suggested</extra>"
                        )
                    fig.add_trace(go.Scattergl(
                        x=x_sug, y=y_sug, mode="markers",
                        marker=dict(size=9, color="cyan", line=dict(color="black", width=1.2), symbol="star"),
                        name="suggested", legendgroup="suggested",
                        showlegend=(r == 0 and c == 0),
                        customdata=custom_s, hovertemplate=hover_s
                    ), row=r+1, col=c+1)

        _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="x", centers=x_vals, is_log=_is_log_feature(j))
        _update_axis_type_and_range(fig, row=r+1, col=c+1, axis="y", centers=y_vals, is_log=_is_log_feature(i))

    # labels
    free_names = [feature_names[j] for j in free_idx]
    for c, nm in enumerate(free_names):
        fig.update_xaxes(title_text=nm, row=k, col=c+1)
    for r, nm in enumerate(free_names):
        fig.update_yaxes(title_text=nm, row=r+1, col=1)

    # title
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

    title_parts = ["2D Partial Dependence of Expected Target"]
    # numeric constraints for display (keys already canonical)
    for name, val in kw_num.items():
        title_parts.append(f"{name}={_fmt_c(val)}")
    # categorical fixed
    for base, lab in cat_fixed.items():
        title_parts.append(f"{base}={lab}")
    title = " — ".join([title_parts[0], ", ".join(title_parts[1:])]) if len(title_parts) > 1 else title_parts[0]

    # layout
    cell = 250
    z_title = "E[target|success]" + (" (log10)" if use_log_scale_for_target else "")
    if use_log_scale_for_target and global_shift > 0:
        z_title += f" (shift Δ={global_shift:.3g})"

    width = width if (width and width > 0) else cell * k
    width = max(width, 400)
    height = height if (height and height > 0) else cell * k
    height = max(height, 400)

    fig.update_layout(
        coloraxis=dict(colorscale=colorscale, cmin=cmin_t, cmax=cmax_t,
                       colorbar=dict(title=z_title)),
        template="simple_white",
        width=width,
        height=height,
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


def plot1d(
    model: xr.Dataset | Path | str,
    output: Path | None = None,
    csv_out: Path | None = None,
    n_points_1d: int = 300,
    line_color: str = "rgb(31,119,180)",
    band_alpha: float = 0.25,
    figure_height_per_row_px: int = 320,
    show: bool = False,
    use_log_scale_for_target_y: bool = True,   # log-y for target
    log_y_epsilon: float = 1e-9,
    optimal: bool = True,
    suggest: int = 0,
    width:int|None = None,
    height:int|None = None,
    **kwargs,
) -> go.Figure:
    """
    Vertical 1D PD panels of E[target|success] vs each *free* numeric feature.
    Scalars (fix & hide), slices (restrict sweep & x-range), lists/tuples (discrete grids).
    Categorical constraints use the base name, e.g. language="Linear A".
    """
    ds = model if isinstance(model, xr.Dataset) else xr.load_dataset(model)
    pred_success, pred_loss = _build_predictors(ds)

    feature_names = [str(n) for n in ds["feature"].values.tolist()]
    transforms    = [str(t) for t in ds["feature_transform"].values.tolist()]
    X_mean = ds["feature_mean"].values.astype(float)
    X_std  = ds["feature_std"].values.astype(float)

    # Detect one-hot categorical groups from model features
    groups = opt._onehot_groups(feature_names)   # { base: {"labels":[...], "name_by_label":{label:member}, "members":[...]} }
    bases  = set(groups.keys())
    name_to_idx = {name: j for j, name in enumerate(feature_names)}

    # Raw dataframe + training X (we may filter these by categorical selection)
    df_raw   = _raw_dataframe_from_dataset(ds)
    Xn_train = ds["Xn_train"].values.astype(float)
    n_rows   = Xn_train.shape[0]
    p        = Xn_train.shape[1]

    # --- canonicalize numeric kwargs to real feature names; collect categorical kwargs by base ---
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

    # --- resolve categorical constraints to a single fixed label per base ---
    cat_fixed: dict[str, str] = {}
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
            if len(chosen) > 1:
                console.print(f"[yellow]Warning: multiple categories for {base!r} supplied; using {chosen[0]!r}[/yellow]")
            cat_fixed[base] = chosen[0]
        else:
            raise ValueError(f"Categorical constraint for {base!r} must be a string or list/tuple of strings.")

    # --- filter rows to chosen categories (affects medians/percentiles & overlay points) ---
    row_mask = np.ones(n_rows, dtype=bool)
    for base, label in cat_fixed.items():
        if base in df_raw.columns:  # prefer stored raw string column
            row_mask &= (df_raw[base].astype("string") == pd.Series([label]*len(df_raw), dtype="string")).to_numpy()
        else:
            member_name = groups[base]["name_by_label"][label]
            j = name_to_idx[member_name]
            raw_j = feature_raw_from_artifact_or_reconstruct(ds, j, member_name, transforms[j]).astype(float)
            row_mask &= (raw_j >= 0.5)

    if cat_fixed:
        df_raw_f = df_raw.loc[row_mask].reset_index(drop=True)
        Xn_train_f = Xn_train[row_mask, :]
    else:
        df_raw_f = df_raw
        Xn_train_f = Xn_train

    # --- helpers to move between original and standardized for a given feature j ---
    def _orig_to_std(j: int, x, transforms, mu, sd):
        x = np.asarray(x, dtype=float)
        if transforms[j] == "log10":
            x = np.where(x <= 0, np.nan, x)
            x = np.log10(x)
        return (x - mu[j]) / sd[j]

    # --- split numeric constraints into scalar fixes, range windows, discrete choices (STANDARDIZED) ---
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

    # --- overlays (pass original kwargs so optimal/suggest are conditioned the same way, incl. categorical) ---
    optimal_df  = opt.optimal(model, count=1, **kwargs) if optimal else None
    suggest_df  = opt.suggest(model, count=suggest, **kwargs) if (suggest and suggest > 0) else None

    # --- base standardized point (median of filtered rows), then apply scalar fixes ---
    base_std = np.median(Xn_train_f, axis=0)
    for j, vstd in fixed_scalars.items():
        base_std[j] = vstd

    # --- features to PLOT: exclude scalar-fixed + all one-hot member features ---
    onehot_members = set()
    for base, g in groups.items():
        onehot_members.update(g["members"])
    free_idx = [j for j in range(p) if (j not in fixed_scalars) and (feature_names[j] not in onehot_members)]
    if not free_idx:
        raise ValueError("All features are fixed (or categorical only); nothing to plot. Fix fewer variables.")

    # --- empirical 1–99% from filtered rows for sane bounds ---
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

    # sweeps (standardized)
    sweeps_std = {j: _grid_1d(j, n_points_1d) for j in free_idx}

    # --- overlay masks/data from filtered rows ---
    tgt_col = str(ds.attrs["target_column"])
    success_mask = ~pd.isna(df_raw_f[tgt_col]).to_numpy()
    fail_mask    = ~success_mask
    losses_success = df_raw_f.loc[success_mask, tgt_col].to_numpy().astype(float)
    trial_ids_success = df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[success_mask]
    trial_ids_fail    = df_raw_f.get("trial_id", pd.Series(np.arange(len(df_raw_f)))).to_numpy()[fail_mask]

    band_fill_rgba = _rgb_to_rgba(line_color, band_alpha)

    free_names = [feature_names[j] for j in free_idx]
    fig = make_subplots(
        rows=len(free_idx), cols=1, shared_xaxes=False,
        subplot_titles=[f"{name}" for name in free_names]
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

        # experimental points (filtered df)
        if feature_names[j] in df_raw_f.columns:
            x_data_all = df_raw_f[feature_names[j]].to_numpy().astype(float)
        else:
            full_vals = feature_raw_from_artifact_or_reconstruct(ds, j, feature_names[j], transforms[j]).astype(float)
            x_data_all = full_vals[row_mask] if cat_fixed else full_vals

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
        fig.update_yaxes(title_text=f"{tgt_col}", row=row_pos, col=1)
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

    # title w/ constraints summary (numeric + categorical)
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

    title_parts = [f"1D partial dependence of expected {tgt_col}"]
    if kw_num_raw:
        title_parts.append(", ".join(f"{k}={_fmt_c(v)}" for k, v in kw_num_raw.items()))
    if cat_fixed:
        title_parts.append(", ".join(f"{b}={lab}" for b, lab in cat_fixed.items()))
    title = " — ".join(title_parts) if len(title_parts) > 1 else title_parts[0]

    width = width if (width and width > 0) else 1200
    height = height if (height and height > 0) else figure_height_per_row_px * len(free_idx)

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
