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

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .model import run_model
from .viz import make_pairplot, make_partial_dependence1D
from .opt import suggest_candidates, find_optimal

__version__ = "0.1.0"

console = Console()
app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")


class Direction(str, Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"
    AUTO = "auto"


def _ensure_parent_dir(path: Path) -> None:
    if path.suffix and path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _force_netcdf_suffix(path: Path) -> Path:
    """Ensure path ends with .nc (normalize .netcdf to .nc)."""
    if path.suffix.lower() in {".nc", ".netcdf"}:
        return path.with_suffix(".nc")
    return path.with_suffix(".nc")


def _default_out_for(model_path: Path, stem_suffix: str, ext: str) -> Path:
    """model_path='foo.nc', stem_suffix='__pairplot', ext='.html' -> 'foo__pairplot.html'"""
    base = model_path.with_suffix("")  # strip .nc
    return base.with_name(base.name + stem_suffix).with_suffix(ext)


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit.", is_eager=True),
):
    if version:
        console.print(f"[bold]psyop[/] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------
@app.command(help="Fit the model on a CSV and save a single model artifact.")
def model(
    input: Path = typer.Argument(..., help="Input CSV file."),
    output: Path = typer.Argument(..., help="Path to save model artifact (.nc)."),
    target: str = typer.Option("loss", "--target", "-t", help="Target column name."),
    exclude: list[str] = typer.Option([], "--exclude", help="Feature columns to exclude."),
    direction: Direction = typer.Option(
        Direction.AUTO, "--direction", "-d",
        help="Optimization direction for the target."
    ),
    success_column: Optional[str] = typer.Option(
        None, "--success-column",
        help="Optional boolean/int column for success (1) / fail (0). "
             "If omitted, success is inferred as ~isna(target)."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for fitting/sampling."),
    compress: bool = typer.Option(
        True, "--compress/--no-compress",
        help="Apply compression inside the artifact."
    ),
):
    if not input.exists():
        raise typer.BadParameter(f"Input CSV not found: {input.resolve()}")
    if input.suffix.lower() != ".csv":
        console.print(":warning: [yellow]Input does not end with .csv[/]")

    output = _force_netcdf_suffix(output)
    _ensure_parent_dir(output)

    run_model(
        input_path=input,
        output_path=output,
        target_column=target,
        exclude_columns=exclude,
        direction=direction.value,
        success_column=success_column,
        random_seed=seed,
        compress=compress,
    )
    console.print(f"[green]Wrote model artifact →[/] {output}")


@app.command(help="Suggest BO candidates (constrained EI + exploration).")
def suggest(
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Where to save candidates CSV (defaults relative to model)."
    ),
    count: int = typer.Option(12, "--count", "-n", help="Number of candidates to propose."),
    p_success_threshold: float = typer.Option(
        0.8, "--p-threshold",
        help="Feasibility threshold for constrained EI."
    ),
    explore_fraction: float = typer.Option(
        0.34, "--explore-fraction",
        help="Fraction of suggestions reserved for exploration."
    ),
    candidates_pool: int = typer.Option(
        5000, "--pool",
        help="Random candidate pool size to score."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for proposals."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__bo_proposals", ext=".csv")
    _ensure_parent_dir(output)

    suggest_candidates(
        model_path=_force_netcdf_suffix(model),
        output_path=output,
        count=count,
        p_success_threshold=p_success_threshold,
        explore_fraction=explore_fraction,
        candidates_pool=candidates_pool,
        random_seed=seed,
    )
    console.print(f"[green]Wrote proposals →[/] {output}")


@app.command(help="Rank points by probability of being the best feasible minimum.")
def optimal(
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Where to save top candidates CSV (defaults relative to model)."
    ),
    count: int = typer.Option(10, "--count", "-k", help="How many top rows to keep."),
    draws: int = typer.Option(2000, "--draws", help="Monte Carlo draws."),
    min_success_probability: float = typer.Option(
        0.0, "--min-p-success",
        help="Hard feasibility cutoff (0 disables)."
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for MC."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__bo_best_probable", ext=".csv")
    _ensure_parent_dir(output)

    find_optimal(
        model_path=_force_netcdf_suffix(model),
        output_path=output,
        top_k=count,
        n_draws=draws,
        min_success_probability=min_success_probability,
        random_seed=seed,
    )
    console.print(f"[green]Wrote top probable minima →[/] {output}")


@app.command(help="Create a 2D pairplot PD heatmap with contours & data points.")
def plot2d(
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, "-o", help="Output HTML (defaults relative to model)."),
    n_points_1d: int = typer.Option(300, help="Points along 1D sweeps (diagonal)."),
    n_points_2d: int = typer.Option(70, help="Grid size per axis for 2D panels."),
    use_log_scale_for_target: bool = typer.Option(False, "--log-target", help="Log10 colours for target."),
    log_shift_epsilon: float = typer.Option(1e-9, "--log-eps", help="Epsilon shift for log colours."),
    colourscale: str = typer.Option("RdBu", "--colourscale", help="Plotly colourscale name."),
    show: bool = typer.Option(False, "--show", help="Open the figure in a browser."),
    n_contours: int = typer.Option(12, "--n-contours", help="Number of contour levels."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__pairplot", ext=".html")
    _ensure_parent_dir(output)

    make_pairplot(
        model=_force_netcdf_suffix(model),
        output=output,
        n_points_1d=n_points_1d,
        n_points_2d=n_points_2d,
        use_log_scale_for_target=use_log_scale_for_target,
        log_shift_epsilon=log_shift_epsilon,
        colourscale=colourscale,
        show=show,
        n_contours=n_contours,
    )
    console.print(f"[green]Wrote pairplot →[/] {output}")


@app.command(name="partial-dependence", help="Create 1D PD panels with shading & experimental points.")
def plot1d(
    model: Path = typer.Argument(..., help="Path to the model artifact (.nc)."),
    output: Optional[Path] = typer.Option(None, "-o", help="Output HTML (defaults relative to model)."),
    csv_out: Optional[Path] = typer.Option(None, help="Optional CSV export of tidy PD data."),
    n_points_1d: int = typer.Option(300, help="Points along 1D sweep."),
    line_color: str = typer.Option("rgb(31,119,180)", help="Line/band color (consistent across variables)."),
    band_alpha: float = typer.Option(0.25, help="Fill alpha for ±2σ."),
    figure_height_per_row_px: int = typer.Option(320, help="Pixels per PD row."),
    show_figure: bool = typer.Option(False, "--show", help="Open the figure in a browser."),
    use_log_scale_for_target_y: bool = typer.Option(True, "--log-y/--no-log-y", help="Log scale for target (Y)."),
    log_y_epsilon: float = typer.Option(1e-9, "--log-y-eps", help="Clamp for log-Y."),
):
    if not model.exists():
        raise typer.BadParameter(f"Model artifact not found: {model.resolve()}")
    if model.suffix.lower() not in {".nc", ".netcdf"}:
        console.print(":warning: [yellow]Model path does not end with .nc[/]")

    if output is None:
        output = _default_out_for(model, stem_suffix="__pd_1d", ext=".html")
    if csv_out is None:
        csv_out = _default_out_for(model, stem_suffix="__pd_1d", ext=".csv")
    _ensure_parent_dir(output)
    _ensure_parent_dir(csv_out)

    make_partial_dependence1D(
        model=_force_netcdf_suffix(model),
        output=output,
        csv_out=csv_out,
        n_points_1d=n_points_1d,
        line_color=line_color,
        band_alpha=band_alpha,
        figure_height_per_row_px=figure_height_per_row_px,
        show_figure=show_figure,
        use_log_scale_for_target_y=use_log_scale_for_target_y,
        log_y_epsilon=log_y_epsilon,
    )
    console.print(f"[green]Wrote PD HTML →[/] {output}")
    console.print(f"[green]Wrote PD CSV  →[/] {csv_out}")


if __name__ == "__main__":
    app()
