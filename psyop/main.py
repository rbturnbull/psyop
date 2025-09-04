import os
for _env_var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import typer
from pathlib import Path
from enums import Enum

from .model import run_model
from .viz import make_pairplot, make_partial_dependence1D
from .opt import suggest_candidates, find_optimal

app = typer.Typer()


class Direction(Enum):
    MINIMIZE = "min"
    MAXIMIZE = "max"
    AUTO = "auto"


@app.command()
def model(
    input: Path,
    output: Path,
    target: str = typer.Option("loss", help="Target column name"),
    exclude: list[str] = typer.Option([], help="Columns to exclude from features"),
    direction: Direction = typer.Option(Direction.AUTO, help="Optimization direction"),
):
    run_model(
        input_path=input,
        output_path=output,
        target_column=target,
        exclude_columns=exclude,
        direction=direction.value,
    )


@app.command()
def suggest(
    model: Path = typer.Argument(..., help="Path to the model output netcdf"),
    output: Path = typer.Option(None, help="Path to save the candidates"),
    count: int = typer.Option(1, help="Number of candidates to present"),
):
    suggest_candidates(model, output=output, count=count)


@app.command()
def optimal(
    model: Path = typer.Argument(..., help="Path to the model output netcdf"),
    output: Path = typer.Option(None, help="Path to save the candidates"),
    count: int = typer.Option(1, help="Number of candidates to present"),
):
    find_optimal(model, output=output, count=count)


@app.command()
def pairplot(
    model: Path = typer.Argument(..., help="Path to the model output netcdf"),
    output: Path = typer.Option(None, help="Path to save the pairplot"),
    n_points_1d: int = 300,
    n_points_2d: int = 70,
    use_log_scale_for_target: bool = False,
    log_shift_epsilon: float = 1e-9,
    colourscale: str = "RdBu",
    show: bool = False,
    n_contours: int = 12,
):
    make_pairplot(
        model=model,
        output=output,
        n_points_1d=n_points_1d,
        n_points_2d=n_points_2d,
        use_log_scale_for_target=use_log_scale_for_target,
        log_shift_epsilon=log_shift_epsilon,
        colourscale=colourscale,
        show=show,
        n_contours=n_contours,
    )


@app.command()
def partial_dependence(
    model: Path = typer.Argument(..., help="Path to the model output netcdf"),
    output: Path = typer.Option(None, help="Path to save the plot"),
    n_points_1d: int = 300,
    line_color: str = "rgb(31,119,180)",     # same color for all variables
    band_alpha: float = 0.25,                # fill alpha for ±2σ
    figure_height_per_row_px: int = 320,
    show_figure: bool = True,
    use_log_scale_for_target_y: bool = False,
    log_y_epsilon: float = 1e-9,             # clamp for log-y safety
):
    make_partial_dependence1D(
        model=model,
        output=output,
        n_points_1d=n_points_1d,
        line_color=line_color,
        band_alpha=band_alpha,
        figure_height_per_row_px=figure_height_per_row_px,
        show_figure=show_figure,
        use_log_scale_for_target_y=use_log_scale_for_target_y,
        log_y_epsilon=log_y_epsilon,
    )