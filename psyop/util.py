import numpy as np
import pandas as pd
from rich.table import Table

def get_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    """ Get a numpy random number generator from a seed. """
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
    elif seed is None:
        rng = np.random.default_rng()
    else:
        raise TypeError(
            f"seed must be None, int, or numpy.random.Generator, not {type(seed)}"
        )

    return rng


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table | None = None,
    show_index: bool = True,
    index_name: str | None = None,
    transpose: bool = True,
    sig_figs: int = 3,
    heading_style: str = "magenta",
) -> Table:
    """Convert a pandas.DataFrame to a rich.Table with optional transpose and sig-fig formatting."""

    def _fmt_cell(x) -> str:
        try:
            import numpy as _np
            if isinstance(x, (float, _np.floating)):
                if _np.isnan(x) or _np.isinf(x):
                    return str(x)
                return f"{float(x):.{sig_figs}g}"
            if isinstance(x, (int, _np.integer)) and not isinstance(x, bool):
                return str(int(x))
        except Exception:
            pass
        return str(x)

    rich_table = rich_table or Table(show_header=not transpose, header_style="bold magenta")

    df = pandas_dataframe

    if not transpose:
        # Original orientation
        if show_index:
            rich_table.add_column(str(index_name) if index_name else "", style=heading_style)
        for col in df.columns:
            rich_table.add_column(str(col))
        for idx, row in df.iterrows():
            cells = ([str(idx)] if show_index else []) + [_fmt_cell(v) for v in row.tolist()]
            rich_table.add_row(*cells)
        return rich_table

    # Transposed-like view (columns as rows)
    left_header = str(index_name) if index_name is not None else ""
    rich_table.add_column(left_header, style=heading_style)  # magenta left column

    # Column headers across (not displayed when show_header=False, but keeps widths aligned)
    across_headers = [str(i) for i in df.index] if show_index else ["" for _ in range(len(df.index))]
    for h in across_headers:
        rich_table.add_column(h)

    for col in df.columns:
        row_vals = [_fmt_cell(v) for v in df[col].tolist()]
        rich_table.add_row(str(col), *row_vals)

    return rich_table
