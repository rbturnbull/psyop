.. image:: https://rbturnbull.github.io/psyop/_images/psyop-banner.jpg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge|

.. |testing badge| image:: https://github.com/rbturnbull/psyop/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/psyop/actions

.. |docs badge| image:: https://github.com/rbturnbull/psyop/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/psyop
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/d3a9e5f1b7d7b8593c9df1cd46fe7557/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/psyop/coverage/
    
.. end-badges

.. start-quickstart

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install psyop

Or

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/psyop.git

.. warning::

    Psyop is currently in alpha testing phase. More updates are coming soon.

Quick help
----------

.. code-block:: bash

    psyop --help
    psyop <command> --help


Model artifact
--------------

Most commands take a path to a ``.nc`` artifact produced by ``psyop model``.

The artifact bundles:

- raw columns (for plotting & metadata),
- standardized design matrices,
- per-feature transforms & standardization stats,
- two GP heads (success probability; expected target conditional on success),
- convenience predictions and metadata (direction, seed, etc.).


Constraint syntax (used by ``suggest``, ``optimal``, ``plot2d``, ``plot1d``)
-----------------------------------------------------------------------------

These commands accept **extra CLI options** that are *not predeclared*—they are parsed
into **feature constraints**. Constraints are always interpreted in **original units**
(before any internal log/standardization).

Pass any mix of:

- **Fixed value** *(number)* — clamp a feature at a single value and (for plots) remove it from the axes:

  .. code-block:: bash

      --epochs 20
      --learning-rate 0.001

- **Range** *(slice)* — inclusive lower/upper bound:

  .. code-block:: bash

      --dropout 0.0:0.2
      --x 0..2          # same as 0:2
      --width 64:512:64 # optional step token; used where applicable

- **Choices** *(finite set)* — list/tuple or Python-like ``range()`` call:

  .. code-block:: bash

      --batch-size "(16, 32, 64)"
      --optimizer "[adam, sgd, adamw]"
      --layers "range(2,8,2)"        # -> (2, 4, 6, 8)

Rules:

- Unknown keys are ignored with a warning (feature names are matched case-insensitively; hyphens/underscores are normalized).
- If you pass both a fixed value and a range/choices for the same feature, the **fixed value wins**.
- For **suggest**/**optimal**, bounds/choices are enforced strictly when sampling candidates.
- For **plot2d**/**plot1d**, fixed features are **clamped** and **not shown** on axes; range constraints **restrict the sweep domain** even if historical points exist outside the range.

Tip (shells): quote lists/tuples and anything that contains commas or parentheses to avoid shell expansion.


Commands
========

1) Fit a model
--------------

.. code-block:: bash

    psyop model INPUT.csv OUTPUT.nc [OPTIONS]

**Arguments**

- ``INPUT`` *(CSV)* — your experiment log.
- ``OUTPUT`` *(.nc)* — where to save the model artifact.

**Options**

- ``--target, -t TEXT`` — target column name (default: ``loss``).
- ``--exclude TEXT`` — repeatable; columns to exclude from features.
- ``--direction, -d [min|max|auto]`` — optimization direction for the target (default: ``auto``).
- ``--success-column TEXT`` — optional boolean/int column; if omitted, success is inferred as ``~isna(target)``.
- ``--seed INTEGER`` — RNG seed (default: 0).
- ``--compress / --no-compress`` — compress numeric arrays inside the artifact (default: on).

**Example**

.. code-block:: bash

    psyop model runs.csv output/trials.nc \
      --target loss --exclude run_id --exclude notes --direction auto --seed 42


2) Suggest candidates (constrained EI + exploration)
----------------------------------------------------

.. code-block:: bash

    psyop suggest MODEL.nc [OPTIONS] [EXTRA_CONSTRAINTS...]

**Options**

- ``--output, -o PATH`` — write suggestions CSV (if omitted, prints the table).
- ``--count, -k INTEGER`` — number of suggestions (default: 10).
- ``--p-success-threshold FLOAT`` — feasibility threshold in cEI (default: 0.8).
- ``--explore FLOAT`` — fraction of suggestions reserved for exploration (default: 0.34).
- ``--candidates-pool INTEGER`` — random candidate pool size to score (default: 5000).
- ``--seed INTEGER`` — RNG seed (default: 0).

**Constraints** — see *Constraint syntax* above.

**Output CSV columns**

``rank``, feature columns, ``pred_p_success``, ``pred_target_mean``, ``pred_target_sd``,
``acq_cEI``, ``acq_explore``, ``novelty_norm``, ``direction``, ``conditioned_on``.

**Examples**

.. code-block:: bash

    # Fix epochs; bound dropout
    psyop suggest output/trials.nc --epochs 20 --dropout 0.0:0.2 -k 12 -o output/suggest.csv

    # Discrete choices and integer grid:
    psyop suggest output/trials.nc \
      --batch-size "(16, 32, 64)" \
      --layers "range(2,8,2)" \
      --optimizer "[adam, sgd]"


3) Rank probable optima (winner-take-all MC)
--------------------------------------------

.. code-block:: bash

    psyop optimal MODEL.nc [OPTIONS] [EXTRA_CONSTRAINTS...]

**Options**

- ``--output PATH`` — write top rows CSV (prints table if omitted).
- ``--count, -k INTEGER`` — how many top rows to keep (default: 10).
- ``--draws INTEGER`` — Monte-Carlo draws (default: 2000).
- ``--min-p-success FLOAT`` — hard feasibility cutoff; set to 0.0 to disable (default: 0.0).
- ``--seed INTEGER`` — RNG seed (default: 0).

**Constraints** — see *Constraint syntax* above.

**Output CSV columns**

``rank_prob_best``, feature columns, ``pred_p_success``, ``pred_target_mean``,
``pred_target_sd``, ``prob_best_feasible``, ``wins``, ``n_draws_effective``, ``conditioned_on``.

**Example**

.. code-block:: bash

    psyop optimal output/trials.nc \
      --epochs 12 --dropout 0.0:0.2 --min-p-success 0.5 -k 5 -o output/optimal.csv


4) 2D Partial Dependence (pairwise features)
--------------------------------------------

.. code-block:: bash

    psyop plot2d MODEL.nc [OPTIONS] [EXTRA_CONSTRAINTS...]

**Options**

- ``--output PATH`` — HTML file.
- ``--n-points-1d INTEGER`` — diagonal sweep resolution (default: 300).
- ``--n-points-2d INTEGER`` — grid size per axis for 2D panels (default: 70).
- ``--use-log-scale-for-target`` — enable log10 colors for the target (toggle flag; default: off).
- ``--log-shift-epsilon FLOAT`` — epsilon shift for log colors (default: 1e-9).
- ``--colorscale TEXT`` — Plotly colorscale (default: ``RdBu``).
- ``--show`` — open in a browser.
- ``--n-contours INTEGER`` — contour levels (default: 12).
- ``--optimal / --no-optimal`` — overlay the current best-probable optimum (default: on).
- ``--suggest INTEGER`` — overlay up to N suggested points (default: 0).
- ``--width INTEGER`` / ``--height INTEGER`` — panel dimensions (pixels).

**Constraints**

- **Fixed** features are clamped and **removed** from the axes.
- **Ranges** restrict the sweep domain for that feature.

**Examples**

.. code-block:: bash

    # Clamp epochs; restrict dropout domain
    psyop plot2d output/trials.nc --epochs 20 --dropout 0.0:0.2 --show

    # Discrete choices for batch size
    psyop plot2d output/trials.nc --batch-size "(16,32,64)" -o pairplot.html


5) 1D Partial Dependence (per-feature)
--------------------------------------

.. code-block:: bash

    psyop plot1d MODEL.nc [OPTIONS] [EXTRA_CONSTRAINTS...]

**Options**

- ``--output PATH`` — HTML file.
- ``--csv-out PATH`` — tidy CSV export of PD values.
- ``--n-points-1d INTEGER`` — sweep resolution (default: 300).
- ``--line-color TEXT`` — Plotly color string for mean/band (default: ``rgb(31,119,180)``).
- ``--band-alpha FLOAT`` — fill alpha for ±2σ (default: 0.25).
- ``--figure-height-per-row-px INTEGER`` — pixels per PD row (default: 320).
- ``--show`` — open in a browser.
- ``--log-y / --no-log-y`` — log scale for target axis (default: log).
- ``--log-y-eps FLOAT`` — clamp for log-Y (default: 1e-9).
- ``--optimal / --no-optimal`` — overlay the current best-probable optimum (default: on).
- ``--suggest INTEGER`` — overlay up to N suggested points (default: 0).
- ``--width INTEGER`` / ``--height INTEGER`` — panel dimensions (pixels).

**Constraints**

Same as *Constraint syntax*. Fixed features are **not plotted**; ranges **clip** the sweep domain.

**Examples**

.. code-block:: bash

    psyop plot1d output/trials.nc --epochs 20 --dropout 0.0:0.2 \
      --csv-out output/pd.csv -o output/pd.html --show


Notes
-----

- **Colorscales** are Plotly names (e.g. ``RdBu``, ``Viridis``, ``Inferno``).
- For plots, historical points are drawn even if outside your specified *range*,
  but the **sweep domain** (and axes) respect your bounds.
- All constraint parsing is printed once as ``Constraints: ...`` for sanity checking.


Examples at a glance
--------------------

.. code-block:: bash

    # Fit
    psyop model runs.csv output/trials.nc -t loss --exclude run_id --seed 0

    # Suggest inside bounds, with discrete choices
    psyop suggest output/trials.nc \
      --epochs 20 \
      --dropout 0.0:0.2 \
      --batch-size "(16,32,64)" \
      -k 12 -o output/suggest.csv

    # Rank optima with a minimum feasibility threshold
    psyop optimal output/trials.nc --min-p-success 0.6 -k 5

    # Pairwise PD conditioned on epochs
    psyop plot2d output/trials.nc --epochs 20 --show

    # 1D PD with CSV export
    psyop plot1d output/trials.nc --csv-out output/pd.csv -o output/pd.html


Programmatic API
================

All functionality is also exposed as Python functions. You can work directly with
``xarray.Dataset`` objects or file paths.

Import paths:

.. code-block:: python

    import xarray as xr
    from pathlib import Path
    from psyop import build_model, optimal, suggest, plot1d, plot2d

Build a model
-------------

.. code-block:: python

    build_model(
        input=Path("runs.csv"),
        output=Path("output/trials.nc"),
        target_column="loss",
        exclude_columns=["run_id", "notes"],
        direction="auto",          # "min", "max", or "auto"
        success_column=None,       # infer success as ~isna(target)
        random_seed=42,
        compress=True,             # compress numeric arrays within the .nc
    )

Load a model
------------

.. code-block:: python

    ds = xr.load_dataset("output/trials.nc")

Suggest candidates
------------------

.. code-block:: python

    # Constraints are passed as kwargs in ORIGINAL units:
    # - fixed: number
    # - range: slice(lo, hi)          (inclusive semantics for the endpoints)
    # - choices: list/tuple (finite)  (e.g. tuple(range(...)))
    suggestions = suggest(
        model=ds,                    # or "output/trials.nc"
        output=None,                 # optional CSV path; None to return only the DataFrame
        count=12,
        p_success_threshold=0.8,
        explore_fraction=0.34,
        candidates_pool=5000,
        random_seed=0,
        epochs=20,                   # fixed
        dropout=slice(0.0, 0.2),     # range
        batch_size=(16, 32, 64),     # choices
    )
    print(suggestions.head())

Rank probable optima
--------------------

.. code-block:: python

    top = optimal(
        model=ds,                    # or "output/trials.nc"
        output=None,                 # optional CSV path
        count=10,
        n_draws=2000,
        min_success_probability=0.5, # 0.0 disables the hard cutoff
        random_seed=0,
        epochs=12,
        dropout=slice(0.0, 0.2),
    )
    print(top[["prob_best_feasible", "pred_target_mean"]].head())

2D Partial Dependence (HTML)
----------------------------

.. code-block:: python

    # Fixed features are clamped and removed from axes.
    # Ranges clip the sweep domain even if historical points exist outside the range.
    plot2d(
        model=ds,                    # xarray.Dataset
        output=Path("pairplot.html"),
        n_points_1d=300,
        n_points_2d=70,
        use_log_scale_for_target=False,
        log_shift_epsilon=1e-9,
        colorscale="RdBu",
        show=False,
        n_contours=12,
        optimal=True,                # overlay current best-probable optimum
        suggest=5,                   # overlay top-N suggestions
        width=None,
        height=None,
        epochs=20,
        dropout=slice(0.0, 0.2),
    )

1D Partial Dependence (HTML + tidy CSV)
---------------------------------------

.. code-block:: python

    plot1d(
        model=ds,
        output=Path("pd.html"),
        csv_out=Path("pd.csv"),
        n_points_1d=300,
        line_color="rgb(31,119,180)",
        band_alpha=0.25,
        figure_height_per_row_px=320,
        show=False,
        use_log_scale_for_target_y=True,
        log_y_epsilon=1e-9,
        optimal=True,
        suggest=3,
        width=None,
        height=None,
        epochs=20,
        dropout=slice(0.0, 0.2),
    )

Return types and side effects
-----------------------------

- ``build_model(...)`` → ``None`` (writes a ``.nc`` file).
- ``suggest(...)`` → ``pandas.DataFrame`` (and optionally writes a CSV if ``output`` is provided).
- ``optimal(...)`` → ``pandas.DataFrame`` (and optionally writes a CSV if ``output`` is provided).
- ``plot2d(...)`` → ``None`` (writes HTML if ``output`` is provided; may open a browser if ``show=True``).
- ``plot1d(...)`` → ``None`` (writes HTML/CSV if paths are provided; may open a browser if ``show=True``).

Constraint objects in Python
----------------------------

- **Fixed**: ``epochs=20`` or ``learning_rate=1e-3``.
- **Range**: ``dropout=slice(0.0, 0.2)`` (inclusive ends).
- **Choices**: ``batch_size=(16, 32, 64)`` (tuple/list of finite values).
- **Integer grids**: ``layers=tuple(range(2, 9, 2))``  → ``(2, 4, 6, 8)``.

All constraints are interpreted in **original units** of your data. Bounds are enforced
for candidate sampling and sweep ranges; fixed values remove the feature from PD axes.


.. end-quickstart


Credits
==================================

.. start-credits

Robert Turnbull
For more information contact: <robert.turnbull@unimelb.edu.au>

.. end-credits

