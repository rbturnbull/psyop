
import numpy as np
import pandas as pd

from psyop import build_model, optimal, suggest, plot1d, plot2d


def test_quadratic(tmpdir):
    output = tmpdir/"quadratic.nc"

    x = np.linspace(0, 2.0, 20)
    y = (x - 0.5)**2
    df = pd.DataFrame({
        "x": x,
        "y": y
    })
    model = build_model(df, target_column="y", output=output)
    assert output.exists()

    optimal_df = optimal(model)
    assert len(optimal_df) == 10
    assert np.allclose(optimal_df['x'], 0.5, atol=0.05)
    assert "pred_p_success" in optimal_df
    assert "pred_target_mean" in optimal_df
    assert "pred_target_sd" in optimal_df

    suggest_df = suggest(model, count=6, x=slice(0.0,2.0), explore=0.0)
    assert len(suggest_df) == 6
    assert np.allclose(suggest_df['x'], 0.5, atol=0.05)
    assert "pred_p_success" in suggest_df
    assert "pred_target_mean" in suggest_df
    assert "pred_target_sd" in suggest_df

    show = False
    fig = plot1d(model, show=show)
    assert fig is not None

    fig = plot2d(model, show=show)
    assert fig is not None

