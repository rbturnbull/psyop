import pandas as pd

from psyop import suggest, optimal

from .test_model import TEST_DATA_DIR
TEST_XARRAY = TEST_DATA_DIR/"trials.nc"
TEST_QUADRATIC_XARRAY = TEST_DATA_DIR/"quadratic.nc"

def test_suggest(tmpdir):
    output = tmpdir/"output.csv"
    suggest(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 10


def test_optimal(tmpdir):
    output = tmpdir/"output.csv"
    optimal(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 10


def test_suggest_quadratic_exploit():
    result = suggest(TEST_QUADRATIC_XARRAY, explore=0.0, count=30)
    assert len(result) == 30
    assert all(abs(result['x'] - 0.5) < 0.05)


def test_suggest_quadratic_explore():
    result = suggest(TEST_QUADRATIC_XARRAY, explore=0.4, count=30)
    assert len(result) == 30
    assert (abs(result['x'] - 0.5) > 0.05).mean() > 0.3


