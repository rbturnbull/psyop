import pandas as pd

from psyop import suggest, optimal

from .test_model import TEST_DATA_DIR
TEST_XARRAY = TEST_DATA_DIR/"trials.nc"

def test_suggest(tmpdir):
    output = tmpdir/"output.csv"
    suggest(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 12


def test_optimal(tmpdir):
    output = tmpdir/"output.csv"
    optimal(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 10