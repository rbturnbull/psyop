import pandas as pd

from psyop.opt import suggest_candidates, find_optimal

from .test_model import TEST_DATA_DIR
TEST_XARRAY = TEST_DATA_DIR/"trials.nc"

def test_suggest_candidates(tmpdir):
    output = tmpdir/"output.csv"
    suggest_candidates(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 12


def test_find_optimal(tmpdir):
    output = tmpdir/"output.csv"
    find_optimal(TEST_XARRAY, output=output)
    assert output.exists()
    df = pd.read_csv(output)
    assert len(df) == 10