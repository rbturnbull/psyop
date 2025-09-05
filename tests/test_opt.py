import pandas as pd

from psyop.opt import suggest_candidates, find_optimal

from .test_model import TEST_DATA_DIR
TEST_XARRAY = TEST_DATA_DIR/"trials.nc"

def test_suggest_candidates(tmpdir):
    output_path = tmpdir/"output.csv"
    suggest_candidates(TEST_XARRAY, output_path=output_path)
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 12


def test_find_optimal(tmpdir):
    output_path = tmpdir/"output.csv"
    find_optimal(TEST_XARRAY, output_path=output_path)
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 10