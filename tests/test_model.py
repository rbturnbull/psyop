from pathlib import Path

from psyop.model import run_model

TEST_DATA_DIR = Path(__file__).parent / "test-data"
TEST_CSV = TEST_DATA_DIR / "trials.csv"

def test_run_model(tmpdir):
    output = tmpdir/"output.nc"
    run_model(TEST_CSV, target_column="loss", output=output, exclude_columns="trial_id")
    assert output.exists()

