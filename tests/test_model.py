from pathlib import Path

from psyop.model import build_model

TEST_DATA_DIR = Path(__file__).parent / "test-data"
TEST_CSV = TEST_DATA_DIR / "trials.csv"

def test_build_model(tmpdir):
    output = tmpdir/"output.nc"
    build_model(
        TEST_CSV, 
        target="loss", 
        output=output, 
        exclude="trial_id",
    )
    assert output.exists()

