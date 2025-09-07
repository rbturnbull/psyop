from typer.testing import CliRunner
from psyop.main import app

runner = CliRunner()

from .test_model import TEST_CSV
from .test_opt import TEST_XARRAY


def test_app_model(tmpdir):
    output = tmpdir/"output.nc"
    result = runner.invoke(app, ["model", str(TEST_CSV), str(output), "--target", "loss", "--exclude", "trial_id"])
    assert result.exit_code == 0
    assert output.exists()


def test_app_suggest(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["suggest", str(TEST_XARRAY), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_app_optimal(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["optimal", str(TEST_XARRAY), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
