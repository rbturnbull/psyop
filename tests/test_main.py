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


def test_app_model_prior(tmpdir):
    output = tmpdir/"output.nc"
    result = runner.invoke(app, ["model", str(TEST_CSV), str(output), "--target", "loss", "--exclude", "trial_id", "--prior-model", str(TEST_XARRAY)])
    assert result.exit_code == 0
    assert output.exists()


def test_app_suggest(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["suggest", str(TEST_XARRAY), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_app_suggest_constraints(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["suggest", str(TEST_XARRAY), "--output", str(output), "--batch_size", "20", "--dropout", "0.1:0.2"])
    assert result.exit_code == 0
    assert output.exists()


def test_app_optimal(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["optimal", str(TEST_XARRAY), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_app_plot2d(tmpdir):
    output_path = tmpdir/"output.png"
    result = runner.invoke(app, ["plot2d", str(TEST_XARRAY), "--output", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_app_plot1d(tmpdir):
    output_path = tmpdir/"output.png"
    result = runner.invoke(app, ["plot1d", str(TEST_XARRAY), "--output", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_export(tmpdir):
    output = tmpdir/"output.csv"
    result = runner.invoke(app, ["export", str(TEST_XARRAY), str(output)])
    assert result.exit_code == 0
    assert output.exists()
    text = output.read_text('utf8')
    assert text.startswith('trial_id,batch_size,dropout,epochs,learning_rate,r_drop_alpha,loss\n1,16,')
