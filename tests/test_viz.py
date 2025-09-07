from psyop.viz import plot2d, plot1d

from .test_opt import TEST_XARRAY


def test_plot2d(tmpdir):
    output_path = tmpdir/"output.png"
    plot2d(TEST_XARRAY, output=output_path)
    assert output_path.exists()


def test_plot1d(tmpdir):
    output_path = tmpdir/"output.png"
    plot1d(TEST_XARRAY, output=output_path)
    assert output_path.exists()
