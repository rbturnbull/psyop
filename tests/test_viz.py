from psyop.viz import make_pairplot, make_partial_dependence1D

from .test_opt import TEST_XARRAY


def test_make_pairplot(tmpdir):
    output_path = tmpdir/"output.png"
    make_pairplot(TEST_XARRAY, output=output_path)
    assert output_path.exists()


def test_make_partial_dependence1D(tmpdir):
    output_path = tmpdir/"output.png"
    make_partial_dependence1D(TEST_XARRAY, output=output_path)
    assert output_path.exists()
