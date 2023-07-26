import numpy
from hypothesis import given
from vbeam_test_helpers.generators import sector_scans

from vbeam.scan import SectorScan


def _get_cartesian_bounds_brute_force(scan: SectorScan):
    """Just generate the points from the scan and calculate the minimum and maximum
    x-, y- (if 3D), and z-values."""
    points = scan.get_points()
    points -= scan.apex
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    if scan.is_3d:
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        return (min_x, max_x, min_y, max_y, min_z, max_z)
    if scan.is_2d:
        return (min_x, max_x, min_z, max_z)


@given(sector_scans([200, 2]))
def test_cartesian_bounds_2D(scan: SectorScan):
    bounds_brute_force = _get_cartesian_bounds_brute_force(scan)
    bounds = scan.cartesian_bounds
    numpy.testing.assert_allclose(bounds_brute_force, bounds, rtol=1e-4, atol=1e-12)


@given(sector_scans([10, 10, 10]))
def test_cartesian_bounds_3D(scan: SectorScan):
    bounds_brute_force = _get_cartesian_bounds_brute_force(scan)
    bounds = scan.cartesian_bounds
    numpy.testing.assert_allclose(bounds_brute_force, bounds, rtol=1e-4, atol=1e-12)
