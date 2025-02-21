"""Test geometry utilities."""

import numpy as np
import hypothesis
import hypothesis.strategies as st
import pytest
from vbeam.util.geometry.v2 import rotate_az_el
from vbeam.util.coordinate_systems import az_el_to_cartesian


def test_rotate_az_el_input_validation():
    """Test that the function properly validates input dimensions."""
    with pytest.raises(ValueError, match="Point must be 3D"):
        rotate_az_el(point=np.array([1.0, 2.0]), azimuth=0, elevation=0)


def test_rotate_az_el_zero_rotation():
    """Test that rotating by 0 angles doesn't change the point."""
    point = np.array([1.0, 2.0, 3.0])
    rotated = rotate_az_el(point, azimuth=0, elevation=0)
    np.testing.assert_array_almost_equal(rotated, point)


def test_rotate_az_el_xz_plane_azimuth():
    """Test rotating a point in the xz-plane with azimuth.
    The point should stay in the xz-plane (y=0)."""
    point = np.array([1.0, 0.0, 1.0])  # point in xz-plane
    azimuth = np.pi / 4  # 45 degrees
    rotated = rotate_az_el(point, azimuth=azimuth, elevation=0)

    # Point should stay in xz-plane (y=0)
    np.testing.assert_almost_equal(rotated[1], 0)
    # Length should be preserved
    np.testing.assert_almost_equal(np.linalg.norm(rotated), np.linalg.norm(point))


def test_rotate_az_el_yz_plane_elevation():
    """Test rotating a point in the yz-plane with elevation.
    The point should stay in the yz-plane (x=0)."""
    point = np.array([0.0, 1.0, 1.0])  # point in yz-plane
    elevation = np.pi / 3  # 60 degrees
    rotated = rotate_az_el(point, azimuth=0, elevation=elevation)

    # Point should stay in yz-plane (x=0)
    np.testing.assert_almost_equal(rotated[0], 0)
    # Length should be preserved
    np.testing.assert_almost_equal(np.linalg.norm(rotated), np.linalg.norm(point))


def test_rotate_az_el_xz_plane_elevation():
    """Test rotating a point in the xz-plane with elevation.
    The point should move out of the xz-plane."""
    point = np.array([1.0, 0.0, 1.0])  # point in xz-plane
    elevation = np.pi / 4  # 45 degrees
    rotated = rotate_az_el(point, azimuth=0, elevation=elevation)

    # Point should move out of xz-plane (y≠0)
    assert abs(rotated[1]) > 1e-10
    # Length should be preserved
    np.testing.assert_almost_equal(np.linalg.norm(rotated), np.linalg.norm(point))


# https://github.com/magnusdk/vbeam/pull/44#issuecomment-2504705798
@pytest.mark.parametrize(
    "point,azimuth,elevation,expected",
    [
        # Point along z-axis, rotate 90° azimuth -> should go to x-axis
        (np.array([0.0, 0.0, 1.0]), np.pi / 2, 0, np.array([1.0, 0.0, 0.0])),
        # Point along x-axis, rotate 90° elevation -> should go to positive y while keeping x
        (np.array([0.0, 0.0, 1.0]), 0, np.pi / 2, np.array([0.0, 1.0, 0.0])),
        # 180° azimuth should flip x and z coordinates
        (np.array([1.0, 0.0, 0.0]), np.pi, 0, np.array([-1.0, 0.0, 0.0])),
    ],
)
def test_rotate_az_el_specific_angles(point, azimuth, elevation, expected):
    """Test specific rotation angles with known expected outcomes."""
    rotated = rotate_az_el(point, azimuth, elevation)
    np.testing.assert_array_almost_equal(rotated, expected)


def test_rotate_az_el_multiple_points():
    """Test rotating multiple points simultaneously."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T  # Shape: (3, 2)

    azimuth = np.pi / 4
    elevation = np.pi / 6

    rotated = rotate_az_el(points, azimuth, elevation)

    # Check shape is preserved
    assert rotated.shape == points.shape


@hypothesis.settings(
    max_examples=10,  # Limit test cases for shorter run-time
    deadline=1000,  # milliseconds
    # Only try to find examples that fail, no need to reproduce
    phases=[
        hypothesis.Phase.explicit,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
    ],
)
@hypothesis.given(
    azimuth=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
    elevation=st.floats(min_value=-np.pi / 2, max_value=np.pi / 2),
)
@hypothesis.example(azimuth=np.pi / 2, elevation=0)
@hypothesis.example(azimuth=0, elevation=np.pi / 2)
@hypothesis.example(azimuth=np.pi / 4, elevation=0)
@hypothesis.example(azimuth=0, elevation=np.pi / 4)
@hypothesis.example(azimuth=np.pi / 4, elevation=np.pi / 6)
def test_rotate_matches_cartesian(azimuth, elevation):
    """Test that rotate_az_el of the unit-depth vector matches az_el_to_cartesian.

    This checks consistency that the rotation matches the equations:
    https://github.com/magnusdk/vbeam/pull/44#issuecomment-2504705798

    i.e., angle=0 is the unit-depth vector [0, 0, 1]
    rotatign the unit-depth vector should be equivalent to converting azimuth-elevation-angles
    to cartesian coordinates.
    """
    unit_depth_vector = np.array([0.0, 0.0, 1.0])

    zero_angle_rotated = rotate_az_el(
        point=unit_depth_vector, azimuth=azimuth, elevation=elevation
    )
    direction_vector = az_el_to_cartesian(azimuth=azimuth, elevation=elevation)

    np.testing.assert_array_almost_equal(zero_angle_rotated, direction_vector)
