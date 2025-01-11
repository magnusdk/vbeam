import numpy.testing as t
from hypothesis import given
from spekk import ops
from vbeam_test_helpers.generators.geometry import (
    angles,
    directions,
    orientations,
    points,
)

from vbeam.geometry import Direction, Orientation


@given(direction=directions())
def test_direction_normalized_vector(direction: Direction):
    """Assert that rotation operations are performed in the expected order, according
    to ultrasound conventions: xz-plane first (azimuth), then zy-plane (elevation)."""
    expected_normalized_vector = ops.stack(
        [
            ops.sin(direction.azimuth) * ops.cos(direction.elevation),
            ops.sin(direction.elevation),
            ops.cos(direction.azimuth) * ops.cos(direction.elevation),
        ],
        axis="xyz",
    )
    t.assert_allclose(
        direction.normalized_vector, expected_normalized_vector, atol=1e-8
    )


@given(direction=directions())
def test_direction_normalized_vector_and_from_array(direction: Direction):
    """Assert that converting back and forth between a Direction and the corresponding
    normalized vector produces the same Direction."""
    direction2 = Direction.from_array(direction.normalized_vector)
    t.assert_allclose(
        direction.normalized_vector,
        direction2.normalized_vector,
        atol=1e-8,
    )


@given(orientation=orientations(), point=points())
def test_rotate_and_rotate_inverse(orientation: Orientation, point: ops.array):
    "Assert that rotating and un-rotating a point gives the same point."
    point2 = orientation.rotate_inverse(orientation.rotate(point))
    t.assert_allclose(point, point2, atol=1e-8)


@given(direction=directions(), roll=angles())
def test_orientation_from_direction_and_roll(direction: Direction, roll: float):
    """Assert that an Orientation object created from a direction and a roll angle has
    the same direction (normal) as the original direction."""
    orientation = Orientation.from_direction_and_roll(direction, roll)
    # Sanity check
    assert orientation.roll == roll
    # NOTE: roll rotates the xy-plane, so it's not trivial that the resulting direction
    # is the same as the original one (it should be, unless there is a bug).
    t.assert_allclose(
        direction.normalized_vector,
        orientation.direction.normalized_vector,
        atol=1e-8,
    )
