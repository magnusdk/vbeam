import numpy.testing as t
from hypothesis import given
from spekk import ops
from vbeam_test_helpers.generators.geometry import angles, directions, points, rotations

from vbeam.geometry import Direction, Rotation


@given(direction=directions())
def test_direction_normalized_vector(direction: Direction):
    """Assert that rotations are performed in the expected order, according to
    ultrasound conventions: xz-plane first (azimuth), then zy-plane (elevation)."""
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


@given(rotation=rotations(), point=points())
def test_rotate_and_rotate_inverse(rotation: Rotation, point: ops.array):
    "Assert that rotating and un-rotating a point gives the same point."
    point2 = rotation.rotate_inverse(rotation.rotate(point))
    t.assert_allclose(point, point2, atol=1e-8)


@given(direction=directions(), roll=angles())
def test_rotation_from_direction_and_roll(direction: Direction, roll: float):
    """Assert that a Rotation object created from a direction and a roll angle has the
    same direction (normal) as the original direction."""
    rotation = Rotation.from_direction_and_roll(direction, roll)
    # Sanity check
    assert rotation.roll == roll
    # NOTE: roll rotates the xy-plane, so it's not trivial that the resulting direction
    # is the same as the original one (it should be, unless there is a bug).
    t.assert_allclose(
        direction.normalized_vector,
        rotation.direction.normalized_vector,
        atol=1e-8,
    )
