import numpy.testing as t
from hypothesis import assume, given
from hypothesis import strategies as st
from spekk import ops
from vbeam_test_helpers.generators.geometry import directions, points, rotations

from vbeam.geometry import Direction, Plane, Rotation


@given(
    origin=points(),
    orientation=rotations(),
    point=points(),
    along=st.one_of(directions(), st.none()),
)
def test_plane_coordinate_transformations(
    origin: ops.array,
    orientation: Rotation,
    point: ops.array,
    along: Direction,
):
    """Assert that transforming a point to plane coordinates and back to global
    coordinates results in the original point."""
    plane = Plane(origin, orientation)
    # Ensure that the point lies on the plane.
    projected_point = plane.project(point, along=along)
    # Assume the point isn't projected really far off, like when the plane normal and
    # the projection direction are (close to) perpendicular.
    assume(ops.linalg.vector_norm(projected_point, axis="xyz") < 1e6)

    # Assert that going back and forth between plane coordinates produces the same point.
    x, y = plane.to_plane_coordinates(projected_point, is_already_projected=True)
    reconstructed_point = plane.from_plane_coordinates(x, y)
    t.assert_allclose(reconstructed_point, projected_point, atol=1e-8)


@given(
    origin=points(),
    orientation=rotations(),
    point=points(),
    along=st.one_of(directions(), st.none()),
)
def test_distance_equals_projection_distance(
    origin: ops.array,
    orientation: Rotation,
    point: ops.array,
    along: Direction,
):
    """Assert that the distance to the plane equals the distance between the original
    point and the projected point."""
    plane = Plane(origin, orientation)
    projected_distance = ops.abs(plane.signed_distance(point, along=along))
    projected_point = plane.project(point, along=along)
    true_distance = ops.linalg.vector_norm(projected_point - point, axis="xyz")
    t.assert_allclose(projected_distance, true_distance, atol=1e-8)
