from typing import Optional, Tuple

from spekk import Module, ops

from vbeam.geometry.rotation import Direction, Rotation


class Plane(Module):
    """An oriented Euclidian 2D plane.

    Attributes:
        origin (ops.array): The center of the plane. In local plane coordinates, the
            origin is at (0, 0).
        orientation (Rotation): The 3D orientation of the plane, including azimuth,
            elevation, and roll. See :class:`vbeam.geometry.rotation.Rotation` for more
            information.

    A plane is oriented in the order of:
    - Azimuth: Rotation around the y-axis.
    - Elevation: Rotation around the x-axis.
    - Roll: Rotation around the z-axis.
    """

    origin: ops.array
    orientation: Rotation

    @property
    def normal(self) -> Direction:
        """The normal direction of the plane.

        You can use `plane.normal.normalized_vector` to get the normal vector."""
        return self.orientation.direction

    def signed_distance(
        self, point: ops.array, *, along: Optional[Direction] = None
    ) -> float:
        """Return the distance from a `point` to the plane, optionally `along` a
        direction. If `along` is not given, return the distance to the closest point
        on the plane."""
        normal_vector = self.normal.normalized_vector
        distance = ops.linalg.vecdot(point - self.origin, normal_vector, axis="xyz")
        if along is not None:
            alignment = ops.vecdot(along.normalized_vector, normal_vector, axis="xyz")
            distance /= alignment
        return distance

    def project(self, point: ops.array, *, along: Optional[Direction] = None):
        """Project the `point` onto the plane, optionally `along` a direction,
        returning a 3D point. If `along` is not given, return the closest point on
        the plane."""
        distance = self.signed_distance(point, along=along)
        if along is None:
            along = self.orientation
        return point - along.normalized_vector * distance

    def to_plane_coordinates(
        self,
        point: ops.array,
        *,
        along: Optional[Direction] = None,
        is_already_projected: bool = False,
    ) -> Tuple[float, float]:
        """Project the `point` onto the plane, optionally `along` a direction,
        returning a tuple of x and y, representing the 2D point in the coordinates of
        the plane.

        Use `is_already_projected=True` if the point is already projected onto the
        plane."""
        if not is_already_projected:
            point = self.project(point, along=along)
        # Subtract origin and un-rotate the orientation.
        point -= self.origin
        point = self.orientation.rotate_inverse(point)
        x = ops.take(point, 0, axis="xyz")
        y = ops.take(point, 1, axis="xyz")
        return x, y

    def from_plane_coordinates(self, x: float, y: float) -> ops.array:
        """Return a 3D point from the 2D point at (`x`, `y`) in local plane
        coordinates.

        For example, `plane.from_plane_coordinates(0, 0)` would return the origin of
        the plane in 3D."""
        point = ops.stack([x, y, ops.zeros_like(x)], axis="xyz")
        point = self.orientation.rotate(point)
        point += self.origin
        return point

    def orient(self, direction: Direction, roll: Optional[float] = None) -> "Plane":
        """Rotate the plane such that it points in the given direction. By default,
        roll is left unchanged."""
        if roll is None:
            roll = self.orientation.roll
        return Plane(self.origin, Rotation.from_direction_and_roll(direction, roll))
