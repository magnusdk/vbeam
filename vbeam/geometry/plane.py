from typing import Optional, Tuple

from spekk import Module, ops

from vbeam.geometry.util import get_rotation_matrix, normalize_vector


class Plane(Module):
    """An oriented Euclidian 2D plane.

    Attributes:
        origin (ops.array): The center of the plane. In local plane coordinates, the
            origin is at (0, 0).
        orientation (Orientation): The 3D orientation of the plane, including azimuth,
            elevation, and roll. See :class:`vbeam.geometry.orientation.Orientation`
            for more information.

    A plane is oriented in the order of:
    - Azimuth: Rotation around the y-axis.
    - Elevation: Rotation around the x-axis.
    - Roll: Rotation around the z-axis.
    """

    origin: ops.array
    basis_x: ops.array
    basis_y: ops.array
    normal: ops.array

    def signed_distance(
        self,
        point: ops.array,
        *,
        along: Optional[ops.array] = None,
        along_is_normalized: bool = False,
    ) -> float:
        """Return the distance from a `point` to the plane, optionally `along` a
        direction. If `along` is not given, return the distance to the closest point
        on the plane."""
        distance = ops.linalg.vecdot(point - self.origin, self.normal, axis="xyz")
        if along is not None:
            if not along_is_normalized:
                along = normalize_vector(along)
            alignment = ops.vecdot(along, self.normal, axis="xyz")
            distance /= alignment
        return distance

    def project(
        self,
        point: ops.array,
        *,
        along: Optional[ops.array] = None,
        along_is_normalized: bool = False,
    ):
        """Project the `point` onto the plane, optionally `along` a direction,
        returning a 3D point. If `along` is not given, return the closest point on
        the plane."""
        if along is not None and not along_is_normalized:
            along = normalize_vector(along)
        distance = self.signed_distance(point, along=along, along_is_normalized=True)
        if along is None:
            along = self.normal
        return point - along * distance

    def to_plane_coordinates(self, point: ops.array) -> Tuple[float, float]:
        """Project the `point` onto the plane, returning a tuple of x and y,
        representing the 2D point in the coordinates of the plane."""
        point = point - self.origin
        x = ops.vecdot(point, self.basis_x, axis="xyz")
        y = ops.vecdot(point, self.basis_y, axis="xyz")
        return x, y

    def from_plane_coordinates(self, x: float, y: float) -> ops.array:
        """Return a 3D point from the 2D point at (`x`, `y`) in local plane
        coordinates.

        For example, `plane.from_plane_coordinates(0, 0)` would return the origin of
        the plane in 3D."""
        return x * self.basis_x + y * self.basis_y + self.origin

    def orient(
        self, normal: ops.array, *, normal_is_normalized: bool = False
    ) -> "Plane":
        """Return an oriented copy of the plane such that it has the given normal.

        NOTE: Orienting by a normal that points in the opposite direction of the
        current normal results in NaNs."""
        if not normal_is_normalized:
            normal = normalize_vector(normal)

        scaled_rotation_axis = ops.linalg.cross(self.normal, normal, axis="xyz")
        cos_angle = ops.linalg.vecdot(self.normal, normal, axis="xyz")

        basis_x = _rotate_using_rodrigues(self.basis_x, scaled_rotation_axis, cos_angle)
        basis_y = _rotate_using_rodrigues(self.basis_y, scaled_rotation_axis, cos_angle)
        return Plane(self.origin, basis_x, basis_y, normal)

    @staticmethod
    def from_origin_and_angles(
        origin: ops.array | None = None,
        *,
        azimuth: float = 0,
        elevation: float = 0,
    ) -> "Plane":
        if origin is None:
            origin = ops.array([0, 0, 0], ["xyz"])
        basis_x, basis_y, normal = get_rotation_matrix(
            azimuth=azimuth, elevation=elevation
        )
        return Plane(origin, basis_x, basis_y, normal)


def _rotate_using_rodrigues(
    vector: ops.array,
    scaled_rotation_axis: ops.array,
    cos_angle: ops.array,
) -> ops.array:
    "Rotate a vector around a given axis using Rodrigues' rotation formula."
    cross1 = ops.linalg.cross(scaled_rotation_axis, vector, axis="xyz")
    cross2 = ops.linalg.cross(scaled_rotation_axis, cross1, axis="xyz")
    return vector + cross1 + cross2 / (1 + cos_angle)
