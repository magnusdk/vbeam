from typing import Optional, Tuple

from spekk import Module, ops

from vbeam.geometry.util import get_rotation_matrix, get_yz


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
                along = along / ops.linalg.vector_norm(along, axis="xyz")
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
            along = along / ops.linalg.vector_norm(along, axis="xyz")
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

    @staticmethod
    def from_origin_and_angles(
        origin: ops.array,
        *,
        azimuth: float,
        elevation: float,
    ) -> "Plane":
        basis_x, basis_y, normal = get_rotation_matrix(
            azimuth=azimuth, elevation=elevation
        )
        return Plane(origin, basis_x, basis_y, normal)

    @staticmethod
    def from_origin_and_normal(
        origin: ops.array, normal: ops.array, *, normal_is_normalized: bool = False
    ) -> "Plane":
        if not normal_is_normalized:
            normal = normal / ops.linalg.vector_norm(normal, axis="xyz")

        # TODO: This is unstable and the basis vectors switches signs at specific angles
        normal_y, normal_z = get_yz(normal)
        basis_y = ops.stack([0, normal_z, -normal_y], axis="xyz")
        basis_x = -ops.linalg.cross(normal, basis_y, axis="xyz")
        basis_y /= ops.linalg.vector_norm(basis_y, axis="xyz")
        basis_x /= ops.linalg.vector_norm(basis_x, axis="xyz")
        return Plane(origin, basis_x, basis_y, normal)
