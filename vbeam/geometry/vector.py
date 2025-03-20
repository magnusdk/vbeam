from spekk import Module, ops

from vbeam.geometry.coordinate_systems import rotate_xz, rotate_yz


class Vector(Module):
    magnitude: float
    direction: ops.array

    def to_array(self) -> ops.array:
        return self.magnitude * self.direction

    @staticmethod
    def from_array(v: ops.array) -> "Vector":
        magnitude = ops.linalg.vector_norm(v, axis="xyz")
        v = v / magnitude
        return Vector(magnitude, v)

    @staticmethod
    def from_angles(depth: float, *, azimuth: float, elevation: float) -> "Vector":
        x, y, z = 0.0, 0.0, 1.0
        y, z = rotate_yz(y, z, elevation)
        x, z = rotate_xz(x, z, azimuth)
        direction = ops.stack([x, y, z], axis="xyz")
        return Vector(depth, direction)


class VectorWithInfiniteMagnitude(Vector):
    direction: ops.array

    def to_array(self) -> ops.array:
        raise ValueError(
            "Can not convert a vector with infinite magnitude to an array."
        )

    @staticmethod
    def from_angles(*, azimuth: float, elevation: float) -> "Vector":
        x, y, z = 0.0, 0.0, 1.0
        y, z = rotate_yz(y, z, elevation)
        x, z = rotate_xz(x, z, azimuth)
        direction = ops.stack([x, y, z], axis="xyz")
        return Vector(direction)
