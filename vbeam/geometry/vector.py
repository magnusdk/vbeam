from spekk import Module, ops

from vbeam.geometry.orientation import Direction


class Vector(Module):
    """A vector defined by its `magnitude` and `direction`.

    Attributes:
        magnitude (float): The magnitude, or length, of the vector.
        direction (Direction): The direction of the vector.

    The magnitude may be infinite, in which case the vector is effectively just a
    direction.
    """

    magnitude: float
    direction: Direction

    @property
    def azimuth(self):
        return self.direction.azimuth

    @property
    def elevation(self):
        return self.direction.elevation

    def to_array(self) -> ops.array:
        return self.direction.normalized_vector * self.magnitude

    def __sub__(self, other: ops.array) -> "Vector":
        # Return self if magnitude is infinite, because subtracting a finite number
        # from infinity is a no-op.
        if not isinstance(other, ops.array):
            raise TypeError()

        diff = Vector.from_array(self.to_array() - other)
        magnitude = ops.where(
            ops.isfinite(self.magnitude),
            diff.magnitude,
            self.magnitude,
        )
        azimuth = ops.where(
            ops.isfinite(self.magnitude),
            diff.direction.azimuth,
            self.direction.azimuth,
        )
        elevation = ops.where(
            ops.isfinite(self.magnitude),
            diff.direction.elevation,
            self.direction.elevation,
        )
        return Vector(magnitude, Direction(azimuth, elevation))

    @staticmethod
    def from_array(v: ops.array) -> "Vector":
        return Vector(ops.linalg.vector_norm(v, axis="xyz"), Direction.from_array(v))
