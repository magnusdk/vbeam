from spekk import Module, ops

from vbeam.geometry.rotation import Direction


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
        result = Vector.from_array(self.to_array() - other)
        # Return self if magnitude is infinite, because subtracting a finite number
        # from infinity is a no-op.
        return ops.where(ops.isfinite(self.magnitude), result, self)

    @staticmethod
    def from_array(v: ops.array) -> "Vector":
        return Vector(ops.linalg.vector_norm(v, axis="xyz"), Direction.from_array(v))
