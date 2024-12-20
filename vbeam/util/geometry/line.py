from spekk import Module, ops


class Line(Module):
    """A line as defined by a*x + b*z + c = 0.

    Note that only 2D lines are supported but all point-parameters and returned points
    are 3D (has x-, y-, and z-components).
    """

    a: float
    b: float
    c: float

    @staticmethod
    def passing_through(point1: ops.array, point2: ops.array) -> "Line":
        "Construct a line that passes through both point1 and point2."
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        a = z1 - z2
        b = x2 - x1
        c = x1 * z2 - x2 * z1
        return Line(a, b, c)

    @staticmethod
    def from_anchor_and_angle(anchor: ops.array, angle: float) -> "Line":
        "Construct a line that passes through the anchor and has the given angle."
        return Line.passing_through(
            anchor,
            anchor
            + ops.stack(
                [
                    ops.cos(angle),
                    ops.zeros(shape=angle.shape, dims=angle.dims, dtype=angle.dtype),
                    ops.sin(angle),
                ],
                axis="xyz",
            ),
        )

    def intersection(l1: "Line", l2: "Line") -> ops.array:
        "Return the point where the lines l1 and l2 intersect."
        x = (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
        y = ops.zeros_like(x)
        z = (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
        return ops.stack([x, y, z], axis="xyz")

    @property
    def angle(self):
        "The angle of the line."
        return ops.atan2(self.b, self.a)

    def signed_distance(self, point: ops.array, axis="xyz") -> float:
        """Return the signed distance between the point and the nearest point on the
        line.

        The sign of the distance will be positive on the left-hand side and negative on
        the right-hand side of the line.

        The y-component of the point is ignored.

        >>> vertical_line = Line.passing_through(np.array([0,0,0]), np.array([0,0,1]))
        >>> vertical_line.signed_distance(np.array([0,0,5]))
        0.0
        >>> vertical_line.signed_distance(np.array([-1,0,5]))  # The left-hand side
        1.0
        >>> vertical_line.signed_distance(np.array([1,0,5]))  # The right-hand side
        -1.0
        """
        norm = ops.sqrt(self.a**2 + self.b**2)
        return (
            self.a * ops.take(point, 0, axis=axis)
            + self.b * ops.take(point, 2, axis=axis)
            + self.c
        ) / norm
