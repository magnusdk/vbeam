from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass(("a", "b", "c"))
class Line:
    """A line as defined by a*x + b*z + c = 0.

    Note that only 2D lines are supported but all point-parameters and returned points
    are 3D (has x-, y-, and z-components).
    """

    a: float
    b: float
    c: float

    @staticmethod
    def passing_through(point1: np.ndarray, point2: np.ndarray) -> "Line":
        "Construct a line that passes through both point1 and point2."
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        a = z1 - z2
        b = x2 - x1
        c = x1 * z2 - x2 * z1
        return Line(a, b, c)

    @staticmethod
    def from_anchor_and_angle(anchor: np.ndarray, angle: float) -> "Line":
        "Construct a line that passes through the anchor and has the given angle."
        return Line.passing_through(
            anchor, anchor + np.array([np.cos(angle), 0, np.sin(angle)])
        )

    def intersection(l1: "Line", l2: "Line") -> np.ndarray:
        "Return the point where the lines l1 and l2 intersect."
        x = (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
        z = (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
        return np.array([x, 0, z])

    @property
    def angle(self):
        "The angle of the line."
        return np.arctan2(self.b, self.a)

    def signed_distance(self, point: np.ndarray) -> float:
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
        norm = np.sqrt(self.a**2 + self.b**2)
        return (self.a * point[..., 0] + self.b * point[..., 2] + self.c) / norm
