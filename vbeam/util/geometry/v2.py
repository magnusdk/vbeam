"""Differential 2D curves and relevant functions.

It's all in this one module because it's all so closely related."""

from abc import abstractmethod
from typing import Optional, Tuple, Union

from fastmath import Module
from spekk import ops

####
# Functions for calculating intersections.
# Note, not all combinations of curves are supported.


def intersect_line_line(
    line1: "Line", line2: "Line"
) -> Tuple[int, Tuple[ops.array, ops.array]]:
    direction_det = ops.linalg.det(
        ops.stack([line1.direction, line2.direction], axis="xyz2"),
        axes=["xyz", "xyz2"],
    )
    anchor_det = ops.linalg.det(
        ops.stack([line2.anchor - line1.anchor, line2.direction], axis="xyz2"),
        axes=["xyz", "xyz2"],
    )
    num_intersections = ops.where(direction_det == 0, 0, 1)
    return num_intersections, (
        line1.anchor + anchor_det / direction_det * line1.direction
    )


def intersect_circle_line(
    circle: "Circle", line: "Line"
) -> Tuple[int, Tuple[ops.array, ops.array]]:
    # http://paulbourke.net/geometry/circlesphere/
    a = ops.sum(line.direction**2)
    b = 2 * ops.sum((line.direction * (line.anchor - circle.center)))
    c = (
        ops.sum(circle.center**2)
        + ops.sum(line.anchor**2)
        - 2 * ops.sum(circle.center * line.anchor)
        - circle.radius**2
    )

    discriminant = b**2 - 4 * a * c + 0j
    u1 = (-b + ops.sqrt(discriminant)) / (2 * a)
    u2 = (-b - ops.sqrt(discriminant)) / (2 * a)

    num_intersections = 0
    num_intersections = ops.where(discriminant == 0, 1, num_intersections)
    num_intersections = ops.where(discriminant > 0, 2, num_intersections)
    return num_intersections, ops.array([line(u1), line(u2)]).T


####
# Some helper functions


def distance(point1: ops.array, point2: Optional[ops.array] = None, axis: int = "xyz"):
    diff = point1 if point2 is None else point2 - point1
    return ops.sqrt(ops.sum(diff**2, axis=axis))


def rotate(point: ops.array, theta: float, phi: float) -> ops.array:
    # Define the rotation matrices
    rotation_matrix_theta = ops.array(
        [
            [ops.cos(theta), 0, ops.sin(theta)],
            [0, 1, 0],
            [-ops.sin(theta), 0, ops.cos(theta)],
        ]
    )
    rotation_matrix_phi = ops.array(
        [
            [ops.cos(phi), -ops.sin(phi), 0],
            [ops.sin(phi), ops.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    # Rotate the point
    return rotation_matrix_theta @ rotation_matrix_phi @ point


####
# Curve classes


class Curve(Module):
    """A 2D curve. See method docstrings for details."""

    @abstractmethod
    def __call__(self, t: Union[float, ops.array]) -> Tuple[float, float]:
        """Evaluate the curve at a given parameter value.

        For example, evaluating a circle at t=0 gives the point at angle 0, t=pi/2 gives
        the point at 90 degrees angle, etc.

        Differentiating this, e.g. via jax.jvp, gives the tangent of the curve at a
        given point."""

    @abstractmethod
    def signed_distance(self, point: ops.array) -> ops.array:
        """Return the signed distance from a point to the nearest point on the curve.

        The sign of the returned value gives information about which side of the curve
        the point is on. The sign is positive on the right-hand side and negative on the
        left-hand side of the curve. For a circle, the distance is positive on the
        outside, and negative on the inside."""

    @abstractmethod
    def intersect(self, other: "Curve") -> Tuple[int, Tuple[ops.array, ops.array]]:
        """Return the number of intersections between this curve and another curve and
        the (potentially none) intersection points."""


class Line(Curve):
    anchor: ops.array
    direction: ops.array

    def __post_init__(self):
        # Normalize direction vector (tangent)
        self.direction = self.direction / distance(self.direction)

    def __call__(self, t: Union[float, ops.array]) -> Tuple[float, float]:
        if ops.is_ndarray(t) and t.ndim != 0:
            points = ops.moveaxis(self.anchor + t[..., None] * self.direction, -1, 0)
        else:
            points = self.anchor + t * self.direction
        return points

    def signed_distance(self, point: ops.array) -> float:
        vector_to_point = point - self.anchor
        perpendicular_vector = ops.array([-1, 1], ["xyz"]) * ops.flip(
            self.direction,
            axis="xyz",
        )
        signed_distance = ops.sum(vector_to_point * perpendicular_vector, axis="xyz")
        return signed_distance

    @staticmethod
    def passing_through(point1: ops.array, point2: ops.array) -> "Line":
        return Line(point1, point2 - point1)

    @staticmethod
    def with_angle(anchor: ops.array, angle: float) -> "Line":
        return Line(anchor, ops.stack([ops.cos(angle), ops.sin(angle)], axis="xyz"))

    def intersect(self, other: Curve) -> Tuple[int, Tuple[ops.array, ops.array]]:
        if isinstance(other, Line):
            return intersect_line_line(self, other)
        else:
            raise NotImplementedError

    @property
    def angle(self) -> float:
        return ops.atan2(self.direction[1], self.direction[0])

    @property
    def normal(self) -> ops.array:
        direction = ops.moveaxis(self.direction, "xyz", 0)
        return ops.stack([-direction[1], direction[0]], axis="xyz")


class Circle(Curve):
    center: ops.array
    radius: float

    def __call__(self, t: float) -> Tuple[float, float]:
        center = self.center
        if ops.is_ndarray(t) and t.ndim != 0:
            center = self.center[:, None]
        return ops.array([ops.cos(t), ops.sin(t)]) * self.radius + center

    def signed_distance(self, point: ops.array) -> ops.array:
        return distance(point, self.center) - self.radius

    def intersect(self, other: Curve) -> Tuple[int, Tuple[ops.array, ops.array]]:
        if isinstance(other, Line):
            return intersect_circle_line(self, other)
        else:
            raise NotImplementedError


class Ellipse(Curve):
    """An ellipse defined by two focus points and the summed distance from each to
    points on the ellipse.

    In other words: if p is a point on the ellipse, then d = distance(p, f1) + distance(p, f2)

    An ellipse is just a transformed unit circle, and that's how we implement Curve
    behavior for ellipses. As a consequence, the signed distance function is not yet
    implemented because transforming space also transforms the distance function. We
    should also not depend on the derivative of the __call__ method to compute a
    normalized tangent."""

    f1: ops.array  # Focus point 1
    f2: ops.array  # Focus point 2
    d: float  # If p is a point on the ellipse, then d = distance(p, f1) + distance(p, f2)

    def __call__(self, t: ops.array) -> ops.array:
        # An ellipse is just a transformed circle
        circle = Circle(ops.array([0, 0]), 1)
        circle_transform = self._circle_transform
        if ops.is_ndarray(t) and t.ndim != 0:
            circle_transform = ops.vmap(circle_transform, [1])
        return circle_transform(circle(t)).T

    def signed_distance(self, point: ops.array) -> ops.array:
        # Not implemented yet. Need to think about how to normalize the distance since
        # the space is transformed.
        raise NotImplementedError

    def intersect(self, other: Curve) -> Tuple[int, Tuple[ops.array, ops.array]]:
        if isinstance(other, Line):
            other = Line.passing_through(
                self._circle_undo_transform(other(0)),
                self._circle_undo_transform(other(1)),
            )
            circle = Circle(ops.array([0, 0]), 1)
            num_intersections, intersections = intersect_circle_line(circle, other)
            i1, i2 = intersections.real.T
            return (
                num_intersections,
                ops.array([self._circle_transform(i1), self._circle_transform(i2)]).T,
            )
        else:
            raise NotImplementedError

    def _circle_rotation(self) -> float:
        return ops.atan2(self.f2[1] - self.f1[1], self.f2[0] - self.f1[0])

    def _circle_scale(self) -> ops.array:
        a = self.d / 2
        b = ops.sqrt(a**2 - distance(self.f2, self.f1) ** 2 / 4)
        return ops.array([a, b])

    def _circle_translation(self) -> ops.array:
        return (self.f1 + self.f2) / 2

    def _circle_transform(self, point: ops.array) -> ops.array:
        point *= self._circle_scale()
        point = rotate(point, self._circle_rotation())
        point += self._circle_translation()
        return point

    def _circle_undo_transform(self, point: ops.array) -> ops.array:
        point -= self._circle_translation()
        point = rotate(point, -self._circle_rotation())
        point /= self._circle_scale()
        return point
