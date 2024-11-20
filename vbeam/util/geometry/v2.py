"""Differential 2D curves and relevant functions.

It's all in this one module because it's all so closely related."""

from abc import abstractmethod
from typing import Optional, Tuple, Union

from fastmath import Array, Module

from vbeam.fastmath import numpy as api

####
# Functions for calculating intersections.
# Note, not all combinations of curves are supported.


def intersect_line_line(
    line1: "Line", line2: "Line"
) -> Tuple[int, Tuple[Array, Array]]:
    num_intersections = api.where(api.cross(line1.direction, line2.direction) == 0, 0, 1)
    return num_intersections, (
        line1.anchor
        + api.cross(line2.anchor - line1.anchor, line2.direction)
        / api.cross(line1.direction, line2.direction)[..., None]
        * line1.direction
    )


def intersect_circle_line(
    circle: "Circle", line: "Line"
) -> Tuple[int, Tuple[Array, Array]]:
    # http://paulbourke.net/geometry/circlesphere/
    a = api.sum(line.direction**2)
    b = 2 * api.sum((line.direction * (line.anchor - circle.center)))
    c = (
        api.sum(circle.center**2)
        + api.sum(line.anchor**2)
        - 2 * api.sum(circle.center * line.anchor)
        - circle.radius**2
    )

    discriminant = b**2 - 4 * a * c + 0j
    u1 = (-b + api.sqrt(discriminant)) / (2 * a)
    u2 = (-b - api.sqrt(discriminant)) / (2 * a)

    num_intersections = 0
    num_intersections = api.where(discriminant == 0, 1, num_intersections)
    num_intersections = api.where(discriminant > 0, 2, num_intersections)
    return num_intersections, api.array([line(u1), line(u2)]).T


####
# Some helper functions


def distance(
    point1: Array, point2: Optional[Array] = None, axis: int = -1
):
    diff = point1 if point2 is None else point2 - point1
    return api.sqrt(api.sum(diff**2, axis=axis))


def rotate(point: Array, theta: float, phi: float) -> Array:
    # Define the rotation matrices
    rotation_matrix_theta = api.array(
        [
            [api.cos(theta), 0, api.sin(theta)],
            [0, 1, 0],
            [-api.sin(theta), 0, api.cos(theta)],
        ]
    )
    rotation_matrix_phi = api.array(
        [
            [api.cos(phi), -api.sin(phi), 0],
            [api.sin(phi), api.cos(phi), 0],
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
    def __call__(self, t: Union[float, Array]) -> Tuple[float, float]:
        """Evaluate the curve at a given parameter value.

        For example, evaluating a circle at t=0 gives the point at angle 0, t=pi/2 gives
        the point at 90 degrees angle, etc.

        Differentiating this, e.g. via jax.jvp, gives the tangent of the curve at a
        given point."""

    @abstractmethod
    def signed_distance(self, point: Array) -> Array:
        """Return the signed distance from a point to the nearest point on the curve.

        The sign of the returned value gives information about which side of the curve
        the point is on. The sign is positive on the right-hand side and negative on the
        left-hand side of the curve. For a circle, the distance is positive on the
        outside, and negative on the inside."""

    @abstractmethod
    def intersect(
        self, other: "Curve"
    ) -> Tuple[int, Tuple[Array, Array]]:
        """Return the number of intersections between this curve and another curve and
        the (potentially none) intersection points."""


class Line(Curve):
    anchor: Array
    direction: Array

    def __post_init__(self):
        # Normalize direction vector (tangent)
        self.direction = self.direction / distance(self.direction)

    def __call__(self, t: Union[float, Array]) -> Tuple[float, float]:
        if api.is_ndarray(t) and t.ndim != 0:
            points = api.moveaxis(self.anchor + t[..., None] * self.direction, -1, 0)
        else:
            points = self.anchor + t * self.direction
        return points

    def signed_distance(self, point: Array) -> float:
        return api.cross(point - self.anchor, self.direction)

    @staticmethod
    def passing_through(point1: Array, point2: Array) -> "Line":
        return Line(point1, point2 - point1)

    @staticmethod
    def with_angle(anchor: Array, angle: float) -> "Line":
        return Line(anchor, api.array([api.cos(angle), api.sin(angle)]))

    def intersect(
        self, other: Curve
    ) -> Tuple[int, Tuple[Array, Array]]:
        if isinstance(other, Line):
            return intersect_line_line(self, other)
        else:
            raise NotImplementedError

    @property
    def angle(self) -> float:
        return api.arctan2(self.direction[1], self.direction[0])

    @property
    def normal(self) -> Array:
        return api.array([-self.direction[1], self.direction[0]])


class Circle(Curve):
    center: Array
    radius: float

    def __call__(self, t: float) -> Tuple[float, float]:
        center = self.center
        if api.is_ndarray(t) and t.ndim != 0:
            center = self.center[:, None]
        return api.array([api.cos(t), api.sin(t)]) * self.radius + center

    def signed_distance(self, point: Array) -> Array:
        return distance(point, self.center) - self.radius

    def intersect(
        self, other: Curve
    ) -> Tuple[int, Tuple[Array, Array]]:
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

    f1: Array  # Focus point 1
    f2: Array  # Focus point 2
    d: float  # If p is a point on the ellipse, then d = distance(p, f1) + distance(p, f2)

    def __call__(self, t: Array) -> Array:
        # An ellipse is just a transformed circle
        circle = Circle(api.array([0, 0]), 1)
        circle_transform = self._circle_transform
        if api.is_ndarray(t) and t.ndim != 0:
            circle_transform = api.vmap(circle_transform, [1])
        return circle_transform(circle(t)).T

    def signed_distance(self, point: Array) -> Array:
        # Not implemented yet. Need to think about how to normalize the distance since
        # the space is transformed.
        raise NotImplementedError

    def intersect(
        self, other: Curve
    ) -> Tuple[int, Tuple[Array, Array]]:
        if isinstance(other, Line):
            other = Line.passing_through(
                self._circle_undo_transform(other(0)),
                self._circle_undo_transform(other(1)),
            )
            circle = Circle(api.array([0, 0]), 1)
            num_intersections, intersections = intersect_circle_line(circle, other)
            i1, i2 = intersections.real.T
            return (
                num_intersections,
                api.array([self._circle_transform(i1), self._circle_transform(i2)]).T,
            )
        else:
            raise NotImplementedError

    def _circle_rotation(self) -> float:
        return api.arctan2(self.f2[1] - self.f1[1], self.f2[0] - self.f1[0])

    def _circle_scale(self) -> Array:
        a = self.d / 2
        b = api.sqrt(a**2 - distance(self.f2, self.f1) ** 2 / 4)
        return api.array([a, b])

    def _circle_translation(self) -> Array:
        return (self.f1 + self.f2) / 2

    def _circle_transform(self, point: Array) -> Array:
        point *= self._circle_scale()
        point = rotate(point, self._circle_rotation())
        point += self._circle_translation()
        return point

    def _circle_undo_transform(self, point: Array) -> Array:
        point -= self._circle_translation()
        point = rotate(point, -self._circle_rotation())
        point /= self._circle_scale()
        return point
