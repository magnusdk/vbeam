"""Differential 2D curves and relevant functions.

It's all in this one module because it's all so closely related."""


from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass

####
# Functions for calculating intersections.
# Note, not all combinations of curves are supported.


def intersect_line_line(
    line1: "Line", line2: "Line"
) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
    num_intersections = np.where(np.cross(line1.direction, line2.direction) == 0, 0, 1)
    return num_intersections, (
        line1.anchor
        + np.cross(line2.anchor - line1.anchor, line2.direction)
        / np.cross(line1.direction, line2.direction)[..., None]
        * line1.direction
    )


def intersect_circle_line(
    circle: "Circle", line: "Line"
) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
    # http://paulbourke.net/geometry/circlesphere/
    a = np.sum(line.direction**2)
    b = 2 * np.sum((line.direction * (line.anchor - circle.center)))
    c = (
        np.sum(circle.center**2)
        + np.sum(line.anchor**2)
        - 2 * np.sum(circle.center * line.anchor)
        - circle.radius**2
    )

    discriminant = b**2 - 4 * a * c + 0j
    u1 = (-b + np.sqrt(discriminant)) / (2 * a)
    u2 = (-b - np.sqrt(discriminant)) / (2 * a)

    num_intersections = 0
    num_intersections = np.where(discriminant == 0, 1, num_intersections)
    num_intersections = np.where(discriminant > 0, 2, num_intersections)
    return num_intersections, np.array([line(u1), line(u2)]).T


####
# Some helper functions


def distance(point1: np.ndarray, point2: Optional[np.ndarray] = None):
    diff = point1 if point2 is None else point2 - point1
    return np.sqrt(np.sum(diff**2, axis=-1))


def rotate(
    point: np.ndarray, radians: float, anchor: Optional[np.ndarray] = None
) -> np.ndarray:
    """Rotate a point around an anchor point."""
    if anchor is None:
        anchor = np.array([0, 0])
    return (
        np.array(
            [
                np.cos(radians) * (point[0] - anchor[0])
                - np.sin(radians) * (point[1] - anchor[1]),
                np.sin(radians) * (point[0] - anchor[0])
                + np.cos(radians) * (point[1] - anchor[1]),
            ]
        )
        + anchor
    )


####
# Curve classes


class Curve(ABC):
    """A 2D curve. See method docstrings for details."""

    @abstractmethod
    def __call__(self, t: Union[float, np.ndarray]) -> Tuple[float, float]:
        """Evaluate the curve at a given parameter value.

        For example, evaluating a circle at t=0 gives the point at angle 0, t=pi/2 gives
        the point at 90 degrees angle, etc.

        Differentiating this, e.g. via jax.jvp, gives the tangent of the curve at a
        given point."""

    @abstractmethod
    def signed_distance(self, point: np.ndarray) -> np.ndarray:
        """Return the signed distance from a point to the nearest point on the curve.

        The sign of the returned value gives information about which side of the curve
        the point is on. The sign is positive on the right-hand side and negative on the
        left-hand side of the curve. For a circle, the distance is positive on the
        outside, and negative on the inside."""

    @abstractmethod
    def intersect(self, other: "Curve") -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        """Return the number of intersections between this curve and another curve and
        the (potentially none) intersection points."""


@traceable_dataclass(("anchor", "direction"))
class Line(Curve):
    anchor: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        # Normalize direction vector (tangent)
        self.direction = self.direction / distance(self.direction)

    def __call__(self, t: Union[float, np.ndarray]) -> Tuple[float, float]:
        if np.is_ndarray(t) and t.ndim != 0:
            points = np.moveaxis(self.anchor + t[..., None] * self.direction, -1, 0)
        else:
            points = self.anchor + t * self.direction
        return points

    def signed_distance(self, point: np.ndarray) -> float:
        return np.cross(point - self.anchor, self.direction)

    @staticmethod
    def passing_through(point1: np.ndarray, point2: np.ndarray) -> "Line":
        return Line(point1, point2 - point1)

    @staticmethod
    def with_angle(anchor: np.ndarray, angle: float) -> "Line":
        return Line(anchor, np.array([np.cos(angle), np.sin(angle)]))

    def intersect(self, other: Curve) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(other, Line):
            return intersect_line_line(self, other)
        else:
            raise NotImplementedError

    @property
    def angle(self) -> float:
        return np.arctan2(self.direction[1], self.direction[0])

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])


@traceable_dataclass(("center", "radius"))
class Circle(Curve):
    center: np.ndarray
    radius: float

    def __call__(self, t: float) -> Tuple[float, float]:
        center = self.center
        if np.is_ndarray(t) and t.ndim != 0:
            center = self.center[:, None]
        return np.array([np.cos(t), np.sin(t)]) * self.radius + center

    def signed_distance(self, point: np.ndarray) -> np.ndarray:
        return distance(point, self.center) - self.radius

    def intersect(self, other: Curve) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(other, Line):
            return intersect_circle_line(self, other)
        else:
            raise NotImplementedError


@traceable_dataclass(("f1", "f2", "d"))
class Ellipse(Curve):
    """An ellipse defined by two focus points and the summed distance from each to
    points on the ellipse.

    In other words: if p is a point on the ellipse, then d = distance(p, f1) + distance(p, f2)

    An ellipse is just a transformed unit circle, and that's how we implement Curve
    behavior for ellipses. As a consequence, the signed distance function is not yet
    implemented because transforming space also transforms the distance function. We
    should also not depend on the derivative of the __call__ method to compute a
    normalized tangent."""

    f1: np.ndarray  # Focus point 1
    f2: np.ndarray  # Focus point 2
    d: float  # If p is a point on the ellipse, then d = distance(p, f1) + distance(p, f2)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        # An ellipse is just a transformed circle
        circle = Circle(np.array([0, 0]), 1)
        circle_transform = self._circle_transform
        if np.is_ndarray(t) and t.ndim != 0:
            circle_transform = np.vmap(circle_transform, [1])
        return circle_transform(circle(t)).T

    def signed_distance(self, point: np.ndarray) -> np.ndarray:
        # Not implemented yet. Need to think about how to normalize the distance since
        # the space is transformed.
        raise NotImplementedError

    def intersect(self, other: Curve) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        if isinstance(other, Line):
            other = Line.passing_through(
                self._circle_undo_transform(other(0)),
                self._circle_undo_transform(other(1)),
            )
            circle = Circle(np.array([0, 0]), 1)
            num_intersections, intersections = intersect_circle_line(circle, other)
            i1, i2 = intersections.real.T
            return (
                num_intersections,
                np.array([self._circle_transform(i1), self._circle_transform(i2)]).T,
            )
        else:
            raise NotImplementedError

    def _circle_rotation(self) -> float:
        return np.arctan2(self.f2[1] - self.f1[1], self.f2[0] - self.f1[0])

    def _circle_scale(self) -> np.ndarray:
        a = self.d / 2
        b = np.sqrt(a**2 - distance(self.f2, self.f1) ** 2 / 4)
        return np.array([a, b])

    def _circle_translation(self) -> np.ndarray:
        return (self.f1 + self.f2) / 2

    def _circle_transform(self, point: np.ndarray) -> np.ndarray:
        point *= self._circle_scale()
        point = rotate(point, self._circle_rotation())
        point += self._circle_translation()
        return point

    def _circle_undo_transform(self, point: np.ndarray) -> np.ndarray:
        point -= self._circle_translation()
        point = rotate(point, -self._circle_rotation())
        point /= self._circle_scale()
        return point
