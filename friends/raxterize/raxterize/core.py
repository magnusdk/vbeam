from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from vbeam.fastmath import backend_manager
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry import Line

backend_manager.active_backend = "jax"

# TODO: Make vbeam.util.geometry.Line work with 2D points
@traceable_dataclass(("a", "b", "c"))
class LineNB:
    a: float
    b: float
    c: float

    @staticmethod
    def passing_through(point1: np.ndarray, point2: np.ndarray) -> "Line":
        "Construct a line that passes through both point1 and point2."
        x1, z1 = point1
        x2, z2 = point2
        a = z1 - z2
        b = x2 - x1
        c = x1 * z2 - x2 * z1
        return Line(a, b, c)

    @staticmethod
    def from_anchor_and_angle(anchor: np.ndarray, angle: float) -> "Line":
        "Construct a line that passes through the anchor and has the given angle."
        return Line.passing_through(
            anchor, anchor + np.array([np.cos(angle), np.sin(angle)])
        )

    def intersection(l1: "Line", l2: "Line") -> np.ndarray:
        "Return the point where the lines l1 and l2 intersect."
        x = (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
        z = (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
        return np.array([x, z])

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
        return (self.a * point[..., 0] + self.b * point[..., 1] + self.c) / norm


Vertex = namedtuple("Vertex", ["xyz", "value"])
Polygon = namedtuple("Polygon", ["v1", "v2", "v3", "v4"])


@jax.jit
def image2polygons(coords, image) -> Polygon:
    @partial(jax.vmap, in_axes=[0, 0])
    def polygon_from_coord(xi: int, zi: int) -> Polygon:
        return Polygon(
            *(
                Vertex(coords[xi + xd, zi + zd], image[xi + xd, zi + zd])
                for (xd, zd) in [(0, 0), (1, 0), (1, 1), (0, 1)]
            )
        )

    nx, nz = image.shape
    X, Z = jnp.meshgrid(jnp.arange(nx - 1), jnp.arange(nz - 1), indexing="ij")
    return polygon_from_coord(jnp.ravel(X), jnp.ravel(Z))


def contains(polygon: Polygon, point: jnp.ndarray):
    "Return True if the point is within the polygon."
    l1 = Line.passing_through(polygon.v1.xyz, polygon.v2.xyz)
    l2 = Line.passing_through(polygon.v2.xyz, polygon.v3.xyz)
    l3 = Line.passing_through(polygon.v3.xyz, polygon.v4.xyz)
    l4 = Line.passing_through(polygon.v4.xyz, polygon.v1.xyz)
    return (
        (l1.signed_distance(point) >= 0)
        & (l2.signed_distance(point) >= 0)
        & (l3.signed_distance(point) >= 0)
        & (l4.signed_distance(point) >= 0)
    )


def lerp_triangle(point, v1: Vertex, v2: Vertex, v3: Vertex):
    # https://codeplea.com/triangular-interpolation
    denominator = (v2.xyz[2] - v3.xyz[2]) * (v1.xyz[0] - v3.xyz[0]) + (
        v3.xyz[0] - v2.xyz[0]
    ) * (v1.xyz[2] - v3.xyz[2])
    w1 = (
        (v2.xyz[2] - v3.xyz[2]) * (point[0] - v3.xyz[0])
        + (v3.xyz[0] - v2.xyz[0]) * (point[2] - v3.xyz[2])
    ) / denominator
    w2 = (
        (v3.xyz[2] - v1.xyz[2]) * (point[0] - v3.xyz[0])
        + (v1.xyz[0] - v3.xyz[0]) * (point[2] - v3.xyz[2])
    ) / denominator
    w3 = 1 - w1 - w2
    value = w1 * v1.value + w2 * v2.value + w3 * v3.value
    return jnp.where((w1 >= 0) & (w2 >= 0) & (w3 >= 0), value, 0)


def lerp(poly: Polygon, point: jnp.ndarray):
    return lerp_triangle(point, poly.v1, poly.v2, poly.v3) + lerp_triangle(
        point, poly.v1, poly.v3, poly.v4
    )


@jax.jit
@partial(jax.vmap, in_axes=(None, 0))
def lerp_all(polygons: Polygon, points: jnp.ndarray):
    vmapped_lerp = jax.vmap(lerp, (0, None))
    return jnp.sum(vmapped_lerp(polygons, points))
