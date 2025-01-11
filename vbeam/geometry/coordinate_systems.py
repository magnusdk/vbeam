from spekk import ops

from vbeam.geometry.util import get_xyz
from vbeam.geometry.orientation import Direction
from vbeam.geometry.vector import Vector


def as_cartesian(point: ops.array) -> ops.array:
    azimuth, elevation, depth = get_xyz(point)
    return Vector(depth, Direction(azimuth, elevation)).to_array()


def as_polar(point: ops.array) -> ops.array:
    v = Vector.from_array(point)
    return ops.stack([v.azimuth, v.elevation, v.magnitude], axis="xyz")
