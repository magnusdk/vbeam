import functools
import operator
from typing import TYPE_CHECKING, Optional, Union

from spekk import ops

if TYPE_CHECKING:
    from vbeam.geometry.vector import Vector, VectorWithInfiniteMagnitude


def get_x(v: ops.array):
    return ops.take(v, 0, axis="xyz")


def get_y(v: ops.array):
    return ops.take(v, 1, axis="xyz")


def get_z(v: ops.array):
    return ops.take(v, 2, axis="xyz")


def get_xy(v: ops.array):
    return get_x(v), get_y(v)


def get_xz(v: ops.array):
    return get_x(v), get_z(v)


def get_yz(v: ops.array):
    return get_y(v), get_z(v)


def get_xyz(v: ops.array):
    return get_x(v), get_y(v), get_z(v)


def distance(
    point1: Union[ops.array, "Vector", "VectorWithInfiniteMagnitude"],
    point2: Union[ops.array, "Vector", "VectorWithInfiniteMagnitude"],
) -> float:
    from vbeam.geometry.vector import Vector, VectorWithInfiniteMagnitude

    if isinstance(point1, VectorWithInfiniteMagnitude) or isinstance(
        point2, VectorWithInfiniteMagnitude
    ):
        return ops.inf

    if isinstance(point1, Vector):
        point1 = point1.to_array()
    if isinstance(point2, Vector):
        point2 = point2.to_array()
    return ops.linalg.vector_norm(point1 - point2, axis="xyz")


def get_rotation_matrix(
    *,
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
) -> ops.array:
    transformation_chain = []
    if azimuth is not None:
        cos_azimuth, sin_azimuth = ops.cos(azimuth), ops.sin(azimuth)
        transformation_chain.append(
            ops.array(
                [
                    [cos_azimuth, 0, -sin_azimuth],
                    [0, 1, 0],
                    [sin_azimuth, 0, cos_azimuth],
                ],
                ["xyz_new_basis", "xyz"],
            )
        )
    if elevation is not None:
        cos_elevation, sin_elevation = ops.cos(elevation), ops.sin(elevation)
        transformation_chain.append(
            ops.array(
                [
                    [1, 0, 0],
                    [0, cos_elevation, -sin_elevation],
                    [0, sin_elevation, cos_elevation],
                ],
                ["xyz_new_basis", "xyz"],
            )
        )

    if len(transformation_chain) == 0:
        return ops.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            ["xyz_new_basis", "xyz"],
            dtype="float32",
        )
    return functools.reduce(operator.matmul, transformation_chain)
