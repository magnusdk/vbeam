import functools
import operator
from typing import TYPE_CHECKING, Optional, Union

from spekk import ops

if TYPE_CHECKING:
    from vbeam.geometry.vector import Vector, VectorWithInfiniteMagnitude


def get_x(v: ops.array):
    return v["xyz", 0]


def get_y(v: ops.array):
    return v["xyz", 1]


def get_z(v: ops.array):
    return v["xyz", 2]


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


def normalize_vector(v: ops.array) -> ops.array:
    magnitude = ops.linalg.vector_norm(v, axis="xyz")
    return v / magnitude


def get_rotation_matrix(
    *,
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
) -> ops.array:
    transformation_chain = []
    if azimuth is not None:
        cos_azimuth, sin_azimuth = ops.cos(azimuth), ops.sin(azimuth)
        transformation_chain.append(
            ops.stack(
                [
                    ops.stack([cos_azimuth, 0, -sin_azimuth], axis="xyz"),
                    ops.stack([0, 1, 0], axis="xyz"),
                    ops.stack([sin_azimuth, 0, cos_azimuth], axis="xyz"),
                ],
                axis="xyz_new_basis",
            )
        )
    if elevation is not None:
        cos_elevation, sin_elevation = ops.cos(elevation), ops.sin(elevation)
        transformation_chain.append(
            ops.stack(
                [
                    ops.stack([1, 0, 0], axis="xyz"),
                    ops.stack([0, cos_elevation, -sin_elevation], axis="xyz"),
                    ops.stack([0, sin_elevation, cos_elevation], axis="xyz"),
                ],
                axis="xyz_new_basis",
            )
        )

    if len(transformation_chain) == 0:
        return ops.stack(
            [
                ops.stack([1.0, 0.0, 0.0], axis="xyz"),
                ops.stack([0.0, 1.0, 0.0], axis="xyz"),
                ops.stack([0.0, 0.0, 1.0], axis="xyz"),
            ],
            axis="xyz_new_basis",
        )
    return functools.reduce(operator.matmul, transformation_chain)
