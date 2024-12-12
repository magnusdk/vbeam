from typing import Sequence, Union

from spekk import ops


def ensure_positive_index(n: int, index: Union[int, Sequence[int]]) -> int:
    """Take an index that may be negative, following numpy's negative index semantics,
    and ensure that it is positive. That is, if the index=-1, then it will be the last
    index of an array."""
    if isinstance(index, int):
        return n + index if index < 0 else index
    else:
        return [ensure_positive_index(n, i) for i in index]


def broadcast_to_axis(a: ops.array, axis: int, ndims: int) -> ops.array:
    assert a.ndim <= 1
    shape = [1] * ndims
    shape[axis] = -1
    return a.reshape(shape)


def ensure_2d_point(point: ops.array) -> ops.array:
    if point.dim_sizes["xyz"] == 2:
        return point
    elif point.dim_sizes["xyz"] == 3:
        return ops.take(point, ops.array([0, 2]), axis="xyz")
    else:
        raise ValueError(f"Expected 2D or 3D point, got shape={point.shape}.")
