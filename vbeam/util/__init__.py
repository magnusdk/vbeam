from typing import Sequence, Union

from vbeam.fastmath import numpy as np


def ensure_positive_index(n: int, index: Union[int, Sequence[int]]) -> int:
    """Take an index that may be negative, following numpy's negative index semantics,
    and ensure that it is positive. That is, if the index=-1, then it will be the last
    index of an array."""
    if isinstance(index, int):
        return n + index if index < 0 else index
    else:
        return [ensure_positive_index(n, i) for i in index]


def ensure_2d_point(point: np.ndarray) -> np.ndarray:
    if point.shape[-1] == 2:
        return point
    elif point.shape[-1] == 3:
        return point[..., np.array([0, 2])]  # Return x- and z-coordinates
    else:
        raise ValueError(f"Expected 2D or 3D point, got shape={point.shape}.")
