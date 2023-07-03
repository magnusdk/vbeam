from typing import Optional, Sequence

from vbeam.fastmath import numpy as np


def grid(*axes: np.ndarray, shape: Optional[Sequence[int]] = None) -> np.ndarray:
    """Return an array of each point position, organized in a grid.

    >>> x, z = np.array([1,2,3]), np.array([1,2,3,4])
    >>> points = grid(x, z)
    >>> points.shape == (3, 4, 2)
    True
    >>> points.shape == (x.size, z.size, 2)
    True
    >>> (points[:, 0, 0] == x).all()
    True
    >>> (points[0, :, 1] == z).all()
    True

    We can control the final shape of the grid by specifying the shape argument:
    >>> points_3d = grid(x, np.array([0.0]), z, shape=(3, 4, 3))
    >>> (points_3d[:, :, [0, 2]] == points).all()
    True
    """
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    if shape:
        points = points.reshape(shape)
    return points
