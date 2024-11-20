from fastmath import Array

from vbeam.core import InterpolationSpace1D
from vbeam.fastmath import numpy as api


class NearestInterpolation(InterpolationSpace1D):
    """Interpolate by rounding (nearest neighbour).

    >>> interp = NearestInterpolation(0, 1, 10)
    >>> fp = np.arange(10)
    >>> x = np.array([0, 0.5, 0.51, 2.3])
    >>> interp(x, fp)
    array([0, 0, 1, 2])
    """

    min: float
    d: float
    n: int
    left: float = 0
    right: float = 0

    def __call__(self, x: Array, fp: Array) -> Array:
        index = api.round((x - self.min) / self.d)
        return api.select(
            [index < 0, index >= self.n],
            [self.left, self.right],
            fp[index.astype(int)],
        )

    @property
    def start(self) -> float:
        return self.min

    @property
    def end(self) -> float:
        return (self.d * self.n) + self.min
