from typing import Tuple

from vbeam.core import InterpolationSpace1D
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util import ensure_positive_index


@traceable_dataclass(data_fields=("min", "d", "n"))
class FastInterpLinspace(InterpolationSpace1D):
    """Interpolation for linspace.

    If we assume that it is uniformly spaced (linspace), an array can be interpolated
    really quickly by subtracting the minimum value and dividing by the step-size."""

    min: float
    d: float
    n: int

    @staticmethod
    def from_array(arr: np.ndarray) -> "FastInterpLinspace":
        # Assumes that arr was created as a linspace.
        return FastInterpLinspace(arr[0], arr[1] - arr[0], len(arr))

    def to_array(self) -> np.ndarray:
        return np.linspace(self.min, self.min + self.d * self.n, self.n)

    def interp1d_indices(self, x: np.ndarray) -> Tuple[float, int, int, float, float]:
        """Return a tuple of 5 elements with information about how to interpolate the
        array.

        Returns a tuple of:
          0. bounds_flag: If 0, then x is within bounds. If -1, then x is outside of
               bounds on the left side. If 1, then x is out of bounds on the right side.
          1. clipped_i1: The index of the first value used in the interpolation.
          2. clipped_i2: The index of the second value used in the interpolation.
          3. p1: The weighting of the value at index clipped_i1.
          4. p2: The weighting of the value at index clipped_i2.

        The final interpolated value can be found as such:
        >>> fp = np.array([0, 1, 2])
        >>> interp = FastInterpLinspace.from_array(fp)
        >>> x = np.array([0.5, 1.5])
        >>> bounds_flag, clipped_i1, clipped_i2, p1, p2 = interp.interp1d_indices(x)
        >>> interpolated_values = fp[clipped_i1]*p1 + fp[clipped_i2]*p2
        >>> interpolated_values[1]
        1.5
        """
        pseudo_index = (x - self.min) / self.d
        i_floor = np.floor(pseudo_index)
        di = pseudo_index - i_floor

        bounds_flag = 0
        bounds_flag = np.where(pseudo_index < 0, -1, bounds_flag)
        bounds_flag = np.where(pseudo_index > (self.n - 1), 1, bounds_flag)
        clipped_i1 = np.clip(i_floor, 0, self.n - 1).astype("int32")
        clipped_i2 = np.clip(i_floor + 1, 0, self.n - 1).astype("int32")
        p1, p2 = (1 - di), di
        return bounds_flag, clipped_i1, clipped_i2, p1, p2

    def interp1d(
        self,
        x: np.ndarray,
        fp: np.ndarray,
        left: int = 0,
        right: int = 0,
    ) -> np.ndarray:
        bounds_flag, clipped_i1, clipped_i2, p1, p2 = self.interp1d_indices(x)
        bounds_flag = np.expand_dims(bounds_flag, tuple(range(1, fp.ndim)))
        p1 = np.expand_dims(p1, tuple(range(1, fp.ndim)))
        p2 = np.expand_dims(p2, tuple(range(1, fp.ndim)))
        v = fp[clipped_i1] * p1 + fp[clipped_i2] * p2
        v = np.where(bounds_flag == -1, left, v)
        v = np.where(bounds_flag == 1, right, v)
        return v

    # InterpolationSpace1D interface
    def __call__(self, x, fp):
        return self.interp1d(x, fp)

    @property
    def start(self) -> float:
        return self.min

    @property
    def end(self) -> float:
        return (self.d * self.n) + self.min

    @staticmethod
    def interp2d(
        x: np.ndarray,
        y: np.ndarray,
        xp: "FastInterpLinspace",
        yp: "FastInterpLinspace",
        z: np.ndarray,
        azimuth_axis: int = 0,
        depth_axis: int = 1,
        *,  # Remaining args must be passed by name (to avoid confusion)
        padding: float = 0.0,
    ) -> np.ndarray:
        # Ensure that the axes are positive numbers
        azimuth_axis = ensure_positive_index(z.ndim, azimuth_axis)
        depth_axis = ensure_positive_index(z.ndim, depth_axis)

        # Ensure that the azimuth and depth axes are the first two axes
        if depth_axis == 0 and azimuth_axis == 1:
            z = np.swapaxes(z, azimuth_axis, depth_axis)
        else:
            z = np.moveaxis(z, azimuth_axis, 0)
            z = np.moveaxis(z, depth_axis, 1)

        # Interpolate along the axes
        bounds_flag_x, clipped_xi1, clipped_xi2, px1, px2 = xp.interp1d_indices(x)
        bounds_flag_y, clipped_yi1, clipped_yi2, py1, py2 = yp.interp1d_indices(y)

        # Make weighting values and flags broadcastable with respect to the shape of
        # elements in z. This makes is it so that we can interpolate grids of vectors;
        # not just grids of scalars.
        num_value_dims = len(z.shape[2:])
        broadcastable = lambda a: np.expand_dims(
            a, tuple(range(a.ndim, a.ndim + num_value_dims))
        )
        px1, px2, py1, py2 = [broadcastable(p) for p in [px1, px2, py1, py2]]
        bounds_flag_x = broadcastable(bounds_flag_x)
        bounds_flag_y = broadcastable(bounds_flag_y)

        v0 = z[clipped_xi1, clipped_yi1] * px1 + z[clipped_xi2, clipped_yi1] * px2
        v1 = z[clipped_xi1, clipped_yi2] * px1 + z[clipped_xi2, clipped_yi2] * px2
        v = v0 * py1 + v1 * py2
        v = np.where(np.logical_or(bounds_flag_x != 0, bounds_flag_y != 0), padding, v)

        # Swap axes back to their original positions
        if depth_axis == 0 and azimuth_axis == 1:
            v = np.swapaxes(v, azimuth_axis, depth_axis)
        elif v.ndim >= 2:
            v = np.moveaxis(v, 1, depth_axis)
            v = np.moveaxis(v, 0, azimuth_axis)
        return v

    @property
    def shape(self):
        return self.min.shape

    @property
    def ndim(self):
        return len(self.shape)
