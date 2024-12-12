from typing import Tuple, Union

from spekk import ops

from vbeam.interpolation import FastInterpLinspace


def coherence_factor(beamformed_data: ops.array, receivers_axis: int):
    coherent_sum = ops.abs(ops.sum(beamformed_data, receivers_axis)) ** 2
    incoherent_sum = ops.sum(ops.abs(beamformed_data) ** 2, receivers_axis)
    num_receivers = beamformed_data.shape[receivers_axis]
    return ops.nan_to_num(coherent_sum / incoherent_sum / num_receivers)


def normalized_decibels(data: ops.array):
    "Convert the data into decibels normalized for dynamic range."
    data_db = 20 * ops.nan_to_num(ops.log10(ops.abs(data)))
    return data_db - data_db.max()


def _upsampling_indices(n: int, data_size: int) -> ops.array:
    """Return the sampling indices of data with size data_size after upsampling by n.

    The formula is a bit complicated because it is as general as possible. Check out
    the notebook `docs/tutorials/postprocess/upsampling.ipynb` for some visualizations.
    """
    is_even = n % 2 == 0
    is_odd = not is_even
    d = (n // 2 + is_odd) * 2
    return ops.arange(n * data_size - d + is_odd) / n + (is_even + d - n) / (2 * n)


def upsample_by_interpolation(
    data: ops.array,
    n: int,
    axis: Union[int, Tuple[int, ...]] = 0,
) -> ops.array:
    """Upsample the data along the given axes. Multiple axes may be given."""
    # No axis has been given: upsample all axes
    if axis is None:
        axis = tuple(range(data.ndim))

    # Multiple axes have been given: upsample each axis separately
    if isinstance(axis, tuple):
        if isinstance(n, int):
            n = (n,) * len(axis)
        # Upsample each axis separately
        for axis, _n in zip(axis, n):
            data = upsample_by_interpolation(data, _n, axis)
        return data  # ...and return the data that has been upsampled along all axes

    # Only one axis has been given: upsample that axis
    sample_indices = _upsampling_indices(n, data.shape[axis])
    data = ops.swapaxes(data, axis, 0)
    interpolator = FastInterpLinspace(0, 1, data.shape[0])
    interpolated_data = interpolator.interp1d(
        sample_indices, data, left=data[0], right=data[-1]
    )
    return ops.swapaxes(interpolated_data, 0, axis)
