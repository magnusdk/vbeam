from spekk import Dim, ops

from vbeam.interpolation import LinearCoordinates, LinearNDInterpolator


def upsample_grid(data: ops.array, n: int, axis: Dim):
    """IQ-upsampling.

    You should always upsample a Nyquist-sampled grid before envelope detection.
    """
    assert n == 2
    last_index = data.dim_sizes[axis] - 1
    dim_size = data.dim_sizes[axis]
    sample_indices = ops.linspace(0.25, last_index - 0.25, last_index * 2, dim=axis)

    # Upsample the data by interpolation at the sample indices
    data_coordinates = {axis: LinearCoordinates(0, last_index, dim_size)}
    interpolator = LinearNDInterpolator(data_coordinates, data, fill_value=None)
    return interpolator({axis: sample_indices})