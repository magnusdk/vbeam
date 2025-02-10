from spekk import Dim, ops

from vbeam.interpolation import LinearInterpolator, LinearlySampledData


def upsample_grid(data: ops.array, n: int, axis: Dim):
    """IQ-upsampling.

    You should always upsample a Nyquist-sampled grid before envelope detection.
    """
    # Create the n indices to sample the data at.
    sample_indices = ops.arange(
        n // 2,
        n * data.dim_sizes[axis] - n // 2,
        dim=axis,
    )
    sample_indices /= n
    # Move the points slightly off the grid to get better noise statistics. For
    # example, when n=2, the sampled indices are offset by 0.25
    sample_indices -= (n - 1) / (2 * n)

    # Upsample the data by interpolation at the sample indices
    interpolator = LinearInterpolator()
    interpolable = LinearlySampledData.from_array(data, axis=axis)
    return interpolator(interpolable, sample_indices, axis)
