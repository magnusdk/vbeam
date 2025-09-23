from spekk import Dim, ops

from vbeam.interpolation import LinearCoordinate, LinearNDInterpolator


def iq_upsample(data: ops.array, axis: Dim | list[Dim] | tuple[Dim]):
    """IQ-upsample the data by a factor of 2 along the given axis or multiple axes.

    You should always upsample a Nyquist-sampled grid before envelope detection.
    """
    if not isinstance(axis, (list, tuple)):
        axis = [axis]

    coordinates = {}
    indices = {}
    for ax in axis:
        dim_size = data.dim_sizes[ax]
        last_index = dim_size - 1
        indices[ax] = ops.linspace(0.25, last_index - 0.25, last_index * 2, dim=ax)
        coordinates[ax] = LinearCoordinate(0, last_index, dim_size)

    interpolator = LinearNDInterpolator(coordinates, data, fill_value=None)
    return interpolator(indices)
