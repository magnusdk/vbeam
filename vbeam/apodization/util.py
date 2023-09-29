from typing import Sequence

from spekk import Spec

from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.util.transformations import *


def get_apodization_values(
    apodization: Apodization,
    sender: ElementGeometry,
    point_position: np.ndarray,
    receiver: ElementGeometry,
    wave_data: WaveData,
    spec: Spec,
    dimensions: Sequence[str],
    average_overlap: bool = False,
):
    """Return the apodization values for the dimensions. All other relevant
    dimensions are (by default) summed over.

    If you want the apodization values for each transmit and point, you'd call
    setup.get_apodization_values(["transmits", "points"]). Likewise, if you only
    want the points, you'd call setup.get_apodization_values(["points"]). All other
    dimensions are summed over.

    If average_overlap is True, then the apodization values are averaged instead of
    summed."""
    kwargs = {
        "apodization": apodization,
        "sender": sender,
        "point_position": point_position,
        "receiver": receiver,
        "wave_data": wave_data,
    }

    # Define what dimensions to vmap and sum over and how
    vmap_dimensions = (
        spec["sender"].dimensions
        | spec["point_position"].dimensions
        | spec["receiver"].dimensions
        | spec["wave_data"].dimensions
    )
    sum_dimensions = vmap_dimensions - set(dimensions)
    reduce_sum_dimension = spec["wave_data"].dimensions & sum_dimensions
    if reduce_sum_dimension:
        # Make it only one of the dimensions (doesn't matter which one)
        reduce_sum_dimension = {reduce_sum_dimension.pop()}
        vmap_dimensions -= reduce_sum_dimension

    # Define how to calculate the apodization values
    calculate_apodization = compose(
        lambda apodization, *args, **kwargs: apodization(*args, **kwargs),
        *[ForAll(dim) for dim in vmap_dimensions],
        Apply(np.sum, [Axis(dim) for dim in sum_dimensions - reduce_sum_dimension]),
        # [*reduce_sum_dimension][0] gets the "first element" of the set
        Reduce.Sum([*reduce_sum_dimension][0]) if reduce_sum_dimension else do_nothing,
        # Put the dimensions in the order defined by keep
        Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        Wrap(np.jit),  # Make it run faster, if the backend supports it.
    ).build(spec)

    # Calculate the apodization values
    values = calculate_apodization(**kwargs)
    # Divide by the number of points summed over, if average_overlap is True
    if average_overlap:
        values /= sum(spec.size(kwargs, dim) for dim in sum_dimensions)
    return values
