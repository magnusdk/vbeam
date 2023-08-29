from typing import Callable, Literal, Sequence, Union

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
    sum_fn: Union[
        Literal["sum"], Callable[[np.ndarray, Sequence[int]], np.ndarray]
    ] = "sum",
):
    """Return the apodization values for the dimensions. All other relevant
    dimensions are (by default) summed over.

    If you want the apodization values for each transmit and point, you'd call
    setup.get_apodization_values(["transmits", "points"]). Likewise, if you only
    want the points, you'd call setup.get_apodization_values(["points"]). All other
    dimensions are summed over.

    By default, all dimensions not specified in the list are summed over. You can
    override this by passing in a sum_fn."""
    sum_fn = np.sum if sum_fn == "sum" else sum_fn
    all_dimensions = (
        spec["sender"].dimensions
        | spec["point_position"].dimensions
        | spec["receiver"].dimensions
        | spec["wave_data"].dimensions
    )
    calculate_apodization = compose(
        lambda apodization, *args, **kwargs: apodization(*args, **kwargs),
        *[ForAll(dim) for dim in all_dimensions],
        Apply(sum_fn, [Axis(dim) for dim in all_dimensions - set(dimensions)]),
        # Put the dimensions in the order defined by keep
        Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        Wrap(np.jit),  # Make it run faster, if the backend supports it.
    ).build(spec)
    return calculate_apodization(
        apodization=apodization,
        sender=sender,
        point_position=point_position,
        receiver=receiver,
        wave_data=wave_data,
    )
