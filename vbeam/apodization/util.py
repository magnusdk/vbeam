import math
from typing import Optional, Sequence

from spekk import Spec

from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.util import _deprecations
from vbeam.util.transformations import *


@_deprecations.renamed_kwargs("1.0.5", average_overlap="average")
def get_apodization_values(
    apodization: Apodization,
    sender: ElementGeometry,
    point_position: np.ndarray,
    receiver: ElementGeometry,
    wave_data: WaveData,
    spec: Spec,
    dimensions: Optional[Sequence[str]] = None,
    reduce_sum_dimension: Optional[Sequence[str]] = None,
    average: bool = False,
    jit: bool = True,
):
    """
    Calculate and return the apodization values based on the provided arguments
    (``sender``, ``point_position``, ``receiver``, and ``wave_data``).

    The ``dimensions`` argument determines what dimensions to keep; all others are
    summed over (except when ``dimesions`` is None where we keep all dimensions
    instead). ``spec`` describes the dimensions of the data. E.g., if ``dimensions`` is
    ``["transmits", "x", "z"]``, the result will be a 3D array with shape (Nt, Nx, Nz),
    where Nt, Nx, Nz are the sizes of the given dimensions in the data.

    Args:
        apodization (Apodization): The apodization function to use.
        sender (ElementGeometry): The sender argument to ``apodization``.
        point_position (np.ndarray): The point_position argument to ``apodization``.
        receiver (ElementGeometry): The receiver argument to ``apodization``.
        wave_data (WaveData): The wave data argument to ``apodization``.
        spec (Spec): A spec describing the dimensions/shape of the arguments.
        dimensions (Optional[Sequence[str]]): The dimensions to keep in the returned
            result. If it is an empty list, all dimensions are summed over. If it is
            None, all dimensions from the spec are kept.
        average (bool): If True, the result is averaged instead of summed.
        jit (bool): If True, the process is JIT-compiled (if the backend supports it).

    Returns:
        np.ndarray: The calculated apodization values with shape corresponding to the
        dimensions defined in ``dimensions``.
    """
    kwargs = {
        "apodization": apodization,
        "sender": sender,
        "point_position": point_position,
        "receiver": receiver,
        "wave_data": wave_data,
    }

    # Return the full datacube if dimensions are not given
    # Careful! This may allocate a lot of memory.
    if dimensions is None:
        dimensions = list(spec.dimensions)

    # Define what dimensions to vmap and sum over and how
    vmap_dimensions = (
        spec["apodization"].dimensions
        | spec["sender"].dimensions
        | spec["point_position"].dimensions
        | spec["receiver"].dimensions
        | spec["wave_data"].dimensions
    )
    sum_dimensions = vmap_dimensions - set(dimensions)
    if reduce_sum_dimension is None:
        reduce_sum_dimension = spec["wave_data"].dimensions & sum_dimensions
        if reduce_sum_dimension:
            # Default to iterative summation over one of the dimensions (doesn't matter which one)
            reduce_sum_dimension = {reduce_sum_dimension.pop()}
    reduce_sum_dimension = set(reduce_sum_dimension)
    # Some dimensions will be iterated over instead of vmap
    vmap_dimensions: str = vmap_dimensions - reduce_sum_dimension
    vmap_sum_dimensions: set[str] = sum_dimensions - reduce_sum_dimension

    # Define how to calculate the apodization values
    calculate_apodization = compose(
        lambda apodization, *args, **kwargs: apodization(*args, **kwargs),
        *[ForAll(dim) for dim in vmap_dimensions],
        Apply(np.sum, [Axis(dim) for dim in vmap_sum_dimensions])
        if vmap_sum_dimensions
        else do_nothing,
        # [*reduce_sum_dimension][0] gets the "first element" of the set
        *[Reduce.Sum(dim) for dim in reduce_sum_dimension],
        # Put the dimensions in the order defined by keep
        Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        # Make it run faster if `jit` is True and if the backend supports it.
        Wrap(np.jit) if jit else do_nothing,
    ).build(spec)

    # Calculate the apodization values
    values = calculate_apodization(**kwargs)
    # Divide by the number of points summed over, if average is True
    if average:
        values /= max(1, math.prod(spec.size(kwargs, dim) for dim in sum_dimensions))
    return values
