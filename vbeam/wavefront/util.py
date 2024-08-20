import math
from typing import Optional, Sequence

from spekk import Spec

from vbeam.core import (
    ElementGeometry,
    ReflectedWavefront,
    TransmittedWavefront,
    WaveData,
)
from vbeam.fastmath import numpy as np
from vbeam.util.transformations import *


def get_transmitted_wavefront_values(
    wavefront: TransmittedWavefront,
    sender: ElementGeometry,
    point_position: np.ndarray,
    wave_data: WaveData,
    spec: Spec,
    dimensions: Optional[Sequence[str]] = None,
    jit: bool = True,
):
    """
    Calculate and return the (transmit) wavefront distance values based on the provided
    arguments (``sender``, ``point_position``, and ``wave_data``).

    The ``dimensions`` argument determines what dimensions to keep; all others are
    summed over (except when ``dimesions`` is None where we keep all dimensions
    instead). ``spec`` describes the dimensions of the data. E.g., if ``dimensions`` is
    ``["transmits", "x", "z"]``, the result will be a 3D array with shape (Nt, Nx, Nz),
    where Nt, Nx, Nz are the sizes of the given dimensions in the data.

    Args:
        wavefront (TransmittedWavefront): The wavefront function to use.
        sender (ElementGeometry): The sender argument to ``wavefront``.
        point_position (np.ndarray): The point_position argument to ``wavefront``.
        wave_data (WaveData): The wave data argument to ``wavefront``.
        spec (Spec): A spec describing the dimensions/shape of the arguments.
        dimensions (Optional[Sequence[str]]): The dimensions to keep in the returned
            result. If it is an empty list, all dimensions are summed over. If it is
            None, all dimensions from the spec are kept.
        jit (bool): If True, the process is JIT-compiled (if the backend supports it).

    Returns:
        np.ndarray: The calculated wavefront distance values with shape corresponding
        to the dimensions defined in ``dimensions``.
    """
    kwargs = {
        "transmitted_wavefront": wavefront,
        "sender": sender,
        "point_position": point_position,
        "wave_data": wave_data,
    }

    # Return the full datacube if dimensions are not given
    # Careful! This may allocate a lot of memory.
    if dimensions is None:
        dimensions = list(spec.dimensions)

    # Define what dimensions to vmap and sum over and how
    vmap_dimensions = (
        spec["transmitted_wavefront"].dimensions
        | spec["sender"].dimensions
        | spec["point_position"].dimensions
        | spec["wave_data"].dimensions
    )
    sum_dimensions = vmap_dimensions - set(dimensions)
    reduce_sum_dimension = spec["wave_data"].dimensions & sum_dimensions
    if reduce_sum_dimension:
        # Make it only one of the dimensions (doesn't matter which one)
        reduce_sum_dimension = {reduce_sum_dimension.pop()}
        vmap_dimensions -= reduce_sum_dimension

    # Define how to calculate the wavefront values
    calculate_wavefront_values = compose(
        lambda transmitted_wavefront, *args, **kwargs: transmitted_wavefront(
            *args, **kwargs
        ),
        *[ForAll(dim) for dim in vmap_dimensions],
        Apply(np.sum, [Axis(dim) for dim in sum_dimensions - reduce_sum_dimension]),
        # [*reduce_sum_dimension][0] gets the "first element" of the set
        Reduce.Sum([*reduce_sum_dimension][0]) if reduce_sum_dimension else do_nothing,
        # Put the dimensions in the order defined by keep
        Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        # Make it run faster if `jit` is True and if the backend supports it.
        Wrap(np.jit) if jit else do_nothing,
    ).build(spec)

    # Calculate the wavefront values
    values = calculate_wavefront_values(**kwargs)
    # Return the average delay values
    values /= max(1, math.prod(spec.size(kwargs, dim) for dim in sum_dimensions))
    return values


def get_reflected_wavefront_values(
    wavefront: ReflectedWavefront,
    point_position: np.ndarray,
    receiver: ElementGeometry,
    spec: Spec,
    dimensions: Optional[Sequence[str]] = None,
    jit: bool = True,
):
    """
    Calculate and return the (receive) wavefront distance values based on the provided
    arguments (``point_position`` and ``receiver``).

    The ``dimensions`` argument determines what dimensions to keep; all others are
    summed over (except when ``dimesions`` is None where we keep all dimensions
    instead). ``spec`` describes the dimensions of the data. E.g., if ``dimensions`` is
    ``["receivers", "x", "z"]``, the result will be a 3D array with shape (Nr, Nx, Nz),
    where Nr, Nx, Nz are the sizes of the given dimensions in the data.

    Args:
        wavefront (ReflectedWavefront): The wavefront function to use.
        point_position (np.ndarray): The point_position argument to ``wavefront``.
        receiver (ElementGeometry): The receiver argument to ``wavefront``.
        spec (Spec): A spec describing the dimensions/shape of the arguments.
        dimensions (Optional[Sequence[str]]): The dimensions to keep in the returned
            result. If it is an empty list, all dimensions are summed over. If it is
            None, all dimensions from the spec are kept.
        jit (bool): If True, the process is JIT-compiled (if the backend supports it).

    Returns:
        np.ndarray: The calculated wavefront distance values with shape corresponding
        to the dimensions defined in ``dimensions``.
    """
    kwargs = {
        "reflected_wavefront": wavefront,
        "point_position": point_position,
        "receiver": receiver,
    }

    # Return the full datacube if dimensions are not given
    # Careful! This may allocate a lot of memory.
    if dimensions is None:
        dimensions = list(spec.dimensions)

    # Define what dimensions to vmap and sum over and how
    vmap_dimensions = (
        spec["reflected_wavefront"].dimensions
        | spec["point_position"].dimensions
        | spec["receiver"].dimensions
    )
    sum_dimensions = vmap_dimensions - set(dimensions)
    reduce_sum_dimension = spec["wave_data"].dimensions & sum_dimensions
    if reduce_sum_dimension:
        # Make it only one of the dimensions (doesn't matter which one)
        reduce_sum_dimension = {reduce_sum_dimension.pop()}
        vmap_dimensions -= reduce_sum_dimension

    # Define how to calculate the wavefront values
    calculate_wavefront_values = compose(
        lambda reflected_wavefront, *args, **kwargs: reflected_wavefront(
            *args, **kwargs
        ),
        *[ForAll(dim) for dim in vmap_dimensions],
        Apply(np.sum, [Axis(dim) for dim in sum_dimensions - reduce_sum_dimension]),
        # [*reduce_sum_dimension][0] gets the "first element" of the set
        Reduce.Sum([*reduce_sum_dimension][0]) if reduce_sum_dimension else do_nothing,
        # Put the dimensions in the order defined by keep
        Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        # Make it run faster if `jit` is True and if the backend supports it.
        Wrap(np.jit) if jit else do_nothing,
    ).build(spec)

    # Calculate the wavefront values
    values = calculate_wavefront_values(**kwargs)
    # Return the average delay values
    values /= max(1, math.prod(spec.size(kwargs, dim) for dim in sum_dimensions))
    return values
