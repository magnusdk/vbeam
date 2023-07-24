import operator
from typing import Sequence, Set, Union

from spekk.transformations import Transformation

from vbeam.data_importers.setup import SignalForPointSetup
from vbeam.fastmath import numpy as np
from vbeam.scan import SectorScan, cartesian_map
from vbeam.util.transformations import *
from vbeam.util.vmap import apply_binary_operation_across_axes


def do_nothing(f):
    return f


def vectorize_over_datacube(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that vectorizes it to run for all points,
    receivers, transmits, and frames (if part of the spec).

    It assumes that the spec follows the convention of naming the dimensions "points",
    "receivers", "transmits", and "frames". A check is made whether the spec has the
    dimension before vectorizing over it — if the spec does not have the dimension then
    it is ignored."""
    return compose(
        *[
            ForAll(dim)
            for dim in ["points", "receivers", "transmits", "frames"]
            if setup.spec.has_dimension(dim)
        ]
    )


def sum_over_dimensions(
    setup: SignalForPointSetup,
    *,  # Always use keyword arguments for remaining args for clarity/readability
    keep: Union[Sequence[str], Set[str]],
) -> Transformation:
    """Return a :class:`Transformation` that makes the function sum over receivers and
    transmits, unless they should be kept (as specified by :param:`keep`).

    Any dimension in :param:`keep` is not summed over. Dimensions that are not in the
    spec are also ignored.

    Dimensions needed to recombine the points using the points optimizer are always
    kept."""
    keep = set(keep)
    if setup.points_optimizer is not None:
        keep |= set(setup.points_optimizer.shape_info.required_for_recombine)
    return Apply(
        np.sum,
        [
            Axis(dim)
            for dim in ["receivers", "transmits"]
            if dim not in keep and setup.spec.has_dimension(dim)
        ],
    )


def unflatten_points(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function convert the flattened
    points back to the original scan shape with "width" and "height" dimensions
    (assuming a 2D scan).

    Automatically handles point optimizers."""
    if setup.points_optimizer is None:
        return Apply(setup.scan.unflatten, Axis("points", becomes=["width", "height"]))
    else:
        required_dimensions = setup.points_optimizer.shape_info.required_for_recombine
        axes = [
            *(Axis(dim) for dim in required_dimensions[:-1]),
            Axis(required_dimensions[-1], becomes=["width", "height"]),
        ]
        return Apply(setup.points_optimizer.recombine, setup.scan, *axes)


def compensate_for_apodization_overlap(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function divide its result by
    the RTB apodization overlap for each pixel."""
    if setup.points_optimizer is None:
        apodization_overlap = setup.get_apodization_values(["points"])
        apodization_overlap = setup.scan.unflatten(apodization_overlap)
    else:
        apodization_overlap = setup.get_apodization_values(["transmits", "points"])
        apodization_overlap = setup.points_optimizer.recombine(
            apodization_overlap, setup.scan, 0, 1
        )
    return Apply(
        lambda result, width_axis, height_axis: apply_binary_operation_across_axes(
            result, apodization_overlap, operator.truediv, [width_axis, height_axis]
        ),
        Axis("width", keep=True),
        Axis("height", keep=True),
    )


def scan_convert(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function apply scan conversion
    to its result if the scan is a :class:`SectorScan`."""
    if not isinstance(setup.scan, SectorScan):
        return do_nothing
    return Apply(
        cartesian_map,
        setup.scan,
        Axis("width", keep=True),
        Axis("height", keep=True),
    )
