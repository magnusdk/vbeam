import operator
from typing import Sequence, Set, Union

from spekk.transformations import Transformation

from vbeam.data_importers.setup import SignalForPointSetup
from vbeam.fastmath import numpy as np
from vbeam.scan import CoordinateSystem
from vbeam.scan import util as scan_util
from vbeam.scan.advanced import ExtraDimsScanMixin
from vbeam.util.transformations import *
from vbeam.util.vmap import apply_binary_operation_across_axes


def do_nothing(f):
    return f


def vectorize_over_datacube(
    setup: SignalForPointSetup,
    *,  # Always use keyword arguments for remaining args for clarity/readability
    ignore: Union[Sequence[str], Set[str]] = (),
) -> Transformation:
    """Return a :class:`Transformation` that vectorizes it to run for all points,
    receivers, transmits, and frames (if part of the spec).

    It assumes that the spec follows the convention of naming the dimensions "points",
    "receivers", "transmits", and "frames". A check is made whether the spec has the
    dimension before vectorizing over it — if the spec does not have the dimension then
    it is ignored."""
    ignore = set(ignore)
    return compose(
        *[
            ForAll(dim)
            for dim in ["points", "receivers", "senders", "transmits", "frames"]
            if setup.spec.has_dimension(dim) and dim not in ignore
        ]
    )


def sum_over_dimensions(
    setup: SignalForPointSetup,
    *,  # Always use keyword arguments for remaining args for clarity/readability
    keep: Union[Sequence[str], Set[str]] = (),
) -> Transformation:
    """Return a :class:`Transformation` that makes the function sum over receivers and
    transmits, unless they should be kept (as specified by :param:`keep`).

    Any dimension in :param:`keep` is not summed over. Dimensions that are not in the
    spec are also ignored.

    Dimensions needed to recombine the points using the points optimizer are always
    kept."""
    keep = set(keep)
    if isinstance(setup.scan, ExtraDimsScanMixin):
        keep |= set(setup.scan.required_dimensions_for_unflatten)
    return Apply(
        np.sum,
        [
            Axis(dim)
            for dim in ["receivers", "senders", "transmits"]
            if dim not in keep and setup.spec.has_dimension(dim)
        ],
    )


def unflatten_points(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function convert the flattened
    points back to the original scan shape with "width" and "height" dimensions
    (assuming a 2D scan).

    Automatically handles point optimizers."""
    if isinstance(setup.scan, ExtraDimsScanMixin):
        return Apply(
            setup.scan.unflatten,
            *[Axis(dim) for dim in setup.scan.required_dimensions_for_unflatten[:-1]],
            Axis(
                setup.scan.required_dimensions_for_unflatten[-1],
                becomes=setup.scan.dimensions_after_unflatten,
            ),
        )
    else:
        return Apply(setup.scan.unflatten, Axis("points", becomes=["width", "height"]))


def compensate_for_apodization_overlap(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function divide its result by
    the RTB apodization overlap for each pixel."""
    if isinstance(setup.scan, ExtraDimsScanMixin):
        apodization_overlap = setup.get_apodization_values(
            setup.scan.required_dimensions_for_unflatten
        )
        apodization_overlap = setup.scan.unflatten(
            apodization_overlap,
            *range(len(setup.scan.required_dimensions_for_unflatten)),
        )
    else:
        apodization_overlap = setup.get_apodization_values(["points"])
        apodization_overlap = setup.scan.unflatten(apodization_overlap)

    return Apply(
        lambda result, width_axis, height_axis: apply_binary_operation_across_axes(
            result, apodization_overlap, operator.truediv, [width_axis, height_axis]
        ),
        Axis("width", keep=True),
        Axis("height", keep=True),
    )


def scan_convert(setup: SignalForPointSetup) -> Transformation:
    """Return a :class:`Transformation` that makes the function apply scan conversion
    to its result if the scan is defined in polar coordinates."""
    if setup.scan.coordinate_system != CoordinateSystem.POLAR:
        return do_nothing
    return Apply(
        scan_util.scan_convert,
        setup.scan,
        Axis("width", keep=True),
        Axis("height", keep=True),
    )
