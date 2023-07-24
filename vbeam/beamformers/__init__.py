from typing import Sequence, Set, Union

from vbeam.beamformers.building_blocks import (
    compensate_for_apodization_overlap,
    do_nothing,
    scan_convert,
    sum_over_dimensions,
    unflatten_points,
    vectorize_over_datacube,
)
from vbeam.core import signal_for_point
from vbeam.data_importers import SignalForPointSetup
from vbeam.postprocess import coherence_factor, normalized_decibels
from vbeam.util.transformations import *


def get_das_beamformer(
    setup: SignalForPointSetup,
    *,
    log_compress: bool = True,
    apply_scan_conversion: bool = True,
    keep_dimensions: Union[Sequence[str], Set[str]] = (),
):
    return compose(
        signal_for_point,
        vectorize_over_datacube(setup),
        sum_over_dimensions(setup, keep=keep_dimensions),
        unflatten_points(setup),
        compensate_for_apodization_overlap(setup),
        Apply(normalized_decibels) if log_compress else do_nothing,
        scan_convert(setup) if apply_scan_conversion else do_nothing,
    ).build(setup.spec)


def get_coherence_beamformer(
    setup: SignalForPointSetup,
    *,
    apply_scan_conversion: bool = True,
    keep_dimensions: Union[Sequence[str], Set[str]] = (),
):
    keep_dimensions = set(keep_dimensions) | {"receivers"}
    return compose(
        signal_for_point,
        vectorize_over_datacube(setup),
        sum_over_dimensions(setup, keep=keep_dimensions),
        unflatten_points(setup),
        Apply(coherence_factor, Axis("receivers")),
        scan_convert(setup) if apply_scan_conversion else do_nothing,
    ).build(setup.spec)
