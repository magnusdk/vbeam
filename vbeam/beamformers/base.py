from functools import partial
from typing import Literal, Union

from spekk import Spec

from vbeam.beamformers.transformations import *
from vbeam.core import signal_for_point
from vbeam.data_importers.setup import SignalForPointSetup
from vbeam.fastmath import numpy as np
from vbeam.postprocess import coherence_factor, normalized_decibels
from vbeam.scan import SectorScan
from vbeam.scan.sector_scan import cartesian_map as cartesian_map_impl

specced_signal_for_point = Specced(
    signal_for_point,
    lambda _: Spec(
        speed_of_sound=[],
        t_axis_interp=[],
        signal=["signal_time"],
        modulation_frequency=[],
        receiver=[],
        sender=[],
        point_pos=[],
        wavefront=[],
        wave_data=[],
        apodization=[],
    ),
    lambda _: Spec(()),
)


def get_beamformer(
    setup: SignalForPointSetup,
    *,
    cartesian_map: bool = True,
    postprocess: Union[
        Literal["normalized_decibels"], Literal["coherence"], None
    ] = "normalized_decibels",
) -> TransformedFunction:
    # Create a datacube
    beamformer = compose(
        specced_signal_for_point,
        *[
            ForAll(dim)
            for dim in ["points", "receivers", "transmits", "frames"]
            if setup.spec.has_dimension(dim)
        ],
    )

    # Sum over transmits and unflatten points
    if setup.points_optimizer is None:
        beamformer = compose(
            beamformer,
            Apply(np.sum, Axis("transmits")),
            Apply(setup.scan.unflatten, Axis("points", becomes=("width", "height"))),
        )
    else:
        shape_info = setup.points_optimizer.shape_info
        beamformer = compose(
            beamformer,
            Apply(
                setup.points_optimizer.recombine,
                setup.scan,
                *[
                    Axis(
                        dim,
                        keep=dim not in shape_info.removed_dimensions,
                        becomes=("width", "height") if dim == "points" else (),
                    )
                    for dim in shape_info.required_for_recombine
                ],
            ),
        )
        if "transmits" not in shape_info.removed_dimensions:
            beamformer = compose(beamformer, Apply(np.sum, Axis("transmits")))

    # Postprocess
    if postprocess == "normalized_decibels":
        beamformer = compose(
            beamformer,
            Apply(np.sum, Axis("receivers")),
            Apply(normalized_decibels),
        )
    elif postprocess == "coherence":
        beamformer = compose(beamformer, Apply(coherence_factor, Axis("receivers")))
    elif postprocess is None:
        beamformer = compose(beamformer, Apply(np.sum, Axis("receivers")))
    else:
        raise ValueError(
            f'postprocess argument must be one of "normalized_decibels" (default),\
"coherence", or None, but it is "{postprocess}".'
        )

    if cartesian_map and isinstance(setup.scan, SectorScan):
        beamformer = compose(
            beamformer,
            Apply(
                cartesian_map_impl,
                setup.scan,
                Axis("width", keep=True),
                Axis("height", keep=True),
            ),
        )

    beamformer = compose(beamformer, Wrap(partial, **setup.data))
    return beamformer.build(setup.spec)
