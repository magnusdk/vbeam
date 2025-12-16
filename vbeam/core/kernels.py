"""A module containing the main beamforming function:
:func:`~vbeam.core.kernels.signal_for_point`.
"""

from typing import Type

from spekk import Module, ops

from vbeam.core.apodization import Apodization
from vbeam.core.channel_data import TChannelData
from vbeam.core.delay_models import DelayModel
from vbeam.core.interpolation import NDInterpolator
from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave

from vbeam.core.scan import TScan

class Setup(Module):
    scan: TScan
    transmitting_probe: Probe
    receiving_probe: Probe
    transmitted_wave: TransmittedWave
    channel_data: TChannelData
    interpolator_type: Type[NDInterpolator]
    delay_model: DelayModel
    apodization: Apodization


def signal_for_point(setup: Setup) -> ops.array:
    """Delay and interpolate channel data from the given `setup` and return it.

    Return an :class:`~vbeam.core.kernels.Output` object which also has metadata such
    as the calculated weights.
    """

    # Get the delay in seconds between when the wave passed through
    # transmitted_wave.origin, to when it reached a given point, and to when it was
    # reflected back to a receiving element.
    delays = setup.delay_model(
        setup.scan.points,
        setup.transmitted_wave,
        setup.transmitting_probe,
        setup.receiving_probe,
    )

    # Delay, interpolate, and remodulate the channel data (if IQ).
    interpolator = setup.interpolator_type(
        setup.channel_data.data_coordinates,
        setup.channel_data.data,
        fill_value=None,
    )
    values = interpolator({"time": delays})
    values = setup.channel_data.remodulate_if_iq(values, delays)

    # Apply apodization, if given
    if setup.apodization is not None:
        weights = setup.apodization(
            setup.transmitting_probe,
            setup.receiving_probe,
            setup.scan.points,
            setup.transmitted_wave,
        )
        values *= weights

    return values
