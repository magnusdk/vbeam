"""A module containing the main beamforming function:
:func:`~vbeam.core.kernels.signal_for_point`.
"""

from typing import Type, Union

from spekk import Module, ops

from vbeam.core.apodization import Apodization
from vbeam.core.channel_data import TChannelData
from vbeam.core.delay_models import ReflectedWaveDelayModel, TransmittedWaveDelayModel
from vbeam.core.interpolation import NDInterpolator
from vbeam.core.points_getter import PointsGetter
from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave


class Setup(Module):
    points: Union[ops.array, PointsGetter]
    transmitting_probe: Probe
    receiving_probe: Probe
    transmitted_wave: TransmittedWave
    channel_data: TChannelData
    interpolator_type: Type[NDInterpolator]
    transmitted_wave_delay_model: TransmittedWaveDelayModel
    reflected_wave_delay_model: ReflectedWaveDelayModel
    speed_of_sound: float
    apodization: Apodization


class Output(Module):
    data: ops.array
    weight: ops.array


def signal_for_point(setup: Setup) -> Output:
    """Delay and interpolate channel data from the given `setup` and return it.

    Return an :class:`~vbeam.core.kernels.Output` object which also has metadata such
    as the calculated weights.
    """
    points = (
        setup.points.get_points()
        if isinstance(setup.points, PointsGetter)
        else setup.points
    )

    # Get the delay in seconds between when the wave was transmitted from the
    # transmitting probe to when it reached the given point(s), and the same for the
    # reflected wave.
    tx_delays = setup.transmitted_wave_delay_model(
        setup.transmitting_probe,
        points,
        setup.transmitted_wave,
        setup.speed_of_sound,
    )
    rx_delays = setup.reflected_wave_delay_model(
        points,
        setup.receiving_probe,
        setup.speed_of_sound,
    )

    # Delay, interpolate, and remodulate the channel data (if IQ).
    delays = tx_delays + rx_delays
    interpolator = setup.interpolator_type(
        setup.channel_data.data_coordinates,
        setup.channel_data.data,
        fill_value=None,
    )
    values = interpolator({"time": delays})
    values = setup.channel_data.remodulate_if_iq(values, delays)
    weights = setup.apodization(
        setup.transmitting_probe, setup.receiving_probe, points, setup.transmitted_wave
    )

    return Output(values*weights, weights)
