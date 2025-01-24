"""A module containing the main beamforming function:
:func:`~vbeam.core.kernels.signal_for_point`.
"""

from typing import Union

from spekk import Module, ops

from vbeam.core.apodization import Apodization
from vbeam.core.channel_data import ChannelData
from vbeam.core.delay_models import ReflectedWaveDelayModel, TransmittedWaveDelayModel
from vbeam.core.interpolation import TInterpolator
from vbeam.core.points_getter import PointsGetter
from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave


class Setup(Module):
    points: Union[ops.array, PointsGetter]
    transmitting_probe: Probe
    receiving_probe: Probe
    transmitted_wave: TransmittedWave
    channel_data: ChannelData
    interpolator: TInterpolator
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

    # Delay, interpolate, and remodulate the channel data.
    delays = tx_delays + rx_delays
    values = setup.interpolator(setup.channel_data, delays, axis="time")
    if setup.channel_data.modulation_frequency is not None:
        w0 = ops.pi * 2 * setup.channel_data.modulation_frequency
        values = values * ops.exp(1j * w0 * (delays - setup.channel_data.t0))

    # Apply weighting to the signal.
    apodization_values = setup.apodization(
        setup.transmitting_probe,
        setup.receiving_probe,
        points,
        setup.transmitted_wave,
    )
    values = values * apodization_values

    return Output(values, apodization_values)
