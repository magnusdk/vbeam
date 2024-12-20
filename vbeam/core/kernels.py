"""Function (:func:`.signal_for_point`) for beamforming a single point, which is 
usually repeated over all :term:`points<Point>`, :term:`receivers<Receiver>` and 
:term:`transmits<Transmit>` (or any other arbitrary dimensions).

See also:
    :mod:`vbeam.beamformers`
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from spekk import Module, ops

from vbeam.core.apodization import Apodization
from vbeam.core.interpolation import InterpolationSpace1D
from vbeam.core.probe_geometry import ProbeGeometry
from vbeam.core.speed_of_sound import SpeedOfSound
from vbeam.core.wave_data import WaveData
from vbeam.core.wavefront import (
    MultipleTransmitDistances,
    ReflectedWavefront,
    TransmittedWavefront,
)


class Setup(Module):
    probe: ProbeGeometry
    sender: ops.array
    receiver: ops.array
    point_position: ops.array
    signal: ops.array
    transmitted_wavefront: TransmittedWavefront
    reflected_wavefront: ReflectedWavefront
    speed_of_sound: Union[float, SpeedOfSound]
    wave_data: WaveData
    interpolate: InterpolationSpace1D
    modulation_frequency: Optional[float]
    apodization: Apodization


class Output(Module):
    value: ops.array
    apodization: ops.array


class PointsGetter(ABC):
    @abstractmethod
    def get_points(self) -> ops.array: ...


def signal_for_point(setup: Setup) -> Output:
    point_position = (
        setup.point_position.get_points()
        if isinstance(setup.point_position, PointsGetter)
        else setup.point_position
    )
    tx_distance = setup.transmitted_wavefront(
        setup.probe,
        setup.sender,
        point_position,
        setup.wave_data,
    )
    rx_distance = setup.reflected_wavefront(point_position, setup.receiver)

    # Potentially sample the speed-of-sound using a SpeedOfSound instance.
    if isinstance(setup.speed_of_sound, SpeedOfSound):
        speed_of_sound = setup.speed_of_sound.average(
            setup.sender,
            point_position,
            setup.receiver,
        )
    else:
        speed_of_sound = setup.speed_of_sound

    delay = (tx_distance + rx_distance) / speed_of_sound - setup.wave_data.t0
    signal = setup.interpolate(delay, setup.signal, axis="time")

    if setup.modulation_frequency is not None:
        w0 = ops.pi * 2 * setup.modulation_frequency
        signal = signal * ops.exp(1j * w0 * delay)
    if isinstance(tx_distance, MultipleTransmitDistances):
        signal = tx_distance.aggregate_samples(signal)
    apodization = setup.apodization(
        setup.probe,
        setup.sender,
        setup.receiver,
        point_position,
        setup.wave_data,
    )
    return Output(signal * apodization, apodization)
