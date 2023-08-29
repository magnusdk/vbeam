"""Function (:func:`.signal_for_point`) for beamforming a single point, which is usually repeated over all 
:term:`points<Point>`, :term:`receivers<Receiver>` and :term:`transmits<Transmit>`.

See also:
    :mod:`vbeam.beamformers`
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from vbeam.core.apodization import Apodization
from vbeam.core.element_geometry import ElementGeometry
from vbeam.core.interpolation import InterpolationSpace1D
from vbeam.core.kernel_data import KernelData
from vbeam.core.speed_of_sound import SpeedOfSound
from vbeam.core.wave_data import WaveData
from vbeam.core.wavefront import MultipleTransmitDistances, Wavefront
from vbeam.fastmath import numpy as np


def signal_for_point(
    speed_of_sound: Union[float, SpeedOfSound],
    t_axis_interp: InterpolationSpace1D,
    signal: np.ndarray,
    modulation_frequency: Optional[float],
    receiver: ElementGeometry,
    sender: ElementGeometry,
    point_position: np.ndarray,
    wavefront: Wavefront,
    wave_data: WaveData,
    apodization: Apodization,
) -> np.ndarray:
    """Delay-and-sum kernel. Return the delayed signal from a single transmit, for a
    single receiver, for a single point.

    Args:
      speed_of_sound: Speed of sound. If it is a float, it is a constant speed of sound.
        If it is an instance of the class SpeedOfSound then the speed of sound may be
        heterogeneous, and we need to sample from it.
      t_axis_interp: Used for interpolating the signal using the calculated delay.
      signal: The receiver signal data.
      modulation_frequency: The modulation frequency that we have to correct in the
        delayed signal. If None and the signal is real, then the signal remains real.
      receiver_pos: The position of the receiver (x, y, z).
      sender_pos: The position of the sender (x, y, z).
      point_position: The position of the point to be imaged (x, z).
      wavefront: A Wavefront object for calculating the wavefront propagation distance,
        which varies depending on the wavefront model. See Wavefront class documentation
        for details.
      wave_data: Data specific to a given transmitted wave. See WaveData class fields.
      apodization: Apodization function (combined call both for transmit and receive).

    Returns:
      The delayed signal from a single transmit, for a single receiver, for a single
      point.
    """
    if isinstance(speed_of_sound, SpeedOfSound):
        speed_of_sound = speed_of_sound.average(
            sender.position, point_position, receiver.position
        )
    transmit_distance = wavefront.transmit_distance(sender, point_position, wave_data)
    receive_distance = wavefront.receive_distance(point_position, receiver, wave_data)
    delay = (transmit_distance + receive_distance) / speed_of_sound
    signal = t_axis_interp(delay, signal)
    if modulation_frequency is not None:
        signal = phase_correction(signal, delay, modulation_frequency)
    if isinstance(transmit_distance, MultipleTransmitDistances):
        signal = transmit_distance.aggregate_samples(signal)
    return signal * apodization(sender, point_position, receiver, wave_data)


def phase_correction(signal: float, delay: float, modulation_frequency: float):
    w0 = np.pi * 2 * modulation_frequency
    return signal * np.exp(1j * w0 * delay)


@dataclass
class SignalForPointData(KernelData):
    """Data needed for signal_for_point.

    See the docstring of signal_for_point for documentation of each field."""

    speed_of_sound: Union[float, SpeedOfSound]
    t_axis_interp: InterpolationSpace1D
    signal: np.ndarray
    modulation_frequency: float
    receiver: ElementGeometry
    sender: ElementGeometry
    point_position: np.ndarray
    wavefront: Wavefront
    wave_data: WaveData
    apodization: Apodization

    def copy(self) -> "SignalForPointData":
        return SignalForPointData(**self)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.__dataclass_fields__.keys())
