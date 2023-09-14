"""Function (:func:`.signal_for_point`) for beamforming a single point, which is 
usually repeated over all :term:`points<Point>`, :term:`receivers<Receiver>` and 
:term:`transmits<Transmit>` (or any other arbitrary dimensions).

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
from vbeam.core.wavefront import (
    MultipleTransmitDistances,
    ReflectedWavefront,
    TransmittedWavefront,
)
from vbeam.fastmath import numpy as np


def signal_for_point(
    sender: ElementGeometry,
    point_position: np.ndarray,
    receiver: ElementGeometry,
    signal: np.ndarray,
    transmitted_wavefront: TransmittedWavefront,
    reflected_wavefront: ReflectedWavefront,
    speed_of_sound: Union[float, SpeedOfSound],
    wave_data: WaveData,
    interpolate: InterpolationSpace1D,
    modulation_frequency: Optional[float],
    apodization: Apodization,
) -> np.ndarray:
    """The core beamforming function. Return the delayed and interpolated signal from a
    single transmit, for a single receiver, for a single point (pixel).

    To make a full beamformer, this function should be made to run (in parallel) for a
    ll points in the image, all receiving elements, all transmitted waves, etc.

    Args:
      sender: The element or array that the transmitted wave passed through at time 0.
        It "sends" the transmitted wave.
      point_position: The point/pixel to be imaged. Is always 3D and in cartesian
        coordinates (x, y, z).
      receiver: The element that received the reflected/backscattered signal.
      signal: The recorded signal of the backscattered signal for the receiver.
      transmitted_wavefront: A wavefront model for calculating the distance that the
        transmitted wave travels before reaching the point position.
      reflected_wavefront: A wavefront model for calculating the distance that the
        reflected wave travels from a point position back to a receiver. Usually, this
        is just the euclidian distance.
      speed_of_sound: The speed-of-sound of the medium. It is used to convert from
        distance in meters returned from the wavefront models to delay in seconds. If
        it is a float, then the medium has a constant speed-of-sound (it is the same
        everywhere). It may also be an instance of :class:`SpeedOfSound` that which can
        model a heterogeneous speed-of-sound medium.
      wave_data: Data about the transmitted wave, for example the position of the
        virtual source (if applicable).
      interpolate: Return the recorded backscattered signal at the time given the
        calculated delay, by interpolation.
      modulation_frequency: Optional value used to remodulate the delayed interpolated
        signal if it was a demodulated IQ signal. If the signal is not a demodulated IQ
        signal, then modulation_frequency can be set to 0 or None.
      apodization: Apodization function that weights the returned delayed interpolated
        signal.

    Returns:
      The delayed and interpolated signal from a single transmit, for a single
      receiver, for a single point (pixel).
    """
    tx_distance = transmitted_wavefront(sender, point_position, wave_data)
    rx_distance = reflected_wavefront(point_position, receiver)

    # Potentially sample the speed-of-sound using a SpeedOfSound instance.
    if isinstance(speed_of_sound, SpeedOfSound):
        speed_of_sound = speed_of_sound.average(
            sender.position, point_position, receiver.position
        )

    delay = (tx_distance + rx_distance) / speed_of_sound - wave_data.t0
    signal = interpolate(delay, signal)

    if modulation_frequency is not None:
        signal = phase_correction(signal, delay, modulation_frequency)
    if isinstance(tx_distance, MultipleTransmitDistances):
        signal = tx_distance.aggregate_samples(signal)
    return signal * apodization(sender, point_position, receiver, wave_data)


def phase_correction(signal: float, delay: float, modulation_frequency: float):
    w0 = np.pi * 2 * modulation_frequency
    return signal * np.exp(1j * w0 * delay)


@dataclass
class SignalForPointData(KernelData):
    """Data needed for signal_for_point.

    See the docstring of signal_for_point for documentation of each field."""

    sender: ElementGeometry
    point_position: np.ndarray
    receiver: ElementGeometry
    signal: np.ndarray
    transmitted_wavefront: TransmittedWavefront
    reflected_wavefront: ReflectedWavefront
    speed_of_sound: Union[float, SpeedOfSound]
    wave_data: WaveData
    interpolate: InterpolationSpace1D
    modulation_frequency: Optional[float]
    apodization: Apodization

    def copy(self) -> "SignalForPointData":
        return SignalForPointData(**self)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self.__dataclass_fields__.keys())
