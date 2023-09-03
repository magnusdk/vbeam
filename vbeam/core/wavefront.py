"""Wavefront models that return the distance that a wave has traveled. 
:class:`TransmittedWavefront` models the transmitted wave while 
:class:`ReflectedWavefront` models a reflected/backscattered wave.

Wavefront models return a distance in meters. This way they are decoupled from the 
speed of sound of the medium."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance

from .element_geometry import ElementGeometry
from .wave_data import WaveData


class TransmittedWavefront(ABC):
    """The base class for defining wavefront models of transmitted waves.

    :class:`TransmittedWavefront` models are used to calculate the *distance (in
    meters)* from a sending element to a point. This distance, along with the distance
    returned from :class:`ReflectedWavefront` and when divided by a speed-of-sound, can
    be used to delay the signal of a receiving element.

    Most :class:`TransmittedWavefront` models return a single distance for a wave.
    However, it may optionally return more than one distance value in the form of a
    :class:`MultipleTransmitDistances` object. This is useful for wavefront models that
    sample multiple distances for a single transmit.

    See also:
        :class:`ReflectedWavefront`
        :class:`MultipleTransmitDistances`
        :func:`~vbeam.core.kernels.signal_for_point`

    Examples:
        We can create a simple synthetic transmit aperture (STA) wavefront model where
        a single sender element sends out a spherical wavefront. The returned distance
        is then simply the euclidean distance between the sender and point. To
        illustrate the use of :class:`MultipleTransmitDistances`, we also add a small
        smoothing to the distance by averaging with two small delays (+/- 0.1
        millimeters):

        >>> class TransmittedWavefront(Wavefront):
        ...     def __call__(self, sender, point_position, wave_data):
        ...         # Simply get the distance from sender to point (assume STA)
        ...         dist = distance(sender.position, point_position)
        ...         # Smooth the dist by averaging with the two neighboring distances
        ...         dist = jnp.array([dist - 1e-4, dist, dist + 1e-4])
        ...         # Weights for the three different delayed signal samples
        ...         weights = jnp.array([0.2, 0.6, 0.2])
        ...         return MultipleTransmitDistances(
        ...             dist,
        ...             # The samples values are aggregated by a weighted sum
        ...             lambda samples: np.sum(samples * weights),
        ...         )
    """

    @abstractmethod
    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> Union[float, "MultipleTransmitDistances"]:
        """Return the *distance (in meters)* from the sender element to the point for a transmit.

        May optionally return a :class:`MultipleTransmitDistances` object for cases
        where the wavefront model samples multiple distances for a single transmit.

        See also:
            :class:`MultipleTransmitDistances`
            :func:`~vbeam.core.kernels.signal_for_point`
        """


@traceable_dataclass()
class ReflectedWavefront:
    """The base class for defining wavefront models of reflected/backscattered waves.

    This is usually just the euclidian distance as there are limits to what we can
    model.

    See also:
        :class:`TransmittedWavefront`
        :func:`~vbeam.core.kernels.signal_for_point`"""

    def __call__(self, point_position: np.ndarray, receiver: ElementGeometry) -> float:
        return distance(point_position, receiver.position)


@dataclass
class MultipleTransmitDistances:
    """Multiple distance values returned from a :class:`TransmittedWavefront`.

    Some more advanced :class:`TransmittedWavefront` models may return multiple
    distances for a transmitted wave. In this case, each returned distance will be used
    to delay the element signals, and the delayed samples will be combined using the
    function set in :attr:`aggregate_samples`. The :attr:`aggregate_samples` function
    may for example weight the delayed signals differently before summing.

    See reference to :class:`MultipleTransmitDistances` in
    :func:`~vbeam.core.kernels.signal_for_point` for implementation details.

    Mathematical operators applied to the :class:`MultipleTransmitDistances` object
    will apply them to the :attr:`values` attribute as if it was just a numpy array.

    Attributes:
        values (np.ndarray): The distance values that will be used to delay the element signals.
        aggregate_samples (Callable[[np.ndarray, np.ndarray], np.ndarray]): A function
            that will be used to combine the delayed samples into one value, for
            example as a weighted sum. If None, the samples will be averaged.

    See also:
        :func:`~vbeam.core.kernels.signal_for_point`
        :class:`TransmittedWavefront`
    """

    values: np.ndarray
    aggregate_samples: Callable[[np.ndarray, np.ndarray], np.ndarray] = None

    def __post_init__(self):
        # If no aggregate function is set, just average the samples
        if self.aggregate_samples is None:
            self.aggregate_samples = np.mean

    # Mathematical operators applied to the :class:`MultipleTransmitDistances` object
    # will apply them to the ``values`` attribute as if it was just a numpy array.
    def __truediv__(self, other) -> np.ndarray:
        return self.values / other

    def __add__(self, other) -> np.ndarray:
        return self.values + other
