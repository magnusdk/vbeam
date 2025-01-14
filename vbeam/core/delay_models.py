"""Delay models are used to calculate the delay values of points (pixels) to
interpolate the channel data with. They are split into two parts:
:class:`~vbeam.core.delay_models.TransmittedWaveDelayModel` and
:class:`~vbeam.core.delay_models.ReflectedWaveDelayModel`.

Transmitted wave delay models return the seconds elapsed since a transmitted wave
passed through its origin (see
:class:`~vbeam.core.transmitted_wave.TransmittedWave.origin`) until it reached a point
in space. Reflected wave delay models return the seconds elapsed for a reflected wave
to travel from a point in space and back up to a receiving element.
"""

from abc import abstractmethod

from spekk import Module, ops

from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave


class TransmittedWaveDelayModel(Module):
    """An object that models the time elapsed (in seconds) since a transmitted wave
    passed through its origin (see
    :class:`~vbeam.core.transmitted_wave.TransmittedWave.origin`) and until it reached
    a point in space.

    See also :class:`~vbeam.core.delay_models.ReflectedWaveDelayModel` which models the
    time elapsed for the reflected/backscattered wave to travel from the point and back
    up to the receiving elements.
    """

    @abstractmethod
    def __call__(
        self,
        transmitting_probe: Probe,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        speed_of_sound: float,
    ) -> float:
        """Return the time elapsed (in seconds) since the `transmitted_wave` passed
        through its origin (see
        :class:`~vbeam.core.transmitted_wave.TransmittedWave.origin`) and until it
        reached the given `point` in space.
        """


class ReflectedWaveDelayModel(Module):
    """An object that models the time elapsed (in seconds) since a wave was
    reflected/backscattered from the given `point` and traveled back up to the
    receiving elements of the given `receiving_probe`.

    See also :class:`~vbeam.core.delay_models.TransmittedWaveDelayModel` which models
    the time elapsed from transmitting a wave to the wave reaching a given point.
    """

    def __call__(
        self,
        point: ops.array,
        receiving_probe: Probe,
        speed_of_sound: float,
    ) -> float:
        """Return the time elapsed (in seconds) since a wave was
        reflected/backscattered from the given `point` and traveled back up to the
        receiving elements of the given `receiving_probe`.
        """
        distance = ops.linalg.vector_norm(
            point - receiving_probe.active_elements.position, axis="xyz"
        )
        return distance / speed_of_sound
