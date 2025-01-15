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
from typing import TYPE_CHECKING, Optional

from spekk import Module, ops

from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave

# For adding type hints to TransmittedWaveDelayModel.plot:
if TYPE_CHECKING:
    try:
        from matplotlib.axes import Axes
    except ImportError:
        pass


def _plot(
    delay_values: ops.array, transmitting_probe: Probe, points: ops.array, ax: "Axes"
) -> "Axes":
    from vbeam.probe import RectangularPlanarAperture

    extent_mm = [
        1000 * ops.min(ops.take(points, 0, axis="xyz")),
        1000 * ops.max(ops.take(points, 0, axis="xyz")),
        1000 * ops.max(ops.take(points, 2, axis="xyz")),
        1000 * ops.min(ops.take(points, 2, axis="xyz")),
    ]

    ax.imshow(delay_values, extent=extent_mm, cmap="gray")
    if isinstance(transmitting_probe.active_aperture, RectangularPlanarAperture):
        aperture_bounds = transmitting_probe.active_aperture.bounds
        center_left_mm = aperture_bounds.center_left * 1000
        center_right_mm = aperture_bounds.center_right * 1000
        ax.plot(
            [center_left_mm[0], center_right_mm[0]],
            [center_left_mm[2], center_right_mm[2]],
            c="blue",
        )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    return ax


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

        Args:
            transmitting_probe (Probe): The transmitting probe.
            point (ops.array): The point, or pixel, that we want to calculate the delay
                for.
            transmitted_wave (TransmittedWave): Information about the transmitted wave,
                such as its origin or virtual source (if geometrically focused).
            speed_of_sound (float): The speed of sound of the medium.
        """

    def plot(
        self,
        *,
        transmitting_probe: Optional[Probe] = None,
        points: Optional[ops.array] = None,
        transmitted_wave: Optional[TransmittedWave] = None,
        speed_of_sound: Optional[float] = None,
        ax: Optional["Axes"] = None,
    ) -> "Axes":
        import matplotlib.pyplot as plt

        from vbeam.util import _default_values

        if transmitting_probe is None:
            transmitting_probe = _default_values.transmitting_probe()
        if points is None:
            points = _default_values.points()
        if transmitted_wave is None:
            transmitted_wave = _default_values.transmitted_wave()
        if speed_of_sound is None:
            speed_of_sound = _default_values.speed_of_sound()
        if ax is None:
            _, ax = plt.subplots()
        values = self(transmitting_probe, points, transmitted_wave, speed_of_sound)
        return _plot(values, transmitting_probe, points, ax)


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

        Args:
            point (ops.array): The point, or pixel, that we want to calculate the delay
                for.
            receiving_probe (Probe): The receiving probe.
            speed_of_sound (float): The speed of sound of the medium.
        """
        distance = ops.linalg.vector_norm(
            point - receiving_probe.active_elements.position,
            axis="xyz",
        )
        delay = distance / speed_of_sound
        return delay
