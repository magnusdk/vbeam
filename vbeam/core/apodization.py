"Spatial weighting functions for beamforming."

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from spekk import Module, field, ops

from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave

# For adding type hints to Apodization.plot:
if TYPE_CHECKING:
    try:
        from matplotlib.axes import Axes
    except ImportError:
        pass


class Apodization(Module):
    """An apodization (or weighting) function for a beamformed point.

    An :class:`~apodization` is more than just a tapering window over channels. It can
    represent RTB weighting, expanding aperture, or any other spatial weighting
    function.
    """

    @abstractmethod
    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: TransmittedWave,
    ) -> float:
        "Return the spatial weighting at the given `point`."

    def combine(
        apodization,
        *more_apodizations,
        combiner: Optional[Callable] = None,
    ) -> "CombinedApodization":
        """Combine multiple apodizations using `combiner` into a single apodization
        object.
        """
        return CombinedApodization([apodization, *more_apodizations], combiner)

    def plot(
        self,
        *,
        transmitting_probe: Optional[Probe] = None,
        receiving_probe: Optional[Probe] = None,
        points: Optional[ops.array] = None,
        transmitted_wave: Optional[TransmittedWave] = None,
        ax: Optional["Axes"] = None,
    ) -> "Axes":
        import matplotlib.pyplot as plt

        from vbeam.util import _default_values

        if ax is None:
            _, ax = plt.subplots()

        if transmitting_probe is None:
            transmitting_probe = _default_values.transmitting_probe()
        if receiving_probe is None:
            receiving_probe = _default_values.receiving_probe()
        if points is None:
            points = _default_values.points()
        if transmitted_wave is None:
            transmitted_wave = _default_values.transmitted_wave()

        values = self(transmitting_probe, receiving_probe, points, transmitted_wave)
        extent_mm = [
            1000 * ops.min(ops.take(points, 0, axis="xyz")),
            1000 * ops.max(ops.take(points, 0, axis="xyz")),
            1000 * ops.max(ops.take(points, 2, axis="xyz")),
            1000 * ops.min(ops.take(points, 2, axis="xyz")),
        ]
        aperture_bounds = transmitting_probe.active_aperture.bounds

        ax.imshow(values, extent=extent_mm, cmap="gray")
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


class CombinedApodization(Apodization):
    apodizations: Iterable[Apodization]
    combiner: Optional[Callable] = field(
        default_factory=lambda: ops.prod,  # Combine by multiplication by default
        static=True,
    )

    def __call__(self, *args, **kwargs) -> float:
        """Multiply the result of calling the Apodization objects."""
        values = ops.stack(
            [apod(*args, **kwargs) for apod in self.apodizations],
            axis="combinable_apodizations",
        )
        return self.combiner(values, axis="combinable_apodizations")
