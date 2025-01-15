from dataclasses import replace
from typing import Optional

from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, GeometricallyFocusedWave, Probe
from vbeam.util._transmitted_wave import raise_if_not_geometrically_focused_wave


class ConstantWidthApodization(Apodization):
    """Spatially weights the delayed data using a constant-sized line starting from the
    projected wave origin and passing through the virtual source.

    See it visually in a notebook by running this code:
    >>> from vbeam.apodization import ConstantWidthApodization, window
    >>> apodization = ConstantWidthApodization(window.Hamming(), 0.01, 0.01)
    >>> apodization.plot()
    """

    window: Window
    beam_width: float
    beam_height: Optional[float] = None

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
    ) -> float:
        raise_if_not_geometrically_focused_wave(transmitted_wave)

        # Use the same width and height if height is not provided.
        beam_height = self.beam_height
        if beam_height is None:
            beam_height = self.beam_width

        # Get the effective aperture facing towards the virtual source.
        aperture = transmitting_probe.get_effective_aperture(
            transmitted_wave.virtual_source
        )

        # Set the specified constant width and height.
        aperture = replace(aperture, width=self.beam_width, height=beam_height)
        return aperture.project_and_apply_window(point, self.window)
