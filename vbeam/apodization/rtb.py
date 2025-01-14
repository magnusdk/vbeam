from dataclasses import replace

from spekk import ops

from vbeam import geometry
from vbeam.apodization.window import Window
from vbeam.core import Apodization, GeometricallyFocusedWave, Probe
from vbeam.util._transmitted_wave import raise_if_not_geometrically_focused_wave


def _minimum_aperture_size(
    wavelength: float,
    aperture_size: float,
    depths: ops.array,
    mainlobe_width_coefficient: float = 2 * 1.22,
):
    return mainlobe_width_coefficient * wavelength * depths / aperture_size


class RTBApodization(Apodization):
    """Spatially weights the delayed data according to a (simple) focused wave model.

    First, it projects the aperture towards the virtual source. Then it resizes the
    aperture as a function of depth and projects the imaged points onto the aperture
    and applies the window function.

    See it visually in a notebook by running this code:
    >>> from vbeam.apodization import RTBApodization, window
    >>> apodization = RTBApodization(window.Hamming(), 0.5e-3)
    >>> apodization.plot()
    """

    window: Window
    wavelength: float

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
    ) -> float:
        raise_if_not_geometrically_focused_wave(transmitted_wave)

        projected_aperture = transmitting_probe.active_aperture.project_aperture(
            transmitted_wave.virtual_source
        )
        virtual_source_depth = geometry.distance(
            transmitted_wave.virtual_source.to_array(), projected_aperture.center
        )
        depths = projected_aperture.plane.signed_distance(point)

        # focusing_scale creates the familiar RTB "hourglass" shape.
        focusing_scale = ops.abs(1 - depths / virtual_source_depth)

        # _minimum_aperture_size gives the opening apertude size close to the focus
        # point as a function of full width at half maximum (FWHM).
        width = ops.maximum(
            projected_aperture.width * focusing_scale,
            _minimum_aperture_size(self.wavelength, projected_aperture.width, depths),
        )
        height = ops.maximum(
            projected_aperture.height * focusing_scale,
            _minimum_aperture_size(self.wavelength, projected_aperture.height, depths),
        )

        # Set the width and height
        projected_aperture = replace(projected_aperture, width=width, height=height)
        return projected_aperture.project_and_apply_window(point, self.window)
