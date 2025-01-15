from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, GeometricallyFocusedWave, Probe
from vbeam.util._transmitted_wave import raise_if_not_geometrically_focused_wave


class PlaneWaveTransmitApodization(Apodization):
    """Spatially weights the delayed data according to a (simple) plane wave model.

    First, it projects the aperture towards the virtual source, then it projects all
    imaged points onto the aperture and applies the window function.

    See it visually in a notebook by running this code:
    >>> from vbeam.apodization import PlaneWaveTransmitApodization, window
    >>> apodization = PlaneWaveTransmitApodization(window.Hamming())
    >>> apodization.plot()
    """

    window: Window

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
    ) -> float:
        raise_if_not_geometrically_focused_wave(transmitted_wave)

        aperture = transmitting_probe.get_effective_aperture(
            transmitted_wave.virtual_source
        )
        return aperture.project_and_apply_window(point, self.window)
