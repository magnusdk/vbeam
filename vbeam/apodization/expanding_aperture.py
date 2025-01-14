from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, GeometricallyFocusedWave, Probe

# TODO
class ExpandingAperture(Apodization):
    window: Window

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
    ) -> float:
        depths = ops.vecdot(
            point - receiving_probe.active_elements.position,
            receiving_probe.active_elements.normal.normalized_vector,
            axis="xyz",
        )
        expanding_aperture = receiving_probe.active_aperture.set_origin(
            receiving_probe.active_elements.position
        )
        expanding_aperture = expanding_aperture.set_size(1, 1)
        expanding_aperture = expanding_aperture.scale(depths * 0.2)

        return expanding_aperture.project_and_apply_window(point, self.window)
