# NOTE: Work-in-progress :)
# TODO: Support curved probes. How does that work? 🤔

from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, GeometricallyFocusedWave, Probe


class ExpandingAperture(Apodization):
    window: Window
    f_number: float

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
    ) -> float:
        # TODO: Something smells bad with this line of code. What does it mean to the
        # effective aperture from the probe for expanding aperture?
        aperture = receiving_probe.get_effective_aperture(
            receiving_probe.active_elements.normal
        )

        depth = aperture.plane.signed_distance(point)

        # Ensure width and height does not exceed the full aperture or is smaller than
        # four elements (NOTE: the number "four" chosen a bit arbitrarily).
        min_width = receiving_probe.active_elements.width * 4
        min_height = receiving_probe.active_elements.height * 4
        # Include two additional elements when finding the max width. Otherwise, the
        # edge elements will be weighted to 0, because Tukey window becomes zero at
        # either side.
        max_width = aperture.width + receiving_probe.active_elements.width
        max_height = aperture.height + receiving_probe.active_elements.height
        width = ops.clip(depth / self.f_number, min_width, max_width)
        height = ops.clip(depth / self.f_number, min_height, max_height)

        x, y = aperture.plane.to_plane_coordinates(point)
        x = ops.clip(x, (-max_width + width) / 2, (max_width - width) / 2)
        y = ops.clip(y, (-max_height + height) / 2, (max_height - height) / 2)

        elements_x, elements_y = aperture.plane.to_plane_coordinates(
            receiving_probe.active_elements.position
        )
        elements_x, elements_y = elements_x - x, elements_y - y
        aperture = aperture.set_size(width=width, height=height)
        return aperture.apply_window(elements_x, elements_y, self.window)
