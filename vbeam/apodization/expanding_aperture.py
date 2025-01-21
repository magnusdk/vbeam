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

        depth = ops.vecdot(
            point - receiving_probe.active_elements.position,
            receiving_probe.active_elements.normal.normalized_vector,
            axis="xyz",
        )

        width = ops.abs(depth / self.f_number)
        height = ops.abs(depth / self.f_number)

        # Ensure width and height does not exceed the full aperture or is smaller than
        # two elements (NOTE: the number "two" chosen a bit arbitrarily).
        width = ops.clip(
            width,
            receiving_probe.active_elements.width * 2,
            aperture.width,
        )
        height = ops.clip(
            height,
            receiving_probe.active_elements.height * 2,
            aperture.height,
        )

        # TODO: This "bounding" logic should be part of the aperture, because not all
        # apertures are rectangular. They may be circular.
        # Convert to local plane 2D coordinates because that simplifies things a lot :)
        x, y = aperture.plane.to_plane_coordinates(point)

        # Clamp x and y so that the aperture doesn't fall outside of the full aperture.
        x = ops.clip(x, (-aperture.width + width) / 2, (aperture.width - width) / 2)
        y = ops.clip(y, (-aperture.height + height) / 2, (aperture.height - height) / 2)

        # Convert back to glbal 3D coordinates.
        point = aperture.plane.from_plane_coordinates(x, y)

        # Transform the aperture into an expanding aperture originating from the
        # receiving elements.
        aperture = aperture.set_origin(receiving_probe.active_elements.position)
        aperture = aperture.set_size(width=width, height=height)
        # Project the point onto the aperture and apply window.
        return aperture.project_and_apply_window(point, self.window)
