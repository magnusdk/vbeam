from typing import TYPE_CHECKING

from vbeam.core import Aperture

if TYPE_CHECKING:
    from vbeam.apodization.window import Window


class RectangularAperture(Aperture):
    """A flat, rectangular aperture lying on an oriented Euclidian 2D plane.

    Attributes:
        plane (Plane): The oriented Euclidian 2D plane that the aperture lies on.
        width (float): The width of the rectangular aperture.
        height (float): The height of the rectangular aperture.
    """

    def apply_window(self, x: float, y: float, window: "Window") -> float:
        return window(x / self.width) * window(y / self.height)
