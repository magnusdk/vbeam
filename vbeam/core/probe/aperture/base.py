from abc import abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, TypeVar

from spekk import Module, ops

from vbeam.geometry import Direction, Orientation, Plane, RectangularBounds, Vector

if TYPE_CHECKING:
    from vbeam.apodization.window import Window


TPlanarAperture = TypeVar("TPlanarAperture", bound="PlanarAperture")


class Aperture(Module):
    """An Aperture is the region of an ultrasound probe that is either actively
    transmitting or receiving.

    This class is meant to simplify the implementation of various apodization
    functions and delay models (or really anything that involves geometric focusing of
    transmitted waves.)
    """

    @abstractmethod
    def project_aperture(self, source: Vector) -> "PlanarAperture":
        """Return the effective aperture when geometrically focusing towards a virtual
        source.

        This is done by projecting the aperture onto a flat plane that is oriented
        towards the virtual source. The returned aperture is a
        :class:`~vbeam.core.probe.aperture.PlanarAperture` which is a flat, oriented
        aperture with a width and a height.
        """


class PlanarAperture(Aperture):
    """A flat aperture with a width and a height lying on a plane.

    Attributes:
        plane (Plane): The flat, oriented Euclidian 2D plane that the aperture lies on.
        width (float): The width of the aperture.
        height (float): The height of the aperture.

    A PlanarAperture can represent an actually flat aperture or an aperture that has
    been projected towards a virtual source (in the context of geometric focusing).
    """

    plane: Plane
    width: float
    height: float

    @property
    def center(self) -> ops.array:
        "The center of the aperture."
        return self.plane.origin

    @property
    def orientation(self) -> Orientation:
        "The orientation of the aperture."
        return self.plane.orientation

    @property
    def normal(self) -> Direction:
        "The direction that the aperture points in."
        return self.plane.normal

    @property
    def bounds(self) -> RectangularBounds:
        "The 2D rectangular bounds of the aperture in plane coordinates."
        return RectangularBounds(self.plane, self.width, self.height)

    def set_origin(self: TPlanarAperture, origin: ops.array) -> TPlanarAperture:
        "Set the center of the probe to a new value."
        return replace(self, plane=Plane(origin, self.plane.orientation))

    def scale(
        self: TPlanarAperture,
        scale_width: float,
        scale_height: Optional[float] = None,
    ) -> TPlanarAperture:
        """Scale the width and height of the aperture.

        Args:
            scale_width (float): How much to scale the width of the aperture
            scale_height (Optional[float]): How much to scale the height of the
                aperture. If `scale_height` is `None` (default), then it is scaled
                equally in width and height.
        """
        if scale_height is None:
            scale_height = scale_width
        return replace(
            self,
            width=self.width * scale_width,
            height=self.height * scale_height,
        )

    def project_and_apply_window(self, point: ops.array, window: "Window") -> float:
        """Project the `point` onto the plane and apply the `window` function according
        to where the projected point lies on the plane.
        """
        x, y = self.plane.to_plane_coordinates(point)
        return self.apply_window(x, y, window)

    @abstractmethod
    def apply_window(self, x: float, y: float, window: "Window") -> float:
        "Apply the given `window`, given the 2D point (`x`, `y`) in plane coordinates."
