from typing import TYPE_CHECKING

from spekk import ops

from vbeam.core import PlanarAperture
from vbeam.geometry import (
    Orientation,
    Plane,
    Vector,
    average_directions,
)

if TYPE_CHECKING:
    from vbeam.apodization.window import Window


class RectangularPlanarAperture(PlanarAperture):
    """A flat, rectangular aperture lying on an oriented Euclidian 2D plane.

    Attributes:
        plane (Plane): The oriented Euclidian 2D plane that the aperture lies on.
        width (float): The width of the rectangular aperture.
        height (float): The height of the rectangular aperture.
    """

    def project_aperture(self, virtual_source: Vector) -> "RectangularPlanarAperture":
        # We find the normal vector (Direction) of the projected aperture by averaging
        # the directions pointing from each corner of the original aperture to the
        # virtual source.
        corners = self.bounds.corners
        corners_to_source_direction = (virtual_source - corners).direction
        projected_plane_normal = average_directions(
            corners_to_source_direction, axis="bounds_corners"
        )

        # Orient the plane of the projected aperture towards the direction of the
        # virtual source from the projected origin, but keep the roll unchanged.
        plane_orientation = Orientation.from_direction_and_roll(
            projected_plane_normal, self.plane.orientation.roll
        )
        oriented_plane = Plane(self.center, plane_orientation)

        # Project the corners onto the oriented plane along the directions pointing
        # from them to the virtual source. The projected corners are the corners of the
        # projected aperture.
        projected_corners = oriented_plane.project(
            corners, along=corners_to_source_direction
        )
        # The center of the projected corners is the center of the projected aperture.
        projected_origin = ops.mean(projected_corners, axis="bounds_corners")
        projected_plane = Plane(projected_origin, plane_orientation)

        # Find the width and height of the projected aperture.
        diff_azimuth = self.plane.normal.azimuth - projected_plane_normal.azimuth
        diff_elevation = self.plane.normal.elevation - projected_plane_normal.elevation
        projected_width = self.width * ops.cos(diff_azimuth)
        projected_height = self.height * ops.cos(diff_elevation)

        # Return the projected aperture :)
        return RectangularPlanarAperture(
            projected_plane, projected_width, projected_height
        )

    def apply_window(self, x: float, y: float, window: "Window") -> float:
        return window(x / self.width) * window(y / self.height)
