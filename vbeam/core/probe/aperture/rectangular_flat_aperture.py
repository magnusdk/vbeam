from typing import Optional

from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core.probe.aperture.base import Aperture
from vbeam.geometry import (
    Direction,
    Plane,
    RectangularBounds,
    Rotation,
    Vector,
    average_directions,
)


class RectangularAperture(Aperture):
    """A flat, rectangular aperture.

    Attributes:
        plane (Plane): The plane that the flat aperture lies upon.
        width (float): The width of the rectangular aperture.
        height (float): The height of the rectangular aperture.
    """

    plane: Plane
    width: float
    height: float

    @property
    def origin(self) -> ops.array:
        return self.plane.origin

    @property
    def bounds(self) -> RectangularBounds:
        return RectangularBounds(self.plane, self.width, self.height)

    @property
    def orientation(self) -> Rotation:
        return self.plane.orientation

    @property
    def normal(self) -> Direction:
        return self.plane.normal

    def project_aperture(self, source: Vector) -> "RectangularAperture":
        # We find the normal vector (Direction) of the projected aperture by averaging
        # the directions pointing from each corner of the original aperture to the
        # virtual source.
        corners = self.bounds.corners
        corners_to_source_direction = (source - corners).direction
        projected_plane_normal = average_directions(
            corners_to_source_direction, axis="bounds_corners"
        )

        # Orient the plane of the projected aperture towards the direction of the
        # virtual source from the projected origin, but keep the roll unchanged.
        plane_orientation = Rotation.from_direction_and_roll(
            projected_plane_normal, self.plane.orientation.roll
        )
        oriented_plane = Plane(self.origin, plane_orientation)

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
        return RectangularAperture(projected_plane, projected_width, projected_height)

    def signed_distance(self, point: ops.array) -> float:
        return self.plane.signed_distance(point)

    def project_and_apply_window(self, point: ops.array, window: Window) -> float:
        x, y = self.plane.to_plane_coordinates(point)
        return window(x / self.width) * window(y / self.height)

    def scale(
        self,
        scale_width: float,
        scale_height: Optional[float] = None,
    ) -> "RectangularAperture":
        if scale_height is None:
            scale_height = scale_width
        return RectangularAperture(
            self.plane,
            self.width * scale_width,
            self.height * scale_height,
        )
