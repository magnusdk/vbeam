from spekk import ops

from vbeam.core.probe import Probe
from vbeam.core.probe.aperture import Aperture
from vbeam.geometry import (
    Orientation,
    Plane,
    RectangularBounds,
    Vector,
    average_directions,
)
from vbeam.probe.aperture.rectangular_aperture import RectangularAperture


class FlatRectangularProbe(Probe):
    plane: Plane
    width: float
    height: float

    @property
    def bounds(self) -> RectangularBounds:
        return RectangularBounds(self.plane, self.width, self.height)

    def get_effective_aperture(self, virtual_source: Vector) -> Aperture:
        # We find the normal vector (Direction) of the projected aperture by averaging
        # the directions pointing from each corner of the probe to the virtual source.
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
        oriented_plane = Plane(self.plane.origin, plane_orientation)

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

        return RectangularAperture(projected_plane, projected_width, projected_height)
