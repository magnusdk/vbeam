from typing import Union

from spekk import ops, replace

from vbeam.core.probe import Probe
from vbeam.core.probe.aperture import Aperture
from vbeam.geometry import (
    Plane,
    RectangularBounds,
    Vector,
    VectorWithInfiniteMagnitude,
)
from vbeam.probe.aperture.rectangular_aperture import RectangularAperture


class FlatRectangularProbe(Probe):
    plane: Plane
    width: float
    height: float

    @property
    def bounds(self) -> RectangularBounds:
        return RectangularBounds(self.plane, self.width, self.height)

    def get_effective_aperture(
        self, virtual_source: Union[ops.array, Vector, VectorWithInfiniteMagnitude]
    ) -> Aperture:
        # We find the normal vector (Direction) of the projected aperture by averaging
        # the directions pointing from each corner of the probe to the virtual source.
        corners = self.bounds.corners
        if isinstance(virtual_source, VectorWithInfiniteMagnitude):
            projected_plane_normal = virtual_source.direction
            corners_to_source_vectors = projected_plane_normal
        elif isinstance(virtual_source, Vector):
            corners_to_source_vectors = virtual_source.to_array() - corners
            projected_plane_normal = ops.mean(
                corners_to_source_vectors, axis="bounds_corners"
            )
        else:
            corners_to_source_vectors = virtual_source - corners
            projected_plane_normal = ops.mean(
                corners_to_source_vectors, axis="bounds_corners"
            )

        # Orient the plane of the projected aperture towards the direction of the
        # virtual source from the projected origin, but keep the roll unchanged.
        oriented_plane = self.plane.from_origin_and_normal(
            self.plane.origin,
            projected_plane_normal,
            normal_is_normalized=True,
        )

        # Project the corners onto the oriented plane along the directions pointing
        # from them to the virtual source. The projected corners are the corners of the
        # projected aperture.
        projected_corners = oriented_plane.project(
            corners,
            along=corners_to_source_vectors,
            along_is_normalized=True,
        )
        # The center of the projected corners is the center of the projected aperture.
        projected_origin = ops.mean(projected_corners, axis="bounds_corners")
        projected_plane = replace(oriented_plane, origin=projected_origin)

        # Find the width and height of the projected aperture.
        cos_azimuth = ops.vecdot(
            self.plane.basis_x,
            projected_plane.basis_x,
            axis="xyz",
        )
        cos_elevation = ops.vecdot(
            self.plane.basis_y,
            projected_plane.basis_y,
            axis="xyz",
        )
        projected_width = self.width * ops.abs(cos_azimuth)
        projected_height = self.height * ops.abs(cos_elevation)
        return RectangularAperture(projected_plane, projected_width, projected_height)
