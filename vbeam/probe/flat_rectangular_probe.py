from typing import Union

from spekk import ops, replace

from vbeam import geometry
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
    def center(self) -> ops.array:
        return self.plane.origin

    @property
    def bounds(self) -> RectangularBounds:
        return RectangularBounds(self.plane, self.width, self.height)

    def get_effective_aperture(
        self, virtual_source: Union[ops.array, Vector, VectorWithInfiniteMagnitude]
    ) -> Aperture:
        if isinstance(virtual_source, VectorWithInfiniteMagnitude):
            projected_plane = self.plane.from_origin_and_normal(
                self.plane.origin,
                normal=virtual_source.direction,
                normal_is_normalized=True,
            )

        else:
            if isinstance(virtual_source, Vector):
                virtual_source = virtual_source.to_array()

            # We find the normal vector (Direction) of the projected aperture by
            # averaging the directions pointing from each corner of the probe to the
            # virtual source.
            corners_to_source_vectors = virtual_source - self.bounds.corners
            projected_plane_normal = ops.mean(
                geometry.util.normalize_vector(corners_to_source_vectors),
                axis="bounds_corners",
            )

            # Orient the plane of the projected aperture towards the direction of the
            # virtual source from the projected origin, but keep the roll unchanged.
            oriented_plane = self.plane.from_origin_and_normal(
                self.plane.origin, projected_plane_normal, normal_is_normalized=True
            )

            # Get the origin of the projected plane by projecting the virtual source
            # onto the oriented plane.
            projected_origin = oriented_plane.project(virtual_source)
            projected_plane = replace(oriented_plane, origin=projected_origin)

        # Update the width and height of the projected aperture.
        cos_azimuth = ops.vecdot(
            self.plane.basis_x, projected_plane.basis_x, axis="xyz"
        )
        cos_elevation = ops.vecdot(
            self.plane.basis_y, projected_plane.basis_y, axis="xyz"
        )
        projected_width = self.width * ops.abs(cos_azimuth)
        projected_height = self.height * ops.abs(cos_elevation)

        return RectangularAperture(projected_plane, projected_width, projected_height)
