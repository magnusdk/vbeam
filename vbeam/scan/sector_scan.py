from typing import Callable, Literal, Optional, Tuple, Union, overload

from spekk import ops

from vbeam.scan.base import CoordinateSystem, Scan
from vbeam.scan.util import parse_axes, polar_bounds_to_cartesian_bounds, scan_convert
from vbeam.util import _deprecations
from vbeam.util.arrays import grid
from vbeam.util.coordinate_systems import as_cartesian


class SectorScan(Scan):
    azimuths: ops.array
    elevations: Optional[ops.array]  # May be None for 2D scans
    depths: ops.array
    apex: ops.array

    def get_points(self, flatten: bool = True) -> ops.array:
        polar_axis = (
            self.elevations
            if self.is_3d
            else ops.array([0.0], dtype=self.azimuths.dtype)
        )
        points = grid(self.azimuths, polar_axis, self.depths, shape=(*self.shape, 3))
        points = as_cartesian(points)
        # Ensure that points and apex are broadcastable
        apex = (
            ops.expand_dims(self.apex, axis=tuple(range(1, self.ndim)))
            if self.apex.ndim > 1
            else self.apex
        )
        points = points + apex
        if flatten:
            points = points.reshape((self.num_points, 3))
        return points

    def replace(
        self,
        # "unchanged" means that the axis will not be changed.
        azimuths: Union[ops.array, None, Literal["unchanged"]] = "unchanged",
        elevations: Union[ops.array, None, Literal["unchanged"]] = "unchanged",
        depths: Union[ops.array, None, Literal["unchanged"]] = "unchanged",
        apex: Union[ops.array, None, Literal["unchanged"]] = "unchanged",
    ) -> "SectorScan":
        return SectorScan(
            azimuths=azimuths if azimuths != "unchanged" else self.azimuths,
            elevations=elevations if elevations != "unchanged" else self.elevations,
            depths=depths if depths != "unchanged" else self.depths,
            apex=apex if apex != "unchanged" else self.apex,
        )

    def update(
        self,
        azimuths: Optional[Callable[[ops.array], ops.array]] = None,
        elevations: Optional[Callable[[ops.array], ops.array]] = None,
        depths: Optional[Callable[[ops.array], ops.array]] = None,
        apex: Optional[Callable[[ops.array], ops.array]] = None,
    ) -> "SectorScan":
        return self.replace(
            azimuths(self.azimuths) if azimuths is not None else "unchanged",
            elevations(self.elevations) if elevations is not None else "unchanged",
            depths(self.depths) if depths is not None else "unchanged",
            apex(self.apex) if apex is not None else "unchanged",
        )

    def resize(
        self,
        azimuths: Optional[int] = None,
        elevations: Optional[int] = None,
        depths: Optional[int] = None,
    ) -> "SectorScan":
        if elevations is not None and self.elevations is None:
            raise ValueError(
                "Cannot resize elevations because it is not defined on this scan"
            )
        return self.replace(
            azimuths=(
                ops.linspace(self.azimuths[0], self.azimuths[-1], azimuths)
                if azimuths is not None
                else "unchanged"
            ),
            elevations=(
                ops.linspace(self.elevations[0], self.elevations[-1], elevations)
                if elevations is not None
                else "unchanged"
            ),
            depths=(
                ops.linspace(self.depths[0], self.depths[-1], depths)
                if depths is not None
                else "unchanged"
            ),
        )

    @property
    def axes(self) -> Tuple[ops.array, ...]:
        if self.elevations is not None:
            return self.azimuths, self.elevations, self.depths
        else:
            return self.azimuths, self.depths

    @property
    def cartesian_bounds(self):
        """Get the bounds of the scan in cartesian coordinates. It is the same as
        bounding box of the scan-converted image."""
        if self.is_3d:
            raise NotImplementedError(
                "Cartesian bounds are not implemented for 3D scans yet.",
                "Please create an issue on Github if this is something you need.",
            )
        return polar_bounds_to_cartesian_bounds(self.bounds)

    def cartesian_axes(self, shape) -> Tuple[ops.array, ops.array]:
        """Get the azimuth and depth vectos of the scan in cartesian coordinates."""
        if self.is_3d:
            raise NotImplementedError(
                "Cartesian bounds are not implemented for 3D scans yet.",
                "Please create an issue on Github if this is something you need.",
            )

        min_x, max_x, min_z, max_z = polar_bounds_to_cartesian_bounds(self.bounds)
        x_axis = ops.linspace(min_x, max_x, shape[0], dim="xs")
        z_axis = ops.linspace(min_z, max_z, shape[1], dim="zs")
        
        return (x_axis, z_axis)

    @_deprecations.renamed_kwargs("1.0.5", imaged_points="image")
    def scan_convert(
        self,
        image: ops.array,
        azimuth_axis: int = -2,
        depth_axis: int = -1,
        *,  # Remaining args must be passed by name (to avoid confusion)
        shape: Optional[Union[Tuple[int, int], str]] = None,
        default_value: Optional[ops.array] = 0.0,
        edge_handling: str = "Value",
        cartesian_axes: Optional[Tuple[ops.array, ops.array]] = None,
    ):
        return scan_convert(
            image,
            self,
            azimuth_axis,
            depth_axis,
            shape=shape,
            default_value=default_value,
            edge_handling=edge_handling,
            cartesian_axes=cartesian_axes,
        )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.POLAR

    def __repr__(self):
        return f"SectorScan(<shape={self.shape}>, apex={self.apex})"


@overload
def sector_scan(
    azimuths: ops.array,
    depths: ops.array,
    apex: Union[ops.array, float] = 0.0,
) -> SectorScan: ...  # 2D scan


@overload
def sector_scan(
    azimuths: ops.array,
    elevations: ops.array,
    depths: ops.array,
    apex: Union[ops.array, float] = 0.0,
) -> SectorScan: ...  # 3D scan


def sector_scan(*axes: ops.array, apex: Union[ops.array, float] = 0.0) -> SectorScan:
    "Construct a sector scan. See SectorScan documentation for more details."
    azimuths, elevations, depths = parse_axes(axes)
    return SectorScan(azimuths, elevations, depths, ops.array(apex))
