from typing import Callable, Literal, Optional, Tuple, Union, overload

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.scan.base import CoordinateSystem, Scan
from vbeam.scan.util import parse_axes, polar_bounds_to_cartesian_bounds, scan_convert
from vbeam.util import _deprecations
from vbeam.util.arrays import grid
from vbeam.util.coordinate_systems import as_cartesian


@traceable_dataclass(("azimuths", "elevations", "depths", "apex"))
class SectorScan(Scan):
    azimuths: np.ndarray
    elevations: Optional[np.ndarray]  # May be None for 2D scans
    depths: np.ndarray
    apex: np.ndarray

    def get_points(self, flatten: bool = True) -> np.ndarray:
        polar_axis = self.elevations if self.is_3d else np.array([0.0])
        points = grid(self.azimuths, polar_axis, self.depths, shape=(*self.shape, 3))
        points = as_cartesian(points)
        # Ensure that points and apex are broadcastable
        apex = (
            np.expand_dims(self.apex, axis=tuple(range(1, self.ndim)))
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
        azimuths: Union[np.ndarray, None, Literal["unchanged"]] = "unchanged",
        elevations: Union[np.ndarray, None, Literal["unchanged"]] = "unchanged",
        depths: Union[np.ndarray, None, Literal["unchanged"]] = "unchanged",
        apex: Union[np.ndarray, None, Literal["unchanged"]] = "unchanged",
    ) -> "SectorScan":
        return SectorScan(
            azimuths=azimuths if azimuths != "unchanged" else self.azimuths,
            elevations=elevations if elevations != "unchanged" else self.elevations,
            depths=depths if depths != "unchanged" else self.depths,
            apex=apex if apex != "unchanged" else self.apex,
        )

    def update(
        self,
        azimuths: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        elevations: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        depths: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        apex: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
                np.linspace(self.azimuths[0], self.azimuths[-1], azimuths)
                if azimuths is not None
                else "unchanged"
            ),
            elevations=(
                np.linspace(self.elevations[0], self.elevations[-1], elevations)
                if elevations is not None
                else "unchanged"
            ),
            depths=(
                np.linspace(self.depths[0], self.depths[-1], depths)
                if depths is not None
                else "unchanged"
            ),
        )

    @property
    def axes(self) -> Tuple[np.ndarray, ...]:
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

    @_deprecations.renamed_kwargs("1.0.5", imaged_points="image")
    def scan_convert(
        self,
        image: np.ndarray,
        azimuth_axis: int = -2,
        depth_axis: int = -1,
        *,  # Remaining args must be passed by name (to avoid confusion)
        shape: Optional[Tuple[int, int]] = None,
        default_value: Optional[np.ndarray] = 0.0,
        edge_handling:str ="Value",
    ):
        return scan_convert(
            image, self, azimuth_axis, depth_axis, shape=shape, default_value=default_value,edge_handling=edge_handling,
        )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.POLAR

    def __repr__(self):
        return f"SectorScan(<shape={self.shape}>, apex={self.apex})"


@overload
def sector_scan(
    azimuths: np.ndarray,
    depths: np.ndarray,
    apex: Union[np.ndarray, float] = 0.0,
) -> SectorScan: ...  # 2D scan


@overload
def sector_scan(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    depths: np.ndarray,
    apex: Union[np.ndarray, float] = 0.0,
) -> SectorScan: ...  # 3D scan


def sector_scan(*axes: np.ndarray, apex: Union[np.ndarray, float] = 0.0) -> SectorScan:
    "Construct a sector scan. See SectorScan documentation for more details."
    azimuths, elevations, depths = parse_axes(axes)
    return SectorScan(azimuths, elevations, depths, np.array(apex))
