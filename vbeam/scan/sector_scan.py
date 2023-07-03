from typing import Callable, Literal, Optional, Tuple, Union, overload

import numpy

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan.base import Scan, _parse_axes
from vbeam.util import ensure_positive_index
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
        apex = np.expand_dims(self.apex, axis=tuple(range(1, self.ndim))) if self.apex.ndim > 1 else self.apex
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
    def shape(self) -> Tuple[int, ...]:
        return (
            (len(self.azimuths), len(self.elevations), len(self.depths))
            if self.elevations is not None
            else (len(self.azimuths), len(self.depths))
        )

    @property
    def bounds(self):
        bounds = []
        for axis in [self.azimuths, self.elevations, self.depths]:
            if axis is not None:
                bounds += [axis[0], axis[-1]]
        return tuple(bounds)

    @property
    def cartesian_bounds(self):
        points = self.get_points()
        # Ensure that points and apex are broadcastable
        apex = np.expand_dims(self.apex, axis=1) if self.apex.ndim > 1 else self.apex
        points -= apex
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        min_z, max_z = points[:, 2].min(), points[:, 2].max()
        if self.is_3d:
            return (min_x, max_x, min_y, max_y, min_z, max_z)
        if self.is_2d:
            return (min_x, max_x, min_z, max_z)

    def __repr__(self):
        return f"SectorScan(<shape={self.shape}>, apex={self.apex})"


# TODO: Research proper scan conversion
def cartesian_map(
    imaged_points: np.ndarray,
    scan: SectorScan,
    azimuth_axis: int = -2,
    depth_axis: int = -1,
    scale_azimuth: float = 1.0,
    scale_depth: Optional[float] = None,
):
    """Convert the imaged points to cartesian coordinates (scan-conversion)."""
    if not imaged_points.ndim >= 2:
        raise ValueError(
            "Image must be 2D in order to perform scan conversion. Did you forget to \
unflatten the imaged points?"
        )

    scale_depth = scale_azimuth if scale_depth is None else scale_depth

    # Use the sizes of the imaged_points, assuming they are defined from this scan.
    # The points may have been resampled, but the bounds of the scan should still
    # be the same.
    azimuth_size = imaged_points.shape[azimuth_axis]
    depth_size = imaged_points.shape[depth_axis]
    # Use numpy.ceil instead of np.ceil to avoid dynamic shapes when JIT-compiling
    new_azimuth_size = numpy.ceil(azimuth_size * scale_azimuth).astype("int")
    new_depth_size = numpy.ceil(depth_size * scale_depth).astype("int")

    min_x, max_x, min_z, max_z = scan.cartesian_bounds

    new_points = grid(
        np.linspace(min_x, max_x, new_azimuth_size),
        np.linspace(min_z, max_z, new_depth_size),
    )
    x, z = new_points[..., 0], new_points[..., -1]
    angles = np.arctan2(x, z)
    radii = np.sqrt(x**2 + z**2)

    # Ensure that x-axis comes first, followed by y-axis.
    azimuth_axis = ensure_positive_index(imaged_points.ndim, azimuth_axis)
    depth_axis = ensure_positive_index(imaged_points.ndim, depth_axis)
    if depth_axis == 0 and azimuth_axis == 1:
        imaged_points = np.swapaxes(imaged_points, azimuth_axis, depth_axis)
    else:
        imaged_points = np.moveaxis(imaged_points, azimuth_axis, 0)
        imaged_points = np.moveaxis(imaged_points, depth_axis, 1)

    # Interpolate image using the cartesian points
    min_az, max_az, min_depth, max_depth = scan.bounds
    interp_x = FastInterpLinspace(
        min_az, (max_az - min_az) / (azimuth_size - 1), azimuth_size
    )
    interp_z = FastInterpLinspace(
        min_depth, (max_depth - min_depth) / (depth_size - 1), depth_size
    )
    imaged_points = FastInterpLinspace.interp2d(
        angles, radii, interp_x, interp_z, imaged_points
    )

    # Swap axes back to original shape
    if depth_axis == 0 and azimuth_axis == 1:
        imaged_points = np.swapaxes(imaged_points, azimuth_axis, depth_axis)
    else:
        imaged_points = np.moveaxis(imaged_points, 1, depth_axis)
        imaged_points = np.moveaxis(imaged_points, 0, azimuth_axis)
    return imaged_points


@overload
def sector_scan(
    azimuths: np.ndarray,
    depths: np.ndarray,
    apex: Union[np.ndarray, float] = 0.0,
) -> SectorScan:
    ...  # 2D scan


@overload
def sector_scan(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    depths: np.ndarray,
    apex: Union[np.ndarray, float] = 0.0,
) -> SectorScan:
    ...  # 3D scan


def sector_scan(*axes: np.ndarray, apex: Union[np.ndarray, float] = 0.0) -> SectorScan:
    "Construct a sector scan. See SectorScan documentation for more details."
    azimuths, elevations, depths = _parse_axes(axes)
    return SectorScan(azimuths, elevations, depths, np.array(apex))
