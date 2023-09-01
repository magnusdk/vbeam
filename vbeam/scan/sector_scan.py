from typing import Callable, Literal, Optional, Tuple, Union, overload

import numpy

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan.base import CoordinateSystem, Scan
from vbeam.scan.util import parse_axes
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
        """Get the bounds of the scan in cartesian coordinates. It is the same as
        bounding box of the scan-converted image."""
        if self.is_3d:
            raise NotImplementedError(
                "Cartesian bounds are not implemented for 3D scans yet.",
                "Please create an issue on Github if this is something you need.",
            )
        min_az, max_az, min_d, max_d = self.bounds
        # We get the bounds by calculating the bound for each edge of the bounding box
        # individually. _right_bound gets the right-most x coordinate of the bounding
        # box and we can get the other sides by rotating the azimuth bounds by 90, 180,
        # and 270 degrees. Because in ultrasound, "straight down" is at 0 degrees, we
        # have to rotate everything by an additional 90 degrees.
        quarter_turn = np.pi / 2
        half_turn = np.pi
        # Return (left, right, top, bottom)
        return (
            -_right_bound(min_az + quarter_turn, max_az + quarter_turn, min_d, max_d),
            _right_bound(min_az - quarter_turn, max_az - quarter_turn, min_d, max_d),
            -_right_bound(min_az + half_turn, max_az + half_turn, min_d, max_d),
            _right_bound(min_az, max_az, min_d, max_d),
        )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.POLAR

    def __repr__(self):
        return f"SectorScan(<shape={self.shape}>, apex={self.apex})"


def _right_bound(
    min_azimuth: float,
    max_azimuth: float,
    min_depth: float,
    max_depth: float,
) -> float:
    """For the two arcs defined by the azimuth bounds (one for ``min_depth`` and one
    for ``max_depth``), find the right-most x coordinate of the two arcs.

    This will be the right-edge of a bounding box that encompasses the arcs. You can
    get the other edges of the bounding box by rotating the azimuth bounds by 90, 180,
    and 270 degrees.

    See ``docs/tutorials/scan/sector_scan_bounds.ipynb`` for a visualization of the 
    bounding box of the arcs."""
    cos_min, cos_max = np.cos(min_azimuth), np.cos(max_azimuth)
    sin_min, sin_max = np.sin(min_azimuth), np.sin(max_azimuth)

    # Get the maximum x coordinate of the corners of both the inner and outer arc.
    max_corner_x = np.max(
        np.array(
            [
                cos_min * min_depth,  # Inner arc
                cos_max * min_depth,  # Inner arc
                cos_min * max_depth,  # Outer arc
                cos_max * max_depth,  # Outer arc
            ]
        )
    )

    # The right-most part of the arcs may either be ``max_corner_x``, or it may be on
    # the right-most *tangent* of the outer arc. We have to make some additional checks
    # to make this work. The code for this is a bit terse, so just trust the generative
    # unit tests for :attr:`SectorScan.cartesian_bounds` :)
    return np.where(
        (max_azimuth - min_azimuth) < np.pi,
        np.where(np.logical_and(sin_min < 0, sin_max > 0), max_depth, max_corner_x),
        np.where(np.logical_or(sin_min < 0, sin_max > 0), max_depth, max_corner_x),
    )


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
    azimuths, elevations, depths = parse_axes(axes)
    return SectorScan(azimuths, elevations, depths, np.array(apex))
