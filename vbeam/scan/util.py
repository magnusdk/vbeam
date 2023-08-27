from typing import TYPE_CHECKING, Optional, Tuple

from vbeam.fastmath import numpy as np
from vbeam.interpolation import FastInterpLinspace
from vbeam.util.arrays import grid

if TYPE_CHECKING:
    from vbeam.scan import SectorScan


def scan_convert(
    imaged_points: np.ndarray,
    sector_scan: "SectorScan",
    azimuth_axis: int = -2,
    depth_axis: int = -1,
    *,  # Remaining args must be passed by name (to avoid confusion)
    shape: Optional[Tuple[int, int]] = None,
    padding: Optional[np.ndarray] = 0.0,
):
    if shape is None:
        shape = sector_scan.shape

    # Get the points in the cartesian grid
    min_x, max_x, min_z, max_z = sector_scan.cartesian_bounds
    points = grid(
        np.linspace(min_x, max_x, shape[0]),
        np.linspace(min_z, max_z, shape[1]),
    )
    x, z = points[..., 0], points[..., 1]  # (Ignore y; scan_convert only supports 2D!)
    # and transform each point to polar coordinates.
    angles = np.arctan2(x, z)
    radii = np.sqrt(x**2 + z**2)

    # Interpolate the imaged points, sampled at the transformed points.
    return FastInterpLinspace.interp2d(
        angles,
        radii,
        FastInterpLinspace.from_array(sector_scan.azimuths),
        FastInterpLinspace.from_array(sector_scan.depths),
        imaged_points,
        azimuth_axis,
        depth_axis,
        padding=padding,
    )


def parse_axes(xyz):
    "Internal utility function for parsing axes. Handles both 2D and 3D scans."
    if len(xyz) == 3:
        x, y, z = xyz
    elif len(xyz) == 2:
        x, z = xyz
        y = None
    else:
        raise ValueError(
            f"Provide either x, y, and z (3D) or only x and z (2D). Got {len(xyz)} axes"
        )
    return x, y, z
