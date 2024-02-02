from typing import TYPE_CHECKING, Optional, Tuple, Union

from vbeam.fastmath import numpy as np
from vbeam.interpolation import FastInterpLinspace
from vbeam.util import _deprecations
from vbeam.util.arrays import grid

if TYPE_CHECKING:
    from vbeam.scan import SectorScan


def polar_bounds_to_cartesian_bounds(
    bounds: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Take a tuple representing the polar coordinate bounds of a grid and return the
    cartesian coordinate bounds.

    Polar coordinate bounds is of form [min_azimuth, max_azimuth, min_depth, max_depth].
    Returned cartesian coordinate bounds is of form [min_x, max_x, min_z, max_z].
    """
    min_az, max_az, min_d, max_d = bounds
    # Ensure that the min and max are actually min and max
    min_az, max_az = _ensure_min_and_max(min_az, max_az)
    min_d, max_d = _ensure_min_and_max(min_d, max_d)

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


@_deprecations.renamed_kwargs("1.0.5", imaged_points="image", sector_scan="bounds")
def scan_convert(
    image: np.ndarray,
    bounds: Union[Tuple[float, float, float, float], "SectorScan"],
    azimuth_axis: int = -2,
    depth_axis: int = -1,
    *,  # Remaining args must be passed by name (to avoid confusion)
    shape: Optional[Tuple[int, int]] = None,
    padding: Optional[np.ndarray] = 0.0,
):
    from vbeam.scan import SectorScan

    if isinstance(bounds, SectorScan):
        bounds = bounds.bounds
    if shape is None:
        shape = image.shape

    width, height = image.shape
    min_az, max_az, min_depth, max_depth = bounds

    # Get the points in the cartesian grid
    min_x, max_x, min_z, max_z = polar_bounds_to_cartesian_bounds(bounds)
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
        FastInterpLinspace(min_az, (max_az - min_az) / width, width),
        FastInterpLinspace(min_depth, (max_depth - min_depth) / height, height),
        image,
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
