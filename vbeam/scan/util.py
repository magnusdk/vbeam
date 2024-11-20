from typing import TYPE_CHECKING, Optional, Tuple, Union

from fastmath import Array

from vbeam.fastmath import numpy as api
from vbeam.interpolation import FastInterpLinspace
from vbeam.util import _deprecations
from vbeam.util.arrays import grid

if TYPE_CHECKING:
    from vbeam.scan import SectorScan


def _ensure_min_and_max(min_x: float, max_x: float) -> Tuple[float, float]:
    "Swap ``min_x`` and ``max_x`` if ``min_x`` > ``max_x``."
    return tuple(
        api.where(
            min_x > max_x,
            api.array([max_x, min_x]),
            api.array([min_x, max_x]),
        )
    )


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
    cos_min, cos_max = api.cos(min_azimuth), api.cos(max_azimuth)
    sin_min, sin_max = api.sin(min_azimuth), api.sin(max_azimuth)

    # Get the maximum x coordinate of the corners of both the inner and outer arc.
    max_corner_x = api.max(
        api.array(
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
    return api.where(
        (max_azimuth - min_azimuth) < api.pi,
        api.where(api.logical_and(sin_min < 0, sin_max > 0), max_depth, max_corner_x),
        api.where(api.logical_or(sin_min < 0, sin_max > 0), max_depth, max_corner_x),
    )


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
    quarter_turn = api.pi / 2
    half_turn = api.pi
    # Return (left, right, top, bottom)
    return (
        -_right_bound(min_az + quarter_turn, max_az + quarter_turn, min_d, max_d),
        _right_bound(min_az - quarter_turn, max_az - quarter_turn, min_d, max_d),
        -_right_bound(min_az + half_turn, max_az + half_turn, min_d, max_d),
        _right_bound(min_az, max_az, min_d, max_d),
    )


@_deprecations.renamed_kwargs("1.0.5", imaged_points="image", sector_scan="bounds")
def scan_convert(
    image: Array,
    bounds: Union[Tuple[float, float, float, float], "SectorScan"],
    azimuth_axis: int = -2,
    depth_axis: int = -1,
    *,  # Remaining args must be passed by name (to avoid confusion)
    shape: Optional[Tuple[int, int]] = None,
    default_value: Optional[Array] = 0.0,
    edge_handling: str = "Value",
):
    from vbeam.scan import CoordinateSystem, Scan

    if isinstance(bounds, Scan):
        if not bounds.coordinate_system == CoordinateSystem.POLAR:
            raise ValueError("You may only scan convert from polar coordinates.")
        bounds = bounds.bounds_beamspace
    if shape is None:
        shape = image.shape[azimuth_axis], image.shape[depth_axis]

    width, height = image.shape[azimuth_axis], image.shape[depth_axis]
    min_az, max_az, min_depth, max_depth = bounds

    # Get the points in the cartesian grid
    min_x, max_x, min_z, max_z = polar_bounds_to_cartesian_bounds(bounds)
    points = grid(
        api.linspace(min_x, max_x, shape[0]),
        api.linspace(min_z, max_z, shape[1]),
    )
    x, z = points[..., 0], points[..., 1]  # (Ignore y; scan_convert only supports 2D!)
    # and transform each point to polar coordinates.
    angles = api.arctan2(x, z)
    radii = api.sqrt(x**2 + z**2)

    # Interpolate the imaged points, sampled at the transformed points.
    return FastInterpLinspace.interp2d(
        angles,
        radii,
        FastInterpLinspace(min_az, (max_az - min_az) / (width - 1), width),
        FastInterpLinspace(min_depth, (max_depth - min_depth) / (height - 1), height),
        image,
        azimuth_axis,
        depth_axis,
        default_value=default_value,
        edge_handling=edge_handling,
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
