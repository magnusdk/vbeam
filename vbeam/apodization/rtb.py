from typing import Optional, Tuple

from vbeam.apodization.window import Window
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util import ensure_2d_point
from vbeam.util.geometry.v2 import Line, distance


def rtb_apodization(
    point: np.ndarray,
    array_left: np.ndarray,
    array_right: np.ndarray,
    focus_point: np.ndarray,
    minimum_aperture: float,
    window: Optional[Window] = None,
):
    """Focused apodization used for retrospective transmit beamforming (RTB).

    A focused apodization has two main parts:
    - An hourglass shape constructed by the area between two lines drawn from either
      side of the array and passing through the focus point (line_left and line_right).
    - A line with a thickness given by minimum_aperture that passes through the focus point,
      to account for the fact that the hourglass apodization is infinitely narrow at the
      focus point.

    A point that lies both outside of the hourglass and outside of the thickness of the
    line is weighted by 0.
    """
    # Ensure that all points are 2D (only 2D is supported)
    point = ensure_2d_point(point)
    array_left = ensure_2d_point(array_left)
    array_right = ensure_2d_point(array_right)
    focus_point = ensure_2d_point(focus_point)

    # Set up geometry
    line_left = Line.passing_through(array_left, focus_point)
    line_right = Line.passing_through(array_right, focus_point)
    line_mid = Line.with_angle(focus_point, (line_left.angle + line_right.angle) / 2)
    line_mid_perpendicular = Line(point, line_mid.normal)

    # Short circuit if no window is given (same as giving a Rectangular window)
    if window is None:
        return np.logical_or(
            # Hourglass apodization
            line_left.signed_distance(point) * line_right.signed_distance(point) <= 0,
            # Middle line apodization
            np.abs(line_mid.signed_distance(point)) < minimum_aperture,
        )

    # Calculate distances
    _, intersection_left = line_left.intersect(line_mid_perpendicular)
    _, intersection_right = line_right.intersect(line_mid_perpendicular)
    distance_left = distance(intersection_left, point)
    total_distance = distance(intersection_left, intersection_right)
    distance_mid = np.abs(line_mid.signed_distance(point))

    # Calculate apodization
    p = distance_left / total_distance  # Window parameter
    valid = line_left.signed_distance(point) * line_right.signed_distance(point) <= 0
    hourglass_apodization = window(np.abs(p - 0.5)) * valid
    mid_line_apodization = window(distance_mid / minimum_aperture)

    # Take the max of the two apodizations
    return np.where(
        hourglass_apodization >= mid_line_apodization,
        hourglass_apodization,
        mid_line_apodization,
    )


@traceable_dataclass((("array_bounds", "minimum_aperture", "window")))
class RTBApodization(Apodization):
    array_bounds: Tuple[float, float]
    minimum_aperture: float = 0.001  # TODO: Calculate this based on F# and wavelength
    window: Optional[Window] = None

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        return rtb_apodization(
            point_position,
            self.array_bounds[0],
            self.array_bounds[1],
            wave_data.source,
            self.minimum_aperture,
            self.window,
        )
