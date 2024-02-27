from typing import Optional, Tuple

from vbeam.apodization.expanding_aperture import expanding_aperture
from vbeam.apodization.window import Rectangular, Window
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util import ensure_2d_point
from vbeam.util.geometry.v2 import Line, distance

# TODO:
# - Fix docstring referencing minimum_aperture
# - Use something more intuitive than f_number — possibly opening_angle?
# - Fix geometry for a more smooth combination of 2D and 3D geometry

def rtb_apodization(
    point: np.ndarray,
    array_left2d: np.ndarray,
    array_right2d: np.ndarray,
    focus_point2d: np.ndarray,
    f_number: Tuple[float, float],
    maximum_aperture: Optional[float] = None,
    window: Optional[Window] = None,
):
    """Focused apodization used for retrospective transmit beamforming (RTB).

    A focused apodization has three main parts:
    - An hourglass shape constructed by the area between two lines drawn from either
      side of the array and passing through the focus point (line_left and line_right).
    - A line with a thickness given by minimum_aperture that passes through the focus
      point, to account for the fact that the hourglass apodization is infinitely
      narrow at the focus point.
    - A line with a thickness given by maximum_aperture that passes through the focus
      point that sets the maximum width of the hourglass apodization.
    """
    # Ensure that all points are 2D (only 2D is supported)
    point2d = ensure_2d_point(point)
    array_left2d = ensure_2d_point(array_left2d)
    array_right2d = ensure_2d_point(array_right2d)
    focus_point2d = ensure_2d_point(focus_point2d)

    # Set up geometry
    line_left = Line.passing_through(array_left2d, focus_point2d)
    line_right = Line.passing_through(array_right2d, focus_point2d)
    line_mid = Line.with_angle(focus_point2d, (line_left.angle + line_right.angle) / 2)
    line_mid_perpendicular = Line(point2d, line_mid.normal)
    line_array = Line.passing_through(array_left2d, array_right2d)
    _, origin = line_mid.intersect(line_array)

    # NOTE: expanding_aperture supports 3D points
    expanding_aperture_weight = expanding_aperture(
        point,
        np.array([origin[0], 0, origin[1]]),
        np.arctan2(origin[0] - focus_point2d[0], origin[1] - focus_point2d[1]),
        0 * np.arctan2(distance(origin - focus_point2d), origin[1] - focus_point2d[1]),
        f_number,
        window,
    )

    # Short circuit if no window is given (same as giving a Rectangular window)
    if window is None:
        is_within_hourglass = (
            line_left.signed_distance(point2d) * line_right.signed_distance(point2d)
            <= 0
        )
        is_within_max_aperture = (
            maximum_aperture is None
            or np.abs(line_mid.signed_distance(point2d)) < maximum_aperture
        )
        return (
            np.logical_and(is_within_hourglass, is_within_max_aperture)
            * expanding_aperture_weight
        )

    # Calculate distances
    _, intersection_left = line_left.intersect(line_mid_perpendicular)
    _, intersection_right = line_right.intersect(line_mid_perpendicular)
    distance_left = distance(intersection_left, point2d)
    total_distance = distance(intersection_left, intersection_right)
    distance_mid = np.abs(line_mid.signed_distance(point2d))

    # Calculate apodizations
    valid = (
        line_left.signed_distance(point2d) * line_right.signed_distance(point2d) <= 0
    )
    hourglass_apodization = window(np.abs(distance_left / total_distance - 0.5)) * valid
    # Combine hourglass, minimum aperture, and maximum aperture (assumes window is
    # monotonically decreasing from 0 to 0.5).
    value = np.maximum(hourglass_apodization, expanding_aperture_weight)
    if maximum_aperture is not None:
        value = np.minimum(value, window(distance_mid / maximum_aperture))
    return value


@traceable_dataclass((("array_bounds", "f_number", "maximum_aperture", "window")))
class RTBApodization(Apodization):
    array_bounds: Tuple[np.ndarray, np.ndarray]
    f_number: Tuple[float, float]
    maximum_aperture: Optional[float] = None
    window: Optional[Window] = None

    def __post_init__(self):
        self.f_number = (
            self.f_number
            if isinstance(self.f_number, tuple) and len(self.f_number) == 2
            else (self.f_number, self.f_number)
        )

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
            self.f_number,
            self.maximum_aperture,
            self.window,
        )
