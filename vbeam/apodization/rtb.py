from typing import Optional, Tuple

from vbeam.apodization.window import Window
from vbeam.core import Apodization, ProbeGeometry, WaveData
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
        is_within_hourglass = (
            line_left.signed_distance(point) * line_right.signed_distance(point) <= 0
        )
        is_within_min_aperture = (
            np.abs(line_mid.signed_distance(point)) < minimum_aperture
        )
        is_within_max_aperture = (
            maximum_aperture is None
            or np.abs(line_mid.signed_distance(point)) < maximum_aperture
        )
        return np.logical_and(
            np.logical_or(is_within_hourglass, is_within_min_aperture),
            is_within_max_aperture,
        )

    # Calculate distances
    _, intersection_left = line_left.intersect(line_mid_perpendicular)
    _, intersection_right = line_right.intersect(line_mid_perpendicular)
    distance_left = distance(intersection_left, point)
    total_distance = distance(intersection_left, intersection_right)
    distance_mid = np.abs(line_mid.signed_distance(point))

    # Calculate apodizations
    valid = line_left.signed_distance(point) * line_right.signed_distance(point) <= 0
    hourglass_apodization = window(np.abs(distance_left / total_distance - 0.5)) * valid
    minimum_aperture_apodization = window(distance_mid / minimum_aperture)

    # Combine hourglass, minimum aperture, and maximum aperture (assumes window is
    # monotonically decreasing from 0 to 0.5).
    value = np.maximum(hourglass_apodization, minimum_aperture_apodization)
    if maximum_aperture is not None:
        value = np.minimum(value, window(distance_mid / maximum_aperture))
    return value

@traceable_dataclass(("minimum_aperture", "maximum_aperture", "window","replacement_sender") )
class RTBApodization(Apodization):
    minimum_aperture: float = 0.001  # TODO: Calculate this based on F# and wavelength
    maximum_aperture: Optional[float] = None
    window: Optional[Window] = None
    replacement_sender: Optional[np.ndarray] = None

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        receiver: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        if self.replacement_sender is None:
            sender_element_position =  sender
        else:
            sender_element_position =  self.replacement_sender
        array_left, array_right,array_up, array_down = probe.get_tx_aperture_borders(sender=sender_element_position)
        return rtb_apodization(
            point_position,
            array_left,
            array_right,
            wave_data.source,
            self.minimum_aperture,
            self.maximum_aperture,
            self.window,
        )

@traceable_dataclass(("minimum_aperture", "window","replacement_sender"))
class SteppingApertureRTBApodization(Apodization):
    minimum_aperture: float = 0.001  # TODO: Calculate this based on F# and wavelength
    window: Optional[Window] = None
    replacement_sender: np.ndarray = None

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        receiver: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        if self.replacement_sender is None:
            sender_element_position =  sender
        else:
            sender_element_position =  self.replacement_sender
        
        array_left, array_right,array_up, array_down = probe.get_tx_aperture_borders(sender=sender_element_position)
        return rtb_apodization(
            point_position,
            array_left,
            array_right,
            wave_data.source,
            self.minimum_aperture,
            self.maximum_aperture,
            self.window,
        )

def get_bounds(
    array_width,
    sender,
    use_parent,
):
    sender_element_position = np.where(
        use_parent, sender.parent_element.position, sender.position
    )
    sender_element_theta = np.where(
        use_parent, sender.parent_element.theta, sender.theta
    )
    sender_normal = np.array(
        [np.sin(sender_element_theta), 0.0, np.cos(sender_element_theta)]
    )
    array_left = (
        sender_element_position
        + np.cross(np.array([0, -1, 0]), sender_normal) * array_width / 2
    )
    array_right = (
        sender_element_position
        + np.cross(np.array([0, 1, 0]), sender_normal) * array_width / 2
    )
    return [array_left, array_right]
