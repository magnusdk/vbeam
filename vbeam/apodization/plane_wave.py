from typing import Optional, Tuple, Union

from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util import ensure_2d_point
from vbeam.util.geometry.v2 import Line, distance

from .window import Window


@traceable_dataclass(("array_bounds", "window"))
class PlaneWaveTransmitApodization(Apodization):
    array_bounds: Tuple[np.ndarray, np.ndarray]
    window: Optional[Window] = None

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        # Set up the geometry. There is one line originating from each side of the
        # array, going in the direction of the transmitted wave. If the point is
        # outside of those lines, then it is weighted by 0.
        array_left, array_right = self.array_bounds

        # We are only supporting 2D points (for now).
        array_left = ensure_2d_point(array_left)
        array_right = ensure_2d_point(array_right)
        point_position = ensure_2d_point(point_position)

        # FIXME: There's something funky going on with the geometry code here 🤔 Angles
        # need to be inverted to make it work.
        azimuth = np.pi / 2 - wave_data.azimuth  # This is a hack to make it work.
        # Construct the lines going from either side of the array
        line_left = Line.with_angle(array_left, azimuth)
        line_right = Line.with_angle(array_right, azimuth)

        # The point is within the line if it has negative (to the left) signed distance
        # of one of the lines and positive (to the right) signed distance of the other.
        # Multiplying the signed distances gives a negative number if that's the case.
        # apodization_value is True if the point is within the lines, False otherwise.
        apodization_value = (
            line_left.signed_distance(point_position)
            * line_right.signed_distance(point_position)
        ) < 0

        if self.window is not None:
            # Apply the window function. It gradually goes to 0 as the point gets
            # closer to the edges of the apodization values.
            point_perp = Line.with_angle(point_position, azimuth + np.pi / 2)
            _, A = line_left.intersect(point_perp)
            _, B = line_right.intersect(point_perp)
            dist_A = distance(A, point_position)
            dist_B = distance(B, point_position)
            total_distance = dist_A + dist_B
            # p is 0 when the point is exactly between the two lines, and 0.5 when it is
            # on top of one of the lines.
            p = 0.5 - np.min(np.array([dist_A, dist_B])) / total_distance
            apodization_value *= self.window(p)

        # Ensure that it the returned value is a float.
        return apodization_value * 1.0


@traceable_dataclass(("window", "f_number"))
class PlaneWaveReceiveApodization(Apodization):
    window: Window
    f_number: Union[float, Tuple[float, float]]

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        f_number = (
            self.f_number
            if isinstance(self.f_number, tuple) and len(self.f_number) == 2
            else (self.f_number, self.f_number)
        )
        dist = point_position - receiver.position
        x_dist, y_dist, z_dist = dist[0], dist[1], dist[2]
        tan_theta = x_dist / z_dist
        tan_phi = y_dist / z_dist
        ratio_theta = np.abs(f_number[0] * tan_theta)
        ratio_phi = np.abs(f_number[1] * tan_phi)
        return np.array(
            self.window(ratio_theta) * self.window(ratio_phi),
            dtype="float32",
        )
