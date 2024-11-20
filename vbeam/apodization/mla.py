from typing import Tuple

from fastmath import Array, Array

from vbeam.core import Apodization, ProbeGeometry, WaveData
from vbeam.fastmath import numpy as api
from vbeam.util.geometry.v2 import Line

from .window import Window


class MLAApodization(Apodization):
    """Perform multiple line acquisition (MLA) in cartesian space.

    The number of lines acquired per transmit is determined by beam_width. If beam_width
    is the same as the delta_x of the scan (assuming of course that the scan dimensions
    are linear) then this apodization would produce approximately the same effect as
    using scanline imaging."""

    window: Window
    beam_width: float
    array_bounds_x: Tuple[float, float]

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        receiver: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        # Geometry assumes 2D points
        point_position = point_position[api.array([0, 2])]
        source = wave_data.source[api.array([0, 2])]

        array_left = api.array([self.array_bounds_x[0], 0])
        array_right = api.array([self.array_bounds_x[1], 0])
        mid_line_angle = (
            Line.passing_through(array_left, source).angle
            + Line.passing_through(array_right, source).angle
        ) / 2
        array_line = Line.passing_through(array_left, array_right)
        _, sender_position = array_line.intersect(
            Line.with_angle(source, mid_line_angle)
        )

        # The distance from point_position to the nearest point on the line that
        # intersects sender_position and wave_data.source.
        scanline = Line.passing_through(sender_position, source)
        dist = api.abs(scanline.signed_distance(point_position))
        return self.window(dist / (self.beam_width))
