from typing import Tuple

from vbeam.core import ElementGeometry, TransmittedWavefront, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry import Line
from vbeam.wavefront import FocusedSphericalWavefront


@traceable_dataclass(("array_bounds", "base_wavefront"))
class UnifiedWavefront(TransmittedWavefront):
    """Implementation of the unified wavefront model

    https://doi.org/10.1109/tmi.2015.2456982"""

    array_bounds: Tuple[np.ndarray, np.ndarray]
    base_wavefront: TransmittedWavefront = FocusedSphericalWavefront()

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        # Set up the geometry
        array_left, array_right = self.array_bounds
        line_left = Line.passing_through(array_left, wave_data.source)
        line_right = Line.passing_through(array_right, wave_data.source)
        scanline = Line.passing_through(sender.position, wave_data.source)
        intersection_line = Line.from_anchor_and_angle(point_position, scanline.angle)

        # The points where the scanline intersects the region boundaries
        A = line_left.intersection(intersection_line)
        B = line_right.intersection(intersection_line)

        # The weighting of the two intersection point values (used for interpolating)
        dist_A = np.sqrt(np.sum((A - point_position) ** 2))
        dist_B = np.sqrt(np.sum((B - point_position) ** 2))
        total_distance = dist_A + dist_B
        weight_A = 1 - (dist_A / total_distance)
        weight_B = 1 - (dist_B / total_distance)

        # Interpolate the distances for the intersection points of the two regions
        R1 = self.base_wavefront(sender, A, wave_data)
        R2 = self.base_wavefront(sender, B, wave_data)
        interpolated_distance = R1 * weight_A + R2 * weight_B

        # Calculate whether the point is in region I or III using XOR
        is_in_focus = (line_left.signed_distance(point_position) > 0) ^ (
            line_right.signed_distance(point_position) > 0
        )
        # Select the correct distance based on which region the point belongs to
        return np.where(
            is_in_focus,
            self.base_wavefront(sender, point_position, wave_data),
            interpolated_distance,
        )
