from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import Line

from .window import Window


@traceable_dataclass(("window", "beam_width"))
class MLAApodization(Apodization):
    """Perform multiple line acquisition (MLA) in cartesian space.

    The number of lines acquired per transmit is determined by beam_width. If beam_width
    is the same as the delta_x of the scan (assuming of course that the scan dimensions
    are linear) then this apodization would produce approximately the same effect as
    using scanline imaging."""

    window: Window
    beam_width: float

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        # Geometry assumes 2D points
        sender_position = sender.position[np.array([0, 2])]
        point_position = point_position[np.array([0, 2])]
        source = wave_data.source[np.array([0, 2])]

        # The distance from point_position to the nearest point on the line that
        # intersects sender_position and wave_data.source.
        scanline = Line.passing_through(sender_position, source)
        dist = np.abs(scanline.signed_distance(point_position))
        return self.window(dist / (self.beam_width))
