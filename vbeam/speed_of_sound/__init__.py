from typing import Optional

import numpy

from vbeam.core import SpeedOfSound
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import Scan
from vbeam.util import ensure_2d_point
from vbeam.util.geometry.v2 import distance


@traceable_dataclass(
    data_fields=("values", "x_axis", "z_axis", "default_speed_of_sound"),
    aux_fields=("n_samples",),
)
class HeterogeneousSpeedOfSound(SpeedOfSound):
    """Sample the speed of sound between sender, point, and receiver position, and
    return the average.

    You should probably only ever use this if you are doing synthetic transmit aperture
    imaging (STAI). For more complex imaging setups, like focused, diverging, or plane
    wave imaging, this class is not appropriate. How should speed of sound be sampled
    for these setups? That's a difficult question :)"""

    values: np.ndarray
    x_axis: FastInterpLinspace
    z_axis: FastInterpLinspace
    n_samples: int
    default_speed_of_sound: float = 1540.0

    def average(
        self,
        sender_position: np.ndarray,
        point_position: np.ndarray,
        receiver_position: np.ndarray,
    ) -> float:
        sender_position = ensure_2d_point(sender_position)
        point_position = ensure_2d_point(point_position)
        receiver_position = ensure_2d_point(receiver_position)

        # Calculate distances to get correct weighting of the averages
        distance1 = distance(sender_position, point_position)
        distance2 = distance(point_position, receiver_position)
        total_distance = distance1 + distance2

        # Get the average speed of sounds
        average1 = self.average_between_two_points(sender_position, point_position)
        average2 = self.average_between_two_points(point_position, receiver_position)

        # Return the weighted average of the total distance
        return (average1 * distance1 + average2 * distance2) / total_distance

    def average_between_two_points(self, p1: np.ndarray, p2: np.ndarray) -> float:
        "Return the averaged sampled speed of sound between point ``p1`` and ``p2``."
        assert p1.shape == p2.shape == (2,), "Expected p1 and p2 to be 2D points."
        x1, z1 = p1[0], p1[1]
        x2, z2 = p2[0], p2[1]
        dx, dz = (x2 - x1) / self.n_samples, (z2 - z1) / self.n_samples
        # We use np.reduce instead of interp2d directly to allocate less GPU memory
        integrated_speed_of_sound = np.reduce(
            lambda carry, i: carry
            + FastInterpLinspace.interp2d(
                x1 + i * dx,
                z1 + i * dz,
                self.x_axis,
                self.z_axis,
                self.values,
                padding=self.default_speed_of_sound,
            ),
            np.arange(self.n_samples),
            np.array(0.0, dtype=self.values.dtype),
        )
        return integrated_speed_of_sound / self.n_samples

    @staticmethod
    def from_scan(
        scan: Scan,
        values: np.ndarray,
        n_samples: Optional[int] = None,
        default_speed_of_sound: float = 1540.0,
    ) -> "HeterogeneousSpeedOfSound":
        if not scan.is_2d:
            raise ValueError(
                "Only 2D scans are supported for HeterogeneousSpeedOfSound"
            )

        from_x, to_x, from_z, to_z = scan.cartesian_bounds
        n_x, n_z = scan.shape
        if n_samples is None:
            n_samples = int(numpy.ceil(numpy.sqrt(n_x**2 + n_z**2)))
        return HeterogeneousSpeedOfSound(
            values,
            x_axis=FastInterpLinspace(from_x, (to_x - from_x) / n_x, n_x),
            z_axis=FastInterpLinspace(from_z, (to_z - from_z) / n_z, n_z),
            n_samples=n_samples,
            default_speed_of_sound=default_speed_of_sound,
        )
