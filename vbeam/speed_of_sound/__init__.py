from typing import Optional

import numpy

from vbeam.core import SpeedOfSound
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import Scan


@traceable_dataclass(
    data_fields=("values", "x_axis", "z_axis", "default_speed_of_sound"),
    aux_fields=("n_samples",),
)
class HeterogeneousSpeedOfSound(SpeedOfSound):
    values: np.ndarray
    x_axis: FastInterpLinspace
    z_axis: FastInterpLinspace
    n_samples: int
    default_speed_of_sound: float = 1540.0

    def average_between(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        x1, _, z1 = pos1[0], pos1[1], pos1[2]
        x2, _, z2 = pos2[0], pos2[1], pos2[2]
        dx, dz = (x2 - x1) / self.n_samples, (z2 - z1) / self.n_samples

        # Numerically integrates speed of sound from pos1 to pos2
        integrated_speed_of_sound = np.reduce(
            lambda carry, i: carry
            + FastInterpLinspace.interp2d(
                x1 + i * dx,
                z1 + i * dz,
                self.x_axis,
                self.z_axis,
                self.values,
                self.default_speed_of_sound,
            ),
            np.arange(self.n_samples),
            0.0,
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
