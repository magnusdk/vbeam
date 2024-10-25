from fastmath import Array

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData
from vbeam.fastmath import numpy as np


class PlaneWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        diff = point_position - sender
        x, y, z = diff[0], diff[1], diff[2]
        return (
            x * np.sin(wave_data.azimuth) * np.cos(wave_data.elevation)
            + y * np.sin(wave_data.elevation)
            + z * np.cos(wave_data.azimuth) * np.cos(wave_data.elevation)
        )
