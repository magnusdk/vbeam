from vbeam.core import TransmittedWavefront, WaveData, ProbeGeometry
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass()
class PlaneWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        diff = point_position - sender
        x, y, z = diff[0], diff[1], diff[2]
        return (
            x * np.sin(wave_data.azimuth) * np.cos(wave_data.elevation)
            + y * np.sin(wave_data.elevation)
            + z * np.cos(wave_data.azimuth) * np.cos(wave_data.elevation)
        )
