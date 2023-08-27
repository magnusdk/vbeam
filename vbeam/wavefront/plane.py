from vbeam.core import ElementGeometry, WaveData, Wavefront
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass()
class PlaneWavefront(Wavefront):
    def transmit_distance(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        diff = point_position - sender.position
        x, y, z = diff[0], diff[1], diff[2]
        return (
            x * np.sin(wave_data.azimuth) * np.cos(wave_data.elevation)
            + y * np.sin(wave_data.elevation)
            + z * np.cos(wave_data.azimuth) * np.cos(wave_data.elevation)
        ) - wave_data.delay_distance
