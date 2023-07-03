from typing import Tuple, Union

from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass

from .window import Window


@traceable_dataclass(("window", "f_number", "tilt"))
class PlaneWaveTransmitApodization(Apodization):
    window: Window
    f_number: Union[float, Tuple[float, float]]
    tilt: Tuple[float, float] = (0.0, 0.0)

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
        tan_theta = np.tan(wave_data.azimuth) - self.tilt[0]
        tan_phi = np.tan(wave_data.elevation) - self.tilt[1]
        ratio_theta = np.abs(f_number[0] * tan_theta)
        ratio_phi = np.abs(f_number[1] * tan_phi)
        return np.array(
            self.window(ratio_theta) * self.window(ratio_phi),
            dtype="float32",
        )


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
