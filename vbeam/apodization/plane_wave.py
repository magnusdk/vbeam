from typing import Optional, Tuple, Union

from vbeam.apodization.window import Rectangular, Window
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.coordinate_systems import az_el_to_cartesian
from vbeam.util.geometry.v2 import distance


@traceable_dataclass(("array_bounds", "window"))
class PlaneWaveTransmitApodization(Apodization):
    # array_bounds relative to the sender
    # Should be centered at the origin
    array_bounds: Union[
        Tuple[np.ndarray, np.ndarray],  # 1D array
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],  # 2D array
    ]
    window: Optional[Window] = None

    def __post_init__(self):
        """Validate the array bounds."""
        if len(self.array_bounds) not in [2, 4]:
            raise ValueError(
                "Expected either left/right or left/right/front/back array bounds"
            )

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        # Normalized vector pointing in the direction of the transmitted wave
        direction = az_el_to_cartesian(
            azimuth=wave_data.azimuth, elevation=wave_data.elevation
        )

        # Project the point onto the xy-plane with origin at the beam at depth z
        point_position = point_position - sender.position
        point_x = point_position[..., 0]
        point_y = point_position[..., 1]
        point_z = point_position[..., 2]
        x_projected = point_x - direction[0] * point_z / direction[2]
        y_projected = point_y - direction[1] * point_z / direction[2]

        # Apply the window over the projected aperture and evaluate point
        window = self.window if self.window is not None else Rectangular()
        array_width = distance(self.array_bounds[1] - self.array_bounds[0])
        weight = window(np.abs(x_projected / array_width))
        if len(self.array_bounds) == 4:  # If we use a 2D probe, also apply in elevation
            array_height = distance(self.array_bounds[3] - self.array_bounds[2])
            weight *= window(np.abs(y_projected / array_height))
        return weight


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
