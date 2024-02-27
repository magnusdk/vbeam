from typing import Optional, Tuple

from vbeam.apodization.window import Rectangular, Window
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance, rotate


def expanding_aperture(
    point: np.ndarray,
    origin: np.ndarray,
    theta: float,
    phi: float,
    f_number: Tuple[float, float],
    window: Window = Rectangular(),
    minimum_aperture: Optional[float] = None,
    maximum_aperture: Optional[float] = None,
) -> float:
    """Return the expanding aperture weighting for the point.

    Args:
        point (np.ndarray): The point to return the weighting for.
        origin (np.ndarray): The origin of the expanding aperture.
        theta (float): The azimuth angle of the expanding aperture. This rotates the
            xz-plane.
        phi (float): The polar angle of the expanding aperture. This rotates the
            yz-plane.
        f_number (Tuple[float, float]): F-number of the expanding aperture. This
            determines the width (opening angle) of the expanding aperture.
        window (Window): The window function to apply to the expanding aperture.
        minimum_aperture (float): The minimum aperture used. This can help ensure that
            the aperture isn't infinitesimally small near the origin.

    Returns:
        float: The expanding aperture weighting for the given point.
    """
    # Translate coordinate system to origin
    point = point - origin
    # Rotate the coordinate system around origin
    point = rotate(point, -theta, -phi)

    # Calculate expanding aperture weight
    tan_theta = point[0] / point[2]
    tan_phi = point[1] / point[2]
    ratio_theta = np.abs(f_number[0] * tan_theta)
    ratio_phi = np.abs(f_number[1] * tan_phi)
    weight = window(ratio_theta) * window(ratio_phi)

    if minimum_aperture is not None or maximum_aperture is not None:
        # To project the point, we can just remove the z-component since we have
        # transformed the coordinate system.
        point_projected_onto_z = point[:2]
        min_aperture_line_dist = distance(point_projected_onto_z)

        # Calculate minimum aperture weight
        if minimum_aperture is not None:
            min_aperture_weight = window(min_aperture_line_dist / minimum_aperture)
            # Combine the weights
            weight = np.maximum(weight, min_aperture_weight)
        if maximum_aperture is not None:
            max_aperture_weight = window(min_aperture_line_dist / minimum_aperture)
            # Combine the weights
            weight = np.minimum(weight, max_aperture_weight)

    return weight


@traceable_dataclass(("window", "f_number", "minimum_aperture"))
class ExpandingApertureApodization(Apodization):
    f_number: Tuple[float, float]
    window: Window
    minimum_aperture: float = 0.0

    def __post_init__(self):
        self.f_number = (
            self.f_number
            if isinstance(self.f_number, tuple) and len(self.f_number) == 2
            else (self.f_number, self.f_number)
        )

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        return expanding_aperture(
            point_position,
            receiver.position,
            receiver.theta if receiver.theta is not None else 0.0,
            receiver.phi if receiver.phi is not None else 0.0,
            self.f_number,
            self.window,
            self.minimum_aperture,
        )
