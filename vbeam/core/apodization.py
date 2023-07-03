"Point-based apodization for weighting the delayed signal."

from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np

from .element_geometry import ElementGeometry
from .wave_data import WaveData


class Apodization(ABC):
    @abstractmethod
    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        """
        Return the weighting for the signal for the given sender, point, receiver
        position, and wave data.

        Args:
          sender: The geometry of the sender (e.g. position (x, y, z)).
          point_position: The position of the point being imaged (x, y, z).
          receiver: The geometry of the receiver (e.g. position (x, y, z)).
          wave_data: Data specific to the wave being sent (see vbeam.core.WaveData).
        """
        ...
