"""Interface for sampling the speed of sound on a line, typically the line between a 
:term:`sender` and a :term:`point`, and a :term:`point` and a :term:`receiver`. """

from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np


class SpeedOfSound(ABC):
    @abstractmethod
    def average(
        self,
        sender_position: np.ndarray,
        point_position: np.ndarray,
        receiver_position: np.ndarray,
    ) -> float:
        """Sample the speed of sound between the sender, the point position, and the
        receiver, and return the average.

        All positions are arrays of three elements: (x, y, z)."""
