"""Interface for sampling the speed of sound on a line, typically the line between a 
:term:`sender` and a :term:`point`, and a :term:`point` and a :term:`receiver`. """

from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np


class SpeedOfSound(ABC):
    @abstractmethod
    def average_between(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Sample the speed of sound between pos1 and pos2 and return the average.

        pos1 and pos2 are arrays of three elements: (x, y, z)."""
        ...
