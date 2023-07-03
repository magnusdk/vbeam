"Interface for interpolating the :term:`signal` given a delay."

from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np


class InterpolationSpace1D(ABC):
    """An interface for interpolating data in 1D."""

    @abstractmethod
    def __call__(self, x: np.ndarray, fp: np.ndarray) -> np.ndarray:
        """Evaluate the points x on the discrete array fp.

        Any point in x that is outside of the range of fp is evaluated as zero."""
        ...

    @property
    @abstractmethod
    def start(self) -> float:
        ...

    @property
    @abstractmethod
    def end(self) -> float:
        ...
