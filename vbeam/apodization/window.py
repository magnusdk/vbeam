"""This module implements some popular window functions (also called apodization 
functions or tapering functions).

See the notebook ``docs/tutorials/apodization/windows.ipynb`` for a visualization of 
the various implemented windows."""

from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


class Window(ABC):
    """A window (also called apodization function or tapering function) is used to get
    more desirable main-lobe/side-lobe characteristics. A Window object can be called
    as afunction. It takes a number between 0 and 0.5 and returns the weight of the
    window, where the function is highest at 0 and lowest at 0.5.

    For example, the output from a Bartlett window may look like this:
    >>> window = Bartlett()
    >>> ratios = np.linspace(0, 0.5, 6)
    >>> weights = [window(r) for r in ratios]
    >>> [f"{w:.1f}" for w in weights]  # Round to 1 decimal to avoid numerical errors
    ['1.0', '0.8', '0.6', '0.4', '0.2', '0.0']
    """

    @abstractmethod
    def __call__(self, ratio: float) -> float:
        """Return the weight for the ratio (between 0 and 0.5). The peak is at ratio=0,
        and it tapers off as the ratio approaches 0.5."""


@traceable_dataclass()
class NoWindow(Window):
    def __call__(self, ratio: float) -> float:
        return np.ones(ratio.shape)


def _within_valid(ratio: float) -> bool:
    return np.logical_and(0 <= ratio, ratio <= 0.5)


@traceable_dataclass()
class Rectangular(Window):
    def __call__(self, ratio: float) -> float:
        return _within_valid(ratio) * 1.0


@traceable_dataclass(("a0", "a1"))
class Hanning(Window):
    a0: float = 0.5
    a1: float = 0.5

    def __call__(self, ratio: float) -> float:
        return _within_valid(ratio) * (self.a0 + self.a1 * np.cos(2 * np.pi * ratio))


@traceable_dataclass()
class Hamming(Window):
    def __call__(self, ratio: float) -> float:
        return Hanning(0.53836, 0.46164)(ratio)


@traceable_dataclass(("roll",))
class Tukey(Window):
    roll: float

    def __call__(self, ratio: float) -> float:
        p1 = ratio <= (1 / 2 * (1 - self.roll))
        p2 = ratio > (1 / 2 * (1 - self.roll))
        p3 = (ratio < (1 / 2)) * 0.5
        p4 = 1 + np.cos(2 * np.pi / self.roll * (ratio - self.roll / 2 - 1 / 2))
        return _within_valid(ratio) * (p1 + p2 * p3 * p4)


def Tukey25() -> Tukey:
    return Tukey(0.25)


def Tukey50() -> Tukey:
    return Tukey(0.5)


def Tukey75() -> Tukey:
    return Tukey(0.75)


def Tukey80() -> Tukey:
    return Tukey(0.8)


@traceable_dataclass()
class Bartlett(Window):
    def __call__(self, ratio: float) -> float:
        return _within_valid(ratio) * (0.5 - ratio) * 2


if __name__ == "__main__":
    import doctest

    from vbeam.fastmath import backend_manager

    with backend_manager.using_backend("numpy"):
        doctest.testmod()
