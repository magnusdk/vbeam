from abc import ABC, abstractmethod

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


class Window(ABC):
    @abstractmethod
    def __call__(self, ratio: float) -> float:
        ...


@traceable_dataclass()
class NoWindow(Window):
    def __call__(self, ratio: float) -> float:
        return 1.0


@traceable_dataclass()
class Rectangular(Window):
    def __call__(self, ratio: float) -> float:
        return (ratio <= 0.5) * 1.0


@traceable_dataclass(("a0", "a1"))
class Hanning(Window):
    a0: float = 0.5
    a1: float = 0.5

    def __call__(self, ratio: float) -> float:
        return (ratio <= 0.5) * (self.a0 + self.a1 * np.cos(2 * np.pi * ratio))


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
        return p1 + p2 * p3 * p4


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
        return (0.5 - ratio) * 2
