"""This module implements some popular window functions (also called tapering functions
or apodization functions; but don't mix up window functions and vbeam's
`~vbeam.core.apodization.Apodization` class!).
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from spekk import Module, ops

# For adding type hints to Apodization.plot:
if TYPE_CHECKING:
    try:
        from matplotlib.axes import Axes
    except ImportError:
        pass


class Window(Module):
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
        """Return the weight for the ratio (between -0.5 and 0.5). The peak is at
        ratio=0, and it tapers off as the ratio approaches ±0.5.

        Try plotting window functions like this:
        >>> from vbeam.apodization import window
        >>> win = window.Hamming()
        >>> win.plot()
        """

    def plot(self) -> "Axes":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(ncols=2, figsize=(10, 3))
        ratios = ops.linspace(-1, 1, 200)
        values = self(ratios)
        ax[0].plot(ratios, values)
        ax[0].set_title(repr(self))
        ax[0].set_xlabel("Ratio")
        ax[0].set_ylabel(f"{repr(self)}(ratio)")

        values_fft = ops.fft.fftshift(ops.fft.fft(self(ops.linspace(-1, 1, 25)), n=400))
        values_fft = 20 * ops.log10(ops.abs(values_fft))
        values_fft -= ops.max(values_fft)
        ax[1].plot(values_fft)
        ax[1].set_ylim([-100, 2])
        ax[1].set_title(f"{repr(self)} in frequency domain")
        ax[1].set_xticks([])
        ax[1].set_ylabel("Amplitude [dB]")

        fig.tight_layout()


class NoWindow(Window):
    def __call__(self, ratio: float) -> float:
        return ops.ones(ratio.shape)


def _within_valid(ratio: float) -> bool:
    return ops.logical_and(-0.5 <= ratio, ratio <= 0.5)


class Rectangular(Window):
    def __call__(self, ratio: float) -> float:
        return _within_valid(ratio) * 1.0


class Hanning(Window):
    a0: float = 0.5
    a1: float = 0.5

    def __call__(self, ratio: float) -> float:
        return _within_valid(ratio) * (self.a0 + self.a1 * ops.cos(2 * ops.pi * ratio))


class Hamming(Window):
    def __call__(self, ratio: float) -> float:
        return Hanning(0.53836, 0.46164)(ratio)


class Tukey(Window):
    roll: float

    def __call__(self, ratio: float) -> float:
        ratio = ops.abs(ratio)
        p1 = ratio <= (0.5 * (1 - self.roll))
        p2 = ratio > (0.5 * (1 - self.roll))
        p3 = (ratio < 0.5) * 0.5
        p4 = 1 + ops.cos(2 * ops.pi / self.roll * (ratio - self.roll / 2 - 0.5))
        return p1 + p2 * p3 * p4


def Tukey25() -> Tukey:
    return Tukey(0.25)


def Tukey50() -> Tukey:
    return Tukey(0.5)


def Tukey75() -> Tukey:
    return Tukey(0.75)


def Tukey80() -> Tukey:
    return Tukey(0.8)


class Bartlett(Window):
    def __call__(self, ratio: float) -> float:
        ratio = ops.abs(ratio)
        return _within_valid(ratio) * (0.5 - ratio) * 2
