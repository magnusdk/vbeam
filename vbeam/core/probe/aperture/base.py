from abc import abstractmethod
from typing import TypeVar

from spekk import AbstractVar, Module, ops

from vbeam.apodization.window import Window
from vbeam.geometry import Vector

TAperture = TypeVar("TAperture", bound="Aperture")


class Aperture(Module):
    """An Aperture is the region of an ultrasound probe that is either actively
    transmitting or receiving.

    This class is meant to simplify the implementation of various apodization
    functions, or anything that involves geometric focusing of transmitted waves.
    """

    # Classes inheriting from Aperture needs to have an attribute or property named origin.
    origin: AbstractVar[ops.array]

    @abstractmethod
    def project_aperture(self: TAperture, source: Vector) -> TAperture:
        """Project the aperture towards the virtual source. Return a new scaled
        aperture that is oriented towards the source."""

    @abstractmethod
    def signed_distance(self, point: ops.array) -> float:
        """Return the distance from the point to the closest point on the aperture,
        negative if behind the aperture (hence signed)."""

    @abstractmethod
    def project_and_apply_window(self, point: ops.array, window: Window) -> float:
        """Project the point onto the aperture and apply the given window function."""

    @abstractmethod
    def scale(self: TAperture, scaling_factor: float) -> TAperture:
        """Return a copy of the aperture, scaled down by `scaling_factor`.

        This is useful for implementing various apodization functions; for example
        expanding aperture, where the aperture is scaled as a function of depth."""
