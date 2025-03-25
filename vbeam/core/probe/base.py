from abc import abstractmethod

from spekk import Module, ops

from vbeam.core.probe.aperture import Aperture
from vbeam.geometry import Plane, Vector, VectorWithInfiniteMagnitude


class ProbeElement(Module):
    plane: Plane
    width: float
    height: float

    @property
    def normal(self) -> ops.array:
        return self.plane.normal

    @property
    def position(self) -> ops.array:
        return self.plane.origin


class Probe(Module):
    active_elements: ProbeElement

    @abstractmethod
    def get_effective_aperture(
        self, virtual_source: Vector | VectorWithInfiniteMagnitude | ops.array
    ) -> Aperture:
        """Return the effective aperture that is geometrically focused towards the
        given virtual source."""
