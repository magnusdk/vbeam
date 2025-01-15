from abc import abstractmethod

from spekk import Module, ops

from vbeam.core.probe.aperture import Aperture
from vbeam.geometry import Direction, Orientation, Vector


class ProbeElement(Module):
    position: ops.array
    orientation: Orientation
    width: float
    height: float

    @property
    def normal(self) -> Direction:
        return self.orientation.direction


class Probe(Module):
    active_elements: ProbeElement

    @abstractmethod
    def get_effective_aperture(self, virtual_source: Vector) -> Aperture:
        """Return the effective aperture that is geometrically focused towards the
        given virtual source."""
