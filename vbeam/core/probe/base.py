from spekk import Module, ops

from vbeam.core.probe.aperture.base import Aperture
from vbeam.geometry import Orientation


class ProbeElement(Module):
    position: ops.array
    orientation: Orientation
    width: float
    height: float


class Probe(Module):
    origin: ops.array
    active_elements: ProbeElement
    active_aperture: Aperture
