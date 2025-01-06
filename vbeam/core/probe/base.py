from spekk import Module, ops

from vbeam.core.probe.aperture.base import Aperture
from vbeam.geometry import Direction


class ProbeElement(Module):
    position: ops.array
    normal: Direction


class Probe(Module):
    active_elements: ProbeElement
    active_aperture: Aperture
