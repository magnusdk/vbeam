
from typing import Protocol, runtime_checkable
from spekk import ops


@runtime_checkable
class TScan(Protocol):
    """A Protocol describing the required behavior of a Scan-like object in
    order to define the cartesian points for beamforming."""

    @property
    def points(self) -> ops.array:
        "The actual cartesian points as in array. It must at least have a dimension named 'xyz'."
        ...


