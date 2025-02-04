from vbeam.core import Apodization
from spekk import ops

class NoApodization(Apodization):
    """Always return 1.0 (no spatial weighting)."""

    def __call__(self, *_, **__) -> float:
        return ops.array(1.0)
