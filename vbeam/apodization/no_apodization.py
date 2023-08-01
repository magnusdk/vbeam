from vbeam.core import Apodization
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass()
class NoApodization(Apodization):
    """Always return 1.0 (no weighting)."""

    def __call__(self, *_, **__) -> float:
        return 1.0
