from vbeam.core import Apodization


class NoApodization(Apodization):
    """Always return 1.0 (no spatial weighting)."""

    def __call__(self, *_, **__) -> float:
        return 1.0
