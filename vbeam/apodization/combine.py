from functools import reduce
from operator import mul
from typing import Iterable

from vbeam.core import Apodization
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass(("apodizations",))
class CombinedApodization(Apodization):
    apodizations: Iterable[Apodization]

    def __call__(self, *args, **kwargs) -> float:
        """Multiply the result of calling the Apodization objects."""
        values = [apodization(*args, **kwargs) for apodization in self.apodizations]
        return reduce(mul, values, 1.0)


def combine_apodizations(
    *apodizations: Iterable[Apodization],
) -> CombinedApodization:
    """Return a new Apodization object that combines all the given apodizations
    by multiplication."""
    return CombinedApodization(apodizations)
