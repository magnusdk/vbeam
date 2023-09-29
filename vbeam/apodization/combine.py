from functools import reduce
from operator import mul
from typing import Callable, Iterable, Optional

from vbeam.core import Apodization
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass(("apodizations",), ("combiner",))
class CombinedApodization(Apodization):
    apodizations: Iterable[Apodization]
    combiner: Optional[Callable[[np.ndarray], float]] = None

    def __call__(self, *args, **kwargs) -> float:
        """Multiply the result of calling the Apodization objects."""
        values = np.array([apod(*args, **kwargs) for apod in self.apodizations])
        if self.combiner is not None:
            return self.combiner(values)
        return np.min(values)


def combine_apodizations(
    *apodizations: Iterable[Apodization],
    combiner: Optional[Callable[[np.ndarray], float]] = None,
) -> CombinedApodization:
    """Return a new Apodization object that combines all the given apodizations.
    
    By default, apodizations are combined by taking the minimum of all their results."""
    return CombinedApodization(apodizations, combiner)
