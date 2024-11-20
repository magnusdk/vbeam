from typing import Callable, Iterable, Optional

from fastmath import Array, field

from vbeam.core import Apodization
from vbeam.fastmath import numpy as api


class CombinedApodization(Apodization):
    apodizations: Iterable[Apodization]
    combiner: Optional[Callable[[Array], float]] = field(
        default=None, static=True
    )

    def __call__(self, *args, **kwargs) -> float:
        """Multiply the result of calling the Apodization objects."""
        values = api.array([apod(*args, **kwargs) for apod in self.apodizations])
        if self.combiner is not None:
            return self.combiner(values)
        return api.prod(values)


def combine_apodizations(
    *apodizations: Iterable[Apodization],
    combiner: Optional[Callable[[Array], float]] = None,
) -> CombinedApodization:
    """Return a new Apodization object that combines all the given apodizations.

    By default, apodizations are combined by taking the product of all their results."""
    return CombinedApodization(apodizations, combiner)
