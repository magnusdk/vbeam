from typing import Callable, Iterable, Optional

from vbeam.core import Apodization
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass(("apodizations",), ("combiner",))
class CombinedApodization(Apodization):
    apodizations: Iterable[Apodization]
    combiner: Optional[Callable[list[np.ndarray], np.ndarray]] = None

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Multiply the result of calling the Apodization objects.

        The product is taken along a new axis to handle broadcasting correctly.
        """
        values = [apod(*args, **kwargs) for apod in self.apodizations]
        if self.combiner is not None:
            return self.combiner(values)
        return np.prod(np.array(values), axis=0)


def combine_apodizations(
    *apodizations: Iterable[Apodization],
    combiner: Optional[Callable[[list[np.ndarray]], np.ndarray]] = None,
) -> CombinedApodization:
    """Return a new Apodization object that combines all the given apodizations.

    By default, apodizations are combined by taking the product of all their results."""
    return CombinedApodization(apodizations, combiner)
