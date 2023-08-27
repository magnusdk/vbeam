from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Sequence, Tuple, TypeVar, Union

from vbeam.fastmath import numpy as np
from vbeam.scan.base import CoordinateSystem, Scan

TSelf = TypeVar("TSelf", bound="WrappedScan")


def _wrap_with_new_base_scan(wrapped_scan: TSelf, new_base_scan: Scan) -> TSelf:
    new_copy = wrapped_scan.copy()
    new_copy.base_scan = new_base_scan
    return new_copy


class WrappedScan(Scan, ABC):
    """A scan that wraps another scan (the base_scan).

    This is useful for composing scans to make more advanced scans."""

    # The wrapped scan (accessed via the base_scan property).
    _base_scan: Scan

    def replace(
        self: TSelf,
        # "unchanged" means that the axis will not be changed.
        **kwargs: Union[np.ndarray, None, Literal["unchanged"]],
    ) -> TSelf:
        return _wrap_with_new_base_scan(self.base_scan.replace(**kwargs))

    def update(
        self: TSelf, **kwargs: Optional[Callable[[np.ndarray], np.ndarray]]
    ) -> TSelf:
        return _wrap_with_new_base_scan(self.base_scan.update(**kwargs))

    def resize(self: TSelf, kwargs: Optional[int]) -> TSelf:
        return _wrap_with_new_base_scan(self.base_scan.resize(**kwargs))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base_scan.shape

    @property
    def bounds(self) -> np.ndarray:
        return self.base_scan.bounds

    @property
    def cartesian_bounds(self) -> np.ndarray:
        return self.base_scan.cartesian_bounds

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self.base_scan.coordinate_system

    @property
    def is_2d(self) -> bool:
        return self.base_scan.is_2d

    @property
    def is_3d(self) -> bool:
        return self.base_scan.is_3d

    @property
    def base_scan(self):
        "Get the wrapped scan."
        return self._base_scan

    @base_scan.setter
    def base_scan(self, new_base_scan):
        "Set the wrapped scan."
        # You should override this setter in your subclass if this changes something.
        self._base_scan = new_base_scan

    @abstractmethod
    def copy(self: TSelf) -> TSelf:
        "Return a copy of the scan."

    def __getattr__(self, name):
        "Forward any other attribute access to the base_scan."
        try:
            return getattr(self.base_scan, name)
        except AttributeError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e


class ExtraDimsScanMixin(ABC):
    """A mixin for scans that work with more than just the "points" dimension. For
    example, the
    :class:`~vbeam.scan.optimized.apodization_filtered_scan.ApodizationFilteredScan`
    may filter out different points for each transmit, so it has an extra "transmits"
    dimension.

    This is just to give vbeam enough information to automatically create beamformers
    from more advanced scan setups."""

    @property
    @abstractmethod
    def flattened_points_dimensions(self) -> Sequence[str]:
        """The named dimensions of the array returned by
        ``Scan.get_points(flatten=True)``"""

    @property
    @abstractmethod
    def required_dimensions_for_unflatten(self) -> Sequence[str]:
        "The named dimensions required as axis-arguments by :meth:`Scan.unflatten`."

    @property
    def dimensions_after_unflatten(self) -> Sequence[str]:
        "The named dimensions of the array returned by :meth:`Scan.unflatten`."
        return ["width", "height"]
