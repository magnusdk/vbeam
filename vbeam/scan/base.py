import operator
from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from typing import Callable, Literal, Optional, Tuple, Union

from vbeam.fastmath import numpy as np
from vbeam.util import ensure_positive_index


class CoordinateSystem(Enum):
    CARTESIAN = "cartesian"
    POLAR = "polar"


class Scan(ABC):
    @abstractmethod
    def get_points(self, flatten: bool = True) -> np.ndarray:
        """Return the points defined by the scan, flattened to a (N, 3) array by
        default, where N is the number of points."""

    @abstractmethod
    def replace(self, *_axes: Union[np.ndarray, Literal["unchanged"]]) -> "Scan":
        "Return a copy of the scan with values replaced."

    @abstractmethod
    def update(self, *_axes: Optional[Callable[[np.ndarray], np.ndarray]]) -> "Scan":
        "Return a copy of the scan with updates applied to the given axes."

    @abstractmethod
    def resize(self, *_axes: Optional[int]) -> "Scan":
        "Return a copy of the scan where the given axes are resized."

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        "Return the shape of the grid of points defined by the scan."

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        "Return the bounds of the axes of the scan."

    @property
    @abstractmethod
    def cartesian_bounds(self) -> np.ndarray:
        """Return the bounds in cartesian coordinates of the axes of the scan (useful
        for sector scans)."""

    @property
    @abstractmethod
    def coordinate_system(self) -> CoordinateSystem:
        """Return the coordinate system of the scan. E.g.: sector scans are in polar
        coordinates and linear scans are in cartesian coordinates."""

    def unflatten(self, imaged_points: np.ndarray, points_axis: int = -1) -> np.ndarray:
        "Unflatten a flattened array of values into the original shape of the scan."
        points_axis = ensure_positive_index(imaged_points.ndim, points_axis)
        return imaged_points.reshape(
            (
                imaged_points.shape[:points_axis]
                + self.shape
                + imaged_points.shape[points_axis + 1 :]
            )
        )

    @property
    def num_points(self) -> int:
        "The number of points in the scan."
        return reduce(operator.mul, self.shape)

    @property
    def ndim(self) -> int:
        "The number of dimensions of the scan."
        return len(self.shape)

    @property
    def is_2d(self) -> bool:
        "True if only two axes are defined."
        return self.ndim == 2

    @property
    def is_3d(self) -> bool:
        "True if all three axes are defined."
        return self.ndim == 3
