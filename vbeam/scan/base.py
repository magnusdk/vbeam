import operator
from abc import abstractmethod
from enum import Enum
from functools import reduce
from typing import Callable, Literal, Optional, Tuple, Union

from fastmath import ArrayOrNumber, Module

from vbeam.util import ensure_positive_index


class CoordinateSystem(Enum):
    CARTESIAN = "cartesian"
    POLAR = "polar"


class Scan(Module):
    @abstractmethod
    def get_points(self, flatten: bool = True) -> ArrayOrNumber:
        """Return the points defined by the scan, flattened to a (N, 3) array by
        default, where N is the number of points."""

    @abstractmethod
    def replace(self, *_axes: Union[ArrayOrNumber, Literal["unchanged"]]) -> "Scan":
        "Return a copy of the scan with values replaced."

    @abstractmethod
    def update(
        self, *_axes: Optional[Callable[[ArrayOrNumber], ArrayOrNumber]]
    ) -> "Scan":
        "Return a copy of the scan with updates applied to the given axes."

    @abstractmethod
    def resize(self, *_axes: Optional[int]) -> "Scan":
        "Return a copy of the scan where the given axes are resized."

    @property
    @abstractmethod
    def axes(self) -> Tuple[ArrayOrNumber, ...]:
        """Return the axes of the scan.

        E.g.: if the scan is a sector scan, return the azimuth and depths axes."""

    @property
    def shape(self) -> Tuple[int, ...]:
        "Return the shape of the grid of points defined by the scan."
        return tuple([len(axis) for axis in self.axes])

    @property
    def bounds(self) -> ArrayOrNumber:
        "Return the bounds of the axes of the scan."
        bounds = []
        for ax in self.axes:
            bounds += [ax[0], ax[-1]]
        return tuple(bounds)

    @property
    @abstractmethod
    def cartesian_bounds(self) -> ArrayOrNumber:
        """Return the bounds in cartesian coordinates of the axes of the scan (useful
        for sector scans)."""

    @property
    @abstractmethod
    def coordinate_system(self) -> CoordinateSystem:
        """Return the coordinate system of the scan. E.g.: sector scans are in polar
        coordinates and linear scans are in cartesian coordinates."""

    def unflatten(
        self, imaged_points: ArrayOrNumber, points_axis: int = -1
    ) -> ArrayOrNumber:
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
