from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple

from vbeam.fastmath import numpy as np
from vbeam.scan import Scan


class PointOptimizer(ABC):
    @abstractmethod
    def reshape(self, point_pos: np.ndarray, scan: Scan):
        """Reshape the points to a more optimized shape for imaging."""

    @abstractmethod
    def recombine(self, imaged_points: np.ndarray, scan: Scan, *axes: int):
        """Recombine the imaged points back into the scan's original shape.

        The axes should correspond to the axes returned by reshape. For example, if
        reshape adds a transmit dimension to the point (for example, by filtering out
        points that are not needed for each transmit) then recombine needs to know what
        axis that dimension was added to."""

    @property
    @abstractmethod
    def shape_info(self) -> "ShapeInfo":
        """Return information about the dimensions of the points after reshaping and
        recombining."""


@dataclass
class ShapeInfo:
    """Information about the dimensions of the points after reshaping and recombining.

    Attributes:
      after_reshape: The dimensions of the points after applying reshape.
      required_for_recombine: The dimensions of the points required for applying
        recombine. These dimensions must be passed as axis-indices.
      after_recombine: The dimensions of the points after applying recombine.
    """

    after_reshape: Sequence[str]
    required_for_recombine: Sequence[str]
    after_recombine: Sequence[str]

    @property
    def removed_dimensions(self) -> Tuple[str, ...]:
        """The dimensions that were removed when undoing the reshape.

        For example, when undoing the reshape for scanlines, the transmits dimension is
        concatenated and thus removed."""
        return tuple(
            dim
            for dim in self.required_for_recombine
            if dim not in self.after_recombine
        )
