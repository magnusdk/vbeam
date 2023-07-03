import operator
from functools import reduce

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.scan import Scan
from vbeam.scan.points_optimizers.base import PointOptimizer, ShapeInfo
from vbeam.util import ensure_positive_index


@traceable_dataclass()
class Scanlines(PointOptimizer):
    """Optimize the points such that only one column (scanline) is imaged per transmit."""

    def reshape(self, point_pos: np.ndarray, scan: Scan):
        num_transmits = scan.shape[0]  # Size of x-axis
        points_per_transmit = reduce(operator.mul, scan.shape[1:])
        return point_pos.reshape((num_transmits, points_per_transmit, 3))

    def recombine(
        self,
        imaged_points: np.ndarray,
        scan: Scan,
        transmits_axis: int,
        points_axis: int,
    ) -> np.ndarray:
        if not isinstance(scan, Scan):
            raise TypeError(f"Expected a Scan, but type(scan) is {str(type(scan))}.")

        transmits_axis = ensure_positive_index(imaged_points.ndim, transmits_axis)
        points_axis = ensure_positive_index(imaged_points.ndim, points_axis)

        # Put the transmit axis as the 0th axis because we are concatenating it
        dim_indices = tuple(range(imaged_points.ndim))
        transposed_axes = (
            (transmits_axis,)
            + (dim_indices[:transmits_axis])
            + (dim_indices[transmits_axis + 1 :])
        )
        imaged_points = np.transpose(imaged_points, transposed_axes)

        # Account for moving the transmits axis to the front when using points_axis
        if transmits_axis > points_axis:
            points_axis += 1

        # Unflatten the points
        unflattened_shape = (
            imaged_points.shape[:points_axis]
            # x-axis has size 1 since it is vectorized over as scanlines
            + (1, *scan.shape[1:])
            + imaged_points.shape[points_axis + 1 :]
        )
        imaged_points = imaged_points.reshape(unflattened_shape)

        # Concatenate transmits into the x-axis (first axis of points).
        # We subtract one (1) because the first dimension disappears from the output,
        # as per concatenation.
        image = np.concatenate(imaged_points, axis=points_axis - 1)

        return image

    @property
    def shape_info(self) -> ShapeInfo:
        return ShapeInfo(
            ["transmits", "points"],
            ["transmits", "points"],
            ["width", "height"],
        )
