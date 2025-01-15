from typing import Optional

from spekk import Dim, ops, util

from vbeam.core import Interpolable, Interpolator


class LinearInterpolator(Interpolator):
    left: Optional[float] = None
    right: Optional[float] = None

    def __call__(self, interpolable: Interpolable, x: float, axis: Dim) -> ops.array:
        # Find the indices around x in the interpolable data
        data_coordinates = interpolable.interpolation_coordinates[axis]
        samples_axis = util.random_dim_name(self, f"{axis}_samples")
        indices = data_coordinates.get_nearest_indices(x, 2, samples_axis)

        # Calculate the weights for the sample values as a function of distance to x.
        weights = ops.abs(indices.positions - x)
        weights = weights / ops.sum(weights, axis=samples_axis)
        weights = 1 - weights

        # Perform a weighted sum (in practice this is a lerp).
        values = interpolable.get_values(indices.indices, axis)
        values = ops.sum(values * weights, axis=samples_axis)

        # Apply boundary conditions.
        if self.left is not None:
            values = ops.where(x < data_coordinates.start, self.left, values)
        if self.right is not None:
            values = ops.where(x > data_coordinates.end, self.right, values)

        return values
