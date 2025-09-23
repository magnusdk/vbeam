import functools
from typing import Dict, Mapping

from spekk import Dim, at, ops

from vbeam.core import IndicesInfo, NDInterpolator


class LinearNDInterpolator(NDInterpolator):
    def _get_weights(self, indices: IndicesInfo) -> ops.array:
        distances_between_sampled_positions = ops.sum(
            indices.offset_distances, axis=indices.dim_name, keepdims=True
        )
        weights = 1 - indices.offset_distances / distances_between_sampled_positions
        return weights

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        values = self.data
        for dim, x in xi.items():
            # Get the 2 nearest indices for each sample position from xi
            coordinate = self.coordinates[dim]
            indices_info = coordinate.get_nearest_indices(x, 2)
            indices_dim = indices_info.dim_name

            # Calculate the weights
            weights = self._get_weights(indices_info)
            # Sum the weighted interpolated samples iteratively (this is faster than
            # indexing it all as one big indexing operation).
            # NOTE: We measured it to be faster to index the array in as small chunks
            # as possible rather than one big indexing operation when using JAX.
            values = sum(
                values[{dim: at(indices_info)[indices_dim, i].get().indices}]
                * at(weights)[indices_dim, i].get()
                for i in range(2)
            )

            # Replace the values that are out of bounds by fill_value if not None.
            if self.fill_value is not None:
                values = ops.where(indices_info.within_bounds, values, self.fill_value)

        return values


class NearestNDInterpolator(NDInterpolator):
    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the values at the interpolated indices
        values = self.data
        for dim, x in xi.items():
            coordinate = self.coordinates[dim]
            indices_info = coordinate.get_nearest_indices(x, 1)
            values = values[
                {dim: ops.squeeze(indices_info.indices, axis=indices_info.dim_name)}
            ]
            values = ops.where(indices_info.within_bounds, values, self.fill_value)

        return values
