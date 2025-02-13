import functools
from typing import Dict

from spekk import Dim, ops

from vbeam.core import IndicesInfo, NDInterpolator


class LinearNDInterpolator(NDInterpolator):
    def _get_weights(self, indices: IndicesInfo) -> ops.array:
        return 1 - indices.offset_distances / ops.sum(
            indices.offset_distances, axis=indices.dim_name
        )

    def __call__(self, xi: Dict[Dim, ops.array]) -> ops.array:
        # Get the 2 nearest indices for each sample position from kwargs
        indices_info_dict: Dict[Dim, IndicesInfo] = {
            dim: self.data_coordinates[dim].get_nearest_indices(x, 2)
            for dim, x in xi.items()
        }

        # Get the values at the interpolated indices
        values = self.data[
            {
                dim: indices_info.indices
                for dim, indices_info in indices_info_dict.items()
            }
        ]

        # Linearly interpolate the values by performing a weighted sum.
        for indices_info in indices_info_dict.values():
            weights = self._get_weights(indices_info)
            values = ops.sum(values * weights, axis=indices_info.dim_name)

        # Replace the values that are out of bounds by fill_value if not None.
        if self.fill_value is not None:
            within_bounds = map(
                lambda indices_info: indices_info.within_bounds,
                indices_info_dict.values(),
            )
            within_bounds = functools.reduce(ops.logical_and, within_bounds)
            values = ops.where(within_bounds, values, self.fill_value)

        return values


class NearestNDInterpolator(NDInterpolator):
    def __call__(self, xi: Dict[Dim, ops.array]) -> ops.array:
        # Get the 2 nearest indices for each sample position from kwargs
        indices_info_dict: Dict[Dim, IndicesInfo] = {
            dim: self.data_coordinates[dim].get_nearest_indices(x, 1)
            for dim, x in xi.items()
        }

        # Get the values at the interpolated indices
        values = self.data[
            {
                # Squeeze to get rid of the dimension with size=1
                dim: ops.squeeze(indices_info.indices, axis=indices_info.dim_name)
                for dim, indices_info in indices_info_dict.items()
            }
        ]

        # Replace the values that are out of bounds by fill_value if not None.
        if self.fill_value is not None:
            within_bounds = map(
                lambda indices_info: indices_info.within_bounds,
                indices_info_dict.values(),
            )
            within_bounds = functools.reduce(ops.logical_and, within_bounds)
            values = ops.where(within_bounds, values, self.fill_value)

        return values
