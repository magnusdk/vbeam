import functools
import itertools
from typing import Dict, Mapping

from spekk import Dim, at, field, ops

from vbeam.core import IndicesInfo, NDInterpolator


class OptimizedLinearNDInterpolator(NDInterpolator):
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


class OptimizedNearestNDInterpolator(NDInterpolator):
    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the values at the interpolated indices
        values = self.data
        for dim, x in xi.items():
            coordinate = self.coordinates[dim]
            indices_info = coordinate.get_nearest_indices(x, 1)
            values = values[
                {dim: ops.squeeze(indices_info.indices, axis=indices_info.dim_name)}
            ]
            if self.fill_value is not None:
                values = ops.where(indices_info.within_bounds, values, self.fill_value)

        return values


class LinearNDInterpolatorFast(NDInterpolator):
    """Fast N-D linear interpolator that avoids the n_samples dimension.

    Instead of gathering both neighbors into a [..., 2] array (which adds an
    extra index dimension and causes slow scatter-add in JAX backward pass),
    this interpolator does two separate gathers per dimension — one for the
    floor index and one for floor+1 — and combines them with weights. This
    keeps all index arrays at the same rank as the input.

    Requires coordinates to have a ``get_index_and_frac(x)`` method (e.g.
    ``LinearCoordinateFast``).
    """

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        values = self.data
        within_bounds_list = []

        for dim, x in xi.items():
            coordinate = self.coordinates[dim]
            idx_floor, frac = coordinate.get_index_and_frac(x)

            # Two separate gathers — keeps indices at same rank as input,
            # so backward pass scatter-add stays 2D (fast path).
            lo = values[{dim: idx_floor}]
            hi = values[{dim: idx_floor + 1}]
            values = lo * (1 - frac) + hi * frac

            if self.fill_value is not None:
                within_bounds_list.append(coordinate.is_within_bounds(x))

        # Apply fill_value for out-of-bounds positions (if requested).
        if self.fill_value is not None and within_bounds_list:
            within_bounds = within_bounds_list[0]
            for wb in within_bounds_list[1:]:
                within_bounds = ops.logical_and(within_bounds, wb)
            values = ops.where(within_bounds, values, self.fill_value)

        return values


class LinearNDInterpolator(NDInterpolator):
    def _get_weights(self, indices: IndicesInfo) -> ops.array:
        distances_between_sampled_positions = ops.sum(
            indices.offset_distances, axis=indices.dim_name, keepdims=True
        )
        weights = 1 - indices.offset_distances / distances_between_sampled_positions
        return weights

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the 2 nearest indices for each sample position from kwargs
        indices_info_dict: Dict[Dim, IndicesInfo] = {
            dim: self.coordinates[dim].get_nearest_indices(x, 2)
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
    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the 2 nearest indices for each sample position from kwargs
        indices_info_dict: Dict[Dim, IndicesInfo] = {
            dim: self.coordinates[dim].get_nearest_indices(x, 1)
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


class FastLinearNDInterpolator(NDInterpolator):
    """Fast N-D linear interpolator using all-at-once gather (no n_samples axis).

    Computes nearest-pair indices and weights for every dimension up-front
    using ``LinearCoordinateFast.get_nearest_indices(x, n_samples=2)``, then
    enumerates all 2^N corner combinations with a plain Python bit-mask loop
    (jit-safe). Each
    combination produces a single dict-index gather over all dims at once, which
    avoids the duplicate-dim crash that the per-dim loop in
    ``LinearNDInterpolatorFast`` triggers when query arrays share dims with data.

    Requires coordinates to have ``get_nearest_indices(x, n_samples)`` and
    ``is_within_bounds(x)`` methods (e.g. ``LinearCoordinateFast``).
    """

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Collect nearest-pair indices/weights and bounds for every dim up-front.
        dims = list(xi.keys())
        n = len(dims)
        indices_lo = []
        indices_hi = []
        weights_lo = []
        weights_hi = []
        within_bounds_list = []

        for dim in dims:
            coordinate = self.coordinates[dim]
            info = coordinate.get_nearest_indices(xi[dim], 2)
            indices_lo.append(info.indices_lo)
            indices_hi.append(info.indices_hi)
            weights_lo.append(info.weights_lo)
            weights_hi.append(info.weights_hi)
            if self.fill_value is not None:
                within_bounds_list.append(info.within_bounds)

        # Enumerate all 2^N corners via bit-masking (static Python loop, jit-safe).
        result = None
        for combo in range(1 << n):
            # Build index dict and weight for this corner.
            index_dict = {}
            weight = None
            for j in range(n):
                bit = (combo >> j) & 1
                if bit:
                    index_dict[dims[j]] = indices_hi[j]
                    w = weights_hi[j]
                else:
                    index_dict[dims[j]] = indices_lo[j]
                    w = weights_lo[j]
                weight = w if weight is None else weight * w

            gathered = self.data[index_dict]
            contribution = gathered * weight
            result = contribution if result is None else result + contribution

        # Apply fill_value for out-of-bounds positions (if requested).
        if self.fill_value is not None and within_bounds_list:
            within_bounds = within_bounds_list[0]
            for wb in within_bounds_list[1:]:
                within_bounds = ops.logical_and(within_bounds, wb)
            result = ops.where(within_bounds, result, self.fill_value)

        return result


class FastNearestNDInterpolator(NDInterpolator):
    """Fast nearest-neighbor N-D interpolator using all-at-once gather.

    Computes the nearest index for each dimension via
    ``LinearCoordinateFast.get_nearest_indices(x, n_samples=1)``.
    A single dict-index gather over all dims avoids the shared-dim
    issues of per-dim looping.

    Requires coordinates to have ``get_nearest_indices(x, n_samples)`` and
    ``is_within_bounds(x)`` methods (e.g. ``LinearCoordinateFast``).
    """

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        index_dict = {}
        within_bounds_list = []

        for dim, x in xi.items():
            coordinate = self.coordinates[dim]
            info = coordinate.get_nearest_indices(x, 1)
            index_dict[dim] = info.indices_lo
            if self.fill_value is not None:
                within_bounds_list.append(info.within_bounds)
        
        values = self.data[index_dict]

        if self.fill_value is not None and within_bounds_list:
            within_bounds = within_bounds_list[0]
            for wb in within_bounds_list[1:]:
                within_bounds = ops.logical_and(within_bounds, wb)
            values = ops.where(within_bounds, values, self.fill_value)

        return values


class GeneralFastLinearNDInterpolator(NDInterpolator):
    """Fast N-D linear interpolator using all-at-once gather.

    Enumerates all 2^ndim corner combinations with independent flat gathers
    and linear interpolation weights.  Works with any Coordinate that
    implements ``get_flat_sample_data`` (e.g. ``LinearCoordinate``).

    ``n_samples`` is a static field: changing it triggers JIT recompilation
    (the Python corner-enumeration loop is unrolled at trace time).
    """

    n_samples: int = field(static=True, default=2)

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        dims = list(xi.keys())
        ndim = len(dims)
        n = self.n_samples

        # Collect per-dim flat indices and Lagrange weights.
        all_indices = []  # all_indices[dim_idx][sample_idx] -> flat index array
        all_weights = []  # all_weights[dim_idx][sample_idx] -> flat weight array
        within_bounds_list = []

        for dim in dims:
            coordinate = self.coordinates[dim]
            indices, weights, within_bounds = coordinate.get_flat_sample_data(
                xi[dim], n
            )
            all_indices.append(indices)
            all_weights.append(weights)
            if self.fill_value is not None:
                within_bounds_list.append(within_bounds)

        # Enumerate all n^ndim corners (static Python loop, jit-safe).
        result = None
        for combo in itertools.product(range(n), repeat=ndim):
            index_dict = {}
            weight = None
            for j, sample_idx in enumerate(combo):
                index_dict[dims[j]] = all_indices[j][sample_idx]
                w = all_weights[j][sample_idx]
                weight = w if weight is None else weight * w

            gathered = self.data[index_dict]
            contribution = gathered * weight
            result = contribution if result is None else result + contribution

        # Apply fill_value for out-of-bounds positions (if requested).
        if self.fill_value is not None and within_bounds_list:
            within_bounds = within_bounds_list[0]
            for wb in within_bounds_list[1:]:
                within_bounds = ops.logical_and(within_bounds, wb)
            result = ops.where(within_bounds, result, self.fill_value)

        return result
