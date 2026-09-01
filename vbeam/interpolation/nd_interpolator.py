from __future__ import annotations

import functools
import itertools
from typing import Mapping

from spekk import Dim, field, ops

from vbeam.core import IndicesInfo, NDInterpolator


class LinearNDInterpolator(NDInterpolator):
    """N-D linear interpolator optimized for reverse-mode differentiation.

    For two or more interpolated dimensions, gathers the 2^ndim corners as one
    indexing operation over an ``n_samples`` axis per dimension, then contracts
    those axes with the interpolation weights. One-dimensional complex data uses
    two independent gathers, which gives JAX a faster reverse-mode graph. Works
    with any :class:`~vbeam.core.interpolation.Coordinate`, including
    non-uniform ones such as
    :class:`~vbeam.interpolation.IrregularSampledCoordinate`.

    Prefer :class:`PolynomialNDInterpolator` for forward-only calls.
    Prefer this interpolator when differentiating through two or more
    interpolated dimensions, where its single gather avoids the 2^ndim
    scatter-adds in ``PolynomialNDInterpolator``'s reverse pass.
    """

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the 2 nearest samples for each sample position from kwargs
        samples: dict[Dim, IndicesInfo] = {
            dim: self.coordinates[dim].get_indices_info(x, 2)
            for dim, x in xi.items()
        }

        if len(samples) == 1:
            dim, sample = next(iter(samples.items()))
            if ops.isdtype(self.data.dtype, "complex floating"):
                values = (
                    self.data[{dim: sample.indices[sample.dim_name, 0]}]
                    * sample.weights[sample.dim_name, 0]
                    + self.data[{dim: sample.indices[sample.dim_name, 1]}]
                    * sample.weights[sample.dim_name, 1]
                )
            else:
                values = self.data[{dim: sample.indices}]
                values = (
                    values[sample.dim_name, 0] 
                    * sample.weights[sample.dim_name, 0]
                    + values[sample.dim_name, 1]
                    * sample.weights[sample.dim_name, 1]
                )
        else:
            values = self.data[
                {dim: sample.indices for dim, sample in samples.items()}
            ]
            for sample in samples.values():
                values = ops.sum(values * sample.weights, axis=sample.dim_name)

        # Replace the values that are out of bounds by fill_value if not None.
        if self.fill_value is not None:
            within_bounds = functools.reduce(
                ops.logical_and, (sample.within_bounds for sample in samples.values())
            )
            values = ops.where(within_bounds, values, self.fill_value)

        return values


class NearestNDInterpolator(NDInterpolator):
    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        # Get the nearest sample for each sample position from kwargs
        samples: dict[Dim, IndicesInfo] = {
            dim: self.coordinates[dim].get_indices_info(x, 1)
            for dim, x in xi.items()
        }

        # Get the values at the interpolated indices
        values = self.data[
            {
                # Squeeze to get rid of the dimension with size=1
                dim: ops.squeeze(sample.indices, axis=sample.dim_name)
                for dim, sample in samples.items()
            }
        ]

        # Replace the values that are out of bounds by fill_value if not None.
        if self.fill_value is not None:
            within_bounds = functools.reduce(
                ops.logical_and, (sample.within_bounds for sample in samples.values())
            )
            values = ops.where(within_bounds, values, self.fill_value)

        return values


class PolynomialNDInterpolator(NDInterpolator):
    """Fast N-D linear interpolator using all-at-once gather.

    Enumerates all n^ndim corner combinations with independent flat gathers
    and Lagrange interpolation weights. Works with any Coordinate, including
    non-uniform ones such as ``IrregularSampledCoordinate``.

    ``n_samples`` is a static field: changing it triggers JIT recompilation
    (the Python corner-enumeration loop is unrolled at trace time).
    """

    n_samples: int = field(static=True, default=2)

    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        dims = list(xi.keys())
        ndim = len(dims)
        n = self.n_samples

        # Collect per-dim flat indices and Lagrange weights. Slicing the stacked
        # n_samples axis apart is free: XLA folds the stack away entirely.
        all_indices = []  # all_indices[dim_idx][sample_idx] -> flat index array
        all_weights = []  # all_weights[dim_idx][sample_idx] -> flat weight array
        within_bounds_list = []

        for dim in dims:
            sample = self.coordinates[dim].get_indices_info(xi[dim], n)
            all_indices.append(
                [sample.indices[sample.dim_name, i] for i in range(n)]
            )
            all_weights.append(
                [sample.weights[sample.dim_name, i] for i in range(n)]
            )
            if self.fill_value is not None:
                within_bounds_list.append(sample.within_bounds)

        data = self.data

        # Enumerate all n^ndim corners (static Python loop, jit-safe).
        result = None
        for combo in itertools.product(range(n), repeat=ndim):
            index_dict = {}
            weight = None
            for j, sample_idx in enumerate(combo):
                index_dict[dims[j]] = all_indices[j][sample_idx]
                w = all_weights[j][sample_idx]
                weight = w if weight is None else weight * w

            gathered = data[index_dict]
            contribution = gathered * weight
            result = contribution if result is None else result + contribution

        # Apply fill_value for out-of-bounds positions (if requested).
        if self.fill_value is not None and within_bounds_list:
            within_bounds = within_bounds_list[0]
            for wb in within_bounds_list[1:]:
                within_bounds = ops.logical_and(within_bounds, wb)
            result = ops.where(within_bounds, result, self.fill_value)

        return result
