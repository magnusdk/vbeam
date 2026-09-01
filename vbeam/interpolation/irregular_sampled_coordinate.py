import math

from spekk import Dim, ops, util

from vbeam.core import Coordinate, IndicesInfo


class IrregularSampledCoordinate(Coordinate):
    """Coordinate sampled at irregular, strictly monotonic positions.

    ``x_data`` must be strictly monotonically increasing or decreasing along
    ``dim``. Unsorted positions and duplicate positions are not supported.
    """

    x_data: ops.array
    dim: Dim = None  # dim in x to interploate

    def __post_init__(self):
        if self.x_data.ndim > 1 and self.dim is None:
            raise ValueError(
                "dim must specify the interpolation axis when x_data.ndim > 1"
            )
        if self.x_data.ndim == 1 and self.dim is None:
            self.dim = self.x_data.dims[0]

    @property
    def start(self):
        return self.x_data[self.dim, 0]

    @property
    def stop(self):
        return self.x_data[self.dim, -1]
    
    @property
    def size(self):
        return self.x_data.dim_sizes[self.dim]

    def is_within_bounds(self, x: ops.array) -> bool:
        lower = ops.minimum(self.start, self.stop)
        upper = ops.maximum(self.start, self.stop)
        return ops.logical_and(lower <= x, x <= upper)

    def get_indices_info(self, x: ops.array, n_samples: int) -> IndicesInfo:
        """When n_samples is odd, returns equally number of indices on both sides of closest index.
        When n_samples is even, retuns equally number of indices on both sides of the new sampled position.
        """

        last_index = self.x_data.dim_sizes[self.dim] - 1

        # Generate a unique dimension name for the new axis with size=n_samples.
        dim_name = util.random_dim_name(self)

        direction = ops.where(self.stop > self.start, 1, -1)
        normalized_positions = self.x_data * direction
        normalized_x = x * direction
        lower_bound = ops.zeros_like(self.start + x, dtype=ops.int32)
        upper_bound = lower_bound + self.size
        for _ in range(math.ceil(math.log2(self.size + 1))):
            midpoint = (lower_bound + upper_bound) // 2
            midpoint_clipped = ops.clip(midpoint, 0, last_index)
            searching = lower_bound < upper_bound
            move_right = (
                normalized_positions[self.dim, midpoint_clipped] <= normalized_x
            )
            lower_bound = ops.where(
                ops.logical_and(searching, move_right), midpoint + 1, lower_bound
            )
            upper_bound = ops.where(
                searching, ops.where(move_right, upper_bound, midpoint), upper_bound
            )

        if n_samples % 2 == 0:
            nearest_index = lower_bound
        else:
            left_index = ops.clip(lower_bound - 1, 0, last_index)
            right_index = ops.clip(lower_bound, 0, last_index)
            left_distance = ops.abs(self.x_data[self.dim, left_index] - x)
            right_distance = ops.abs(self.x_data[self.dim, right_index] - x)
            nearest_index = ops.where(
                left_distance <= right_distance, left_index, right_index
            )

        # Add an array of offsets centered around zero. For example:
        offsets = ops.arange(n_samples, dim=dim_name) - n_samples // 2  # [0,1]
        indices_around_x = nearest_index + offsets

        # Convert to int and ensure that we don't index outside of the range.
        indices_around_x = ops.int32(indices_around_x)
        indices_around_x_clipped = ops.clip(indices_around_x, 0, last_index)

        # Get the actual positions/coordinates of the samples at the indices.
        indices_positions = self.x_data[self.dim, indices_around_x_clipped]

        if n_samples == 2:
            # Two-point form: one division instead of two, and no boundary correction.
            position_0 = indices_positions[dim_name, 0]
            position_1 = indices_positions[dim_name, 1]
            denominator = position_1 - position_0
            # Both samples clipped onto the same edge index makes the denominator
            # exactly zero. The guard keeps the weights summing to 1, and both indices
            # are the edge, so the edge value is still reproduced.
            t = (x - position_0) / ops.where(denominator == 0, 1, denominator)
            weights = [1 - t, t]
        else:
            # Samples clipped at the boundary collapse onto the edge position, which
            # would make the Lagrange denominators zero. Give them a virtual position
            # outside the grid instead, so the positions stay strictly ordered. The
            # duplicated index is harmless: Lagrange weights always sum to 1, so the
            # edge value is reproduced.
            mean_spacing = (self.stop - self.start) / last_index
            indices_positions = indices_positions + (
                indices_around_x - indices_around_x_clipped
            ) * mean_spacing

            # Non-uniform spacing, so the weights are built from the actual positions.
            weights = []
            for j in range(n_samples):
                position_j = indices_positions[dim_name, j]
                w = None
                for k in range(n_samples):
                    if k != j:
                        position_k = indices_positions[dim_name, k]
                        factor = (x - position_k) / (position_j - position_k)
                        w = factor if w is None else w * factor
                if w is None:  # n_samples == 1
                    w = x - x + 1
                weights.append(w)

        return IndicesInfo(
            indices_around_x_clipped,
            ops.stack(weights, axis=dim_name),
            self.is_within_bounds(x),
            dim_name,
        )
