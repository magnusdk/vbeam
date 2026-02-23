from spekk import Dim, ops, util

from vbeam.core import Coordinate, IndicesInfo


class LinearCoordinate(Coordinate):
    start: float
    stop: float
    size: int

    def is_within_bounds(self, x: ops.array) -> bool:
        lower = ops.minimum(self.start, self.stop)
        upper = ops.maximum(self.start, self.stop)
        return ops.logical_and(lower <= x, x <= upper)

    def get_nearest_indices(self, x: float, n_samples: int) -> IndicesInfo:
        width = self.stop - self.start
        last_index = self.size - 1

        # Generate a unique dimension name for the new axis with size=n_samples.
        dim_name = util.random_dim_name(self)

        # Find the (fractional) index of the sample that lies closest to x, i.e. the
        # index before rounding.
        fractional_index_of_x = (x - self.start) / width * last_index
        if n_samples % 2 == 0:
            fractional_index_of_x += 0.5
        nearest_index = ops.round(fractional_index_of_x)

        # Add an array of offsets centered around zero. For example:
        offsets = ops.arange(n_samples, dim=dim_name) - n_samples // 2
        indices_around_x = nearest_index + offsets

        # Get the actual positions/coordinates of the samples at the indices.
        indices_positions = indices_around_x * width / last_index + self.start

        # Convert to int and ensure that we don't index outside of the range.
        indices_around_x = ops.int32(indices_around_x)
        indices_around_x = ops.clip(indices_around_x, 0, last_index)

        return IndicesInfo(
            x,
            indices_around_x,
            indices_positions,
            self.is_within_bounds(x),
            dim_name,
        )

    def to_array(self, *, dim: Dim=None):
        return ops.linspace(self.start, self.stop, self.size, dim=dim)

    def delta(self):
        arr = ops.linspace(self.start, self.stop, self.size)
        return arr[1]-arr[0]


class LinearCoordinateFast(LinearCoordinate):
    """Fast linear coordinate that returns floor index and fraction directly.

    Avoids creating an extra n_samples dimension (which causes slow 3D
    scatter-add in JAX backward pass). Instead returns the floor index and
    interpolation fraction for two-point linear interpolation.
    """

    def get_index_and_frac(self, x):
        """Get floor index and interpolation fraction.

        Args:
            x: physical position(s) — spekk array of any shape

        Returns:
            idx_floor: int32 index of the lower neighbor, clipped to [0, size-2]
            frac: interpolation fraction in [0, 1]
        """
        last_index = self.size - 1
        fractional_index = (x - self.start) / (self.stop - self.start) * last_index
        idx_floor = ops.astype(
            ops.clip(ops.floor(fractional_index), min=0, max=last_index - 1), "int32"
        )
        frac = ops.clip(fractional_index - idx_floor, min=0, max=1)
        return idx_floor, frac
    