from spekk import Dim, Module, ops, util

from vbeam.core import Coordinate, IndicesInfo


class FastNearestIndicesInfo(Module):
    """Compact nearest-index info for fast interpolation paths.

    Unlike `IndicesInfo`, this structure avoids adding an explicit n_samples
    axis. It stores lower/upper neighbor indices and their linear weights at
    the same rank as `x`.
    """

    x: ops.array
    indices_lo: ops.array
    indices_hi: ops.array
    weights_lo: ops.array
    weights_hi: ops.array
    within_bounds: ops.array
    # Compatibility fields used by legacy interpolators.
    indices: ops.array
    indices_positions: ops.array
    dim_name: Dim

    @property
    def offset_distances(self) -> ops.array:
        return ops.abs(self.indices_positions - self.x)


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

    def get_flat_sample_data(self, x, n_samples):
        """Return per-sample flat indices and Lagrange weights (no n_samples dim).

        More efficient than ``get_nearest_indices`` for interpolators that
        enumerate corners independently, because it never creates an
        n_samples dimension.

        Args:
            x: query position(s)
            n_samples: number of neighbor samples (1=nearest, 2=linear, 3=quadratic)

        Returns:
            (indices, weights, within_bounds) where *indices* and *weights* are
            lists of length ``n_samples``, each element having the same shape
            as *x*.
        """
        width = self.stop - self.start
        last_index = self.size - 1
        frac_idx = (x - self.start) / width * last_index

        # Pick center so that n_samples neighbors bracket x.
        if n_samples % 2 == 0:
            center = ops.round(frac_idx + 0.5)
        else:
            center = ops.round(frac_idx)

        offsets = [i - n_samples // 2 for i in range(n_samples)]

        # Clipped integer indices for safe array access.
        indices = [
            ops.astype(ops.clip(center + offset, 0, last_index), "int32")
            for offset in offsets
        ]

        # Lagrange interpolation weights (using fractional offset from center;
        # the grid spacing cancels so denominators are pure integers).
        t = frac_idx - center
        weights = []
        for j, oj in enumerate(offsets):
            w = None
            for k, ok in enumerate(offsets):
                if k != j:
                    factor = (t - ok) / (oj - ok)
                    w = factor if w is None else w * factor
            if w is None:  # n_samples == 1
                w = t - t + 1
            weights.append(w)

        within_bounds = self.is_within_bounds(x)
        return indices, weights, within_bounds


class LinearCoordinateFast(LinearCoordinate):
    """Fast linear coordinate that returns floor index and fraction directly.

    Avoids creating an extra n_samples dimension (which causes slow 3D
    scatter-add in JAX backward pass). Instead returns the floor index and
    interpolation fraction for two-point linear interpolation.
    """

    def get_nearest_indices(self, x: float, n_samples: int) -> FastNearestIndicesInfo:
        """Get nearest sample indices without creating an n_samples axis.

        Supports:
            - n_samples=1: nearest-neighbor index
            - n_samples=2: lower/upper neighbors for linear interpolation
        """
        last_index = self.size - 1
        fractional_index = (x - self.start) / (self.stop - self.start) * last_index
        within_bounds = self.is_within_bounds(x)

        if n_samples == 1:
            dim_name = util.random_dim_name(self)
            nearest = ops.astype(
                ops.clip(ops.round(fractional_index), min=0, max=last_index), "int32"
            )
            zeros = fractional_index * 0
            ones = zeros + 1
            nearest_pos = nearest * (self.stop - self.start) / last_index + self.start
            return FastNearestIndicesInfo(
                x=x,
                indices_lo=nearest,
                indices_hi=nearest,
                weights_lo=ones,
                weights_hi=zeros,
                within_bounds=within_bounds,
                indices=ops.stack([nearest], axis=dim_name),
                indices_positions=ops.stack([nearest_pos], axis=dim_name),
                dim_name=dim_name,
            )

        if n_samples == 2:
            dim_name = util.random_dim_name(self)
            idx_floor = ops.astype(
                ops.clip(ops.floor(fractional_index), min=0, max=last_index - 1),
                "int32",
            )
            frac = ops.clip(fractional_index - idx_floor, min=0, max=1)
            idx_hi = idx_floor + 1
            pos_lo = idx_floor * (self.stop - self.start) / last_index + self.start
            pos_hi = idx_hi * (self.stop - self.start) / last_index + self.start
            return FastNearestIndicesInfo(
                x=x,
                indices_lo=idx_floor,
                indices_hi=idx_hi,
                weights_lo=1 - frac,
                weights_hi=frac,
                within_bounds=within_bounds,
                indices=ops.stack([idx_floor, idx_hi], axis=dim_name),
                indices_positions=ops.stack([pos_lo, pos_hi], axis=dim_name),
                dim_name=dim_name,
            )

        raise ValueError(
            "LinearCoordinateFast.get_nearest_indices supports only n_samples=1 or 2"
        )

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
    