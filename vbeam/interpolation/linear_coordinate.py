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

    def get_indices_info(self, x: ops.array, n_samples: int) -> IndicesInfo:
        width = self.stop - self.start
        last_index = self.size - 1

        # Find the (fractional) index of the sample that lies closest to x, i.e. the
        # index before rounding.
        fractional_index_of_x = (x - self.start) / width * last_index

        # Center the stencil so that n_samples grid points bracket x.
        if n_samples % 2 == 0:
            center = ops.round(fractional_index_of_x + 0.5)
        else:
            center = ops.round(fractional_index_of_x)

        # Offsets centered around zero, e.g. [-1, 0] for n_samples=2.
        offsets = [i - n_samples // 2 for i in range(n_samples)]

        # Ensure that we don't index outside of the range.
        indices = [
            ops.astype(ops.clip(center + offset, 0, last_index), "int32")
            for offset in offsets
        ]

        # Lagrange weights from the fractional offset rather than from the sample
        # positions: the grid spacing cancels, so the denominators are exact integers
        # and the weights stay accurate in float32 even for very long axes.
        t = fractional_index_of_x - center
        if n_samples == 2:
            weights = [-t, 1 + t]
        else:
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

        # Generate a unique dimension name for the new axis with size=n_samples.
        dim_name = util.random_dim_name(self)
        return IndicesInfo(
            ops.stack(indices, axis=dim_name),
            ops.stack(weights, axis=dim_name),
            self.is_within_bounds(x),
            dim_name,
        )

    def to_array(self, *, dim: Dim=None):
        return ops.linspace(self.start, self.stop, self.size, dim=dim)

    def delta(self):
        arr = ops.linspace(self.start, self.stop, self.size)
        return arr[1]-arr[0]
    