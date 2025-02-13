from spekk import ops, util

from vbeam.core import Coordinates, IndicesInfo


class LinearCoordinates(Coordinates):
    size: int

    def get_nearest_indices(self, x: float, n_samples: int) -> IndicesInfo:
        width = self.stop - self.start
        last_index = self.size - 1

        # Generate a unique dimension name for the new axis with size=n_samples.
        dim_name = util.random_dim_name(self)

        # Find the (fractional) index of the sample that lies closest to x, i.e. the
        # index before rounding. Note that we compensate for the half-pixel offset
        # which is needed when interpolating pixels.
        half_pixel_comp = 0.5 * (2 - n_samples % 2)
        fractional_index_of_x = (x - self.start) / width * last_index + half_pixel_comp
        # Floor it to get the integer index (note: the dtype is still float).
        base_index = ops.floor(fractional_index_of_x)

        # Add an array of offsets centered around zero. For example:
        # - with n_samples=2, offsets become [-0.5, 0.5].
        # - with n_samples=3, offsets become [-1,   0,   1].
        # - etc...
        offsets = ops.arange(
            -(n_samples - 1) / 2,
            n_samples / 2,
            dim=dim_name,
            dtype=base_index.dtype,
        )
        sample_indices = base_index + offsets

        # Get the actual positions/coordinates of the samples at the indices. Note that
        # we subtract 0.5 to undo the half-pixel offset that we added earlier.
        indices_positions = (sample_indices - 0.5) * width / last_index + self.start

        # Convert to int and ensure that we don't index outside of the range.
        indices_around_x = ops.clip((sample_indices).int32(), 0, last_index)

        return IndicesInfo(
            x,
            indices_around_x,
            indices_positions,
            self.is_within_bounds(x),
            dim_name,
        )
