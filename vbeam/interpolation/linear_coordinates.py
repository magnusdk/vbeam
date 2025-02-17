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