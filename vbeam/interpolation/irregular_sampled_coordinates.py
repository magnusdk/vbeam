from spekk import ops, util, Dim

from vbeam.core import Coordinates, IndicesInfo


class IrregularSampledCoordinates(Coordinates):
    x_data: ops.array
    dim: Dim = None # dim in x to interploate

    def __post_init__(self):
        if self.x_data.ndim>1 and self.dim is None:
            ValueError("For x_data.ndim>1, self.dim need to be set as interpolation axis. got self.dim={self.dim}")
        if self.x_data.ndim==1 and self.dim is None:
            self.dim = self.x_data.dims[0]

    def get_nearest_indices(self, x: float, n_samples: int) -> IndicesInfo:
        """ When n_samples is odd, returns equally number of indices on both sides of closest index. 
        When n_samples is even, retuns equally number of indices on both sides of the new sampled position. 
          """
        
        last_index = self.x_data.size - 1

        # Generate a unique dimension name for the new axis with size=n_samples.
        dim_name = util.random_dim_name(self)
        
        # Calculate the absolute differences between each element in y and x
        differences = self.x_data - x
        nearest_index = ops.argmin(ops.abs(differences), axis=self.dim)

        if n_samples%2==0:
            differences_at_nearest_index = differences[self.dim, nearest_index]
            shift = ops.where(differences_at_nearest_index>0, 0, 1)
            nearest_index = nearest_index + shift

        # Add an array of offsets centered around zero. For example:
        offsets = ops.arange(n_samples, dim=dim_name) - n_samples // 2 # [0,1]
        indices_around_x = nearest_index + offsets
        
        # Convert to int and ensure that we don't index outside of the range.
        indices_around_x = ops.int32(indices_around_x)
        indices_around_x_clipped = ops.clip(indices_around_x, 0, last_index)   
        
        # Get the actual positions/coordinates of the samples at the indices.
        indices_positions = self.x_data[self.dim, indices_around_x_clipped]

        # Need to pertubate indices positions on data bondary to prevent both offset_distanceses to be zero.
        indices_outside = ops.where(indices_around_x_clipped!=indices_around_x, True, False)

        # indices_positions[indices_outside] += 1
        indices_positions = ops.where(indices_outside, indices_positions+1, indices_positions)

        return IndicesInfo(
            x,
            indices_around_x_clipped,
            indices_positions,
            self.is_within_bounds(x),
            dim_name,
        )
        