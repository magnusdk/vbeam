from abc import abstractmethod
from typing import Callable, Dict, Optional, Union

from spekk import Dim, Module, ops


class InterpolationIndices(Module):
    """Indices of an array and corresponding coordinate positions, useful for
    interpolation.

    Attributes:
        indices (ops.array): The indices of the array.
        positions (ops.array): The coordinate positions for each index.
    """

    indices: ops.array
    positions: ops.array


class InterpolationCoordinates(Module):
    """Coordinates of data, useful for interpolating the data.

    See also :class:`~vbeam.core.interpolation.LinearInterpolationCoordinates`.

    Attributes:
        start (float): The coordinate (e.g. time or position) of the first sample.
        end (float): The coordinate (e.g. time or position) of the last sample.

    `start` and `end` are useful for implementing boundary condition logic.
    """

    start: float
    end: float

    @abstractmethod
    def get_nearest_indices(
        self, x: float, n_samples: int, new_dim: Dim
    ) -> InterpolationIndices:
        """Return the `n_samples` nearest indices around `x` and the corresponding
        positions.

        Args:
            x (float): The coordinate of the data to sample around. For channel data,
                and in the context of delay-and-sum, this would be the delay (i.e.:
                time).
            n_samples (int): The number of samples around `x` to return.
            new_dim (Dim): The name of the dimension of the indices. It will have size
                `n_samples`.
        """


class Interpolable(Module):
    """Interface for things that can be interpolated.

    It needs to let interpolators know the coordinates of the data in order to know
    what indices to sample at, and it needs to be able to get values from an array of
    indices.
    """

    @property
    @abstractmethod
    def interpolation_coordinates(self) -> Dict[Dim, InterpolationCoordinates]:
        """A dictionary from dimension to interpolation coordinates for the data across
        that dimension. Interpolators use this to look up the coordinates for a given
        dimension.
        """

    @abstractmethod
    def get_values(self, indices: ops.array, axis: Dim) -> ops.array:
        """Return the values for the given indices along the given axis."""


class Interpolator(Module):
    @abstractmethod
    def __call__(self, interpolable: Interpolable, x: float, axis: Dim) -> ops.array:
        pass


# The type of an interpolation function
TInterpolator = Union[
    Interpolator,
    Callable[[Interpolable, float], ops.array],
]


class LinearInterpolationCoordinates(InterpolationCoordinates):
    step: float

    def get_nearest_indices(
        self, x: float, n_samples: int, indices_dim: Dim
    ) -> InterpolationIndices:
        # Get the first sample index (fractional, meaning before rounding). It will lie
        # a little to the left of `x` when `n_samples>1`.
        i_frac_first_sample = (x - self.start) / self.step
        i_frac_first_sample -= (n_samples - 1) / 2

        # Round the "fractional" index to make it an integer that we can index by, and
        # add the `n_samples` other indices.
        all_samples_indices = ops.round(i_frac_first_sample)
        all_samples_indices += ops.arange(n_samples, dim=indices_dim)

        # Get back the actual positions/coordinates of the samples at the indices.
        positions = all_samples_indices * self.step + self.start

        return InterpolationIndices(ops.int32(all_samples_indices), positions)


class LinearlySampledData(Interpolable):
    data: ops.array
    start: float
    step: float
    axis: Dim

    @staticmethod
    def from_array(
        data: ops.array,
        xs: Optional[ops.array] = None,
        *,
        axis: Dim,
    ) -> "LinearlySampledData":
        if xs is not None:
            sample_0, sample_1 = xs.slice_dim(axis)[0], xs.slice_dim(axis)[1]
        else:
            sample_0, sample_1 = 0, 1
        return LinearlySampledData(
            data,
            start=sample_0,
            step=sample_1 - sample_0,
            axis=axis,
        )

    @property
    def interpolation_coordinates(self) -> Dict[Dim, LinearInterpolationCoordinates]:
        return {
            self.axis: LinearInterpolationCoordinates(
                self.start,
                self.start + self.step * (self.data.dim_size(self.axis) - 1),
                self.step,
            )
        }

    def get_values(self, indices: ops.array, axis: Dim) -> ops.array:
        return ops.take_along_dim(self.data, indices, axis)
