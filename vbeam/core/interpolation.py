from abc import abstractmethod
from typing import Mapping, Union

from spekk import Dim, Module, ops


class IndicesInfo(Module):
    """Information about the indices used to interpolate an array.

    Attributes:
        x (ops.array): The position used to interpolate.
        indices (ops.array): The nearest indices in the data array.
        indices_positions (ops.array): The corresponding physical coordinates of the
            sampled indices. It has the same shape as `indices`.
        within_bounds (ops.array): A bool array indicating whether the sampled position
            is within bounds. It has the same shape as `x`.
        dim_name (Dim): The name of the dimension along which the indices are sampled.
    """

    x: ops.array
    indices: ops.array
    indices_positions: ops.array
    within_bounds: ops.array
    dim_name: Dim

    @property
    def offset_distances(self) -> ops.array:
        "The distance from the sampled positions to the interpolated position."
        return ops.abs(self.indices_positions - self.x)


class Coordinate(Module):
    """Coordinate of data, useful for interpolating an array of data. It gives
    information about how to map from a physical position to an index in an array, and
    vice-versa.
    """

    @abstractmethod
    def get_nearest_indices(self, x: ops.array, n_samples: int) -> IndicesInfo:
        """Return the `n_samples` nearest indices around `x` and the corresponding
        positions.

        Args:
            x (float): The coordinate of the data to sample around. For channel data,
                and in the context of delay-and-sum, this would be the delay (i.e.:
                time).
            n_samples (int): The number of samples around `x` to return.
        """


class NDInterpolator(Module):
    """A base class for interpolating N-dimensional arrays with named dimensions.

    Attributes:
        coordinates (Dict[Dim, Coordinate]): The coordinates of the data, giving
            information on how to map from a physical position to an index in the data.
        data (ops.array): The data to be interpolated.
        fill_value (Union[float, None]): The value to give if an index is out of bounds
            of the data. If set to None, then we keep whatever was returned after
            indexing.
    """

    coordinates: Mapping[str, Coordinate]
    data: ops.array
    fill_value: Union[float, None] = float("nan")

    @abstractmethod
    def __call__(self, xi: Mapping[str, int | float | ops.array]) -> ops.array:
        """Interpolate the data at the new positions given by `xi`.

        Args:
            xi (Dict[Dim, ops.array]): A dictionary from dimension name to positions
                that we want to sample at.
        """
