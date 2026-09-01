from abc import abstractmethod
from typing import Mapping, Union

from spekk import Dim, Module, ops


class IndicesInfo(Module):
    """The grid samples used to interpolate an array, and their weights.

    Attributes:
        indices (ops.array): The sampled indices into the data array, stacked along
            `dim_name`.
        weights (ops.array): The interpolation weight of each sampled index. It has
            the same shape as `indices` and sums to 1 along `dim_name`.
        within_bounds (ops.array): A bool array indicating whether the sampled position
            is within bounds. It has the same shape as the queried position.
        dim_name (Dim): The name of the dimension the samples are stacked along.
    """

    indices: ops.array
    weights: ops.array
    within_bounds: ops.array
    dim_name: Dim


class Coordinate(Module):
    """Coordinate of data, useful for interpolating an array of data. It gives
    information about how to map from a physical position to an index in an array, and
    vice-versa.
    """

    @abstractmethod
    def get_indices_info(self, x: ops.array, n_samples: int) -> IndicesInfo:
        """Return the `n_samples` grid samples nearest `x`, and their interpolation
        weights.

        Implementations compute the weights themselves, so that each coordinate can
        use whichever basis is most accurate for its own sample spacing.

        Args:
            x (float): The coordinate of the data to sample around. For channel data,
                and in the context of delay-and-sum, this would be the delay (i.e.:
                time).
            n_samples (int): The number of samples around `x` to return. 1 gives
                nearest-neighbor, 2 linear, and 3 quadratic interpolation.
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
