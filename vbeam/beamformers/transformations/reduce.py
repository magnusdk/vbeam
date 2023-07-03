import operator
from dataclasses import dataclass
from typing import Callable, Dict, TypeVar, Union

from spekk import Spec

from vbeam.beamformers.transformations import Transformation
from vbeam.fastmath import numpy as np

T_f_result = TypeVar("T_f_result")
T_reduction_result = TypeVar("T_reduction_result")


def _map_reduce_kwargs_hack(
    map_f: Callable[..., T_f_result],
    reduce_f: Callable[[T_reduction_result, T_f_result], T_reduction_result],
    initial_value: T_reduction_result,
    spec_indices: Dict[str, Union[int, None]],
):
    def wrapped(*args, **kwargs):
        flat_kwargs = list(kwargs.values())
        positional_args = args + tuple(flat_kwargs)
        kwargs_keys = list(kwargs.keys())

        def wrapped_flat_args(carry, *combined_args):
            num_original_args = len(args)
            args_original = combined_args[:num_original_args]
            kwargs_original = dict(zip(kwargs_keys, combined_args[num_original_args:]))
            return reduce_f(carry, map_f(*args_original, **kwargs_original))

        args_axes = [i for k, i in spec_indices.items() if k not in kwargs]
        kwargs_axes = [spec_indices[k] for k in kwargs_keys]
        in_axes = args_axes + kwargs_axes
        reduced_f = np.reduce(wrapped_flat_args, initial_value, in_axes)
        return reduced_f(*positional_args)

    return wrapped


@dataclass
class Reduce(Transformation):
    """Reduce the values of a dimension iteratively.

    A Reduce transformation is generally a ForAll and Apply transformation combined,
    if the Apply transformation somehow aggregates the result (for example by summing
    over the vectorized axis).

    As a concrete example:
    ForAll("transmits") followed by Apply(np.sum, Axis("transmits") is equivalent to
    Reduce.Sum("transmits"), but using Reduce.Sum will likely allocate a lot less
    memory, potentially at the cost of processing time.

    Args:
      dimension: The dimension to reduce over.
      reduce_f: The function to reduce with. For example operator.add for summation.
      initial_value: The initial value for the reduction. For example 0 for summation.
      unroll: The number of iterations to unroll. Unrolling the loop may make it run
        faster at the cost of compilation time.
    """

    dimension: str
    reduce_f: Callable[[T_reduction_result, T_f_result], T_reduction_result]
    initial_value: T_reduction_result

    def transform(self, f: callable, input_spec: Spec, output_spec: Spec) -> callable:
        return _map_reduce_kwargs_hack(
            f,
            self.reduce_f,
            self.initial_value,
            input_spec.index_for(self.dimension),
        )

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec.remove_dimension(self.dimension)

    def transform_output_spec(self, spec: Spec) -> Spec:
        return spec

    def __repr__(self) -> str:
        return f'Reduce("{self.dimension}", {self.reduce_f}, {self.initial_value})'

    @staticmethod
    def Sum(dimension: str, initial_value: T_reduction_result = 0) -> "Reduce":
        return Reduce(dimension, operator.add, initial_value)

    @staticmethod
    def Product(dimension: str, initial_value: T_reduction_result = 1) -> "Reduce":
        return Reduce(dimension, operator.mul, initial_value)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
