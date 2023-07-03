"""Experimental transformations. See individual classes for documentation."""


import operator
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import jax
import jax.numpy as jnp
from spekk import Spec
from spekk.process.transformations import Transformation


def i_at(arr: jnp.ndarray, i: int, ax: int):
    "Index arr at index i along axis ax."
    indices = (slice(None),) * ax + (i,)
    return arr[indices]


@dataclass
class AxisIndexer:
    axes: tuple  # The axis to index for each argument. None means no indexing.

    def __call__(self, args: tuple, i: int):
        return tuple(
            args[j] if ax is None else i_at(args[j], i, ax)
            for j, ax in enumerate(self.axes)
        )


T_f_result = TypeVar("T_f_result")
T_reduction_result = TypeVar("T_reduction_result")


def map_reduce(
    map_f: Callable[..., T_f_result],
    in_axes: Sequence[int],
    reduce_f: Callable[[T_reduction_result, T_f_result], T_reduction_result],
    initial_value: T_reduction_result,
    unroll=1,
) -> T_reduction_result:
    """Return a function that maps map_f over the vectorized axes of its arguments, and
    iteratively reduces the results with reduce_f.

    In most cases this will be more memory-efficient than using jax.vmap, which in turn
    may speed up the process (if the bottleneck is memory allocation).

    >>> import jax.numpy as jnp
    >>> a = jnp.array([1., 2., 3.])
    >>> f = lambda x: x+1

    >>> from operator import add, mul
    >>> f_generalized_reduced_sum = map_reduce(f, [0], add, 0)
    >>> float(f_generalized_reduced_sum(a))
    9.0

    >>> f_generalized_reduced_prod = map_reduce(f, [0], mul, 1)
    >>> float(f_generalized_reduced_prod(a))
    24.0
    """
    v_axes = [(i, ax) for i, ax in enumerate(in_axes) if ax is not None]

    def vectorized_fun(*args):
        v_sizes = [args[i].shape[ax] for i, ax in v_axes]
        v_ax_size = v_sizes[0]
        assert all(
            [v_size == v_ax_size for v_size in v_sizes]
        ), "All vectorized axes must have the same number of elements."

        indexer = AxisIndexer(in_axes)
        carry = reduce_f(initial_value, map_f(*indexer(args, 0)))
        result, _ = jax.lax.scan(
            lambda carry, x: (reduce_f(carry, map_f(*indexer(args, x))), None),
            carry,
            jnp.arange(1, v_ax_size),
            unroll=unroll,
        )
        return result

    return vectorized_fun


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
    unroll: int = 1

    def transform(self, f: Callable, input_spec: Spec, output_spec: Spec) -> Callable:
        in_axes = input_spec.index_for(self.dimension)
        if isinstance(in_axes, dict):
            in_axes = list(in_axes.values())
        return map_reduce(f, in_axes, self.reduce_f, self.initial_value, self.unroll)

    def preprocess_spec(self, spec: Spec) -> Spec:
        return spec.remove_dimension(self.dimension)

    def postprocess_spec(self, spec: Spec) -> Spec:
        return spec

    def __repr__(self) -> str:
        r = f'Reduce("{self.dimension}", {self.reduce_f}, {self.initial_value}'
        if self.unroll != 1:
            r += f", unroll={self.unroll}"
        r += ")"
        return r

    @staticmethod
    def Sum(dimension: str, unroll: int = 1) -> "Reduce":
        return Reduce(dimension, operator.add, 0, unroll)

    @staticmethod
    def Product(dimension: str, unroll: int = 1) -> "Reduce":
        return Reduce(dimension, operator.mul, 1, unroll)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
