from typing import Callable, Sequence

from spekk import Spec

from vbeam.fastmath import numpy as np
from vbeam.util.transformations import *


def apply_binary_operation_across_axes(
    a: np.ndarray,
    b: np.ndarray,
    binary_operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    axes: Sequence[int],
) -> np.ndarray:
    """Apply a binary operation to two arrays, where the second array is potentially
    broadcasted and transposed to match the shape and axis-ordering of the first array.

    Args:
        a: The main array.
        b: The second array with potentially fewer axes than ``a``.
        binary_operation: A function that takes two arrays of the same shape and
            returns an array of the same shape.
        axes: The axes of `a` that corresponds to the axes of `b`.

    Let's say we have two arrays, ``a`` and ``b``. They share a dimension, in this
    case, the first dimension of ``b`` corresponds to the second dimension of ``a``:

    >>> dim0, dim1, dim2 = 2, 3, 4
    >>> a = np.ones([dim0, dim1, dim2])
    >>> b = np.ones([dim1])

    Now we want to add the arrays together, but ``a+b`` would raise an error since they
    are not broadcastable. With :func:`apply_binary_operation_across_axes` we can
    specify which axes of ``b`` should be used when applying the binary operation:

    >>> import operator
    >>> result = apply_binary_operation_across_axes(a, b, operator.add, [1])
    >>> result.shape
    (2, 3, 4)

    It also automatically transposes ``b`` to match the axis ordering of ``a``:

    >>> a = np.ones([dim0, dim1, dim2]) # <- dim2 comes AFTER dim1 in a!
    >>> b = np.ones([dim2, dim1])       # <- dim2 comes BEFORE dim1 in b!
    >>> result = apply_binary_operation_across_axes(a, b, operator.add, [2, 1])
    >>> result.shape
    (2, 3, 4)
    """
    if not isinstance(axes, Sequence):
        raise TypeError(
            f"axes must be a sequence of ints, but got {type(axes)}.",
        )
    if len(axes) != b.ndim:
        raise ValueError(
            f"The number of axes must match the number of dimensions of b. Expected {b.ndim} axes, but got only {len(axes)}.",
        )
    if any(axis < 0 or axis >= a.ndim for axis in axes):
        raise ValueError(
            f"All given axes must be between 0 and {a.ndim-1} since they refer to axes in a, but got {axes}.",
        )

    # Make sure that the axes of b are in the correct order so that we don't have to
    # transpose them later:
    ordering = sorted(enumerate(axes), key=lambda x: x[1])
    axes = [axis for _, axis in ordering]  # Re-order axis indices
    b = np.transpose(b, [i for i, _ in ordering])  # Re-order axes of b

    # Create a Spec for the dimensions of the data (names of dimensions doesn't matter):
    a_dims = [f"dim{axis}" for axis in range(a.ndim)]
    b_dims = [f"dim{axis}" for axis in axes]
    spec = Spec({"a": a_dims, "b": b_dims})

    # Transform f so that it works for the given data:
    tf = compose(
        Specced(
            lambda a, b: binary_operation(a, b),
            lambda input_spec: input_spec["a"],
        ),
        *[ForAll(dim) for dim in a_dims if dim not in b_dims],
        Apply(np.transpose, [Axis(dim, keep=True) for dim in a_dims]),
    ).build(spec)
    return tf(a=a, b=b)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
