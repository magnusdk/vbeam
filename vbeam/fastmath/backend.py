from functools import wraps
from typing import Callable, Sequence, Type, TypeVar, Union

import numpy as np

T_carry = TypeVar("T_carry")
T_x = TypeVar("T_x")


def i_at(arr, i, ax):
    """Return arr with the i-th element along axis ax selected.

    >>> import numpy as np
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> i_at(arr, 1, 0)
    array([4, 5, 6])
    >>> i_at(arr, 1, 1)
    array([2, 5])
    """
    indices = (slice(None),) * ax + (i,)
    return arr[indices]


class Backend:
    """An implementation of the operations needed by vbeam."""

    @property
    @wraps(np.ndarray)
    def ndarray(self) -> Type[np.ndarray]:
        raise NotImplementedError

    def is_ndarray(self, a):
        "Return True if a is an instance of ndarray."
        return isinstance(a, (self.ndarray, np.ndarray))

    @wraps(np.zeros)
    def zeros(self, shape, dtype=None):
        raise NotImplementedError

    @wraps(np.ones)
    def ones(self, shape, dtype=None):
        raise NotImplementedError

    @property
    @wraps(np.pi)
    def pi(self):
        raise NotImplementedError

    @wraps(np.abs)
    def abs(self, x):
        raise NotImplementedError

    @wraps(np.exp)
    def exp(self, x):
        raise NotImplementedError

    @wraps(np.log)
    def log(self, x):
        raise NotImplementedError

    @wraps(np.log10)
    def log10(self, x):
        raise NotImplementedError

    @wraps(np.sin)
    def sin(self, x):
        raise NotImplementedError

    @wraps(np.cos)
    def cos(self, x):
        raise NotImplementedError

    @wraps(np.tan)
    def tan(self, x):
        raise NotImplementedError

    @wraps(np.arcsin)
    def arcsin(self, x):
        raise NotImplementedError

    @wraps(np.arccos)
    def arccos(self, x):
        raise NotImplementedError

    @wraps(np.arctan2)
    def arctan2(self, y, x):
        raise NotImplementedError

    @wraps(np.sqrt)
    def sqrt(self, x):
        raise NotImplementedError

    @wraps(np.sign)
    def sign(self, x):
        raise NotImplementedError

    @wraps(np.nan_to_num)
    def nan_to_num(self, x, nan=0.0):
        raise NotImplementedError

    @wraps(np.min)
    def min(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.max)
    def max(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.sum)
    def sum(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.prod)
    def prod(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.mean)
    def mean(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.cumsum)
    def cumsum(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.cross)
    def cross(self, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        raise NotImplementedError

    @wraps(np.histogram)
    def histogram(self, a, bins=10, weights=None):
        raise NotImplementedError

    @wraps(np.array)
    def array(self, x, dtype=None):
        raise NotImplementedError

    @wraps(np.transpose)
    def transpose(self, a, axes=None):
        raise NotImplementedError

    @wraps(np.swapaxes)
    def swapaxes(self, a, axis1, axis2):
        raise NotImplementedError

    @wraps(np.moveaxis)
    def moveaxis(self, a, source, destination):
        raise NotImplementedError

    @wraps(np.stack)
    def stack(self, arrays, axis=0):
        raise NotImplementedError

    @wraps(np.tile)
    def tile(self, A, reps):
        raise NotImplementedError

    @wraps(np.concatenate)
    def concatenate(self, arrays, axis=0):
        raise NotImplementedError

    @wraps(np.meshgrid)
    def meshgrid(self, *xi, indexing="xy"):
        raise NotImplementedError

    @wraps(np.linspace)
    def linspace(self, start, stop, num=50):
        raise NotImplementedError

    @wraps(np.arange)
    def arange(self, start, stop=None, step=None):
        raise NotImplementedError

    @wraps(np.expand_dims)
    def expand_dims(self, a, axis):
        raise NotImplementedError

    @wraps(np.ceil)
    def ceil(self, x):
        raise NotImplementedError

    @wraps(np.floor)
    def floor(self, x):
        raise NotImplementedError

    @wraps(np.round)
    def round(self, x):
        raise NotImplementedError

    @wraps(np.clip)
    def clip(self, a, a_min, a_max):
        raise NotImplementedError

    @wraps(np.where)
    def where(self, condition, a, b):
        raise NotImplementedError

    @wraps(np.select)
    def select(self, condlist, choicelist, default=0):
        raise NotImplementedError

    @wraps(np.logical_or)
    def logical_or(self, x1, x2):
        raise NotImplementedError

    @wraps(np.logical_or)
    def logical_and(self, x1, x2):
        raise NotImplementedError

    @wraps(np.squeeze)
    def squeeze(self, a, axis=None):
        raise NotImplementedError

    @wraps(np.ravel)
    def ravel(self, a):
        raise NotImplementedError

    @wraps(np.take)
    def take(self, a, indices, axis=None):
        raise NotImplementedError

    @wraps(np.interp)
    def interp(self, x, xp, fp, left=None, right=None, period=None):
        raise NotImplementedError

    class add:
        @staticmethod
        @wraps(np.add.at)
        def at(a, indices, b):
            raise NotImplementedError

    def gather(self, a, indices):
        """Gather slices from array a, according to indices. Same as numpy slicing, but
        is sometimes required when using Tensorflow backend.

        Example:
        >>> a = np.array([0,1,2])
        >>> (a[0,-1] == np.gather(a, [0,-1])).all()
        True
        """
        raise NotImplementedError

    def jit(self, fun, static_argnums=None, static_argnames=None):
        """Just-in-time compile fun, if the backend supports it."""
        raise NotImplementedError

    def vmap(self, fun, in_axes, out_axes=0):
        """Vectorize fun over the axes specified in in_axes, possibly on the GPU, if the
        backend supports it."""
        raise NotImplementedError

    def scan(self, f, init, xs):
        """Scan over the first axis of xs, applying f to a carry and each element of xs.
        f returns the next carry state and the output for that step. the final result is
        the final carry state and the stacked outputs.

        The semantics of scan is roughly equivalent to the following Python code:
        ```
        def scan(f, init, xs):
            carry = init
            ys = []
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            return carry, np.stack(ys)
        ```

        scan can be used to apply a sequential reduction, potentially saving a lot of
        memory. For example, we can compute the sum of an array using scan, where only
        one result array is allocated at a time:
        >>> from vbeam.fastmath import numpy as np
        >>> def f(carry, x):
        ...     return carry + x, None
        >>> init = 0
        >>> xs = np.array([1,2,3,4,5])
        >>> carry, _ = np.scan(f, init, xs)
        >>> carry
        15

        We can also use it as a general-purpose sequential map:
        >>> def f(carry, x):
        ...     return None, x + 1  # Add 1 to each element
        >>> init = None
        >>> xs = np.array([1,2,3,4,5])
        >>> _, ys = np.scan(f, init, xs)
        >>> ys
        array([2, 3, 4, 5, 6])

        Each step will be run sequentially (not in parallel), even on the GPU. If you
        want to vectorize a function to run in parallel you should use vmap instead (if
        the backend supports it). Note that scan and vmap have very different semantics.
        """
        raise NotImplementedError

    def as_traceable_dataclass_obj(self, obj, data_fields, aux_fields):
        """Wrap/modify obj such that it is traceable by the active backend and acts like
        a dataclass.

        See vbeam.fastmath.traceable for details.

        Args:
          obj: The object to be wrapped/modified.
          data_fields: The attributes of the object that should be traced.
          aux_fields: Attributes of the object that should not be traced, but should
            persist through function calls regardless.

        Returns:
          obj, wrapped/modified.
        """
        raise NotImplementedError

    ######################
    # Dependent functions;
    # These do not need to be implemented by subclasses:

    def reduce(
        self,
        f: Callable[[T_carry, T_x], T_carry],
        xs: Union[Sequence[T_x], np.ndarray],
        init_val: T_carry,
    ):
        """Reduce over xs using f, starting with init_val.

        An example that iteratively sums each value in the list ``[1,2,3,4,5]``:

        >>> import operator
        >>> from vbeam.fastmath import numpy as np
        >>> xs = [1,2,3,4,5]
        >>> np.reduce(operator.add, xs, 0)
        15

        It is semantically equivalent to the following Python code:

        >>> def reduce(f, xs, init_val):
        ...     carry = init_val
        ...     for x in xs:
        ...         carry = f(carry, x)
        ...     return carry
        >>> reduce(operator.add, xs, 0)
        15
        """

        def scan_fn(carry, i):
            """Call reduce_fn with the given carry and the i-th element of each arg
            across the axes defined by in_axes."""
            return f(carry, i), i

        xs = self.array(xs)
        carry, _ = self.scan(scan_fn, init_val, xs)
        return carry  # The final carry is the result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
