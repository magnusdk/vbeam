from spekk import Spec

from vbeam.fastmath import numpy as np
from vbeam.fastmath.util.flatten import flatten


def specced_vmap(f: callable, spec: Spec, dimension: str):
    """Similar to vmap, but flattens/decomposes the kwargs to a list that is supported
    by vmap. It also ensures that each input to the vmapped function (vmap(f, ...))
    does not have any field that does not have the given dimension.
    """

    def wrapped(*_unsupported_positional_args, **kwargs):
        if _unsupported_positional_args:
            raise ValueError(
                "Positional arguments are not supported in specced_vmap. Use keyword arguments instead."
            )

        flattened_args, in_axes, unflatten = flatten(kwargs, spec, dimension)

        def f_with_unflattening_args(*args):
            original_kwargs = unflatten(args)
            return f(**original_kwargs)

        vmapped_f = np.vmap(f_with_unflattening_args, in_axes)
        return vmapped_f(*flattened_args)

    return wrapped


if __name__ == "__main__":
    import doctest

    doctest.testmod()
