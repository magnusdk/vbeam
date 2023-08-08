from typing import Callable


def grad_for_argname(f: Callable, arg_name: str):
    """Wrap ``f`` in :func:`jax.grad` such that the gradient is taken with respect to
    ``arg_name``.

    Normally, :func:`jax.grad` can only take the gradient with respect to positional
    arguments. This function allows you to take the gradient with respect to a keyword
    argument.

    >>> import jax
    >>> def f(x, y):
    ...     return x * y
    >>> f_grad1 = jax.grad(f, argnums=1)  # The argument with index=1 is "y"
    >>> f_grad2 = grad_for_argname(f, "y")  # Here we reference it by name instead

    ``f_grad1`` and ``f_grad2`` are equivalent in this case:

    >>> f_grad1(2., 3.) == f_grad2(2., 3.)
    True
    """
    import jax

    def wrapped(**kwargs):
        if arg_name not in kwargs:
            raise ValueError(
                f"An error occurred when calculating the gradient for {f}: \
{arg_name} not found in kwargs."
            )

        def pos_arg_f(arg):
            new_kwargs = kwargs.copy()
            new_kwargs[arg_name] = arg  # Overwrite arg_name with traced value
            return f(**new_kwargs)

        f_grad = jax.grad(pos_arg_f)
        return f_grad(kwargs[arg_name])

    return wrapped


if __name__ == "__main__":
    import doctest

    doctest.testmod()
