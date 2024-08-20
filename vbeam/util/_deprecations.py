from typing import TypeVar


def _get_function_name(f: callable):
    return getattr(f, "__qualname__", getattr(f, "__name__"))


def renamed_kwargs(version: str, **renamed_kwargs: str):
    """Print a deprecation warning when passing an argument by name when that argument
    has been renamed in a recent version.

    >>> @renamed_kwargs("1.0.5", b="renamed_arg")
    ... def f(a, renamed_arg):
    ...   return a + renamed_arg
    >>> f(a=1, b=2)
    Deprecation warning: argument 'b' of f was renamed to 'renamed_arg' in version 1.0.5.
    3

    Using the new name or given positional arguments doesn't print the warning.
    >>> f(a=1, renamed_arg=2)
    3
    >>> f(1, 2)
    3
    """

    def decorator(f):
        def wrapped(*args, **kwargs):
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in renamed_kwargs:
                    print(
                        f"Deprecation warning: argument '{k}' of "
                        f"{_get_function_name(f)} was renamed to '{renamed_kwargs[k]}' "
                        f"in version {version}."
                    )
                    k = renamed_kwargs[k]
                new_kwargs[k] = v
            return f(*args, **new_kwargs)

        return wrapped

    return decorator


if __name__ == "__main__":
    import doctest

    doctest.testmod()
