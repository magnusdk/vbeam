"""Functions for making classes and fields traceable for the backends.

Use `traceable_dataclass` for your custom classes and `traceable` for field types that 
you know will be traceable by the active backend at runtime.

These methods sets attributes on the class or types that can be understood when calling
`np.as_traceable_dataclass_obj` on them.

A (long, but illustrative) example:
>>> from abc import ABC, abstractmethod
>>> from vbeam.fastmath import numpy as np, backend_manager
>>> from vbeam.fastmath.traceable import traceable, traceable_dataclass
>>> import tensorflow as tf
>>> class MyABC(ABC):
...     @abstractmethod
...     def do_thing(self):
...         ...
...
>>> @traceable_dataclass(data_fields=["a"])
... class Item(MyABC):
...     a: traceable(np.ndarray) = 1.0  # Note: wrapping np.ndarray with traceable is actually unneccesary because it is already wrapped in the backends.
...     def do_thing(self):
...         return 2.0
...
>>> @traceable_dataclass(data_fields=["item"])
... class Container(MyABC):
...     item: Item
...     def do_thing(self):
...         return 3.0
...
>>> backend_manager.active_backend = "tensorflow"
>>> item = Item()
>>> container = Container(item)
>>> @tf.function(jit_compile=True)
... def func(container: Container):
...     return container.item.a + container.item.do_thing() + container.do_thing()
...
>>> with tf.GradientTape() as tape:
...     tape.watch(container.item.a)
...     res = func(container)
...
>>> print(res.numpy(), tape.gradient(res, container.item.a).numpy())  # 6.0 1.0
"""
from vbeam.fastmath import numpy as np


def traceable_dataclass(data_fields=(), aux_fields=()):
    """Decorate the class as a traceable dataclass.

    A class decorated with @traceable_dataclass(...) is understood by the active backend as
    traceable. The class also gets some new attributes such that it acts like a
    dataclass.

    All attributes of the class must also be traceable.

    Args:
      data_fields: The attributes of the object that should be traced.
      aux_fields: Attributes of the object that should not be traced, but should persist
        through function calls regardless.

    Returns:
      A decorator that modifies the class' `__new__` function such that new objects are
      traceable dataclasses.

    Example:
    >>> @traceable_dataclass(data_fields=["a"])
    ... # `Item` can now be understood by backends
    ... class Item:
    ...     a: np.ndarray = 1.0
    """

    def as_traceable_dataclass_inner(cls):
        # Mark class as traceable
        setattr(cls, "__vbeam_fastmath_traceable_custom__", True)
        setattr(cls, "__vbeam_fastmath_traceable_data_fields__", data_fields)
        setattr(cls, "__vbeam_fastmath_traceable_aux_fields__", aux_fields)

        # Duck-type class as a spekk treedef
        setattr(
            cls,
            "__spekk_treedef_keys__",
            lambda self: (
                *get_traceable_data_fields(self),
                *get_traceable_aux_fields(self),
            ),
        )
        setattr(cls, "__spekk_treedef_get__", lambda self, key: getattr(self, key))
        setattr(
            cls,
            "__spekk_treedef_create__",
            lambda self, keys, values: cls(**dict(zip(keys, values))),
        )

        # Wrap __new__ with backend-specific wrapper
        original_new = cls.__new__
        cls.__new__ = lambda cls, *args, **kwargs: np.as_traceable_dataclass_obj(
            original_new(cls), data_fields, aux_fields
        )
        return cls

    return as_traceable_dataclass_inner


def traceable(t):
    """Mark the type-annotation of a field as traceable.

    The type annotations of class attributes are static once they have been defined.
    Wrapping it in `traceable` means that the active backend will know that it is
    traceable regardless of whether it was active when the class was defined or not.
    This way we can define custom classes even before we have set the active backend.

    The type annotations of class attributes are only really an issue when using
    Tensorflow (for now) as the active backend, because it performs type checks when the
    traceable object is created.

    Args:
      t: The class/type/annotation to be marked as traceable.

    Returns:
      A new class, `Traceable`, that inherits from `t`, and that the active backend
      understands is traceable."""

    class Traceable(t):
        __vbeam_fastmath_traceable__ = True

    return Traceable


def is_traceable_dataclass(cls):
    """Return True if the class has been decorated with `@traceable_dataclass(...)`."""
    return getattr(cls, "__vbeam_fastmath_traceable_custom__", False)


def is_traceable(t):
    """Return True if the type has been wrapped with `traceable`."""
    return getattr(t, "__vbeam_fastmath_traceable__", False)


def get_traceable_data_fields(cls):
    return getattr(cls, "__vbeam_fastmath_traceable_data_fields__", ())


def get_traceable_aux_fields(cls):
    return getattr(cls, "__vbeam_fastmath_traceable_aux_fields__", ())
