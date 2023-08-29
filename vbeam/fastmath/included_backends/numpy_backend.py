from dataclasses import dataclass

import numpy as np

from vbeam.fastmath.backend import Backend, i_at
from vbeam.fastmath.traceable import (
    get_traceable_aux_fields,
    get_traceable_data_fields,
    is_traceable_dataclass,
)


class NumpyBackend(Backend):
    @property
    def ndarray(self):
        return np.ndarray

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype)

    @property
    def pi(self):
        return np.pi

    def abs(self, x):
        return np.abs(x)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def log10(self, x):
        return np.log10(x)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def tan(self, x):
        return np.tan(x)

    def arcsin(self, x):
        return np.arcsin(x)

    def arccos(self, x):
        return np.arccos(x)

    def arctan2(self, y, x):
        return np.arctan2(y, x)

    def sqrt(self, x):
        return np.sqrt(x)

    def sign(self, x):
        return np.sign(x)

    def nan_to_num(self, x, nan=0.0):
        return np.nan_to_num(x, nan=nan)

    def min(self, a, axis=None):
        return np.min(a, axis=axis)

    def max(self, a, axis=None):
        return np.max(a, axis=axis)

    def sum(self, a, axis=None):
        return np.sum(a, axis=axis)

    def prod(self, a, axis=None):
        return np.prod(a, axis=axis)

    def mean(self, a, axis=None):
        return np.mean(a, axis=axis)

    def cumsum(self, a, axis=None):
        return np.cumsum(a, axis=axis)

    def cross(self, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return np.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def histogram(self, a, bins=10, weights=None):
        return np.histogram(a, bins=bins, weights=weights)

    def array(self, x, dtype=None):
        return np.array(x, dtype=dtype)

    def transpose(self, a, axes=None):
        return np.transpose(a, axes=axes)

    def swapaxes(self, a, axis1, axis2):
        return np.swapaxes(a, axis1, axis2)

    def moveaxis(self, a, source, destination):
        return np.moveaxis(a, source, destination)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return np.tile(A, reps)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def meshgrid(self, *xi, indexing="xy"):
        return np.meshgrid(*xi, indexing=indexing)

    def linspace(self, start, stop, num=50):
        return np.linspace(start, stop, num=num)

    def arange(self, start, stop=None, step=None):
        return np.arange(start, stop, step)

    def expand_dims(self, a, axis):
        return np.expand_dims(a, axis=axis)

    def ceil(self, x):
        return np.ceil(x)

    def floor(self, x):
        return np.floor(x)

    def round(self, x):
        return np.round(x)

    def clip(self, a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def select(self, condlist, choicelist, default=0):
        return np.select(condlist, choicelist, default)

    def logical_or(self, x1, x2):
        return np.logical_or(x1, x2)

    def logical_and(self, x1, x2):
        return np.logical_and(x1, x2)

    def squeeze(self, a, axis=None):
        return np.squeeze(a, axis=axis)

    def ravel(self, a):
        return np.ravel(a)

    def take(self, a, indices, axis=None):
        return np.take(a, indices, axis=axis)

    def interp(self, x, xp, fp, left=None, right=None, period=None):
        return np.interp(x, xp, fp, left, right, period)

    def gather(self, a, indices):
        return a[indices]

    class add:
        @staticmethod
        def at(a, indices, b):
            a = a.copy()
            np.add.at(a, indices, b)
            return a

    def jit(self, fun, static_argnums=None, static_argnames=None):
        return fun  # No-op

    def vmap(self, fun, in_axes, out_axes=0):
        v_axes = [(i, ax) for i, ax in enumerate(in_axes) if ax is not None]

        def vectorized_fun(*args, **kwargs):
            v_sizes = [args[i].shape[ax] for i, ax in v_axes]
            v_ax_size = v_sizes[0]
            assert all(
                [v_size == v_ax_size for v_size in v_sizes]
            ), "All vectorized axes must have the same number of elements."

            results = []
            for i in range(v_ax_size):
                new_args = [
                    args[j] if ax is None else i_at(args[j], i, ax)
                    for j, ax in enumerate(in_axes)
                ]
                results.append(fun(*new_args, **kwargs))
            results = _recombine_traceables(results)
            results = _set_out_axes(results, out_axes)
            return results

        return vectorized_fun

    def scan(self, f, init, xs):
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

    def as_traceable_dataclass_obj(self, obj, data_fields, aux_fields):
        # Numpy is "traceable" by default (because there is no tracing).
        as_dataclass = dataclass(type(obj))  # Make it a dataclass, though.
        obj.__class__ = as_dataclass
        return obj


def _set_out_axes(result: np.ndarray, out_axis: int):
    result_type = type(result)
    if is_traceable_dataclass(result_type):
        data_fields = get_traceable_data_fields(result_type)
        aux_fields = get_traceable_aux_fields(result_type)
        kwargs = {field: getattr(result_type, field) for field in aux_fields}
        for field in data_fields:
            kwargs[field] = _set_out_axes(getattr(result, field), out_axis)
        return result_type(**kwargs)
    elif isinstance(result, tuple):
        return tuple([_set_out_axes(r, out_axis) for r in result])
    else:
        return np.moveaxis(result, 0, out_axis)


def _recombine_traceables(results: list):
    """Ensure that the returned value of a vmapped function is consistent with jax.

    If the results is a list of traceable dataclass objects, then they are combined into
    a single instance of that object.
    If it is a tuple, a tuple is returned, processed recursively.
    Otherwise, the results are simply returned as a numpy array."""
    if is_traceable_dataclass(type(results[0])):
        assert all([is_traceable_dataclass(type(result)) for result in results])
        result_type = type(results[0])
        data_fields = get_traceable_data_fields(result_type)
        aux_fields = get_traceable_aux_fields(result_type)
        kwargs = {field: getattr(result_type, field) for field in aux_fields}
        for field in data_fields:
            results_for_field = [getattr(result, field) for result in results]
            kwargs[field] = np.stack(results_for_field, 0)
        return result_type(**kwargs)
    elif isinstance(results[0], tuple):
        assert all([isinstance(result, tuple) for result in results])
        n_items = len(results[0])
        return tuple(
            [_recombine_traceables([r[i] for r in results]) for i in range(n_items)]
        )
    else:
        return np.array(results)
