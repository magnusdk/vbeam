import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from vbeam.fastmath.backend import Backend

_already_traceable = set()


class JaxBackend(Backend):
    @property
    def ndarray(self):
        return jnp.ndarray

    def zeros(self, shape, dtype=None):
        return jnp.zeros(shape, dtype)

    def ones(self, shape, dtype=None):
        return jnp.ones(shape, dtype)

    @property
    def pi(self):
        return jnp.pi

    def abs(self, x):
        return jnp.abs(x)

    def exp(self, x):
        return jnp.exp(x)

    def log(self, x):
        return jnp.log(x)

    def log10(self, x):
        return jnp.log10(x)

    def sin(self, x):
        return jnp.sin(x)

    def cos(self, x):
        return jnp.cos(x)

    def tan(self, x):
        return jnp.tan(x)

    def arcsin(self, x):
        return jnp.arcsin(x)

    def arccos(self, x):
        return jnp.arccos(x)

    def arctan2(self, y, x):
        return jnp.arctan2(y, x)

    def sqrt(self, x):
        return jnp.sqrt(x)

    def sign(self, x):
        return jnp.sign(x)

    def nan_to_num(self, x, nan=0.0):
        return jnp.nan_to_num(x, nan=nan)

    def min(self, a, axis=None):
        return jnp.min(a, axis=axis)

    def max(self, a, axis=None):
        return jnp.max(a, axis=axis)

    def sum(self, a, axis=None):
        return jnp.sum(a, axis=axis)

    def prod(self, a, axis=None):
        return jnp.prod(a, axis=axis)

    def mean(self, a, axis=None):
        return jnp.mean(a, axis=axis)

    def cumsum(self, a, axis=None):
        return jnp.cumsum(a, axis=axis)

    def cross(self, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return jnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def histogram(self, a, bins=10, weights=None):
        return jnp.histogram(a, bins=bins, weights=weights)

    def array(self, x, dtype=None):
        return jnp.array(x, dtype=dtype)

    def transpose(self, a, axes=None):
        return jnp.transpose(a, axes=axes)

    def swapaxes(self, a, axis1, axis2):
        return jnp.swapaxes(a, axis1, axis2)

    def moveaxis(self, a, source, destination):
        return jnp.moveaxis(a, source, destination)

    def stack(self, arrays, axis=0):
        return jnp.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return jnp.tile(A, reps)

    def concatenate(self, arrays, axis=0):
        return jnp.concatenate(arrays, axis=axis)

    def meshgrid(self, *xi, indexing="xy"):
        return jnp.meshgrid(*xi, indexing=indexing)

    def linspace(self, start, stop, num=50):
        return jnp.linspace(start, stop, num=num)

    def arange(self, start, stop=None, step=None):
        return jnp.arange(start, stop, step)

    def expand_dims(self, a, axis):
        return jnp.expand_dims(a, axis=axis)

    def ceil(self, x):
        return jnp.ceil(x)

    def floor(self, x):
        return jnp.floor(x)

    def round(self, x):
        return jnp.round(x)

    def clip(self, a, a_min, a_max):
        return jnp.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return jnp.where(condition, x, y)

    def select(self, condlist, choicelist, default=0):
        return jnp.select(condlist, choicelist, default)

    def logical_or(self, x1, x2):
        return jnp.logical_or(x1, x2)

    def logical_and(self, x1, x2):
        return jnp.logical_and(x1, x2)

    def squeeze(self, a, axis=None):
        return jnp.squeeze(a, axis=axis)

    def ravel(self, a):
        return jnp.ravel(a)

    def take(self, a, indices, axis=None):
        if self.is_ndarray(a):
            return jnp.take(a, indices, axis=axis)
        else:
            return jax.tree_util.tree_map(lambda a: jnp.take(a, indices, axis=axis), a)

    def interp(self, x, xp, fp, left=None, right=None, period=None):
        return jnp.interp(x, xp, fp, left, right, period)

    def gather(self, a, indices):
        return a[indices]

    class add:
        @staticmethod
        def at(a, indices, b):
            return a.at[indices].add(b)

    def jit(self, fun, static_argnums=None, static_argnames=None):
        return jax.jit(
            fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    def vmap(self, fun, in_axes, out_axes=0):
        return jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def scan(self, f, init, xs):
        return jax.lax.scan(f, init, xs)

    def as_traceable_dataclass_obj(self, original_obj, data_fields, aux_fields):
        cls = type(original_obj)

        # Register the class of the object (if it hasn't been done before) as a
        # pytree-node.
        if cls not in _already_traceable:

            def flatten_fn(obj):
                data = [getattr(obj, child) for child in data_fields]
                aux_data = [getattr(obj, aux_field) for aux_field in aux_fields]
                return data, (cls, aux_data)

            def unflatten_fn(treedef, flattened_obj):
                cls, aux_data = treedef
                children_kwargs = dict(zip(data_fields, flattened_obj))
                aux_kwargs = dict(zip(aux_fields, aux_data))
                kwargs = {**children_kwargs, **aux_kwargs}
                return cls(**kwargs)

            try:
                # May throw ValueError if this module got reloaded and
                # _already_traceable got reset.
                jax.tree_util.register_pytree_node(cls, flatten_fn, unflatten_fn)
            except ValueError as e:
                warnings.warn(
                    f"Received ValueError when trying to register JAX pytree: {e}"
                )
            finally:
                _already_traceable.add(cls)  # Don't register it again.

        as_dataclass = dataclass(cls)  # Make it a dataclass as well.
        original_obj.__class__ = as_dataclass
        return original_obj
