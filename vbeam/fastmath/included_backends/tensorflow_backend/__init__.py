from functools import reduce
from typing import Union

import tensorflow as tf

# Same as `import tensorflow.experimental.numpy as tnp`, but vscode doesn't recognize that for some reason.
import tensorflow._api.v2.experimental.numpy as tnp
from tensorflow.python.util import nest

from vbeam.fastmath.backend import Backend
from vbeam.fastmath.included_backends.tensorflow_backend.vmap import vmap
from vbeam.fastmath.traceable import is_traceable, is_traceable_dataclass

# Enable NumPy behavior on Tensors.
tnp.experimental_enable_numpy_behavior()
_traceable_classes_cache = {}


def swap_axes(t: tf.Tensor, ax1: int, ax2: int):
    return tf.transpose(
        t,
        [
            ax2 if dim_i == ax1 else ax1 if dim_i == ax2 else dim_i
            for dim_i in range(t.ndim)
        ],
    )


def make_type_work_with_tensorflow(T):
    if is_traceable(T):
        return Union[tf.Tensor, tf.experimental.BatchableExtensionType]

    if len(getattr(T, "__args__", ())) > 0:
        tf_types_args = [make_type_work_with_tensorflow(arg) for arg in T.__args__]
        return T.copy_with(tuple(tf_types_args))

    if is_traceable_dataclass(T):
        return tf.experimental.BatchableExtensionType

    # Fallback to tf.Tensor. This solves cases where the type is
    # float but value is scalar Tensor, which (I think) _should_
    # work, but doesn't. Tensorflow is quite strict about types here.
    if T is float:
        return Union[float, tf.Tensor]

    return T


class TensorflowBackend(Backend):
    @property
    def ndarray(self):
        return tnp.ndarray

    def zeros(self, shape, dtype=float):
        return tnp.zeros(shape, dtype)

    def ones(self, shape, dtype=float):
        return tnp.ones(shape, dtype)

    @property
    def pi(self):
        return tnp.pi

    def abs(self, x):
        return tnp.abs(x)

    def exp(self, x):
        return tnp.exp(x)

    def log(self, x):
        return tnp.log(x)

    def log10(self, x):
        return tnp.log10(x)

    def sin(self, x):
        return tnp.sin(x)

    def cos(self, x):
        return tnp.cos(x)

    def tan(self, x):
        return tnp.tan(x)

    def arcsin(self, x):
        return tnp.arcsin(x)

    def arccos(self, x):
        return tnp.arccos(x)

    def arctan2(self, y, x):
        return tnp.arctan2(y, x)

    def sqrt(self, x):
        return tnp.sqrt(x)

    def sign(self, x):
        return tnp.sign(x)

    def nan_to_num(self, x, nan=0.0):
        return tf.where(tnp.isnan(x), tf.ones_like(x) * nan, x)

    def min(self, a, axis=None):
        return tnp.min(a, axis=axis)

    def max(self, a, axis=None):
        return tnp.max(a, axis=axis)

    def sum(self, a, axis=None):
        return tnp.sum(a, axis=axis)

    def prod(self, a, axis=None):
        return tnp.prod(a, axis=axis)

    def mean(self, a, axis=None):
        return tnp.mean(a, axis=axis)

    def cumsum(self, a, axis=None):
        return tnp.cumsum(a, axis=axis)

    def cross(self, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
        return tnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    def histogram(self, a, bins=10, weights=None):
        return tnp.histogram(a, bins=bins, weights=weights)

    def array(self, x, dtype=None):
        return tnp.array(x, dtype=dtype)

    def transpose(self, a, axes=None):
        return tnp.transpose(a, axes=axes)

    def swapaxes(self, a, axis1, axis2):
        return tnp.swapaxes(a, axis1, axis2)

    def moveaxis(self, a, source, destination):
        return tnp.moveaxis(a, source, destination)

    def stack(self, arrays, axis=0):
        return tnp.stack(arrays, axis=axis)

    def tile(self, A, reps):
        return tnp.tile(A, reps)

    def concatenate(self, arrays, axis=0):
        # tnp.concatenate has slightly different behavior from numpy and JAX in that it
        # requires the arrays to be in a list. We can't create a list from a Tensor
        # while JIT-compiling, so we instead roll our own version.
        original_axis = axis
        if not isinstance(arrays, tf.Tensor):
            arrays = tnp.array(arrays)

        if axis < 0:
            axis = arrays.ndim - 1 + axis  # axis is negative so this is subtraction

        if arrays.ndim <= 1:
            raise ValueError("zero-dimensional arrays cannot be concatenated")
        if axis >= (arrays.ndim - 1) or axis < 0:
            raise ValueError(
                f"axis {original_axis} is out of bounds for array of dimension {arrays.ndim-1}"
            )

        arrays = swap_axes(arrays, 1, axis + 1)
        shape = arrays.shape
        arrays = tf.reshape(arrays, (shape[0] * shape[1], *shape[2:]))
        return swap_axes(arrays, 0, axis)

    def meshgrid(self, *xi, indexing="xy"):
        return tnp.meshgrid(*xi, indexing=indexing)

    def linspace(self, start, stop, num=50):
        return tnp.linspace(start, stop, num=num)

    def arange(self, start, stop=None, step=None):
        if step is None:
            step = 1
        return tnp.arange(start, stop, step)

    def expand_dims(self, a, axis):
        # Support multiple axes
        if not isinstance(axis, tuple):
            axis = (axis,)
        return reduce(tnp.expand_dims, axis, a)

    def ceil(self, x):
        return tnp.ceil(x)

    def floor(self, x):
        return tnp.floor(x)

    def round(self, x):
        return tnp.round(x)

    def clip(self, a, a_min, a_max):
        return tnp.clip(a, a_min, a_max)

    def where(self, condition, x, y):
        return tnp.where(condition, x, y)

    def select(self, condlist, choicelist, default=0):
        return tnp.select(condlist, choicelist, default)

    def logical_or(self, x1, x2):
        return tnp.logical_or(x1, x2)

    def logical_and(self, x1, x2):
        return tnp.logical_and(x1, x2)

    def squeeze(self, a, axis=None):
        return tnp.squeeze(a, axis=axis)

    def ravel(self, a):
        return tnp.ravel(a)

    def take(self, a, indices, axis=None):
        if is_traceable_dataclass(a.__class__):
            flattened_a = [
                tnp.take(a_tensor, indices, axis=axis)
                for a_tensor in nest.flatten(a, expand_composites=True)
            ]
            return nest.pack_sequence_as(a, flattened_a, expand_composites=True)
        else:
            return tnp.take(a, indices, axis=axis)

    def interp(self, x, xp, fp, left=None, right=None, period=None):
        # TODO: Implement tensorflow version
        return tnp.interp(x, xp, fp, left, right, period)

    def gather(self, a, indices):
        return tf.gather(a, indices)

    class add:
        @staticmethod
        def at(a, indices, b):
            if indices.dtype == bool:
                indices = tf.where(indices)
            else:
                # Flatten indices and updates
                indices, b = tnp.ravel(indices), tnp.ravel(b)
                # Explicitly convert negative indices to their corresponding positive ones
                neg1_indices = tf.where(indices < 0)
                adds = tf.repeat(a.shape[0], neg1_indices.shape[0])
                indices = tf.tensor_scatter_nd_add(
                    indices, neg1_indices, tf.cast(adds, indices.dtype)
                )
                # Add extra dimension so that tensorflow is happy :)
                indices = tnp.expand_dims(indices, -1)
            return tf.tensor_scatter_nd_add(a, indices, tf.cast(b, a.dtype))

    def jit(self, fun, static_argnums=None, static_argnames=None):
        if static_argnums is not None or static_argnames is not None:
            raise NotImplementedError(
                "Tensorflow backend currently doesn't support static_argnums or \
static_argnames"
            )
        return tf.function(fun, jit_compile=True)

    def vmap(self, fun, in_axes, out_axes=0):
        return tf.function(vmap(fun, in_axes, out_axes))

    def scan(self, f, init, xs):
        # TODO: perhaps use tf.scan instead? However, it has slightly different
        # semantics than jax.lax.scan which is what our backend is based on.
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, tnp.stack(ys)

    def as_traceable_dataclass_obj(self, obj, data_fields, aux_fields):
        # Make the object an instance of Tensorflow's ExtensionType so that it can be
        # traced by Tensorflow.
        #
        # Tensorflow doesn't need to differentiate between child attributes and
        # auxilliary data, so we can ignore the last two arguments: data_fields and
        # aux_fields.

        original_class = obj.__class__
        if original_class not in _traceable_classes_cache:
            # Make all the class attributes traceable.
            traceable_annotations = {
                k: make_type_work_with_tensorflow(v)
                for (k, v) in getattr(original_class, "__annotations__", {}).items()
            }
            # Create the new traceable version of the class
            TensorflowTraceable = type(
                original_class.__name__,
                (original_class, tf.experimental.BatchableExtensionType),
                {
                    "__qualname__": original_class.__qualname__,
                    "__annotations__": traceable_annotations,
                },
            )
            _traceable_classes_cache[original_class] = TensorflowTraceable

        # Overwrite the original class
        object.__setattr__(obj, "__class__", _traceable_classes_cache[original_class])
        return obj
