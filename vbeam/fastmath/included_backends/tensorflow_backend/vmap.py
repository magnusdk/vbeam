# Code is copied (and slightly modified) from Trax: https://github.com/google/trax/blob/78d2b6ef9070f1f267eb4a74d2ca6618c84abbb5/trax/tf_numpy/extensions/extensions.py#L1907
# Trax is licensed under Apache License 2.0: https://github.com/google/trax/blob/master/LICENSE

import sys

import tensorflow as tf

# Same as `import tensorflow.experimental.numpy as tnp`, but vscode doesn't recognize that for some reason.
import tensorflow._api.v2.experimental.numpy as tnp

# Enable NumPy behavior on Tensors.
tnp.experimental_enable_numpy_behavior()


def reraise(tp, value, tb=None):
    # Code is copied from Six: https://github.com/benjaminp/six/blob/3b7efbcca41857da03fb01f004ccc425ab82dfbf/six.py#L713
    # Siz is licensed under MIT License: https://github.com/benjaminp/six/blob/master/LICENSE
    try:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None
        tb = None


def _tree_broadcast(to, s):
    """Broadcasts `s` to the nested structure `to`."""
    if not isinstance(to, (list, tuple, dict)):
        if not isinstance(s, (int, type(None))):
            raise ValueError
        return s
    if isinstance(s, (int, type(None))):
        return tf.nest.map_structure(lambda x: s, to)
    if isinstance(to, (list, tuple)):
        if len(to) != len(s):
            raise ValueError
        new_s = [_tree_broadcast(x, y) for x, y in zip(to, s)]
        if isinstance(to, tuple):
            new_s = tuple(new_s)
        return new_s
    elif isinstance(to, dict):
        return {k: _tree_broadcast(to[k], s[k]) for k in to}
    else:
        raise TypeError("Unsupported type %s" % type(to))


def vmap(f, in_axes=0, out_axes=0):
    """Returns a function that maps `f` over first dimension of inputs."""
    in_axes_flat = tf.nest.flatten(in_axes)
    if not all(isinstance(l, (type(None), int)) for l in in_axes_flat):
        raise TypeError(
            "vmap in_axes must be an int, None, or (nested) container with "
            "those types as leaves, but got {}.".format(in_axes)
        )
    if all(isinstance(l, type(None)) for l in in_axes_flat):
        raise ValueError("vmap must have at least one non-None value in in_axes")

    out_axes_flat = tf.nest.flatten(out_axes)
    if not all(isinstance(l, (type(None), int)) for l in out_axes_flat):
        raise TypeError(
            "vmap out_axes must be an int, None, or (nested) container with "
            "those types as leaves, but got {}.".format(out_axes)
        )

    def _f(*args):
        flat_args = tf.nest.flatten(args)
        try:
            f_in_axes = _tree_broadcast(args, in_axes)
        except ValueError:
            reraise(
                ValueError,
                ValueError(
                    "vmap in_axes specification must be a tree prefix of the "
                    r"corresponding value, got specification %s for value tree %s"
                    % (in_axes, args)
                ),
                sys.exc_info()[2],
            )
        f_in_axes_flat = tf.nest.flatten(f_in_axes)

        def tf_f(tf_args):
            """Function passed to tf.vectorized_map call."""
            # Note that unbatched arguments are not passed to tf_f. Here we fill thos
            # arguments back before calling `f`.
            tf_flat_args = []
            j = 0
            for arg, axis in zip(flat_args, f_in_axes_flat):
                if axis is None:
                    tf_flat_args.append(arg)
                else:
                    tf_flat_args.append(tf_args[j])
                    j += 1
            unbatched_args = tf.nest.pack_sequence_as(args, tf_flat_args)
            return f(*unbatched_args)

        # Constructs arguments to pass to `tf_f`.
        # Unbatch arguments are skipped. Arguments with non-zero axis are
        # transposed.
        tf_args = []
        for arg, axis in zip(flat_args, f_in_axes_flat):
            if axis is None:
                continue
            # arg = tnp.asarray(arg)  # NOTE: This breaks when vectorizing over extension types such as WaveData
            if axis != 0:
                arg = tnp.moveaxis(arg, axis, 0)
            tf_args.append(arg)
        # TODO(agarwal): consider creating a tf.function outside of _f and reusing
        # that to avoid overheads of re-vectorizing the code when running eagerly.
        outputs = tf.vectorized_map(tf_f, tf_args)
        try:
            f_out_axes = _tree_broadcast(outputs, out_axes)
        except ValueError:
            reraise(
                ValueError,
                ValueError(
                    "vmap out_axes specification must be a tree prefix of the "
                    r"corresponding value, got specification %s for value tree %s"
                    % (out_axes, outputs)
                ),
                sys.exc_info()[2],
            )

        def map_output(x, axis):
            """Maps output of tf.vectorized_map to the final output."""
            # x = tnp.asarray(x)  # NOTE: This breaks when vectorizing over extension types such as WaveData
            if axis is None:
                # Note that `tf.vectorized_map always batches the outputs.
                # Here we unbatch it again.
                return x[0, ...]
            elif axis == 0:
                return x
            else:
                # Need to transpose the output.
                return tnp.moveaxis(x, 0, axis)

        new_outputs = [
            map_output(output, axis)
            for output, axis in zip(
                tf.nest.flatten(outputs), tf.nest.flatten(f_out_axes)
            )
        ]
        return tf.nest.pack_sequence_as(outputs, new_outputs)

    return _f
