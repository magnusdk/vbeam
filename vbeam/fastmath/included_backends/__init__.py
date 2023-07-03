def get_numpy_backend():
    from .numpy_backend import NumpyBackend

    return NumpyBackend()


def get_jax_backend():
    from .jax_backend import JaxBackend

    return JaxBackend()


def get_tensorflow_backend():
    from .tensorflow_backend import TensorflowBackend

    return TensorflowBackend()


included_backends = {
    "numpy": get_numpy_backend,
    "jax": get_jax_backend,
    "tensorflow": get_tensorflow_backend,
}
