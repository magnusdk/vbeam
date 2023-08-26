import logging
import warnings

logger = logging.getLogger(__name__)


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
backend_priority = ["jax", "numpy"]


def get_best_available_backend():
    """Attempt to import backends in order of priority and return the first one that
    succeeds."""
    for backend_name in backend_priority:
        try:
            included_backends[backend_name]()
            logger.info(f"Using backend: {backend_name}")
            if backend_name == "numpy":
                warnings.warn(
                    "The numpy backend is not optimized for performance. Consider \
installing JAX unless you are only doing simple processing."
                )
            return backend_name
        except ImportError:
            pass
    raise ImportError(
        "Unable to import any of the supported backends. Please follow the install \
instructions for vbeam for how to install the required dependencies."
    )
