import os
import sys

# Disable GPU for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add helper functions to path
sys.path.append(os.path.join(os.path.dirname(__file__), "vbeam_test_helpers"))

import pytest
from spekk import ops

ops.backend.set_backend("numpy")

# backends = ["numpy", "jax", "tensorflow"]
backends = ["numpy", "jax"]


@pytest.fixture(params=backends)
def np(request):
    """A fixture that provides a fastmath backend to be used.

    Usage:
    def my_test(np: Backend):  # np has type vbeam.fastmath.Backend
        assert np.abs(-1) == 1

    tests/conftest.py (this file) is automatically imported by pytest when running tests
    so this function will be available to all tests automatically.
    """
    with ops.backend.temporary_backend(request.param):
        yield ops


@pytest.fixture(params=("jit", "no_jit"))
def jit_able(request, np):
    def _jit(f):
        if request.param == "no_jit":
            return f
        elif ops.backend.backend_name == "jax":
            import jax

            return jax.jit(f)
        elif ops.backend.backend_name == "tensorflow":
            import tensorflow as tf

            return tf.function(f, jit_compile=True)
        else:
            return f  # Do nothing

    return _jit
