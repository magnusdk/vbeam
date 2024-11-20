import pytest
from numpy import allclose, random
from vbeam.fastmath import backend_manager
from vbeam.fastmath import numpy as api


def test_concatenate_general():
    import tensorflow as tf

    # Test some randomly generated cases
    def assert_equal_results_numpy_tf(arr):
        # Test concatenating along each possible axis
        for axis in range(arr.ndim - 1):
            with backend_manager.using_backend("numpy"):
                expected = api.concatenate(arr, axis)

            with backend_manager.using_backend("tensorflow"):
                assert allclose(expected, api.concatenate(arr, axis))
                jitted_concatenate = tf.function(api.concatenate, jit_compile=True)
                assert allclose(expected, jitted_concatenate(arr, axis))

    assert_equal_results_numpy_tf(random.random((6, 7, 8, 9, 10)))
    assert_equal_results_numpy_tf(random.random((6, 1)))
    assert_equal_results_numpy_tf(random.random((6, 0)))


def test_concatenate_negative_axes():
    arrays = random.random((6, 7, 8, 9, 10))
    with backend_manager.using_backend("numpy"):
        expected0 = api.concatenate(arrays, -1)
        expected1 = api.concatenate(arrays, -2)
        expected2 = api.concatenate(arrays, -3)
        expected3 = api.concatenate(arrays, -4)
    with backend_manager.using_backend("tensorflow"):
        assert allclose(expected0, api.concatenate(arrays, -1))
        assert allclose(expected1, api.concatenate(arrays, -2))
        assert allclose(expected2, api.concatenate(arrays, -3))
        assert allclose(expected3, api.concatenate(arrays, -4))


def test_concatenate_raising_errors():
    # Test raising errors
    with backend_manager.using_backend("tensorflow"):
        with pytest.raises(
            ValueError, match="zero-dimensional arrays cannot be concatenated"
        ):
            api.concatenate(api.ones((2,)))
        with pytest.raises(
            ValueError, match="axis 1 is out of bounds for array of dimension 1"
        ):
            api.concatenate(api.ones((2, 2)), 1)
        with pytest.raises(
            ValueError, match="axis -2 is out of bounds for array of dimension 1"
        ):
            api.concatenate(api.ones((2, 2)), -2)
