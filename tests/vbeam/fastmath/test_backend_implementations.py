from typing import Callable

import numpy
from numpy import allclose
from vbeam.fastmath import Backend


def test_add_at(np: Backend):
    a = np.zeros((5,))

    indices = np.array([], dtype="int32")
    b = np.array([])
    assert allclose(
        np.add.at(a, indices, b), [0, 0, 0, 0, 0], 0, 0
    ), "adding at no indices returns the array as-is"

    indices = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    assert allclose(
        np.add.at(a, indices, b), [3, 0, 0, 0, 0], 0, 0
    ), "adding 1 to the same index three times increases that value by 3"

    indices = np.array([0, 2, 3, 3])
    b = np.array([1.2, 3.14, 4.7, 5.1])
    assert allclose(
        np.add.at(a, indices, b), [1.2, 0.0, 3.14, 9.8, 0.0]
    ), "adding different numbers at different indices works as expected"

    indices = np.array([-1, -2, -3])
    b = np.array([1, 1, 1])
    assert allclose(
        np.add.at(a, indices, b), [0.0, 0.0, 1.0, 1.0, 1.0]
    ), "adding at negative indexes works as expected"


def test_add_at_multidimensional(np: Backend):
    a = np.zeros((5,))

    indices = np.array([[0, 1], [0, 4], [4, 4]], dtype="int32")
    b = np.array([[1, 1], [1, 1], [1, 1]])
    assert allclose(
        np.add.at(a, indices, b), [2, 1, 0, 0, 3]
    ), "the shape of indices and b doesn't matter"


def test_vmap(np: Backend, jit_able: Callable[[Callable], Callable]):
    a = np.ones((2, 3, 4))
    b = np.ones((3, 4))

    sum_first_and_last_axis = lambda a: np.sum(a, (0, -1))
    assert jit_able(np.vmap(sum_first_and_last_axis, [0]))(a).shape == (2,)
    assert jit_able(np.vmap(sum_first_and_last_axis, [1]))(a).shape == (3,)
    assert jit_able(np.vmap(sum_first_and_last_axis, [2]))(a).shape == (4,)

    sum_ab_1 = lambda a, b: a + b
    sum_ab = np.vmap(sum_ab_1, [0, None])
    sum_ab = np.vmap(sum_ab, [1, 0])
    sum_ab = np.vmap(sum_ab, [2, 1])
    assert jit_able(sum_ab)(a, b).shape == (4, 3, 2)

    sum_ab_1 = lambda a, b: a + b
    sum_ab = np.vmap(sum_ab_1, [0, 0])
    sum_ab = np.vmap(sum_ab, [1, 1])
    sum_ab = np.vmap(sum_ab, [0, None])
    assert jit_able(sum_ab)(a, b).shape == (2, 4, 3)


def test_nan_to_num(np: Backend, jit_able: Callable[[Callable], Callable]):
    original_array = np.array(numpy.array([numpy.nan, numpy.nan, 1, numpy.nan]))
    assert allclose(jit_able(np.nan_to_num)(original_array), np.array([0, 0, 1, 0]))
