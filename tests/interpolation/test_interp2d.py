from typing import Callable

from numpy import allclose
from vbeam.fastmath import Backend
from vbeam.interpolation import FastInterpLinspace


def test_interp2d(np: Backend, jit_able: Callable[[Callable], Callable]):
    interp_x = FastInterpLinspace(10, 1, 2)  # x coords = [10, 11]
    interp_y = FastInterpLinspace(20, 1, 2)  # y coords = [20, 21]
    values = np.array([[10, 20], [30, 40]])

    @jit_able
    def i2d(eval_x, eval_y):
        return FastInterpLinspace.interp2d(eval_x, eval_y, interp_x, interp_y, values)

    # Interpolate x-axis
    assert allclose(i2d(np.array([9.5]), np.array([20.0])), [0.0]), "Outside of bounds"
    assert allclose(i2d(np.array([10.0]), np.array([20.0])), [10.0])
    assert allclose(i2d(np.array([10.5]), np.array([20.0])), [20.0])
    assert allclose(i2d(np.array([11.0]), np.array([20.0])), [30.0])
    assert allclose(i2d(np.array([11.5]), np.array([20.0])), [0.0]), "Outside of bounds"
    # Interpolate y-axis
    assert allclose(i2d(np.array([10.0]), np.array([19.5])), [0.0]), "Outside of bounds"
    assert allclose(i2d(np.array([10.0]), np.array([20.0])), [10.0])
    assert allclose(i2d(np.array([10.0]), np.array([20.5])), [15.0])
    assert allclose(i2d(np.array([10.0]), np.array([21.0])), [20.0])
    assert allclose(i2d(np.array([10.0]), np.array([21.5])), [0.0]), "Outside of bounds"
    # Interpolate exact middle
    assert allclose(i2d(np.array([10.5]), np.array([20.5])), [np.mean(values)])


def test_interp2d_nonscalars(np: Backend, jit_able: Callable[[Callable], Callable]):
    interp_x = FastInterpLinspace(10, 1, 2)  # x coords = [10, 11]
    interp_y = FastInterpLinspace(20, 1, 2)  # y coords = [20, 21]
    # Each value consists of 3 elements instead of just being scalars
    values = np.array(
        [
            [[1, 2, 3], [11, 12, 13]],
            [[21, 22, 23], [31, 32, 33]],
        ]
    )

    @jit_able
    def i2d(eval_x, eval_y):
        return FastInterpLinspace.interp2d(eval_x, eval_y, interp_x, interp_y, values)

    assert allclose(
        i2d(np.array([10]), np.array([20])),
        np.array([[1, 2, 3]]),
    ), "Smallest x and smallest y gets first element"
    assert allclose(
        i2d(np.array([11]), np.array([21])),
        np.array([[31, 32, 33]]),
    ), "Biggest x and Biggest y gets last element"
    assert allclose(
        i2d(np.array([10.5]), np.array([20])),
        np.array([[(1 + 21) / 2, (2 + 22) / 2, (3 + 23) / 2]]),
    ), "Interpolation is element-wise between values"
    assert allclose(
        i2d(np.array([10.0, 10.5, 11.0]), np.array([20.0, 20.5, 21.0])),
        np.array([[1, 2, 3], [16, 17, 18], [31, 32, 33]]),
    ), "We can interpolate multiple values at once"


def test_interp2d_extrapolation(np: Backend, jit_able: Callable[[Callable], Callable]):
    interp_x = FastInterpLinspace(10, 1, 2)  # x coords = [10, 11]
    interp_y = FastInterpLinspace(20, 1, 2)  # y coords = [20, 21]
    # Each value consists of 3 elements instead of just being scalars
    values = np.array(
        [
            [[1, 2, 3], [11, 12, 13]],
            [[21, 22, 23], [31, 32, 33]],
        ]
    )

    default_value = np.array([[100, 200, 300]])

    @jit_able
    def i2d_default(eval_x, eval_y):
        return FastInterpLinspace.interp2d(eval_x, eval_y, interp_x, interp_y, values,edge_handling="Value",default_value=default_value)
    
    @jit_able
    def i2d_nearest(eval_x, eval_y):
        return FastInterpLinspace.interp2d(eval_x, eval_y, interp_x, interp_y, values,edge_handling="Nearest",default_value=default_value)

    assert allclose(
        i2d_default(np.array([15]), np.array([20])),
        default_value,
    ), "Too high x gives default value"
    assert allclose(
        i2d_default(np.array([5]), np.array([20])),
        default_value,
    ), "Too low x gives default value"
    assert allclose(
        i2d_default(np.array([10]), np.array([25])),
        default_value,
    ), "Too high y gives default value"
    assert allclose(
        i2d_default(np.array([10]), np.array([15])),
        default_value,
    ), "Too low y gives default value"


    assert allclose(
        i2d_nearest(np.array([15]), np.array([20])),
        np.array([21, 22, 23]),
    ), "Too high x gives highest x value"
    assert allclose(
        i2d_nearest(np.array([5]), np.array([20])),
        np.array([1, 2, 3]),
    ), "Too low x gives lowest x value"
    assert allclose(
        i2d_nearest(np.array([11]), np.array([25])),
        np.array([31, 32, 33]),
    ), "Too high y gives highest y value"
    assert allclose(
        i2d_nearest(np.array([11]), np.array([15])),
        np.array([21, 22, 23]),
    ), "Too low y gives lowerst y value"