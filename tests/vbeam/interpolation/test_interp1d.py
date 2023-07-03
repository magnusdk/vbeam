from typing import Callable

from numpy import allclose
from vbeam.fastmath import Backend
from vbeam.interpolation import FastInterpLinspace


def test_interp1d(np: Backend, jit_able: Callable[[Callable], Callable]):
    interp = FastInterpLinspace(0, 1, 2)  # coords = [0, 1]
    jitted_interp1d = jit_able(
        lambda x: interp.interp1d(x, np.array([np.pi, np.pi * 2]), 42, 1337)
    )
    eval_x = np.linspace(-1, 2, 31)
    result = jitted_interp1d(eval_x)
    assert allclose(result[:10], 42), "Left padding is added"
    assert allclose(
        result[10:21], np.linspace(np.pi, np.pi * 2, 11), 1e-5, 1e-6
    ), "Coordinates within bounds are interpolated"
    assert allclose(result[21:], 1337), "Right padding is added"
