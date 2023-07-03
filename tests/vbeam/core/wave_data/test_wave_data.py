from typing import Callable

from numpy import allclose
from vbeam.core import WaveData
from vbeam.fastmath import Backend


def test_vmapping_wave_data(np: Backend, jit_able: Callable[[Callable], Callable]):
    wave_data = WaveData(source=np.array([[0, 0, 1], [0, 0, 2]]))
    inc_z = lambda wave_data: wave_data.with_updates_to(
        source=lambda source: source + np.array([0, 0, 1])
    )
    result = jit_able(np.vmap(inc_z, [0]))(wave_data)
    assert isinstance(result, WaveData)
    assert allclose(result.source, np.array([[0, 0, 2], [0, 0, 3]]))

    # dup_inc_z returns a tuple of two WaveData objects and one number
    dup_inc_z = lambda wave_data: (inc_z(wave_data), inc_z(wave_data), 1337.0)
    result = jit_able(np.vmap(dup_inc_z, [0]))(wave_data)
    assert isinstance(result[0], WaveData)
    assert isinstance(result[1], WaveData)
    assert np.is_ndarray(result[2])
    assert allclose(result[0].source, np.array([[0, 0, 2], [0, 0, 3]]))
    assert allclose(result[1].source, np.array([[0, 0, 2], [0, 0, 3]]))
    assert allclose(result[2], np.array([1337.0, 1337.0]))
