from typing import Callable

from numpy import allclose, random
from vbeam.fastmath import Backend
from vbeam.util.coordinate_systems import as_cartesian, as_polar


def test_inverse_conversions(np: Backend, jit_able: Callable[[Callable], Callable]):
    pol2cart2pol = jit_able(lambda points: as_polar(as_cartesian(points)))
    cart2pol2cart = jit_able(lambda points: as_cartesian(as_polar(points)))
    points = random.random((10, 3))
    assert allclose(pol2cart2pol(points), points)
    assert allclose(cart2pol2cart(points), points)
