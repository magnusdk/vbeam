from typing import Callable

from numpy import allclose, random
from vbeam.fastmath import Backend
from vbeam.util.coordinate_systems import as_cartesian, as_polar


def test_inverse_conversions(np: Backend, jit_able: Callable[[Callable], Callable]):
    pol2cart2pol = jit_able(lambda points: as_polar(as_cartesian(points)))
    cart2pol2cart = jit_able(lambda points: as_cartesian(as_polar(points)))

    # Set random seed for reproducibility
    random.seed(42)

    # Test cases: origin, unit vector, and random points
    test_points = np.array(
        [
            [0.0, 0.0, 0.0],  # origin
            [1.0, 0.0, 0.0],  # unit vector
            [2.0, np.pi / 4, 1.0],
            [3.0, np.pi / 2, -1.0],
            *random.random((6, 3)),  # random points
        ]
    )

    assert allclose(pol2cart2pol(test_points), test_points)
    assert allclose(cart2pol2cart(test_points), test_points)
