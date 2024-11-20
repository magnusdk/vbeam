from typing import Callable

from numpy import testing, random
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
            [
                1e-6,
                1e-6,
                0.0,
            ],  # near origin, except that theta/phi are not defined for x=y=z=0
            [1.0, 0.0, 0.0],  # unit vector
            [2.0, np.pi / 4, 1.0],
            [3.0, np.pi / 2, -1.0],
            *random.random((6, 3)),  # random points
        ]
    )

    testing.assert_allclose(
        pol2cart2pol(test_points), test_points, rtol=1e-4, atol=1e-6
    )
    testing.assert_allclose(
        cart2pol2cart(test_points), test_points, rtol=1e-4, atol=1e-6
    )
