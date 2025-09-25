import numpy as np
import pytest
from spekk import ops

from vbeam.delay_models.speed_of_sound.ray_tracing import SpeedOfSoundRayTrancing
from vbeam.interpolation import LinearCoordinate


def test_simple_horizontal_case():
    n_grid_x = 6
    n_grid_z = 2
    default_speed_of_sound = 1

    # Linearly increasing speed of sound across xs.
    speed_of_sound_map = (
        ops.ones(shape=(n_grid_z,), dims=["zs"]) * ops.arange(n_grid_x, dim="xs") + 1
    )
    coordinates = {
        "xs": LinearCoordinate(0, n_grid_x - 1, n_grid_x),
        "zs": LinearCoordinate(0, n_grid_z - 1, n_grid_z),
    }
    speed_of_sound = SpeedOfSoundRayTrancing(
        speed_of_sound_map,
        coordinates,
        n_samples=n_grid_x - 1,
        default_speed_of_sound=default_speed_of_sound,
    )

    point1 = ops.asarray([0, 0, 0], dims=["xyz"])
    point2 = ops.asarray([5, 0, 0], dims=["xyz"])
    integrated_delays = speed_of_sound.get_delay_between(point1, point2)
    expected = ops.sum(1 / (speed_of_sound_map["zs", 0, "xs", :-1] + 0.5))

    np.testing.assert_equal(integrated_delays, expected)


def test_simple_diagonal_case():
    n_grid_x = 6
    n_grid_z = 6
    default_speed_of_sound = 1

    # Linearly increasing speed of sound across zs and xs, thus a diagonal line.
    speed_of_sound_map = (
        ops.arange(n_grid_z, dim="zs") + ops.arange(n_grid_x, dim="xs") + 1
    )

    coordinates = {
        "xs": LinearCoordinate(0, n_grid_z - 1, n_grid_z),
        "zs": LinearCoordinate(0, n_grid_x - 1, n_grid_x),
    }
    speed_of_sound = SpeedOfSoundRayTrancing(
        speed_of_sound_map,
        coordinates,
        n_samples=n_grid_x - 1,
        default_speed_of_sound=default_speed_of_sound,
    )

    point1 = ops.asarray([0, 0, 0], dims=["xyz"])
    point2 = ops.asarray([5, 0, 5], dims=["xyz"])
    integrated_delays = speed_of_sound.get_delay_between(point1, point2)
    expected = ops.sum(ops.sqrt(2) / (ops.arange(1, 6) * 2))

    np.testing.assert_allclose(integrated_delays, expected)


if __name__ == "__main__":
    pytest.main([__file__])
