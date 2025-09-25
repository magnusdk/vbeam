from typing import Optional, Type

import numpy
from spekk import Dim, field, ops

from vbeam import geometry
from vbeam.delay_models.speed_of_sound.base import SpeedOfSound
from vbeam.interpolation import (
    Coordinate,
    LinearCoordinate,
    LinearNDInterpolator,
    NDInterpolator,
)
from vbeam.scan import Scan


class SpeedOfSoundRayTrancing(SpeedOfSound):
    """Integrate the delays along a straight line between two points in a speed of
    sound map.
    """

    speed_of_sound_map: ops.array
    coordinates: dict[Dim, Coordinate]
    n_samples: int = field(static=True)
    interpolator_type: Type[NDInterpolator] = LinearNDInterpolator
    default_speed_of_sound: float = 1540.0

    def get_delay_between(self, point1: ops.array, point2: ops.array, /) -> ops.array:
        interpolator = self.interpolator_type(
            coordinates=self.coordinates,
            data=self.speed_of_sound_map,
            fill_value=self.default_speed_of_sound,
        )

        step = (point2 - point1) / self.n_samples
        step_size = ops.linalg.vector_norm(step, axis="xyz")
        # Add 0.5 steps because it is more accurate to sample the midpoints (-x- -x-)
        # than the start (x-- x--) of each sub-interval.
        start_position = point1 + 0.5 * step

        def reduce_fn(carry, i):
            sample_point = start_position + i * step
            x, y, z = geometry.util.get_xyz(sample_point)
            sampled_speed_of_sound = interpolator({"xs": x, "zs": z})
            # Integrate the delay for each step. The step size is constant we multiply
            # by it afterwards (see line before return statement).
            carry = carry + 1 / sampled_speed_of_sound
            return carry

        delay_per_meter = ops.reduce_over_dim(
            reduce_fn,
            ops.arange(self.n_samples, dim="iter"),
            init=0.0,
            dim="iter",
        )
        delays = delay_per_meter * step_size
        return delays

    @staticmethod
    def from_scan(
        scan: Scan,
        speed_of_sound_map: ops.array,
        n_samples: Optional[int] = None,
        default_speed_of_sound: float = 1540.0,
    ) -> "SpeedOfSoundRayTrancing":

        points = scan.get_points()
        from_x = points["xyz", 0].min()
        to_x = points["xyz", 0].max()
        from_z = points["xyz", 2].min()
        to_z = points["xyz", 2].max()

        n_x, n_z = (
            speed_of_sound_map.dim_sizes["xs"],
            speed_of_sound_map.dim_sizes["zs"],
        )
        z_axis = LinearCoordinate(from_z, to_z, n_z)
        x_axis = LinearCoordinate(from_x, to_x, n_x)

        if n_samples is None:
            n_samples = int(numpy.ceil(numpy.sqrt(n_x**2 + n_z**2)))

        coordinates = {
            "xs": x_axis,
            "zs": z_axis,
            "xyz": ["xs", None, "zs"],
        }

        return SpeedOfSoundRayTrancing(
            speed_of_sound_map=speed_of_sound_map,
            coordinates=coordinates,
            n_samples=n_samples,
            default_speed_of_sound=default_speed_of_sound,
        )
