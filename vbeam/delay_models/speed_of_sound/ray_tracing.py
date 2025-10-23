from typing import Optional, Type

from enum import Enum
import numpy
from spekk import Dim, field, ops

from vbeam.delay_models.speed_of_sound.base import SpeedOfSound
from vbeam.interpolation import (
    Coordinate,
    LinearCoordinate,
    LinearNDInterpolator,
    NDInterpolator,
)
from vbeam.scan import Scan


class IntegrationMethod(Enum):
    SPEED_OF_SOUND = "speed_of_sound"
    SLOWNESS = "slowness"


class SpeedOfSoundRayTracing(SpeedOfSound):
    """Integrate the delays along a straight line between two points in a (speed of sound or slowness) map.

    Attributes:
        speed of sound: Array containing the speed of sound values.
        coordinates: Dictionary defining the spatial speed of sound coordinates.
        n_steps: Number of integration steps along the straight ray path.
        interpolator_type: Type of interpolator to use for sampling.
        default_speed of sound: Default value to use when sampling outside the speed of sound coordinates.
        coordinate_names_to_idx: Mapping from coordinate names e.g. ("xs", "ys", "zs")
            to their indices in the xyz dimension.
        unroll: Unrolling factor in a jitted context for the integration loop.
        integration_method: Specifies whether the speed of sound map will be integrated as speed of sound or as
            slowness (1/speed_of_sound) values.
            Note that the output from 'get_delay_between' may not be equal for IntegrationMethod SPEED_OF_SOUND and SLOWNESS.
    """

    speed_of_sound: ops.array
    # map_data: ops.array
    coordinates: dict[Dim, Coordinate]
    n_steps: int = field(static=True)
    interpolator_type: Type[NDInterpolator] = LinearNDInterpolator
    default_speed_of_sound: float = 1540.0
    coordinate_names_to_idx: dict = field(
        default_factory=lambda: {"xs": 0, "ys": 1, "zs": 2}, static=True
    )
    integration_method: IntegrationMethod = IntegrationMethod.SPEED_OF_SOUND
    unroll: int | bool = field(default=1, static=True)

    def get_delay_between(self, point1: ops.array, point2: ops.array, /) -> ops.array:

        if self.integration_method == IntegrationMethod.SPEED_OF_SOUND:
            data = self.speed_of_sound
            default_data = self.default_speed_of_sound
        elif self.integration_method == IntegrationMethod.SLOWNESS:
            data = 1 / self.speed_of_sound
            default_data = 1 / self.default_speed_of_sound

        interpolator = self.interpolator_type(
            coordinates=self.coordinates,
            data=data,
            fill_value=default_data,
        )

        step = (point2 - point1) / self.n_steps
        step_size = ops.linalg.vector_norm(step, axis="xyz")
        # Add 0.5 steps because it is more accurate to sample the midpoints (-x- -x-)
        # than the start (x-- x--) of each sub-interval.
        start_position = point1 + 0.5 * step

        def reduce_fn(carry, i):
            sample_point = start_position + i * step
            in_coordinates = {
                key: sample_point["xyz", self.coordinate_names_to_idx[key]]
                for key in self.coordinates.keys()
            }
            sampled_data = interpolator(in_coordinates)

            # Integrate the delay for each step. The step size is constant we multiply
            # by it afterwards (see line before return statement).
            if self.integration_method == IntegrationMethod.SPEED_OF_SOUND:
                carry = carry + 1 / sampled_data
            elif self.integration_method == IntegrationMethod.SLOWNESS:
                carry = carry + sampled_data
            return carry

        delay_per_meter = ops.reduce_over_dim(
            reduce_fn,
            ops.arange(self.n_steps, dim="iter"),
            init=0.0,
            dim="iter",
            unroll=self.unroll,
        )
        delays = delay_per_meter * step_size
        return delays

    @staticmethod
    def from_scan(
        scan: Scan,
        speed_of_sound_map: ops.array,
        n_samples: Optional[int] = None,
        default_speed_of_sound: float = 1540.0,
    ) -> "SpeedOfSoundRayTracing":
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

        return SpeedOfSoundRayTracing(
            speed_of_sound_map=speed_of_sound_map,
            coordinates=coordinates,
            n_steps=n_samples,
            default_speed_of_sound=default_speed_of_sound,
        )
