from spekk import ops

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData
from vbeam.wavefront.plane import PlaneWavefront


class FocusedSphericalWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: ops.array,
        point_position: ops.array,
        wave_data: WaveData,
    ) -> float:
        sender_source_dist = ops.sqrt(ops.sum((sender - wave_data.source) ** 2))
        source_point_dist = ops.sqrt(ops.sum((wave_data.source - point_position) ** 2))
        return sender_source_dist * ops.sign(
            wave_data.source[2] - sender[2]
        ) - source_point_dist * ops.sign(wave_data.source[2] - point_position[2])


class FocusedHybridWavefront(TransmittedWavefront):
    pw_margin: float = 0.001

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: ops.array,
        point_position: ops.array,
        wave_data: WaveData,
    ) -> float:
        spherical_wavefront = FocusedSphericalWavefront()
        plane_wavefront = PlaneWavefront()
        wave_data.azimuth = ops.arctan2(
            wave_data.source[0] - sender[0],
            wave_data.source[2] - sender[2],
        )
        wave_data.elevation = ops.arctan2(
            wave_data.source[1] - sender.position[1],
            wave_data.source[2] - sender.position[2],
        )
        return ops.where(
            ops.abs(point_position[2] - wave_data.source[2]) > self.pw_margin,
            spherical_wavefront(probe, sender, point_position, wave_data),
            plane_wavefront(probe, sender, point_position, wave_data),
        )


class FocusedBlendedWavefront(TransmittedWavefront):
    blending_power: float = 0.5

    def __call__(
        self,
        sender: ops.array,
        point_position: ops.array,
        wave_data: WaveData,
    ) -> float:
        spherical_wavefront = FocusedSphericalWavefront()
        plane_wavefront = PlaneWavefront()
        wave_data.azimuth = ops.arctan2(
            wave_data.source[0] - sender[0],
            wave_data.source[2] - sender[2],
        )

        wave_data.elevation = ops.arctan2(
            wave_data.source[1] - sender[1],
            wave_data.source[2] - sender[2],
        )
        source_point_dist = ops.sqrt(ops.sum((wave_data.source - point_position) ** 2))
        normalized_distance = ops.clip(
            source_point_dist / ops.sqrt(ops.sum((wave_data.source) ** 2)),
            a_min=0,
            a_max=1,
        )
        plane_distance = plane_wavefront(sender, point_position, wave_data)
        spherical_distance = spherical_wavefront(sender, point_position, wave_data)
        return (
            spherical_distance * normalized_distance**self.blending_power
            + plane_distance * (1 - normalized_distance**self.blending_power)
        )
