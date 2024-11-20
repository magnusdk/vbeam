from fastmath import Array, ops

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData


class PlaneWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        diff = point_position - sender
        x, y, z = diff[0], diff[1], diff[2]
        return (
            x * ops.sin(wave_data.azimuth) * ops.cos(wave_data.elevation)
            + y * ops.sin(wave_data.elevation)
            + z * ops.cos(wave_data.azimuth) * ops.cos(wave_data.elevation)
        )
