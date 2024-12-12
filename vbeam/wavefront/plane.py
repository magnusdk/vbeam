from spekk import ops

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData


class PlaneWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: ops.array,
        point_position: ops.array,
        wave_data: WaveData,
    ) -> float:
        diff = point_position - sender
        v = ops.stack(
            [
                ops.sin(wave_data.azimuth) * ops.cos(wave_data.elevation),
                ops.sin(wave_data.elevation),
                ops.cos(wave_data.azimuth) * ops.cos(wave_data.elevation),
            ],
            axis="xyz",
        )
        return ops.sum(diff * v, axis="xyz")
