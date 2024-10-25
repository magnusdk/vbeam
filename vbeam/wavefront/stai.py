from fastmath import Array

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData
from vbeam.util.geometry.v2 import distance


class STAIWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        return distance(probe.sender_position, point_position)
