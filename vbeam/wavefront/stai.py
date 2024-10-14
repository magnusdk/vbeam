from fastmath import ArrayOrNumber

from vbeam.core import ElementGeometry, TransmittedWavefront, WaveData
from vbeam.util.geometry.v2 import distance


class STAIWavefront(TransmittedWavefront):
    def __call__(
        self,
        sender: ElementGeometry,
        point_position: ArrayOrNumber,
        wave_data: WaveData,
    ) -> float:
        return distance(sender.position, point_position)
