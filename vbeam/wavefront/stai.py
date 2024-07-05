from vbeam.core import ElementGeometry, TransmittedWavefront, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance


@traceable_dataclass()
class STAIWavefront(TransmittedWavefront):
    def __call__(
        self,
        sender: ElementGeometry,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        return distance(sender.position, point_position)
