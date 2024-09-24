from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance


@traceable_dataclass()
class STAIWavefront(TransmittedWavefront):
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        return distance(probe.sender_position, point_position)
