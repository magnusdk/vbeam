from typing import Optional

from vbeam.core import Apodization, ElementGeometry, ProbeGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


@traceable_dataclass(("transmit", "receive"))
class TxRxApodization(Apodization):
    """Apodization for both transmit and receive (just a container of two Apodization
    functions)."""

    transmit: Optional[Apodization] = None
    receive: Optional[Apodization] = None

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: np.ndarray,
        receiver: np.ndarray,
        point_position: np.ndarray,
        wave_data: WaveData,
    ) -> float:
        weight = 1.0
        if self.transmit:
            weight *= self.transmit(probe, sender, receiver, point_position, wave_data)
        if self.receive:
            weight *= self.receive(probe, sender,receiver, point_position, wave_data)
        return weight
