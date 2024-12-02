from typing import Optional

from fastmath import Array

from vbeam.core import Apodization, ProbeGeometry, WaveData


class TxRxApodization(Apodization):
    """Apodization for both transmit and receive (just a container of two Apodization
    functions)."""

    transmit: Optional[Apodization] = None
    receive: Optional[Apodization] = None

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        receiver: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        weight = 1.0
        if self.transmit:
            weight *= self.transmit(probe, sender, receiver, point_position, wave_data)
        if self.receive:
            weight *= self.receive(probe, sender, receiver, point_position, wave_data)
        return weight
