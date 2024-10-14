from typing import Optional

from fastmath import ArrayOrNumber

from vbeam.core import Apodization, ElementGeometry, WaveData


class TxRxApodization(Apodization):
    """Apodization for both transmit and receive (just a container of two Apodization
    functions)."""

    transmit: Optional[Apodization] = None
    receive: Optional[Apodization] = None

    def __call__(
        self,
        sender: ElementGeometry,
        point_position: ArrayOrNumber,
        receiver: ElementGeometry,
        wave_data: WaveData,
    ) -> float:
        weight = 1.0
        if self.transmit:
            weight *= self.transmit(sender, point_position, receiver, wave_data)
        if self.receive:
            weight *= self.receive(sender, point_position, receiver, wave_data)
        return weight
