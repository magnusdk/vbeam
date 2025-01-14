from typing import Optional

from spekk import ops

from vbeam.core import Apodization, Probe, TransmittedWave


class TxRxApodization(Apodization):
    """Apodization for both tx and rx (just a container of two
    `~vbeam.core.apodization.Apodization` objects)."""

    tx: Optional[Apodization] = None
    rx: Optional[Apodization] = None

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: TransmittedWave,
    ) -> float:
        weight = 1.0
        if self.tx:
            weight *= self.tx(
                transmitting_probe, receiving_probe, point, transmitted_wave
            )
        if self.rx:
            weight *= self.rx(
                transmitting_probe, receiving_probe, point, transmitted_wave
            )
        return weight
