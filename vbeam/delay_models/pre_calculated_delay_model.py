from spekk import ops

from vbeam.core import DelayModel, Probe, TransmittedWave


class PreCalculatedDelayModel(DelayModel):
    """Delay model that returns a pre-calculated array of delays."""

    rx_delays: ops.array
    tx_delays: ops.array

    def __call__(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> ops.array:
        return self.tx_delays + self.rx_delays 
