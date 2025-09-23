import abc
from typing import final

from spekk import Module, ops

from vbeam.core.probe.base import Probe
from vbeam.core.transmitted_wave import TransmittedWave


class DelayModel(Module):
    """A DelayModel calculates the time elapsed in seconds since
    1. the transmitted wave passed through its origin (see
        :class:`~vbeam.core.transmitted_wave.TransmittedWave.origin`)
    2. it reached a point in space
    3. and was reflected back to the active element(s) of the receiving probe.
    """

    @abc.abstractmethod
    def __call__(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        """Return the time elapsed in seconds since
        1. the transmitted wave passed through its origin (see
            :class:`~vbeam.core.transmitted_wave.TransmittedWave.origin`)
        2. it reached a point in space
        3. and was reflected back to the active element(s) of the receiving probe.
        """

    def get_visualizer(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ):
        from vbeam.visualization.delay_model_visualization import DelayModelVisualizer

        return DelayModelVisualizer(
            self,
            point=point,
            transmitted_wave=transmitted_wave,
            transmitting_probe=transmitting_probe,
            receiving_probe=receiving_probe,
        )


class SeparableDelayModel(DelayModel):
    @abc.abstractmethod
    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array: ...

    @abc.abstractmethod
    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array: ...

    @final
    def __call__(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        tx_delay = self.get_tx_delay(
            point, transmitted_wave, transmitting_probe, receiving_probe
        )
        rx_delay = self.get_rx_delay(
            point, transmitted_wave, transmitting_probe, receiving_probe
        )
        return tx_delay + rx_delay
