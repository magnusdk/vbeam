from spekk import field, ops

from vbeam import geometry
from vbeam.core import DelayModel, Probe, SeparableDelayModel, TransmittedWave
from vbeam.delay_models.speed_of_sound import SpeedOfSound


class STADelayModel(SeparableDelayModel):
    """Synthetic transmit aperture beamforming."""

    speed_of_sound: float | ops.array

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        element_point_distance = geometry.distance(
            transmitting_probe.active_elements.position,
            point,
        )
        element_origin_distance = geometry.distance(
            transmitting_probe.active_elements.position,
            transmitted_wave.origin,
        )
        tx_distance = element_point_distance - element_origin_distance
        delay = tx_distance / self.speed_of_sound
        return delay

    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        rx_distance = geometry.distance(point, receiving_probe.active_elements.position)
        delay = rx_distance / self.speed_of_sound
        return delay


class STASOSMapDelayModel(DelayModel):
    "Synthetic transmit aperture beamforming."

    speed_of_sound: SpeedOfSound
    same_tx_rx_probe: bool = field(default=False, static=True)

    def __call__(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        delays_tx = self.speed_of_sound.get_delay_between(
            transmitting_probe.active_elements.position, point
        )

        if self.same_tx_rx_probe:
            delays_rx = delays_tx.rename_dim("tx", "rx")
        else:
            delays_rx = self.speed_of_sound.get_delay_between(
                receiving_probe.active_elements.position, point
            )

        delays = delays_tx + delays_rx
        return delays
