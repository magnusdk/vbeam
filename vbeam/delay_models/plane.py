from spekk import ops

from vbeam import geometry
from vbeam.core import (
    GeometricallyFocusedWave,
    Probe,
    SeparableDelayModel,
    TransmittedWave,
)


class PlaneDelayModel(SeparableDelayModel):
    """A simple plane wave delay model."""

    speed_of_sound: float | ops.array

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        if not isinstance(transmitted_wave, GeometricallyFocusedWave):
            raise ValueError(
                "Expected a geometrically focused transmitted wave, but got "
                f"{type(transmitted_wave)}."
            )

        # tx_distance is the distance to the plane that is oriented towards the
        # transmitted wave direction and that passes through transmitted_wave.origin.
        tx_distance = ops.linalg.vecdot(
            transmitted_wave.virtual_source.direction,
            point - transmitted_wave.origin,
            axis="xyz",
        )
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
