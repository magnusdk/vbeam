from spekk import ops

from vbeam import geometry
from vbeam.core import (
    GeometricallyFocusedWave,
    Probe,
    TransmittedWaveDelayModel,
)


class STAIDelayModel(TransmittedWaveDelayModel):
    """Synthetic transmit aperture beamforming.

    NOTE: This delay model does **not** use the transmitted wave object, instead it 
    uses the active elements of the probe.
    """

    def __call__(
        self,
        transmitting_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
        speed_of_sound: float,
    ) -> float:
        distances = geometry.distance(
            transmitting_probe.active_elements.position, point
        )
        return distances / speed_of_sound
