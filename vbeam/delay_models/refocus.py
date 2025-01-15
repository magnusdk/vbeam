from typing import Optional

from spekk import ops

from vbeam import geometry
from vbeam.core import GeometricallyFocusedWave, Probe, TransmittedWaveDelayModel
from vbeam.util._transmitted_wave import raise_if_not_geometrically_focused_wave


class REFoCUSDelayModel(TransmittedWaveDelayModel):
    """Model the spherical waves fired by the individual elements of the transmitting 
    probe.

    Reference:
        N. Bottenus, "Recovery of the Complete Data Set From Focused Transmit Beams," 
        in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 
        vol. 65, no. 1, pp. 30-38, Jan. 2018, doi: 10.1109/TUFFC.2017.2773495.
    """
    base_wavefront: TransmittedWaveDelayModel
    tx_speed_of_sound: Optional[float] = None

    def __call__(
        self,
        transmitting_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
        speed_of_sound: float,
    ) -> float:
        raise_if_not_geometrically_focused_wave(transmitted_wave)

        # Calculate the delays for when each individual element fired.
        tx_speed_of_sound = (
            self.tx_speed_of_sound
            if self.tx_speed_of_sound is not None
            else speed_of_sound
        )
        focusing_compensation = self.base_wavefront(
            transmitting_probe,
            transmitting_probe.active_elements.position,
            transmitted_wave,
            tx_speed_of_sound,
        )

        # Calculate the distances from each element to the imaged point.
        distances = geometry.distance(
            transmitting_probe.active_elements.position, point
        )
        # Calculate the delays with focusing compensation.
        delays = distances / speed_of_sound + focusing_compensation
        return delays
