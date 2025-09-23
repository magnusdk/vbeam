from spekk import field, ops, replace

from vbeam.core import DelayModel, Probe, SeparableDelayModel, TransmittedWave
from vbeam.delay_models.synthetic_transmit_aperture import STADelayModel


class REFoCUSDelayModel(DelayModel):
    """Model the spherical waves fired by the individual elements of the transmitting
    probe.

    Reference:
        N. Bottenus, "Recovery of the Complete Data Set From Focused Transmit Beams,"
        in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control,
        vol. 65, no. 1, pp. 30-38, Jan. 2018, doi: 10.1109/TUFFC.2017.2773495.
    """

    synthetic_element_positions: ops.array
    focusing_delay_model: SeparableDelayModel
    stai_delay_model: DelayModel = field(default_factory=lambda: STADelayModel(1540.0))

    def __call__(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        # The element fired when the original wave passed through it. This is the
        # focusing delay compensation.
        focusing_compensation = self.focusing_delay_model.get_tx_delay(
            self.synthetic_element_positions,
            transmitted_wave,
            transmitting_probe,
            receiving_probe,
        )

        # Calculate the delays with focusing compensation.
        stai_transmitted_wave = replace(
            transmitted_wave, origin=self.synthetic_element_positions
        )
        stai_delays = self.stai_delay_model(
            point, stai_transmitted_wave, transmitting_probe, receiving_probe
        )
        delays = stai_delays + focusing_compensation
        return delays
