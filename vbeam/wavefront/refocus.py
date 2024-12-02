from fastmath import Array

from vbeam.core import ProbeGeometry, TransmittedWavefront, WaveData
from vbeam.util.geometry.v2 import distance


class REFoCUSWavefront(TransmittedWavefront):
    """A time-domain REFoCUS wavefront model.

    This class models the diverging spherical wave from a single transmitting /element/
    that was fired as part of an arbitrary focused transmit. It needs to know about the
    original transmitted wave and the original sender (the point that the wave passed
    through at time 0) in order to compensate for arbitrary transmit sequences. See
    :meth:`__call__` for more details.
    """

    base_wavefront: TransmittedWavefront
    base_sender: Array
    compensation_scalar: float = 1.0

    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        """Return the distance that the wave travels, starting from time 0, to when it
        reaches the point position.

        At its core, this is the distance between the sending element and the point
        position, but we also need to compensate for when the element fired in the
        transmit sequence. This compensation is the distance that the full modeled
        wavefront passed through the sending element."""
        focusing_compensation = self.base_wavefront(
            probe, self.base_sender, sender, wave_data
        )
        return (
            distance(sender, point_position)
            + focusing_compensation * self.compensation_scalar
        )
