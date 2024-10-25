"Point-based apodization for weighting the delayed signal."

from abc import abstractmethod

from fastmath import Array, Module

from vbeam.core.probe_geometry import ProbeGeometry
from vbeam.core.wave_data import WaveData


class Apodization(Module):
    @abstractmethod
    def __call__(
        self,
        probe: ProbeGeometry,
        sender: Array,
        receiver: Array,
        point_position: Array,
        wave_data: WaveData,
    ) -> float:
        """
        Return the weighting for the signal for the given sender, point, receiver
        position, and wave data.

        Args:
          sender: The geometry of the sender (e.g. position (x, y, z)).
          point_position: The position of the point being imaged (x, y, z).
          receiver: The geometry of the receiver (e.g. position (x, y, z)).
          wave_data: Data specific to the wave being sent (see vbeam.core.WaveData).
        """
        ...
