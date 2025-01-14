from typing import Dict, Optional

from spekk import Dim, ops

from vbeam.core.interpolation import Interpolable, LinearInterpolationCoordinates


class ChannelData(Interpolable):
    """The recorded channel data.

    Attributes:
        data (ops.array): The array of recorded channel data. It has at least a
            `"time"` dimension.
        t0 (float): The time of the first recorded sample in the channel data since the
            time when the transmitted wave passed through its origin. I.e.: the time
            between generation and acquistion (formulation taken from USTB
            Uff.Wave.initial_time). See
            :class:`~vbeam.core.transmitted_wave.TransmittedWave.origin` for details 
            about the origin of a transmitted wave.
        sampling_frequency (float): The sampling frequency of the channel data.
        modulation_frequency (Optional[float]): The modulation frequency used to
            base-band the channel data. If the channel data is RF data, then
            `modulation_frequency` can be set to `None`.
    """

    data: ops.array
    t0: float
    sampling_frequency: float
    modulation_frequency: Optional[float] = None

    @property
    def interpolation_coordinates(self) -> Dict[Dim, LinearInterpolationCoordinates]:
        return {
            "time": LinearInterpolationCoordinates(
                self.t0,
                self.t0 + (self.data.dim_size("time") - 1) / self.sampling_frequency,
                1 / self.sampling_frequency,
            )
        }

    def get_values(self, indices: ops.array, axis: Dim) -> ops.array:
        return ops.take_along_dim(self.data, indices, axis)
