from typing import Dict, Optional

from spekk import Dim, Module, ops

from vbeam.interpolation import LinearCoordinates


class LinearlySampledChannelData(Module):
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

    def remodulate_if_iq(self, values: ops.array, delays: ops.array) -> ops.array:
        if self.modulation_frequency is not None:
            w0 = ops.pi * 2 * self.modulation_frequency
            values = values * ops.exp(1j * w0 * (delays - self.t0))
        return values

    @property
    def data_coordinates(self) -> Dict[Dim, LinearCoordinates]:
        n_time_samples = self.data.dim_sizes["time"]
        return {
            "time": LinearCoordinates(
                self.t0,
                self.t0 + (n_time_samples - 1) / self.sampling_frequency,
                n_time_samples,
            )
        }
