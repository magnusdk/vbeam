import numpy as np
import scipy
from spekk import Module, ops, replace

from vbeam.channel_data import LinearlySampledChannelData


class ChannelDataFilterer(Module):
    """Use ChannelDataFilterer.from_firwin to create a filterer with a firwin filter.
    Call plot on the resulting object to see the result in the frequency spectrum.
    """

    band_center_frequency: float
    filter_coefficients: ops.array
    group_delay_in_samples: int
    demodulate: bool
    upsample_factor: int
    downsample_factor: int

    @staticmethod
    def from_firwin(
        n_taps: int,
        cutoff: float,
        band_center_frequency: float,
        sampling_frequency: float,
        *,
        demodulate: bool,
        upsample_factor: int,
        downsample_factor: int,
    ) -> "ChannelDataFilterer":
        filter_coefficients = scipy.signal.firwin(
            n_taps,
            cutoff,
            pass_zero="lowpass",
            fs=float(sampling_frequency),
        )
        filter_coefficients = ops.array(filter_coefficients, ["time"])
        group_delay_in_samples = (filter_coefficients.size - 1) // 2
        return ChannelDataFilterer(
            band_center_frequency,
            filter_coefficients,
            group_delay_in_samples,
            demodulate,
            upsample_factor,
            downsample_factor,
        )

    def __call__(
        self, channel_data: LinearlySampledChannelData
    ) -> LinearlySampledChannelData:
        signal = channel_data.data
        time = channel_data.data_coordinates["time"].to_array(dim="time")
        phasor = ops.exp(-1j * 2 * ops.pi * self.band_center_frequency * time)
        signal = signal * phasor
        signal = ops.convolve1d(
            signal, self.filter_coefficients, mode="same", axis="time"
        )
        if not self.demodulate:
            signal = signal / phasor
        signal = ops.array(
            scipy.signal.resample_poly(
                signal.data,
                up=self.upsample_factor,
                down=self.downsample_factor,
            )
            / self.upsample_factor
            * self.downsample_factor,
            signal.dims,
        )
        delay = self.group_delay_in_samples / channel_data.sampling_frequency

        channel_data = replace(channel_data, data=signal)
        channel_data = replace(channel_data, t0=channel_data.t0 - delay * 0)
        channel_data = replace(
            channel_data,
            sampling_frequency=channel_data.sampling_frequency
            * self.upsample_factor
            / self.downsample_factor,
        )
        if self.demodulate:
            channel_data = replace(
                channel_data,
                modulation_frequency=self.band_center_frequency,
            )
        return channel_data

    def plot(self, channel_data: LinearlySampledChannelData, ax=None):
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        channel_data_fft = ops.fft.fft(channel_data.data, axis="time")
        channel_data_fft = ops.mean(ops.abs(channel_data_fft), axis=["rx", "tx"])
        freqs = ops.fft.fftshift(
            ops.fft.fftfreq(
                channel_data_fft.dim_sizes["time"],
                d=1 / channel_data.sampling_frequency,
            )
        )
        ax.plot(
            freqs,
            np.fft.fftshift(channel_data_fft),
            label="Original signal",
        )

        filter_coefficients_fft = np.abs(
            np.fft.fft(self.filter_coefficients, n=channel_data_fft.dim_sizes["time"])
        )
        filter_coefficients_fft *= ops.max(channel_data_fft)
        impulse_response_freqs = freqs
        impulse_response_freqs = ops.roll(
            freqs,
            -freqs.size / channel_data.sampling_frequency * self.band_center_frequency,
        )
        ax.plot(
            impulse_response_freqs,
            np.fft.fftshift(filter_coefficients_fft),
            label="Filter response",
        )

        channel_data = self(channel_data)
        channel_data_fft = ops.fft.fft(channel_data.data, axis="time")
        channel_data_fft = ops.mean(ops.abs(channel_data_fft), axis=["rx", "tx"])
        freqs = ops.fft.fftshift(
            ops.fft.fftfreq(
                channel_data_fft.dim_sizes["time"],
                d=1 / channel_data.sampling_frequency,
            )
        )
        ax.plot(
            freqs,
            np.fft.fftshift(channel_data_fft),
            label="Filtered signal",
        )
        ax.legend()
