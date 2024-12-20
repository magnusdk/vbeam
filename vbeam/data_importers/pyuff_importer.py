from enum import Enum, auto
from typing import Optional, Tuple, Union

import numpy
import pyuff_ustb as pyuff
from scipy.signal import hilbert
from spekk import Module, ops

from vbeam.apodization import (
    Hamming,
    NoApodization,
    PlaneWaveReceiveApodization,
    PlaneWaveTransmitApodization,
    RTBApodization,
    TxRxApodization,
)
from vbeam.core import (
    Apodization,
    InterpolationSpace1D,
    ProbeGeometry,
    Setup,
    SpeedOfSound,
    TransmittedWavefront,
    WaveData,
)
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import Scan, linear_scan, sector_scan
from vbeam.wavefront import PlaneWavefront, ReflectedWavefront, UnifiedWavefront


def parse_pyuff_scan(scan: pyuff.Scan) -> Scan:
    "Convert a PyUFF Scan to a vbeam Scan."
    if isinstance(scan, Scan):
        return scan
    elif isinstance(scan, pyuff.LinearScan):
        return linear_scan(
            ops.array(scan.x_axis, ["x_axis"]),
            ops.array(scan.z_axis, ["z_axis"]),
        )
    elif isinstance(scan, pyuff.SectorScan):
        origin = (
            ops.array(scan.origin.xyz, ["xyz"])
            if isinstance(scan.origin, pyuff.Point)
            else ops.array([p.xyz for p in scan.origin], ["tx", "xyz"])
        )
        return sector_scan(
            ops.array(scan.azimuth_axis, ["azimuth_axis"]),
            ops.array(scan.depth_axis, ["depth_axis"]),
            apex=origin,
        )
    else:
        raise ValueError("The scan is not an instance of pyuff.Scan")


class DatasetInfo(Module):
    is_base_banded: bool


class PyUFFImporter(Module):
    channel_data: pyuff.ChannelData
    scan: Optional[pyuff.Scan] = None

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            is_base_banded=self.get_modulation_frequency() != 0.0,
        )

    def get_signal(self, frame: Union[int, slice, Tuple[int], None]) -> ops.array:
        if frame is None:
            frame = slice(None)
        if self.channel_data.data.ndim == 3:
            if frame not in {0, slice(None)}:
                raise ValueError(
                    f"There is only one frame in the dataset, but attempted to read {frame=}."
                )
            signal = self.channel_data.data
            dims = ["time", "rx", "tx"]
        else:
            signal = self.channel_data.data[..., frame]
            dims = ["time", "rx", "tx", "frames"]
            if isinstance(frame, int):
                dims.remove("frames")
        if not self.info.is_base_banded:
            signal = hilbert(signal, axis=dims.index("time"))
        return ops.array(signal, dims)

    def get_transmitted_wavefront(self) -> TransmittedWavefront:
        waves = self.channel_data.sequence
        all_wavefronts = {wave.wavefront for wave in waves}
        assert (
            len(all_wavefronts) == 1
        ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
    given {all_wavefronts})."
        (wavefront,) = all_wavefronts

        _wave_xyz = waves[0].source.xyz
        if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
            return PlaneWavefront()
        elif wavefront == pyuff.Wavefront.spherical:
            return UnifiedWavefront()
        else:
            raise ValueError(f"Unrecognized wavefront type: {wavefront}.")

    def get_reflected_wavefront(self) -> ReflectedWavefront:
        return ReflectedWavefront()

    def get_probe(self) -> ProbeGeometry:
        if (
            self.channel_data.probe.__class__
            == pyuff.objects.probes.curvilinear_array.CurvilinearArray
        ):
            ROC_azimuth = self.channel_data.probe.radius
        elif (
            self.channel_data.probe.__class__
            == pyuff.objects.probes.curvilinear_matrix_array.CurvilinearMatrixArray
        ):
            ROC_azimuth = self.channel_data.probe.radius_x
        else:
            ROC_azimuth = 10
        ROC_elevation = 10

        probe = ProbeGeometry(ROC=(ROC_azimuth, ROC_elevation))
        probe.rx_aperture_length_s = (
            ops.max(self.channel_data.probe.x) - ops.min(self.channel_data.probe.x),
            ops.max(self.channel_data.probe.y) - ops.min(self.channel_data.probe.y),
        )
        probe.tx_aperture_length_s = (
            ops.max(self.channel_data.probe.x) - ops.min(self.channel_data.probe.x),
            ops.max(self.channel_data.probe.y) - ops.min(self.channel_data.probe.y),
        )
        return probe

    def get_sender(self) -> ops.array:
        return ops.array([0, 0, 0], ["xyz"], dtype="float32")

    def get_receiver(self) -> ops.array:
        return ops.array(self.channel_data.probe.xyz, ["rx", "xyz"])

    def get_point_position(self) -> Union[Scan, None]:
        if self.scan is not None:
            return parse_pyuff_scan(self.scan)
        return None

    def get_speed_of_sound(self) -> float:
        return float(self.channel_data.sound_speed)

    def get_wave_data(self) -> WaveData:
        waves = self.channel_data.sequence
        return WaveData(
            azimuth=ops.array([wave.source.azimuth for wave in waves], ["tx"]),
            elevation=ops.array([wave.source.elevation for wave in waves], ["tx"]),
            source=ops.array([wave.source.xyz for wave in waves], ["tx", "xyz"]),
            t0=ops.array([wave.delay for wave in waves], ["tx"]),
        )

    def get_interpolate(self) -> InterpolationSpace1D:
        return FastInterpLinspace(
            min=float(self.channel_data.initial_time),
            d=1 / float(self.channel_data.sampling_frequency),
            n=int(self.channel_data.N_samples),
        )

    def get_modulation_frequency(self) -> float:
        return float(self.channel_data.modulation_frequency)

    def get_apodization(self) -> Apodization:
        waves = self.channel_data.sequence
        all_wavefronts = {wave.wavefront for wave in waves}
        assert (
            len(all_wavefronts) == 1
        ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
    given {all_wavefronts})."
        (wavefront,) = all_wavefronts

        _wave_xyz = waves[0].source.xyz
        if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
            return TxRxApodization(
                transmit=PlaneWaveTransmitApodization(),
                receive=PlaneWaveReceiveApodization(Hamming(), 1.7),
            )
        elif wavefront == pyuff.Wavefront.spherical:
            return TxRxApodization(
                transmit=RTBApodization(),
                receive=NoApodization(),
            )
        else:
            raise ValueError(f"Unrecognized wavefront type: {wavefront}.")

    def get_setup(self, frame: Union[int, slice, Tuple[int], None]) -> Setup:
        return Setup(
            self.get_probe(),
            self.get_sender(),
            self.get_receiver(),
            self.get_point_position(),
            self.get_signal(frame),
            self.get_transmitted_wavefront(),
            self.get_reflected_wavefront(),
            self.get_speed_of_sound(),
            self.get_wave_data(),
            self.get_interpolate(),
            self.get_modulation_frequency(),
            self.get_apodization(),
        )


class ChannelDataInfo(Module):
    data: ops.array
    sampling_frequency: Union[float, ops.array]
    t0: Union[float, ops.array]

class RecordingType(Enum):
    focused = auto()
    plane = auto()
    diverging = auto()
    stai = auto()


class BeamInfo(Module):
    origin: ops.array
    azimuth: ops.array
    elevation: ops.array
    depth: ops.array
    t0: Union[float, ops.array]


class TransmitterInfo(Module): ...


class ReceiverInfo(Module): ...


class NeededData(Module):
    def get_channel_data_info(self) -> ops.array: ...
    def get_recording_type(self) -> RecordingType: ...
    def get_beam_info(self) -> BeamInfo: ...
    def get_transmitter_info(self) -> TransmitterInfo: ...
    def get_receiver_info(self) -> ReceiverInfo: ...
    def get_sound_speed(self) -> Union[float, ops.array, SpeedOfSound]: ...
    def get_modulation_frequency(self) -> ops.array: ...


def parse_beamformed_data(beamformed_data: pyuff.BeamformedData) -> ops.array:
    "Parse the beamformed data from a PyUFF file into an array with the correct shape."
    imaged_points = ops.squeeze(beamformed_data.data)
    scan = parse_pyuff_scan(beamformed_data.scan)
    return scan.unflatten(imaged_points, points_axis=0)
