from typing import Optional, Sequence, Union

import numpy
import numpy as np
import pyuff_ustb as pyuff
from scipy.signal import hilbert
from spekk import Module, ops

from vbeam import geometry, interpolation, probe
from vbeam.apodization import (
    ExpandingAperture,
    Hamming,
    PlaneWaveTransmitApodization,
    RTBApodization,
    TxRxApodization,
)
from vbeam.channel_data import LinearlySampledChannelData
from vbeam.core import (
    Apodization,
    DelayModel,
    NDInterpolator,
    Probe,
    Setup,
    TChannelData,
    transmitted_wave,
)
from vbeam.core.aberration_correction import NoAberrationCorrection
from vbeam.delay_models import PlaneDelayModel, SphericalBlendedDelayModel
from vbeam.scan import Scan, linear_scan, sector_scan


def parse_pyuff_scan(scan: pyuff.Scan) -> Scan:
    "Convert a PyUFF Scan to a vbeam Scan."
    if isinstance(scan, Scan):
        return scan
    elif isinstance(scan, pyuff.LinearScan):
        return linear_scan(
            ops.array(np.squeeze(scan.x_axis), ["xs"]),
            ops.array(np.squeeze(scan.z_axis), ["zs"]),
        )
    elif isinstance(scan, pyuff.SectorScan):
        origin = (
            ops.array(scan.origin.xyz, ["xyz"])
            if isinstance(scan.origin, pyuff.Point)
            else ops.array([p.xyz for p in scan.origin], ["tx", "xyz"])
        )
        return sector_scan(
            ops.array(np.squeeze(scan.azimuth_axis), ["azimuths"]),
            ops.array(np.squeeze(scan.depth_axis), ["depths"]),
            apex=origin,
        )
    else:
        raise ValueError("The scan is not an instance of pyuff.Scan")


class DatasetInfo(Module):
    is_plane_wave_imaging: bool
    is_base_banded: bool


class PyUFFImporter(Module):
    channel_data: pyuff.ChannelData
    scan: Optional[pyuff.Scan] = None

    @property
    def info(self) -> DatasetInfo:
        waves = self.channel_data.sequence
        if isinstance(waves, pyuff.Wave):
            waves = [waves]
        all_wavefronts = {wave.wavefront for wave in waves}
        assert (
            len(all_wavefronts) == 1
        ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
    given {all_wavefronts})."
        (wavefront,) = all_wavefronts
        _wave_xyz = waves[0].source.xyz
        if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
            is_plane_wave_imaging = True
        else:
            is_plane_wave_imaging = False

        return DatasetInfo(
            is_plane_wave_imaging=is_plane_wave_imaging,
            is_base_banded=self.channel_data.modulation_frequency != 0,
        )

    def get_channel_data(
        self, frame: int | slice | Sequence[int] = slice(None)
    ) -> TChannelData:
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

        # Convert to IQ signal
        if not self.info.is_base_banded:
            signal = hilbert(signal.data, axis=dims.index("time"))

        signal = ops.array(signal, dims)

        waves = self.channel_data.sequence
        if isinstance(waves, pyuff.Wave):
            waves = [waves]
        t0 = [float(wave.delay) for wave in waves]
        # Check if all t0 are the same. If so, return just a single number.
        if len(set(t0)) == 1:
            t0 = ops.array(t0[0])
        else:  # ...else, return an array with tx dimension.
            t0 = ops.array(t0, ["tx"])

        return LinearlySampledChannelData(
            signal,
            t0,
            float(self.channel_data.sampling_frequency),
            float(self.channel_data.modulation_frequency),
        )

    def get_delay_model(self) -> DelayModel:
        speed_of_sound = float(self.channel_data.sound_speed)
        waves = self.channel_data.sequence
        if isinstance(waves, pyuff.Wave):
            waves = [waves]

        all_wavefronts = {wave.wavefront for wave in waves}
        assert (
            len(all_wavefronts) == 1
        ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
    given {all_wavefronts})."
        (wavefront,) = all_wavefronts

        _wave_xyz = waves[0].source.xyz
        if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
            return PlaneDelayModel(speed_of_sound)
        elif wavefront == pyuff.Wavefront.spherical:
            plane_wave_region_size = float(self.channel_data.wavelength * 4)
            return SphericalBlendedDelayModel(speed_of_sound, plane_wave_region_size)
        else:
            raise ValueError(f"Unrecognized wavefront type: {wavefront}.")

    def get_transmitting_probe(self) -> probe.Probe:
        # assert isinstance(self.channel_data.probe, pyuff.MatrixArray)

        active_elements = probe.ProbeElement(
            geometry.Plane.from_origin_and_angles(
                ops.array(self.channel_data.probe.xyz, ["rx", "xyz"])
            ),
            ops.array(self.channel_data.probe.width, ["rx"]),
            ops.array(self.channel_data.probe.height, ["rx"]),
        )
        probe_width = float(
            np.max(self.channel_data.probe.x)
            - np.min(self.channel_data.probe.x)
            + np.mean(self.channel_data.probe.width)
        )
        probe_height = float(
            np.max(self.channel_data.probe.y)
            - np.min(self.channel_data.probe.y)
            + np.mean(self.channel_data.probe.height)
        )
        return probe.FlatRectangularProbe(
            active_elements,
            geometry.Plane.from_origin_and_angles(),
            probe_width,
            probe_height,
        )

    def get_receiving_probe(self) -> Probe:
        return self.get_transmitting_probe()

    def get_points(self) -> Union[Scan, None]:
        if self.scan is not None:
            return parse_pyuff_scan(self.scan)
        return None

    def get_transmitted_wave(self) -> transmitted_wave.GeometricallyFocusedWave:
        waves = self.channel_data.sequence
        if isinstance(waves, pyuff.Wave):
            waves = [waves]

        if self.info.is_plane_wave_imaging:
            virtual_source = geometry.VectorWithInfiniteMagnitude.from_angles(
                azimuth=ops.array([wave.source.azimuth for wave in waves], ["tx"]),
                elevation=ops.array([wave.source.elevation for wave in waves], ["tx"]),
            )
        else:
            virtual_source = geometry.Vector.from_array(
                ops.array([wave.source.xyz for wave in waves], ["tx", "xyz"])
            )

        return transmitted_wave.GeometricallyFocusedWave(
            origin=ops.array([0, 0, 0], ["xyz"]),
            virtual_source=virtual_source,
        )

    def get_interpolator_type(self) -> type[NDInterpolator]:
        return interpolation.LinearNDInterpolator

    def get_apodization(self) -> Apodization:
        if self.info.is_plane_wave_imaging:
            return TxRxApodization(
                tx=PlaneWaveTransmitApodization(Hamming()),
                rx=ExpandingAperture(Hamming(), 1.7),
            )
        else:
            return TxRxApodization(
                tx=RTBApodization(
                    Hamming(),
                    float(self.channel_data.wavelength),
                    3 * 1.22,
                ),
                rx=ExpandingAperture(Hamming(), 1.7),
            )

    def get_setup(self, frame: Union[int, slice, Sequence[int]]) -> Setup:
        return Setup(
            points=self.get_points() if self.scan is not None else None,
            transmitting_probe=self.get_transmitting_probe(),
            receiving_probe=self.get_receiving_probe(),
            transmitted_wave=self.get_transmitted_wave(),
            channel_data=self.get_channel_data(frame),
            interpolator_type=self.get_interpolator_type(),
            delay_model=self.get_delay_model(),
            apodization=self.get_apodization(),
        )


def parse_beamformed_data(beamformed_data: pyuff.BeamformedData) -> ops.array:
    "Parse the beamformed data from a PyUFF file into an array with the correct shape."
    imaged_points = ops.squeeze(
        beamformed_data.data,
        axis=[i for i, size in beamformed_data.data.shape if size == 1],
    )
    scan = parse_pyuff_scan(beamformed_data.scan)
    return scan.unflatten(imaged_points, points_axis=0)
