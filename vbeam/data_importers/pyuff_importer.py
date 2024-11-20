from typing import List, Literal, Optional, Tuple, Union

import numpy
import pyuff_ustb as pyuff
from fastmath import Array, ops
from scipy.signal import hilbert
from spekk import Spec

from vbeam.apodization import (
    Hamming,
    NoApodization,
    PlaneWaveReceiveApodization,
    PlaneWaveTransmitApodization,
    RTBApodization,
    TxRxApodization,
)
from vbeam.core import ProbeGeometry, WaveData
from vbeam.data_importers.setup import SignalForPointSetup
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import Scan, linear_scan, sector_scan
from vbeam.util.geometry.v2 import distance
from vbeam.wavefront import PlaneWavefront, ReflectedWavefront, UnifiedWavefront


def parse_pyuff_scan(scan: pyuff.Scan) -> Scan:
    "Convert a PyUFF Scan to a vbeam Scan."
    if isinstance(scan, Scan):
        return scan
    elif isinstance(scan, pyuff.LinearScan):
        return linear_scan(ops.squeeze(scan.x_axis), ops.squeeze(scan.z_axis))
    elif isinstance(scan, pyuff.SectorScan):
        origin = (
            scan.origin.xyz
            if isinstance(scan.origin, pyuff.Point)
            else [p.xyz for p in scan.origin]
        )
        return sector_scan(
            ops.squeeze(scan.azimuth_axis),
            ops.squeeze(scan.depth_axis),
            apex=ops.array(origin),
        )
    else:
        raise ValueError("The scan is not an instance of pyuff.Scan")


def import_pyuff(
    channel_data: pyuff.ChannelData,
    scan: Optional[Union[pyuff.Scan, Scan]] = None,
    *,
    frames: Union[Literal["all"], int, Tuple[int], List[int], range] = "all",
) -> SignalForPointSetup:
    """Return an instance of ImportedData based on the available data in channel_data
    and optionally a pyuff scan.

    The scan may also be a vbeam.scan.Scan, in which case the resulting beamformer will
    be optimized for that scan."""
    if scan is not None:
        scan = parse_pyuff_scan(scan)

    speed_of_sound = ops.array(float(channel_data.sound_speed), dtype="float32")
    t_axis_interpolate = FastInterpLinspace(
        min=float(channel_data.initial_time),
        d=1 / float(channel_data.sampling_frequency),
        n=int(channel_data.N_samples),
    )

    assert channel_data.data.ndim in (
        3,
        4,
    ), "Shape of data assumed to be either (n_samples, n_elements, n_waves) or \
(n_samples, n_elements, n_waves, n_frames). "

    # Selecting a single frame
    if isinstance(frames, int):
        if channel_data.data.ndim == 4:
            data = channel_data.data[:, :, :, frames]
        else:
            assert frames == 0, "Only frame 0 is available."
            data = channel_data.data
        receiver_signals = ops.transpose(data, (2, 1, 0))
        has_multiple_frames = False
    # Selecting multiple frames
    elif isinstance(frames, (tuple, list, range)):
        if channel_data.data.ndim == 4:
            data = channel_data.data[:, :, :, frames]
        else:
            assert all(frame == 0 for frame in frames), "Only frame 0 is available."
            data = ops.stack([channel_data.data for _ in frames], -1)
        receiver_signals = ops.transpose(data, (3, 2, 1, 0))
        has_multiple_frames = True
    # Selecting all frames
    else:
        if channel_data.data.ndim == 4:
            receiver_signals = ops.transpose(channel_data.data, (3, 2, 1, 0))
            has_multiple_frames = True
        else:
            receiver_signals = ops.transpose(channel_data.data, (2, 1, 0))
            has_multiple_frames = False

    # Apply hilbert transform if modulation_frequency is 0
    modulation_frequency = ops.array(channel_data.modulation_frequency)
    if ops.abs(modulation_frequency) == 0:
        receiver_signals = ops.array(hilbert(receiver_signals), dtype="complex64")

    if (
        channel_data.probe.__class__
        == pyuff.objects.probes.curvilinear_array.CurvilinearArray
    ):
        ROC_azimuth = channel_data.probe.radius
    elif (
        channel_data.probe.__class__
        == pyuff.objects.probes.curvilinear_matrix_array.CurvilinearMatrixArray
    ):
        ROC_azimuth = channel_data.probe.radius_x
    else:
        ROC_azimuth = 10
    ROC_elevation = 10

    probe = ProbeGeometry(ROC=(ROC_azimuth, ROC_elevation))
    probe.rx_aperture_length_s = (
        ops.max(channel_data.probe.x) - ops.min(channel_data.probe.x),
        ops.max(channel_data.probe.y) - ops.min(channel_data.probe.y),
    )
    probe.tx_aperture_length_s = (
        ops.max(channel_data.probe.x) - ops.min(channel_data.probe.x),
        ops.max(channel_data.probe.y) - ops.min(channel_data.probe.y),
    )

    receiver = ops.array(channel_data.probe.xyz)

    sequence: List[pyuff.Wave] = channel_data.sequence
    all_wavefronts = {wave.wavefront for wave in sequence}
    assert (
        len(all_wavefronts) == 1
    ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
given {all_wavefronts})."
    (wavefront,) = all_wavefronts

    _wave_xyz = sequence[0].source.xyz
    if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
        transmitted_wavefront = PlaneWavefront()
        apodization = TxRxApodization(
            transmit=PlaneWaveTransmitApodization(),
            receive=PlaneWaveReceiveApodization(Hamming(), 1.7),
        )

    elif wavefront == pyuff.Wavefront.spherical:
        transmitted_wavefront = UnifiedWavefront()
        apodization = TxRxApodization(
            transmit=RTBApodization(),
            receive=NoApodization(),
        )

    wave_data = WaveData(
        azimuth=ops.array([wave.source.azimuth for wave in sequence]),
        elevation=ops.array([wave.source.elevation for wave in sequence]),
        source=ops.array([wave.source.xyz for wave in sequence]),
        t0=ops.array([wave.delay for wave in sequence]),
    )

    spec = Spec(
        {
            "signal": ["transmits", "receivers", "signal_time"],
            "receiver": ["receivers"],
            "point_position": ["points"],
            "wave_data": ["transmits"],
        }
    )

    # Set sender
    sender = ops.array(
        list(map(lambda x: x.origin.xyz, channel_data.sequence)), dtype="float32"
    )
    if sender.any():
        # Walking aperture
        # Redefine t0 to be when the wave passes through the sender position
        wave_data = wave_data.with_updates_to(
            t0=lambda t0: t0
            + (distance(sender, wave_data.source) - distance(wave_data.source))
            / speed_of_sound
        )
        spec = spec.at["sender"].set(["transmits"])
    else:
        sender = ops.array([0.0, 0.0, 0.0], dtype="float32")

    # Check if we are dealing with a STAI dataset: is each virtual source placed at
    # exactly at an element position?
    if receiver.shape == wave_data.source.shape and (
        numpy.allclose(probe.receiver_position, wave_data.source)
    ):
        # We are dealing with a STAI dataset! Senders are each element in the array
        sender = receiver.copy()
        # One sending element for each transmitted wave
        spec = spec.at["sender"].set(["transmits"])
        # Redefine t0 to be when the wave passes through the sender position
        wave_data = wave_data.with_updates_to(
            t0=lambda t0: t0 - distance(probe.sender_position) / speed_of_sound
        )

    if has_multiple_frames:
        spec = spec.add_dimension("frames", ["signal"])
    return SignalForPointSetup(
        probe=probe,
        sender=sender,
        receiver=receiver,
        # point_position is dynamically set from the scan in SignalForPointSetup,
        point_position=None,
        signal=receiver_signals,
        transmitted_wavefront=transmitted_wavefront,
        reflected_wavefront=ReflectedWavefront(),
        speed_of_sound=speed_of_sound,
        wave_data=wave_data,
        interpolate=t_axis_interpolate,
        modulation_frequency=modulation_frequency,
        apodization=apodization,
        # Additional setup
        spec=spec,
        scan=scan,
    )


def parse_beamformed_data(beamformed_data: pyuff.BeamformedData) -> Array:
    "Parse the beamformed data from a PyUFF file into an array with the correct shape."
    imaged_points = ops.squeeze(beamformed_data.data)
    scan = parse_pyuff_scan(beamformed_data.scan)
    return scan.unflatten(imaged_points, points_axis=0)
