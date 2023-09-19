from typing import List, Literal, Optional, Tuple, Union

import numpy
import pyuff_ustb as pyuff
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
from vbeam.core import ElementGeometry, WaveData
from vbeam.data_importers.setup import SignalForPointSetup
from vbeam.fastmath import numpy as np
from vbeam.interpolation import FastInterpLinspace
from vbeam.scan import Scan, linear_scan, sector_scan
from vbeam.util.geometry.v2 import distance
from vbeam.wavefront import PlaneWavefront, ReflectedWavefront, UnifiedWavefront


def parse_pyuff_scan(scan: pyuff.Scan) -> Scan:
    "Convert a PyUFF Scan to a vbeam Scan."
    if not isinstance(scan, pyuff.Scan):
        raise ValueError("Scan is not an instance of pyuff.Scan")

    if isinstance(scan, pyuff.LinearScan):
        return linear_scan(np.squeeze(scan.x_axis), np.squeeze(scan.z_axis))
    if isinstance(scan, pyuff.SectorScan):
        origin = (
            scan.origin.xyz
            if isinstance(scan.origin, pyuff.Point)
            else [p.xyz for p in scan.origin]
        )
        return sector_scan(
            np.squeeze(scan.azimuth_axis),
            np.squeeze(scan.depth_axis),
            apex=np.array(origin),
        )


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
    if scan and isinstance(scan, pyuff.Scan):
        scan = parse_pyuff_scan(scan)

    speed_of_sound = np.array(float(channel_data.sound_speed), dtype="float32")
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
        receiver_signals = np.transpose(data, (2, 1, 0))
        has_multiple_frames = False
    # Selecting multiple frames
    elif isinstance(frames, (tuple, list, range)):
        if channel_data.data.ndim == 4:
            data = channel_data.data[:, :, :, frames]
        else:
            assert all(frame == 0 for frame in frames), "Only frame 0 is available."
            data = np.stack([channel_data.data for _ in frames], -1)
        receiver_signals = np.transpose(data, (3, 2, 1, 0))
        has_multiple_frames = True
    # Selecting all frames
    else:
        if channel_data.data.ndim == 4:
            receiver_signals = np.transpose(channel_data.data, (3, 2, 1, 0))
            has_multiple_frames = True
        else:
            receiver_signals = np.transpose(channel_data.data, (2, 1, 0))
            has_multiple_frames = False
    # Apply hilbert transform if modulation_frequency is 0
    modulation_frequency = np.array(channel_data.modulation_frequency)
    if numpy.abs(modulation_frequency) == 0:
        receiver_signals = np.array(hilbert(receiver_signals), dtype="complex64")

    receivers = ElementGeometry(
        np.array(channel_data.probe.xyz),
        np.array(channel_data.probe.theta),
        np.array(channel_data.probe.phi),
    )
    sender = ElementGeometry(np.array([0.0, 0.0, 0.0], dtype="float32"), 0.0, 0.0)

    sequence: List[pyuff.Wave] = channel_data.sequence
    all_wavefronts = {wave.wavefront for wave in sequence}
    assert (
        len(all_wavefronts) == 1
    ), f"There must be exactly one type of wavefront in channel_data.sequence (was \
given {all_wavefronts})."
    (wavefront,) = all_wavefronts

    array_bounds = (
        np.array([np.min(channel_data.probe.x), 0.0, 0.0]),
        np.array([np.max(channel_data.probe.x), 0.0, 0.0]),
    )

    _wave_xyz = sequence[0].source.xyz
    if wavefront == pyuff.Wavefront.plane or numpy.isinf(_wave_xyz).any():
        transmitted_wavefront = PlaneWavefront()
        apodization = TxRxApodization(
            transmit=PlaneWaveTransmitApodization(array_bounds),
            receive=PlaneWaveReceiveApodization(Hamming(), 1.7),
        )

    elif wavefront == pyuff.Wavefront.spherical:
        transmitted_wavefront = UnifiedWavefront(array_bounds)
        apodization = TxRxApodization(
            transmit=RTBApodization(array_bounds),
            receive=NoApodization(),
        )

    wave_data = WaveData(
        azimuth=np.array([wave.source.azimuth for wave in sequence]),
        elevation=np.array([wave.source.elevation for wave in sequence]),
        source=np.array([wave.source.xyz for wave in sequence]),
        t0=np.array([wave.delay for wave in sequence]),
    )
    spec = Spec(
        {
            "signal": ["transmits", "receivers", "signal_time"],
            "receiver": ["receivers"],
            "point_position": ["points"],
            "wave_data": ["transmits"],
        }
    )

    # Check if we are dealing with a STAI dataset: is each virtual source placed at
    # exactly at an element position?
    if receivers.position.shape == wave_data.source.shape and (
        numpy.allclose(receivers.position, wave_data.source)
    ):
        # We are dealing with a STAI dataset! Senders are each element in the array
        sender = receivers.copy()
        # One sending element for each transmitted wave
        spec = spec.at["sender"].set(["transmits"])
        # Redefine t0 to be when the wave passes through the sender position
        wave_data = wave_data.with_updates_to(
            t0=lambda t0: t0 - distance(sender.position) / speed_of_sound
        )

    if has_multiple_frames:
        spec = spec.add_dimension("frames", ["signal"])
    return SignalForPointSetup(
        sender=sender,
        # point_position is dynamically set from the scan in SignalForPointSetup,
        point_position=None,
        receiver=receivers,
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


def parse_beamformed_data(beamformed_data: pyuff.BeamformedData) -> np.ndarray:
    "Parse the beamformed data from a PyUFF file into an array with the correct shape."
    imaged_points = np.squeeze(beamformed_data.data)
    scan = parse_pyuff_scan(beamformed_data.scan)
    return scan.unflatten(imaged_points, points_axis=0)
