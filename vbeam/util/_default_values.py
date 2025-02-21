"""Module for creating reasonable default values for some arguments to
:func:`vbeam.core.signal_for_point`. Intended for internal use only.

Each function returns the default value and its spec."""

from typing import Optional

from spekk import ops

from vbeam.core import (
    Apodization,
    ChannelData,
    GeometricallyFocusedWave,
    Interpolator,
    Probe,
    ProbeElement,
    ReflectedWaveDelayModel,
    TransmittedWave,
    TransmittedWaveDelayModel,
)
from vbeam.geometry import Direction, Orientation, Plane, Vector
from vbeam.probe.aperture.rectangular_planar_aperture import RectangularPlanarAperture


def points(n_x: int = 200, n_z: int = 200) -> ops.array:
    points = ops.stack(
        ops.meshgrid(
            ops.linspace(-40e-3, 40e-3, n_x, dim="xs"),
            ops.array([0.0], ["ys"]),
            ops.linspace(-10e-3, 70e-3, n_z, dim="zs"),
        ),
        axis="xyz",
    )
    points = ops.squeeze(points, "ys")
    points = ops.permute_dims(points, ("zs", "xs", "xyz"))
    return points


def transmitting_probe(n_elements: int = 64) -> Probe:
    element_positions = ops.stack(
        [
            ops.linspace(-50e-3, 50e-3, n_elements, dim="elements"),
            ops.zeros(n_elements, dims=["elements"]),
            ops.zeros(n_elements, dims=["elements"]),
        ],
        axis="xyz",
    )
    origin = ops.array([0, 0, 0], ["xyz"])
    return Probe(
        origin,
        ProbeElement(element_positions, Orientation(0, 0, 0), 0.5e-3, 0.5e-3),
        RectangularPlanarAperture(Plane(origin, Orientation(0, 0, 0)), 20e-3, 10e-3),
    )


def receiving_probe(n_elements: int = 64) -> Probe:
    return transmitting_probe(n_elements)


def transmitted_wave() -> TransmittedWave:
    return GeometricallyFocusedWave(
        ops.array([0, 0, 0], ["xyz"]),
        Vector(50e-3, Direction.from_angles(ops.pi / 8, 0)),
    )


def channel_data(
    n_tx: int = 32,
    n_elements: int = 64,
    n_time_samples: int = 128,
) -> ChannelData:
    import numpy as np

    signal = np.random.normal(0, 1, (n_tx, n_elements, n_time_samples))
    return ChannelData(ops.array(signal, ["tx", "rx", "time"]), 0, 1e6)


def interpolator() -> Interpolator:
    raise NotImplementedError()  # TODO


def modulation_frequency() -> Optional[float]:
    return None


def transmitted_wave_delay_model() -> TransmittedWaveDelayModel:
    from vbeam.delay_models import SphericalDelayModel

    return SphericalDelayModel()


def reflected_wave_delay_model() -> ReflectedWaveDelayModel:
    return ReflectedWaveDelayModel()


def speed_of_sound() -> float:
    return 1540.0


def apodization() -> Apodization:
    from vbeam.apodization import NoApodization

    return NoApodization()
