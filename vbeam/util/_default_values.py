# TODO: Redo with spekk 2.0.0

"""Module for creating reasonable default values for some arguments to 
:func:`vbeam.core.signal_for_point`. Intended for internal use only.

Each function returns the default value and its spec."""

from typing import Tuple

from spekk import ops

from vbeam.core import ProbeGeometry, WaveData
from vbeam.util.arrays import grid


def _default_probe(spec: Spec) -> Tuple[ProbeGeometry, Spec]:
    return ProbeGeometry(ops.zeros(3), ops.zeros(3), 10), spec.at["probe"].set([])


def _default_receiver(spec: Spec) -> Tuple[Array, Spec]:
    return ops.zeros(3), spec.at["receiver"].set([])


def _default_sender(spec: Spec) -> Tuple[Array, Spec]:
    return ops.zeros(3), spec.at["sender"].set([])


def _default_point_position(spec: Spec) -> Tuple[Array, Spec]:
    nx, nz = 200, 200
    point_position = grid(
        ops.linspace(-20e-3, 20e-3, nx),
        ops.array([0.0]),
        ops.linspace(0, 40e-3, nz),
        shape=[nx, nz, 3],
    )
    return point_position, spec.at["point_position"].set(["x", "z"])


def _default_wave_data(spec: Spec) -> Tuple[WaveData, Spec]:
    return WaveData(ops.array([5e-3, 0, 20e-3]), 0.0, 0.0, 0.0), spec.at[
        "wave_data"
    ].set([])
