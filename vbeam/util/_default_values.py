"""Module for creating reasonable default values for some arguments to 
:func:`vbeam.core.signal_for_point`. Intended for internal use only.

Each function returns the default value and its spec."""

from typing import Tuple

from spekk import Spec

from vbeam.core import ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.util.arrays import grid


def _default_sender(spec: Spec) -> Tuple[ElementGeometry, Spec]:
    return ElementGeometry(np.array([0, 0, 0]), 0.0, 0.0), spec.at["sender"].set([])


def _default_point_position(spec: Spec) -> Tuple[np.ndarray, Spec]:
    nx, nz = 200, 200
    point_position = grid(
        np.linspace(-20e-3, 20e-3, nx),
        np.array([0.0]),
        np.linspace(0, 40e-3, nz),
        shape=[nx, nz, 3],
    )
    return point_position, spec.at["point_position"].set(["x", "z"])


def _default_receiver(spec: Spec) -> Tuple[ElementGeometry, Spec]:
    return ElementGeometry(np.array([0, 0, 0]), 0.0, 0.0), spec.at["receiver"].set([])


def _default_wave_data(spec: Spec) -> Tuple[ElementGeometry, Spec]:
    return WaveData(np.array([5e-3, 0, 20e-3]), 0.0, 0.0, 0.0), spec.at[
        "wave_data"
    ].set([])
