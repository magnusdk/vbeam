from typing import Callable, Optional

from spekk import Spec

from vbeam.core import (
    ElementGeometry,
    ReflectedWavefront,
    TransmittedWavefront,
    WaveData,
)
from vbeam.fastmath import numpy as np
from vbeam.util._default_values import (
    _default_point_position,
    _default_receiver,
    _default_sender,
    _default_wave_data,
)
from vbeam.wavefront.util import (
    get_reflected_wavefront_values,
    get_transmitted_wavefront_values,
)


def plot_transmitted_wavefront(
    wavefront: TransmittedWavefront,
    sender: Optional[ElementGeometry] = None,
    point_position: Optional[np.ndarray] = None,
    wave_data: Optional[WaveData] = None,
    spec: Optional[Spec] = None,
    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ax=None,  # : Optional[matplotlib.pyplot.Axes]
):
    """Plot the (transmit) wavefront distance values using ``matplotlib``.

    Try to set reasonable default arguments if they are not given. Use ``postprocess``
    to process the values further before plotting (for example scan conversion)."""
    # Try to set helpful default values
    spec = spec if spec is not None else Spec({})
    if sender is None:
        sender, spec = _default_sender(spec)
    if point_position is None:
        point_position, spec = _default_point_position(spec)
    if wave_data is None:
        wave_data, spec = _default_wave_data(spec)

    # Calculate wavefront values
    vals = get_transmitted_wavefront_values(
        wavefront,
        sender,
        point_position,
        wave_data,
        spec,
        # We want to keep the dimensions of point_position
        spec["point_position"].tree,
    )
    if postprocess is not None:
        vals = postprocess(vals)

    # Plot wavefront values
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt
    return ax.imshow(
        vals.T,
        aspect="auto",
        extent=[
            point_position[..., 0].min(),
            point_position[..., 0].max(),
            point_position[..., 2].max(),
            point_position[..., 2].min(),
        ],
    )


def plot_reflected_wavefront(
    wavefront: ReflectedWavefront,
    point_position: Optional[np.ndarray] = None,
    receiver: Optional[ElementGeometry] = None,
    spec: Optional[Spec] = None,
    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ax=None,  # : Optional[matplotlib.pyplot.Axes]
):
    """Plot the (receive) wavefront distance values using ``matplotlib``.

    Try to set reasonable default arguments if they are not given. Use ``postprocess``
    to process the values further before plotting (for example scan conversion)."""
    # Try to set helpful default values
    spec = spec if spec is not None else Spec({})
    if point_position is None:
        point_position, spec = _default_point_position(spec)
    if receiver is None:
        receiver, spec = _default_receiver(spec)

    # Calculate wavefront values
    vals = get_reflected_wavefront_values(
        wavefront,
        point_position,
        receiver,
        spec,
        # We want to keep the dimensions of point_position
        spec["point_position"].tree,
    )
    if postprocess is not None:
        vals = postprocess(vals)

    # Plot wavefront values
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt
    return ax.imshow(
        vals.T,
        aspect="auto",
        extent=[
            point_position[..., 0].min(),
            point_position[..., 0].max(),
            point_position[..., 2].max(),
            point_position[..., 2].min(),
        ],
    )
