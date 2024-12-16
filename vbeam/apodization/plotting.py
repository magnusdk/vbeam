from typing import Callable, Optional

from spekk import Spec

from vbeam.apodization.util import get_apodization_values
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import numpy as np
from vbeam.util._default_values import (
    _default_point_position,
    _default_receiver,
    _default_sender,
    _default_wave_data,
)


def plot_apodization(
    apodization: Apodization,
    sender: Optional[ElementGeometry] = None,
    point_position: Optional[np.ndarray] = None,
    receiver: Optional[ElementGeometry] = None,
    wave_data: Optional[WaveData] = None,
    spec: Optional[Spec] = None,
    postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    average_overlap: bool = True,
    jit: bool = True,
    ax=None,  # : Optional[matplotlib.pyplot.Axes]
):
    """Plot the apodization values using ``matplotlib``.

    Try to set reasonable default arguments if they are not given. Use ``postprocess``
    to process the apodization values further before plotting (for example scan
    conversion)."""
    # Try to set helpful default values
    spec = spec if spec is not None else Spec({})
    if sender is None:
        sender, spec = _default_sender(spec)
    if point_position is None:
        point_position, spec = _default_point_position(spec)
    if receiver is None:
        receiver, spec = _default_receiver(spec)
    if wave_data is None:
        wave_data, spec = _default_wave_data(spec)

    # Calculate apodization values
    vals = get_apodization_values(
        apodization=apodization,
        sender=sender,
        point_position=point_position,
        receiver=receiver,
        wave_data=wave_data,
        spec=spec,
        # We want to keep the dimensions of point_position
        dimensions=spec["point_position"].tree,
        average=average_overlap,
        jit=jit,
    )
    if postprocess is not None:
        vals = postprocess(vals)

    # Plot apodization values
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
