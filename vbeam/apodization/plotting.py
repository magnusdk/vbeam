# TODO: This should be must simpler now that we have named dimensions


from typing import Callable, Optional

from spekk import Array
from spekk import Spec

from vbeam.apodization.util import get_apodization_values
from vbeam.core import Apodization, ProbeGeometry, WaveData
from vbeam.util._default_values import (
    _default_point_position,
    _default_probe,
    _default_wave_data,
)


def plot_apodization(
    apodization: Apodization,
    probe: Optional[ProbeGeometry] = None,
    point_position: Optional[Array] = None,
    wave_data: Optional[WaveData] = None,
    spec: Optional[Spec] = None,
    postprocess: Optional[Callable[[Array], Array]] = None,
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
    if probe is None:
        probe, spec = _default_probe(spec)
    if point_position is None:
        point_position, spec = _default_point_position(spec)
    if wave_data is None:
        wave_data, spec = _default_wave_data(spec)

    # Calculate apodization values
    vals = get_apodization_values(
        apodization,
        probe,
        point_position,
        wave_data,
        spec,
        # We want to keep the dimensions of point_position
        spec["point_position"].tree,
        average_overlap,
        jit,
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
