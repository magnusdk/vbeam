from hypothesis import strategies as st

from vbeam.fastmath import numpy as np
from vbeam.scan import sector_scan


def linspaces(min: float, max: float, num: int = 50, ensure_increasing: bool = False):
    "Generate linspaces where start <= stop."
    return (
        st.tuples(st.floats(min, max), st.floats(min, max))
        .filter(lambda args: not ensure_increasing or args[0] <= args[1])
        .map(lambda args: np.linspace(*args, num))
    )


def sector_scans(shape):
    "Generate instances of :class:`~vbeam.scan.SectorScan` with the given shape."
    assert len(shape) in (2, 3), "scan must be 2D or 3D"
    return st.builds(
        sector_scan,
        *[linspaces(-np.pi, np.pi, size) for size in shape[:-1]],
        linspaces(0, 2, shape[-1]),
        apex=st.builds(
            lambda x, z: np.array([x, 0.0, z]),
            st.floats(-4, 4),
            st.floats(-4, 4),
        ),
    )
