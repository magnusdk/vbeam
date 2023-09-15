"""A datastructure for data related to a transmitted wave.

The most important field of :class:`WaveData` is :attr:`source`, which represents the 
focal point of the transmitted wave. A transmitted wave is usually one of three types:

* A focused wave, where the focal point is in front of the transducer.
* A diverging wave, where the focal point is behind the transducer.
* A plane wave, which is a wave that is *"focused"* at infinity. In this case, the 
  :attr:`source` field is either ``np.inf`` (infinity) or ``None``.
"""

from typing import Callable, Optional

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass

identity_fn = lambda x: x  # Just return value as-is


@traceable_dataclass(("source", "azimuth", "elevation", "t0"))
class WaveData:
    """A vectorizable container of wave data: anything that is specific to a single
    transmitted wave. See this class' fields for what that could be.

    WaveData is vectorizable, meaning that it works with vmap. This way we can represent
    the wave data for all transmitted waves in a single object.

    If a WaveData object contains multiple transmitted waves then each field will have
    an additional dimension. For example, source may have the shape (64, 3) if there are
    64 transmitted waves in the dataset (each with x, y, and z source coordinates)."""

    # The location (x, y, z) of the virtual source for the transmitted wave
    source: Optional[np.ndarray] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    # The time at which the transmitted wave passed through the "sender" element
    t0: Optional[float] = None

    def __getitem__(self, *args) -> "WaveData":
        """Index the wave data.

        Note that a WaveData instance may be a container of multiple vectorizable
        WaveData "objects". See WaveData's class docstring.

        >>> wave_data = WaveData(np.array([[0,0,0], [1,1,1]]), np.array([0,1]))
        >>> wave_data[1]
        WaveData(source=array([1, 1, 1]), azimuth=1, elevation=None, t0=None)
        """
        _maybe_getitem = (
            lambda attr: attr.__getitem__(*args) if attr is not None else None
        )
        return WaveData(
            _maybe_getitem(self.source),
            _maybe_getitem(self.azimuth),
            _maybe_getitem(self.elevation),
            _maybe_getitem(self.t0),
        )

    @property
    def shape(self) -> tuple:
        if self.source is not None:
            return self.source.shape[:-1]
        if self.azimuth is not None:
            return self.azimuth.shape
        if self.elevation is not None:
            return self.elevation.shape
        if self.t0 is not None:
            return self.t0.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def with_updates_to(
        self,
        *,
        source: Callable[[np.ndarray], np.ndarray] = identity_fn,
        azimuth: Callable[[float], float] = identity_fn,
        elevation: Callable[[float], float] = identity_fn,
        t0: Callable[[float], float] = identity_fn,
    ) -> "WaveData":
        """Return a copy with updated values for the given fields.

        If the given value for a field is a function the updated field will be that
        function applied to the current field. Example:
        >>> wave_data = WaveData(np.array([0, 0, 0]))
        >>> wave_data.with_updates_to(source=lambda x: x+1)
        WaveData(source=array([1, 1, 1]), azimuth=None, elevation=None, t0=None)

        If the given value for a field is not a function then the field will simply be
        set to that value. Example:
        >>> wave_data.with_updates_to(azimuth=1)
        WaveData(source=array([0, 0, 0]), azimuth=1, elevation=None, t0=None)
        """
        return WaveData(
            source=source(self.source) if callable(source) else source,
            azimuth=azimuth(self.azimuth) if callable(azimuth) else azimuth,
            elevation=elevation(self.elevation) if callable(elevation) else elevation,
            t0=t0(self.t0) if callable(t0) else t0,
        )
