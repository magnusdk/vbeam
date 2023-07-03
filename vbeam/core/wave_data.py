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


@traceable_dataclass(("source", "azimuth", "elevation", "delay_distance"))
class WaveData:
    """A vectorizable container of wave data: anything that is specific to a single
    transmitted wave. See this class' fields for what that could be.

    WaveData is vectorizable, meaning that it works with vmap. This way we can represent
    the wave data for all transmitted waves in a single object.

    If a WaveData object contains multiple transmitted waves then each field will have
    an additional dimension. For example, source may have the shape (64, 3) if there are
    64 transmitted waves in the dataset (each with x, y, and z source coordinates)."""

    # The point (x, y, z) where the wave is focused.
    source: Optional[np.ndarray] = None
    # The azimuth angle of a plane wave
    azimuth: Optional[float] = None
    elevation: Optional[float] = None
    delay_distance: Optional[float] = None

    def __getitem__(self, *args) -> "WaveData":
        """Index the wave data.

        Note that a WaveData instance may be a container of multiple vectorizable
        WaveData "objects". See WaveData's class docstring.

        >>> wave_data = WaveData(np.array([[0,0,0], [1,1,1]]), np.array([0,1]))
        >>> wave_data[1]
        WaveData(source=array([1, 1, 1]), azimuth=1, elevation=None, delay_distance=None)
        """
        _maybe_getitem = (
            lambda attr: attr.__getitem__(*args) if attr is not None else None
        )
        return WaveData(
            _maybe_getitem(self.source),
            _maybe_getitem(self.azimuth),
            _maybe_getitem(self.elevation),
            _maybe_getitem(self.delay_distance),
        )

    @property
    def shape(self) -> tuple:
        if self.source is not None:
            return self.source.shape[:-1]
        if self.azimuth is not None:
            return self.azimuth.shape
        if self.elevation is not None:
            return self.elevation.shape
        if self.delay_distance is not None:
            return self.delay_distance.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def with_updates_to(
        self,
        *,
        source: Callable[[np.ndarray], np.ndarray] = identity_fn,
        azimuth: Callable[[float], float] = identity_fn,
        elevation: Callable[[float], float] = identity_fn,
        delay_distance: Callable[[float], float] = identity_fn,
    ) -> "WaveData":
        """Return a copy with updated values for the given fields.

        If the given value for a field is a function the updated field will be that
        function applied to the current field. Example:
        >>> wave_data = WaveData(np.array([0, 0, 0]))
        >>> wave_data.with_updates_to(source=lambda x: x+1)
        WaveData(source=array([1, 1, 1]), azimuth=None, elevation=None, delay_distance=None)

        If the given value for a field is not a function then the field will simply be
        set to that value. Example:
        >>> wave_data.with_updates_to(azimuth=1)
        WaveData(source=array([0, 0, 0]), azimuth=1, elevation=None, delay_distance=None)
        """
        return WaveData(
            source=source(self.source) if callable(source) else source,
            azimuth=azimuth(self.azimuth) if callable(azimuth) else azimuth,
            elevation=elevation(self.elevation) if callable(elevation) else elevation,
            delay_distance=delay_distance(self.delay_distance)
            if callable(delay_distance)
            else delay_distance,
        )
