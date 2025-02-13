from abc import abstractmethod
from typing import Dict, Protocol

from spekk import Dim, ops

from vbeam.core.interpolation import Coordinates


class TChannelData(Protocol):
    """A Protocol describing the required behavior of a channel-data-like object in
    order to perform time-domain delay-and-sum beamforming."""

    @property
    def data(self) -> ops.array:
        "The actual channel data array. It must at least have a dimension named 'time'."

    @property
    def data_coordinates(self) -> Dict[Dim, Coordinates]:
        """A dict of coordinates of the channel data dimensions. It must at least have
        an entry for the 'time' dimension. The coordinates let interpolators know how
        to interpolate the data."""

    @abstractmethod
    def remodulate_if_iq(self, values: ops.array, delays: ops.array) -> ops.array:
        """Remodulate the data if it is IQ-data. If the channel data is RF, then this
        operation does nothing."""
