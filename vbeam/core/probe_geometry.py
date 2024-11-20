"""A datastructure for representing a transducer probe excluding 
the position, orientation, etc.
"""

from typing import Callable, Optional, Tuple

from fastmath import ops

from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.util.geometry.v2 import distance

identity_fn = lambda x: x  # Just return value as-is


@traceable_dataclass(("ROC", "rx_aperture_length_s", "tx_aperture_length_s"))
class ProbeGeometry:
    """A container for probe parameters.
    The aperture lengths are arc lenghts of the probe surface in azimuth and elevation.
    """

    ROC: tuple[float, float]  # (azimuth, elevation)
    rx_aperture_length_s: tuple[float, float] = None  # (width, height)
    tx_aperture_length_s: tuple[float, float] = None  # (width, height)

    def __postinit__(self):
        if type(self.ROC) is not tuple or len(self.ROC) != 2:
            self.ROC = (self.ROC, self.ROC)

    def __getitem__(self, *args) -> "ProbeGeometry":
        _maybe_getitem = lambda attr: (
            attr.__getitem__(*args) if attr is not None else None
        )
        return self.ProbeGeometry(
            _maybe_getitem(self.ROC),
            _maybe_getitem(self.rx_aperture_length_s),
            _maybe_getitem(self.tx_aperture_length_s),
        )

    @property
    def curvature_center(self):
        return (
            ops.array([0.0, 0.0, -self.ROC[0]]),
            ops.array([0.0, 0.0, -self.ROC[1]]),
        )

    def get_sender_normal(self):
        vector = self.sender_position - self.curvature_center
        return vector / distance(vector)

    def get_receiver_normal(self):
        vector = self.receiver_position - self.curvature_center
        return vector / distance(vector)

    def get_theta(self, position):
        return ops.arctan2(position[0], self.ROC[0] + position[2])

    def get_phi(self, position):
        return ops.arctan2(position[1], self.ROC[1] + position[2])

    def aperture_distance(self, position1, position2):
        pos1_s = self.cart2surface(position=position1)
        if position2 is None:
            pos2_s = (0.0, 0.0)
        else:
            pos2_s = self.cart2surface(position=position2)
        return ops.sqrt((pos1_s[0] - pos2_s[0]) ** 2 + (pos1_s[1] - pos2_s[1]) ** 2)

    def cart2surface(self, position):
        return (
            self.ROC[0] * self.get_theta(position=position),
            self.ROC[1] * self.get_phi(position=position),
        )

    def surface2cart(self, position_s):
        return ops.array(
            (
                self.ROC[0] * ops.sin(position_s[0] / self.ROC[0]),
                self.ROC[1] * ops.sin(position_s[1] / self.ROC[1]),
                ops.where(
                    self.ROC[0] - self.ROC[0] * ops.cos(position_s[0] / self.ROC[0])
                    > self.ROC[1] - self.ROC[1] * ops.cos(position_s[1] / self.ROC[1]),
                    self.ROC[0] * ops.cos(position_s[0] / self.ROC[0]) - self.ROC[0],
                    self.ROC[1] * ops.cos(position_s[1] / self.ROC[1]) - self.ROC[1],
                ),
            )
        )

    def get_rx_aperture_borders(self):
        left = self.surface2cart((-self.rx_aperture_length_s[0] / 2, 0.0))
        right = self.surface2cart((self.rx_aperture_length_s[0] / 2, 0.0))
        down = self.surface2cart((0.0, -self.rx_aperture_length_s[1] / 2))
        up = self.surface2cart((0.0, self.rx_aperture_length_s[1] / 2))
        return (left, right, up, down)

    def get_tx_aperture_borders(self, sender):
        left = self.surface2cart(
            (
                self.cart2surface(position=sender)[0]
                - self.tx_aperture_length_s[0] / 2,
                0.0,
            )
        )
        right = self.surface2cart(
            (
                self.cart2surface(position=sender)[0]
                + self.tx_aperture_length_s[0] / 2,
                0.0,
            )
        )
        down = self.surface2cart(
            (
                0.0,
                self.cart2surface(position=sender)[1]
                - self.tx_aperture_length_s[1] / 2,
            )
        )
        up = self.surface2cart(
            (
                0.0,
                self.cart2surface(position=sender)[1]
                + self.tx_aperture_length_s[1] / 2,
            )
        )
        return (left, right, up, down)

    def set_rx_aperture_length(self, min_x, max_x, min_y, max_y):
        width_s = (
            ops.arcsin(max_x / self.ROC[0]) * self.ROC[0]
            - ops.arcsin(min_x / self.ROC[0]) * self.ROC[0]
        )
        height_s = (
            ops.arcsin(max_y / self.ROC[1]) * self.ROC[1]
            - ops.arcsin(min_y / self.ROC[1]) * self.ROC[1]
        )
        self.rx_aperture_length_s = (width_s, height_s)

    def set_tx_aperture_length(self, min_x, max_x, min_y, max_y):
        width_s = (
            ops.arcsin(max_x / self.ROC[0]) * self.ROC[0]
            - ops.arcsin(min_x / self.ROC[0]) * self.ROC[0]
        )
        height_s = (
            ops.arcsin(max_y / self.ROC[1]) * self.ROC[1]
            - ops.arcsin(min_y / self.ROC[1]) * self.ROC[1]
        )
        self.tx_aperture_length_s = (width_s, height_s)

    def with_updates_to(
        self,
        *,
        ROC: Callable[[float], float] = identity_fn,
        rx_aperture_length_s: Callable[[Tuple[float, float]], float] = identity_fn,
        tx_aperture_length_s: Callable[[Tuple[float, float]], float] = identity_fn,
    ) -> "ProbeGeometry":
        return ProbeGeometry(
            ROC=ROC(self.ROC) if callable(ROC) else ROC,
            rx_aperture_length_s=(
                rx_aperture_length_s(self.rx_aperture_length_s)
                if callable(rx_aperture_length_s)
                else rx_aperture_length_s
            ),
            tx_aperture_length_s=(
                tx_aperture_length_s(self.tx_aperture_length_s)
                if callable(tx_aperture_length_s)
                else tx_aperture_length_s
            ),
        )

    def copy(self) -> "ProbeGeometry":
        return self.with_updates_to()
