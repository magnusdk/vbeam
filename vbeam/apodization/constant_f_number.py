from typing import Literal

from spekk import field, ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, Probe, TransmittedWave


def _constant_f_number_weight(
    probe: Probe,
    point: ops.array,
    f_number: float | tuple[float, float],
    window: Window,
):
    if not isinstance(f_number, tuple):
        f_number = (f_number, f_number)

    projected_x, projected_y = probe.active_elements.plane.to_plane_coordinates(point)
    depth = probe.active_elements.plane.signed_distance(point)

    width = depth / f_number[0]
    height = depth / f_number[1]
    return window(projected_x / width) * window(projected_y / height)


class ConstantFNumberApodization(Apodization):
    f_number: float
    window: Window
    apply_on: Literal["tx", "rx", "both"] = field(static=True)

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: TransmittedWave,
    ) -> float:
        if self.apply_on == "tx":
            return _constant_f_number_weight(
                transmitting_probe, point, self.f_number, self.window
            )
        elif self.apply_on == "rx":
            return _constant_f_number_weight(
                receiving_probe, point, self.f_number, self.window
            )
        elif self.apply_on == "both":
            return _constant_f_number_weight(
                transmitting_probe, point, self.f_number, self.window
            ) * _constant_f_number_weight(
                receiving_probe, point, self.f_number, self.window
            )
        else:
            raise ValueError(f"Invalid value for self.apply_on: {self.apply_on!r}")
