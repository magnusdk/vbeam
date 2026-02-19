from turtle import position
from spekk import ops

from vbeam.apodization.window import Window, NoWindow, Rectangular
from vbeam.core import Apodization, Probe


class SyntheticTransmitApertureApodizationFNumber(Apodization):
    f_number_x: float = 0.5
    f_number_y: float = 0.5
    minimum_aperture_width_x: float = 3e-3
    minimum_aperture_width_y: float = 3e-3
    # window: Window = Rectangular

    def get_oneway_apodization(
        self,
        probe: Probe,
        point: ops.array,
    ) -> float:

        dx = ops.abs(point["xyz", 0] - probe.active_elements.position["xyz", 0])
        dy = ops.abs(point["xyz", 1] - probe.active_elements.position["xyz", 1])
        dz = ops.abs(point["xyz", 2] - probe.active_elements.position["xyz", 2])

        # Get valid points based on F-number
        valid_fn_x = dx * 2 * self.f_number_x <= dz
        valid_fn_y = dy * 2 * self.f_number_y <= dz

        # Enforce minimum aperture width
        valid_minimum_aperture_x = dx < self.minimum_aperture_width_x / 2

        # valid_minimum_aperture_x = ops.abs(dx) < self.minimum_aperture_width_x / 2
        valid_minimum_aperture_y = dy < self.minimum_aperture_width_y / 2
   
        return ops.logical_or(valid_fn_x * valid_fn_y, valid_minimum_aperture_x*valid_minimum_aperture_y)

    def tx(
        self,
        transmitting_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(transmitting_probe, point)

    def rx(
        self,
        receiving_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(receiving_probe, point)

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave=None,
    ) -> float:
        "Return the spatial weighting at the given `point`."

        return self.rx(receiving_probe, point) * self.tx(transmitting_probe, point)

    @staticmethod
    def from_max_opening_angle(max_angle_rad: float):
        f_number = 1 / (2 * ops.tan(max_angle_rad))

        return SyntheticTransmitApertureApodizationFNumber(f_number_x=f_number, f_number_y=f_number)

class SyntheticTransmitApertureApodizationMaxAngle(Apodization):
    max_angle_rad: float = 0.5

    def get_oneway_apodization(
        self,
        probe: Probe,
        point: ops.array,
    ) -> float:

        dx = ops.abs(point["xyz", 0] - probe.active_elements.position["xyz", 0])
        dy = ops.abs(point["xyz", 1] - probe.active_elements.position["xyz", 1])
        dz = ops.abs(point["xyz", 2] - probe.active_elements.position["xyz", 2])

        apod_x = ops.abs(ops.atan((dx / dz)) < self.max_angle_rad)
        apod_y = ops.abs(ops.atan((dy / dz)) < self.max_angle_rad)

        return apod_x * apod_y

    def tx(
        self,
        transmitting_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(transmitting_probe, point)

    def rx(
        self,
        receiving_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(receiving_probe, point)

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave= None,
    ) -> float:
        "Return the spatial weighting at the given `point`."

        return self.rx(receiving_probe, point) * self.tx(transmitting_probe, point)

