from turtle import position
from spekk import ops

from vbeam.apodization.window import Window, NoWindow, Rectangular
from vbeam.core import Apodization, Probe


class SyntheticTransmitApertureApodizationFNumber(Apodization):
    f_number: float = 0.5
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
        valid_fn_x = dx * 2 * self.f_number <= dz
        valid_fn_y = dy * 2 * self.f_number <= dz

        # Enforce minimum aperture width
        valid_minimum_aperture_x = dx < self.minimum_aperture_width_x / 2

        # valid_minimum_aperture_x = ops.abs(dx) < self.minimum_aperture_width_x / 2
        valid_minimum_aperture_y = dy < self.minimum_aperture_width_y / 2
   
        return ops.logical_or(valid_fn_x * valid_fn_y, valid_minimum_aperture_x*valid_minimum_aperture_y)

    def get_tx_apodization(
        self,
        transmitting_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(transmitting_probe, point)

    def get_rx_apodization(
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
        transmitted_wave: None,
    ) -> float:
        "Return the spatial weighting at the given `point`."

        return self.get_rx_apodization(
            receiving_probe, point
        ) * self.get_tx_apodization(transmitting_probe, point)

