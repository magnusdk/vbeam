from turtle import position
from spekk import field, ops

from vbeam.apodization.window import Window, NoWindow, Rectangular
from vbeam.core import Apodization, Probe


class SyntheticTransmitApertureApodizationFNumber(Apodization):
    f_number_x: float = 0.5
    f_number_y: float = 0.5
    minimum_aperture_width_x: float = 3e-3
    minimum_aperture_width_y: float = 3e-3
    flatten: bool = field(default=False, static=True)
    tx_rx_first: bool = field(default=True, static=True)
    tukey_alpha: float = field(default=0.0, static=True)
    
    # window: Window = Rectangular

    @staticmethod
    def _tukey_half_window(ratio, alpha):
        """Tukey window for ratio in [0, 1]. Returns 1 in flat region,
        cosine taper near the edge, and 0 outside.

        ratio: normalized distance (0 = center, 1 = F-number edge)
        alpha: Tukey parameter (0 = rectangular, 1 = Hann-like taper)
        """
        taper_start = 1.0 - alpha
        taper_weight = 0.5 * (1 + ops.cos(ops.pi * (ratio - taper_start) / alpha))
        return ops.where(ratio <= taper_start, 1.0,
            ops.where(ratio < 1.0, taper_weight, 0.0))

    def get_oneway_apodization(
        self,
        probe: Probe,
        point: ops.array,
        tx_rx_first: bool,
    ) -> float:

        if self.flatten:
            dims = [d for d in point.dims if d != "xyz"]
            point = ops.merge_dims(point, dims, "points")
            # point = ops.reshape(point, (3,-1), ["xyz", "points"])

        if tx_rx_first:
            dx = ops.abs(probe.active_elements.position["xyz", 0]-point["xyz", 0])
            dy = ops.abs(probe.active_elements.position["xyz", 1]-point["xyz", 1])
            dz = ops.abs(probe.active_elements.position["xyz", 2]-point["xyz", 2])
        else:
            dx = ops.abs(point["xyz", 0] - probe.active_elements.position["xyz", 0])
            dy = ops.abs(point["xyz", 1] - probe.active_elements.position["xyz", 1])
            dz = ops.abs(point["xyz", 2] - probe.active_elements.position["xyz", 2])

        if self.tukey_alpha > 0:
            # Tukey-windowed F-number apodization
            dz_safe = ops.where(dz == 0, 1e-30, dz)
            ratio_x = dx * 2 * self.f_number_x / dz_safe
            ratio_y = dy * 2 * self.f_number_y / dz_safe
            weight_x = self._tukey_half_window(ratio_x, self.tukey_alpha)
            weight_y = self._tukey_half_window(ratio_y, self.tukey_alpha)
            fn_weight = weight_x * weight_y
        else:
            # Original hard rectangular F-number apodization
            valid_fn_x = dx * 2 * self.f_number_x <= dz
            valid_fn_y = dy * 2 * self.f_number_y <= dz
            fn_weight = ops.astype(valid_fn_x * valid_fn_y, "float32")

        # Enforce minimum aperture width
        valid_minimum_aperture_x = dx < self.minimum_aperture_width_x / 2
        valid_minimum_aperture_y = dy < self.minimum_aperture_width_y / 2
        min_ap_weight = ops.astype(valid_minimum_aperture_x * valid_minimum_aperture_y, "float32")
   
        return ops.maximum(fn_weight, min_ap_weight)

    def tx(
        self,
        transmitting_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(transmitting_probe, point, True)

    def rx(
        self,
        receiving_probe: Probe,
        point: ops.array,
    ):
        return self.get_oneway_apodization(receiving_probe, point, self.tx_rx_first)

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
    def from_max_opening_angle(max_angle_rad: float, flatten: bool, tx_rx_first: bool):
        f_number = 1 / (2 * ops.tan(max_angle_rad))

        return SyntheticTransmitApertureApodizationFNumber(f_number_x=f_number, f_number_y=f_number, flatten=flatten, tx_rx_first=tx_rx_first)
