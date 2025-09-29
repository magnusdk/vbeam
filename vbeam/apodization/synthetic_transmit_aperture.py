from spekk import ops

from vbeam.apodization.window import Window
from vbeam.core import Apodization, Probe

class SyntheticTransmitApertureApodizationFNumber(Apodization):
    f_number: float=0.5

    def __call__(
        self,
        transmitting_probe: Probe,
        receiving_probe: Probe,
        point: ops.array,
        transmitted_wave: None,
    ) -> float:
        "Return the spatial weighting at the given `point`."

        # 3D boxcar apodization with f-number criteria
        x_hd = point["xyz", 0]
        y_hd = point["xyz", 1]
        z_hd = point["xyz", 2]
        
        tx_probe_x = transmitting_probe.active_elements.position["xyz", 0]
        tx_probe_y = transmitting_probe.active_elements.position["xyz", 1]
        rx_probe_x = receiving_probe.active_elements.position["xyz", 0]
        rx_probe_y = receiving_probe.active_elements.position["xyz", 1]
        
        # X-axis apodization (lateral)
        apod_rx_x = (ops.abs(ops.atan((x_hd-rx_probe_x) / z_hd)) < self.f_number)
        apod_tx_x = (ops.abs(ops.atan((x_hd-tx_probe_x) / z_hd)) < self.f_number)
        
        # Y-axis apodization (elevational)
        apod_rx_y = (ops.abs(ops.atan((y_hd-rx_probe_y) / z_hd)) < self.f_number)
        apod_tx_y = (ops.abs(ops.atan((y_hd-tx_probe_y) / z_hd)) < self.f_number)
        
        # Combined boxcar apodization: all conditions must be met
        return apod_rx_x * apod_tx_x * apod_rx_y * apod_tx_y

