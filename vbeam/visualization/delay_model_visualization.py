from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spekk import ops
from spekk.module.base import at, dim_sizes

from vbeam.core.delay_models import DelayModel, SeparableDelayModel
from vbeam.core.transmitted_wave import TransmittedWave
from vbeam.probe import Probe, ProbeElement


@dataclass
class DelayModelVisualizer:
    delay_model: DelayModel
    point: ops.array
    transmitted_wave: TransmittedWave
    transmitting_probe: Probe
    receiving_probe: Probe

    def plot(
        self,
        use_tx_or_rx: Literal["tx", "rx", "both"] = "both",
        **indexing_objects: int,
    ):
        # Slice the attributes based on the selected indices.
        delay_model = at(self.delay_model)[indexing_objects].get()
        point = at(self.point)[indexing_objects].get()
        transmitted_wave = at(self.transmitted_wave)[indexing_objects].get()
        transmitting_probe = at(self.transmitting_probe)[indexing_objects].get()
        receiving_probe = at(self.receiving_probe)[indexing_objects].get()

        if isinstance(delay_model, SeparableDelayModel):
            if use_tx_or_rx == "tx":
                delay_model = delay_model.get_tx_delay
                plot_title = f"Tx delays for '{type(self).__name__}'"
            elif use_tx_or_rx == "rx":
                delay_model = delay_model.get_rx_delay
                plot_title = f"Rx delays for '{type(self).__name__}'"
            elif use_tx_or_rx == "both":
                # Keep delay_model as-is
                plot_title = f"Tx + Rx delays for '{type(self).__name__}'"
            else:
                raise ValueError(f"Invalid value for use_tx_or_rx: '{use_tx_or_rx}'")
        elif use_tx_or_rx == "both":
            plot_title = f"Tx + Rx delays for '{type(self).__name__}'"
        else:
            raise ValueError(
                "Only SeparableDelayModel instances can plot tx or rx separately; "
                f"use_tx_or_rx must be 'both', but got '{use_tx_or_rx}'."
            )

        # Get delays for each point in the image.
        image = delay_model(
            point=point,
            transmitted_wave=transmitted_wave,
            transmitting_probe=transmitting_probe,
            receiving_probe=receiving_probe,
        )
        not_indexed_dims = [
            dim for dim in dim_sizes(image).keys() if dim not in {"xs", "zs"}
        ]
        if not_indexed_dims:
            raise ValueError(
                f"You must select index for dimensions {not_indexed_dims} when calling "
                f"plot, e.g.: .plot({", ".join(f"{dim}=0" for dim in not_indexed_dims)})."
            )

        fig, ax = plt.subplots()

        # Plot delays with contour lines.
        im = ax.imshow(
            ops.permute_dims(image, ["zs", "xs"]),
            extent=(
                float(ops.min(point[{"xyz": 0}])),
                float(ops.max(point[{"xyz": 0}])),
                float(ops.max(point[{"xyz": 2}])),
                float(ops.min(point[{"xyz": 2}])),
            ),
        )
        cs = ax.contour(
            point[{"xyz": 0}],
            point[{"xyz": 2}],
            ops.permute_dims(image, ["xs", "zs"]),
            locator=MultipleLocator(20e-6),
            colors="k",
            linewidths=0.5,
            alpha=0.5,
        )
        ax.clabel(cs, fmt=lambda v: f"{v*1e6:.0f}μs")
        # Set x- and z-axis units to millimeters (1e-3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1e3:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*1e3:.0f}"))

        # Add colorbar to an axis of equal height to ax
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label="Delay [μs]")
        # Set delay units to microseconds (1e-6)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*1e6:.1f}"))

        def plot_element(ax, element: ProbeElement, color: str):
            element_left = element.bounds.center_left
            element_right = element.bounds.center_right
            ax.plot(
                [element_left[{"xyz": 0}], element_right[{"xyz": 0}]],
                [element_left[{"xyz": 2}], element_right[{"xyz": 2}]],
                color=color,
            )

        # Plot receiving probe
        for rx_i in range(self.receiving_probe.dim_sizes["rx"]):
            element = self.receiving_probe.active_elements.at["rx", rx_i].get()
            plot_element(ax, element, "white")
        selected_element = receiving_probe.active_elements
        plot_element(ax, selected_element, "red")

        # Plot virtual source
        vs_point = transmitted_wave.virtual_source.to_array()
        ax.plot(vs_point[{"xyz": 0}], vs_point[{"xyz": 2}], "o", color="red")

        # Add custom legend
        custom_lines = [
            Line2D([0], [0], color="white"),
            Line2D([0], [0], color="red"),
            Line2D([0], [0], color="red", marker="o", linestyle="None"),
        ]
        labels = ["Receiving elements", "Selected receiving element", "Virtual source"]
        ax.legend(custom_lines, labels, loc="lower left")

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(plot_title)
        fig.tight_layout()

    def _ipython_display_(self):
        from IPython.display import display
        from ipywidgets import widgets

        kwarg_dim_sizes = dim_sizes(
            [
                self.delay_model,
                self.point,
                self.transmitted_wave,
                self.transmitting_probe,
                self.receiving_probe,
            ]
        )
        controls = {
            dim: widgets.IntSlider(value=size // 4, min=0, max=size, description=dim)
            for dim, size in kwarg_dim_sizes.items()
            if dim not in {"xyz", "xs", "zs"}
        }
        if isinstance(self.delay_model, SeparableDelayModel):
            controls["use_tx_or_rx"] = widgets.RadioButtons(
                options=[("Tx", "tx"), ("Rx", "rx"), ("Both", "both")],
                value="both",
                orientation="horizontal",
            )
        ui = widgets.VBox(list(controls.values()))
        return display(ui, widgets.interactive_output(self.plot, controls))
