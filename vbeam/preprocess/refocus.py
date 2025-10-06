from spekk import ops, replace, traverse, update_at

from vbeam.apodization import ConstantFNumberApodization, window
from vbeam.core import DelayModel, Setup
from vbeam.delay_models import REFoCUSDelayModel


def convert_to_refocus_setup(
    setup: Setup,
    sta_delay_model: DelayModel,
    synthetic_element_positions: ops.array | None = None,
    *,
    refocus_dim_name: str = "refocus_tx",
) -> Setup:
    if synthetic_element_positions is None:
        synthetic_element_positions = (
            setup.transmitting_probe.active_elements.position.rename_dim(
                "rx", refocus_dim_name
            )
        )

    refocus_delay_model = REFoCUSDelayModel(
        synthetic_element_positions,
        setup.delay_model,
        sta_delay_model,
    )
    refocus_transmitting_probe = traverse(
        setup.transmitting_probe,
        map_leaf=lambda x: (
            x.rename_dim("rx", refocus_dim_name)
            if isinstance(x, ops.array) and "rx" in x.dims
            else x
        ),
    )
    apodization = update_at(
        setup.apodization,
        ["tx"],
        lambda apod: apod.combine(
            ConstantFNumberApodization(0.5, window.Hamming(), apply_on="tx")
        ),
    )

    setup = replace(
        setup,
        delay_model=refocus_delay_model,
        transmitting_probe=refocus_transmitting_probe,
        apodization=apodization,
    )
    return setup
