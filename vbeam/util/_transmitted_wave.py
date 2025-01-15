from vbeam.core import GeometricallyFocusedWave, TransmittedWave


def raise_if_not_geometrically_focused_wave(transmitted_wave: TransmittedWave):
    if not isinstance(transmitted_wave, GeometricallyFocusedWave):
        raise ValueError(
            "Expected a geometrically focused transmitted wave, but got "
            f"{type(transmitted_wave)}."
        )
