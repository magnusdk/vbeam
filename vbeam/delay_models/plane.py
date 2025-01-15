from spekk import ops

from vbeam.core import GeometricallyFocusedWave, Probe, TransmittedWaveDelayModel
from vbeam.util._transmitted_wave import raise_if_not_geometrically_focused_wave


class PlaneDelayModel(TransmittedWaveDelayModel):
    """A simple plane wave delay model.

    See it visually in a notebook by running this code:
    >>> from vbeam.delay_models import PlaneDelayModel
    >>> delay_model = PlaneDelayModel()
    >>> delay_model.plot()
    """

    def __call__(
        self,
        transmitting_probe: Probe,
        point: ops.array,
        transmitted_wave: GeometricallyFocusedWave,
        speed_of_sound: float,
    ) -> float:
        raise_if_not_geometrically_focused_wave(transmitted_wave)

        distance = ops.linalg.vecdot(
            transmitted_wave.virtual_source.direction.normalized_vector,
            point - transmitted_wave.origin,
            axis="xyz",
        )
        delay = distance / speed_of_sound
        return delay
