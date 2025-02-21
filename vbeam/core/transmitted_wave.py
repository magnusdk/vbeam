from typing import Union

from spekk import Module, ops

from vbeam.geometry import Vector, VectorWithInfiniteMagnitude


class TransmittedWave(Module):
    """Information about the transmitted wave.

    Attributes:
        origin (ops.array): The point in space that the wave passed through at time=0s.

    Other information could include the virtual source in the case of geometrically
    focused waves (see :class:`~GeometricallyFocusedWave`), or the delays and weights
    for pulse-encoded transmits, or arbitrary waveform information, etc.
    """

    origin: ops.array


class GeometricallyFocusedWave(TransmittedWave):
    """Geometrically focused waves with a virtual source, encompassing plane waves,
    diverging waves, and focused waves.

    Attributes:
        origin (ops.array): The point in space that the wave passed through at time=0s.
        virtual_source (Vector): The "virtual source", aka the focus point for focused
            transmits. For diverging waves, the virtual source lies behind the
            transmitting probe. For plane waves, the virtual source lies at infinity
            (and is effectively just a direction).

    This class is based on the formulation of the Generalized Beamformer from USTB.
    """

    virtual_source: Union[Vector, VectorWithInfiniteMagnitude]
