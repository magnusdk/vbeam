"""Interfaces and kernel function for the core of the beamforming algorithm.

``vbeam`` is built around a small functional core found in this package. It is built 
around the :func:`~vbeam.core.kernels.signal_for_point` kernel [#]_ which depend on a 
set of interfaces to calculate the delayed, interpolated, and weighted signal for a 
given point.

A generalized delay-and-sum beamformer can be expressed as in equation 
:eq:`das_general_beamformer`:

.. math::
    :label: das_general_beamformer

    b_{DAS}(p) = 
    \\sum_{a=0}^{N-1} 
        \\sum_{m=0}^{M-1}
            w(a, m, p)\\,
            s_{a, m}(t)\\,
            e^{i2\\pi f_{mod}t},
    t := \\frac{d(a, m, p)}{c}

where :math:`a`, :math:`m`, and :math:`p` represent a given :term:`transmit`, 
:term:`receiver`, and :term:`point`, respectively. Note that this equation is for a 
full delay-and-sum (DAS) beamformer while the 
:func:`~vbeam.core.kernels.signal_for_point` function is only the inner part of the 
loop, i.e.: it computes the delayed signal for a single transmit, receiver, and point. 
For building up a full DAS beamformer from 
:func:`~vbeam.core.kernels.signal_for_point`, see :mod:`vbeam.beamformers`.

In equation :eq:`das_general_beamformer`:

* :math:`w(a, m, p)` weights the delayed and interpolated signal and is represented by 
  the :class:`~vbeam.core.apodization.Apodization` class in vbeam.

* :math:`s_{a, m}(t)` is the delayed and interpolated :term:`signal` for the 
  :term:`transmit` and :term:`receiver` at time :math:`t` seconds *(defined below)*. 
  Interpolation is represented by the 
  :class:`~vbeam.core.interpolation.InterpolationSpace1D` class.

* :math:`e^{i2\\pi f_{mod}t}` is the modulation frequency correction, which should be 
  applied if the signal is a demodulated IQ signal. :math:`f_{mod}` is the modulation 
  frequency, represented as a number, and given as an argument to 
  :func:`~vbeam.core.kernels.signal_for_point`.

* :math:`d(a, m, p)` is the distance the wave has travelled from the :term:`sender` to 
  the :term:`point`, and back to the :term:`receiver`. This is represented by the 
  :class:`~vbeam.core.wavefront.ReflectedWavefront` and 
  :class:`~vbeam.core.wavefront.TransmittedWavefront` classes.

* :math:`t := \\frac{d(a, m, p)}{c}` is the delay in seconds, computed by dividing the 
  distance by the speed of sound. The speed of sound may be just a number, or in  more 
  advanced cases, an instance of the :class:`~vbeam.core.speed_of_sound.SpeedOfSound` 
  class.

Again, it is important to keep in mind that 
:func:`~vbeam.core.kernels.signal_for_point` runs for just a single :term:`point`, a 
single :term:`receiving element<Receiver>`, a single 
:term:`transmitted wave<Transmit>`, etc. In general, it returns a scalar value.

.. rubric:: Footnotes
.. [#] We refer to :func:`~vbeam.core.kernels.signal_for_point` as a 
       :term:`kernel<Kernel function>` because it is the central function that we 
       repeatedly call when beamforming. However, it is just a normal Python function — 
       it just happens to be a very important function.
"""

from vbeam.core.apodization import Apodization
from vbeam.core.element_geometry import ElementGeometry
from vbeam.core.interpolation import InterpolationSpace1D
from vbeam.core.kernels import KernelData, SignalForPointData, signal_for_point
from vbeam.core.speed_of_sound import SpeedOfSound
from vbeam.core.wave_data import WaveData
from vbeam.core.wavefront import MultipleTransmitDistances, TransmittedWavefront

__all__ = [
    "Apodization",
    "ElementGeometry",
    "InterpolationSpace1D",
    "KernelData",
    "SignalForPointData",
    "signal_for_point",
    "SpeedOfSound",
    "WaveData",
    "MultipleTransmitDistances",
    "TransmittedWavefront",
]
