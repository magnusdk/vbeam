.. _glossary:

Glossary
========

.. glossary::
    
    Point
        A point that we want to beamform/image. It can usually be thought of as a pixel in an image. In ``vbeam``, points are always in cartesian coordinates and 3D, having an x-, y-, and z-component *(unless otherwise specified)*. Points are usually flattened to a ``(N, 3)`` array, with ``N`` being the number of points.

    Sender
        The origin of a transmitted wave represented with an :class:`~vbeam.core.element_geometry.ElementGeometry` object. It is usually the position of the array where the transmitted wave passes through at ``t0``, but may also refer to actual individual transducer elements, as in the case of synthetic transmit aperture (STA).

    Receiver
        A transducer element that has recorded a signal. Usually represented with an :class:`~vbeam.core.element_geometry.ElementGeometry` object and an array of the recorded signal.

    Transmit
        A transmitted wave event representing a wave sent out from the transducer. See also :class:`~vbeam.core.wave_data.WaveData` which contains data associated with a transmit, like the position of the :term:`virtual source`.

    Frame
        A frame in a video.

    Signal
        The recorded signal for a :term:`receiver`. Also called channel-data.

    Virtual source
        The focus point used when transmitting waves. For focused transmits it is in front of the transducer, and the transmitted wave converges into it. For diverging waves it is behind the transducer. For plane waves it is *"at infinity"*, meaning that it only has a direction.

    Kernel function
        A particularly important function in ``vbeam`` that is usually called multiple times in parallel. See :func:`~vbeam.core.kernels.signal_for_point`.

    Vectorization
        TODO

