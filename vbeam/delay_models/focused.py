from spekk import ops

from vbeam import geometry
from vbeam.core import (
    GeometricallyFocusedWave,
    Probe,
    SeparableDelayModel,
    TransmittedWave,
)
from vbeam.delay_models.plane import PlaneDelayModel


class SphericalFocusedDelayModel(SeparableDelayModel):
    """A simple focused wave delay model, modeling a spherical wave originating from a
    virtual source.

    NOTE: This delay model has a harsh discontinuity around the focus depth which may
    create artifacts in your image, often looking like small lines or noise around the
    focus depth. To fix this, you should use either
    :class:`~vbeam.delay_models.focused.SphericalBlendedDelayModel` instead. It has a
    smoother transition in delay values around the focus depth.
    """

    speed_of_sound: float | ops.array

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        if not isinstance(transmitted_wave, GeometricallyFocusedWave):
            raise ValueError(
                "Expected a geometrically focused transmitted wave, but got "
                f"{type(transmitted_wave)}."
            )

        # Calculate distances.
        origin_source_distance = geometry.distance(
            transmitted_wave.origin, transmitted_wave.virtual_source
        )
        virtual_source_point_distance = geometry.distance(
            transmitted_wave.virtual_source, point
        )

        # Get the depths of the virtual source and point along the direction of the
        # transmitted wave. To get the direction we first need the effective aperture
        # facing towards the virtual source.
        effective_aperture = transmitting_probe.get_effective_aperture(
            transmitted_wave.virtual_source
        )
        virtual_source_depth = effective_aperture.plane.signed_distance(
            transmitted_wave.virtual_source.to_array()
        )
        point_depth = effective_aperture.plane.signed_distance(point)

        # Find out whether the point lies before or beyond the virtual source. If it
        # lies before, depth_sign will equal negative 1, otherwise positive 1.
        depth_sign = ops.sign(point_depth - virtual_source_depth)

        # Calculate distance on tx and rx and divide by speed of sound.
        tx_distance = (
            virtual_source_point_distance * depth_sign + origin_source_distance
        )
        delay = tx_distance / self.speed_of_sound
        return delay

    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        distance = geometry.distance(point, receiving_probe.active_elements.position)
        delay = distance / self.speed_of_sound
        return delay


class SphericalDivergingDelayModel(SeparableDelayModel):
    """A simple focused wave delay model, modeling a spherical wave originating from a
    virtual source.

    NOTE: This delay model has a harsh discontinuity around the focus depth which may
    create artifacts in your image, often looking like small lines or noise around the
    focus depth. To fix this, you should use either
    :class:`~vbeam.delay_models.focused.SphericalBlendedDelayModel` instead. It has a
    smoother transition in delay values around the focus depth.
    """

    speed_of_sound: float | ops.array

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        if not isinstance(transmitted_wave, GeometricallyFocusedWave):
            raise ValueError(
                "Expected a geometrically focused transmitted wave, but got "
                f"{type(transmitted_wave)}."
            )

        # Calculate distances.
        origin_source_distance = geometry.distance(
            transmitted_wave.origin, transmitted_wave.virtual_source.to_array()
        )
        virtual_source_point_distance = geometry.distance(
            transmitted_wave.virtual_source.to_array(), point
        )

        tx_distance = virtual_source_point_distance - origin_source_distance
        delay = tx_distance / self.speed_of_sound
        return delay

    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        distance = geometry.distance(point, receiving_probe.active_elements.position)
        delay = distance / self.speed_of_sound
        return delay


class SphericalHybridDelayModel(SeparableDelayModel):
    """A simple focused wave delay model, modeling a spherical wave originating from a
    virtual source, but modeling it as a plane wave around the focus depth. Modeling
    the wave as a plane wave around the focus depth lessens the discontinuity artifacts
    seen in :class:`~vbeam.delay_models.focused.SphericalDelayModel`.

    NOTE: If you still get discontinuity artifacts around the focus depth you can try
    :class:`~vbeam.delay_models.focused.SphericalBlendedDelayModel` instead.

    Attributes:
        plane_wave_region_size (float): The length in meters of the region where the
            wave should be modeled as a plane wave.

    Reference:
        O. M. Hoel Rindal, A. R. -. Molares and A. Austeng, "A Simple, Artifact - Free,
        Virtual Source Model," 2018 IEEE International Ultrasonics Symposium (IUS),
        Kobe, Japan, 2018, pp. 1-4, doi: 10.1109/ULTSYM.2018.8579944.
    """

    speed_of_sound: float | ops.array
    plane_wave_region_size: float

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> ops.array:
        if not isinstance(transmitted_wave, GeometricallyFocusedWave):
            raise ValueError(
                "Expected a geometrically focused transmitted wave, but got "
                f"{type(transmitted_wave)}."
            )

        # Find the depth of the point compared to the depth of the virtual source.
        aperture = transmitting_probe.get_effective_aperture(
            transmitted_wave.virtual_source
        )
        virtual_source_depth = aperture.plane.signed_distance(
            transmitted_wave.virtual_source.to_array()
        )
        point_depth = aperture.plane.signed_distance(point)
        depth_difference = ops.abs(point_depth - virtual_source_depth)

        # Use plane wave delay model when the depth is <= plane_wave_region_size, else
        # spherical wave delay model.
        return ops.where(
            depth_difference <= (self.plane_wave_region_size / 2),
            PlaneDelayModel(self.speed_of_sound).get_tx_delay(
                point, transmitted_wave, transmitting_probe, receiving_probe
            ),
            SphericalFocusedDelayModel(self.speed_of_sound).get_tx_delay(
                point, transmitted_wave, transmitting_probe, receiving_probe
            ),
        )

    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        distance = geometry.distance(point, receiving_probe.active_elements.position)
        delay = distance / self.speed_of_sound
        return delay


class SphericalBlendedDelayModel(SeparableDelayModel):
    """A simple focused wave delay model, modeling a spherical wave originating from a
    virtual source, but gradually modeling it as a plane wave around the focus depth.
    Modeling the wave as a plane wave around the focus depth lessens the discontinuity
    artifacts seen in :class:`~vbeam.delay_models.focused.SphericalDelayModel`.

    The spherical wave and plane wave delay values are blended as a function of
    distance to the focal plane. When the distance is zero, then the wave is modeled as
    a plane wave. When the distance is more than `plane_wave_region_size/2`, then the
    wave is modeled as a spherical wave. The delay models transition linearly from one
    mode to the other.

    Attributes:
        plane_wave_region_size (float): The length in meters of the transition region
            where the wave should gradually be modeled as a plane wave.

    Reference:
        O. M. Hoel Rindal, A. R. -. Molares and A. Austeng, "A Simple, Artifact - Free,
        Virtual Source Model," 2018 IEEE International Ultrasonics Symposium (IUS),
        Kobe, Japan, 2018, pp. 1-4, doi: 10.1109/ULTSYM.2018.8579944.
    """

    speed_of_sound: float | ops.array
    plane_wave_region_size: float

    def get_tx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> ops.array:
        if not isinstance(transmitted_wave, GeometricallyFocusedWave):
            raise ValueError(
                "Expected a geometrically focused transmitted wave, but got "
                f"{type(transmitted_wave)}."
            )

        # Get delay values for spherical wave model and plane wave model.
        spherical_wave_delay = SphericalFocusedDelayModel(
            self.speed_of_sound
        ).get_tx_delay(point, transmitted_wave, transmitting_probe, receiving_probe)
        plane_wave_delay = PlaneDelayModel(self.speed_of_sound).get_tx_delay(
            point, transmitted_wave, transmitting_probe, receiving_probe
        )

        # Find the depth of the point compared to the depth of the virtual source.
        aperture = transmitting_probe.get_effective_aperture(
            transmitted_wave.virtual_source
        )
        virtual_source_depth = aperture.plane.signed_distance(
            transmitted_wave.virtual_source.to_array()
        )
        point_depth = aperture.plane.signed_distance(point)
        depth_difference = ops.abs(point_depth - virtual_source_depth)

        # Get the weight p for the plane wave delay. If p is 1, then we use the plane
        # wave delay, if it is 0, then we use the spherical wave delay. Linearly
        # interpolate between the two.
        p = 1 - depth_difference / (self.plane_wave_region_size / 2)
        # Clip p to [0, 1]. NOTE: p can never be more than 1 here, so we only have to
        # clip it so that it doesn't go below 0.
        p = ops.maximum(p, 0)

        # Linearly blend the two delay models.
        return plane_wave_delay * p + spherical_wave_delay * (1 - p)

    def get_rx_delay(
        self,
        point: ops.array,
        transmitted_wave: TransmittedWave,
        transmitting_probe: Probe,
        receiving_probe: Probe,
    ) -> float | ops.array:
        distance = geometry.distance(point, receiving_probe.active_elements.position)
        delay = distance / self.speed_of_sound
        return delay
