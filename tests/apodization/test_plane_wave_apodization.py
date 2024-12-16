import pytest
import numpy as np

from vbeam.apodization.plane_wave import PlaneWaveTransmitApodization
from vbeam.apodization.window import Hanning
from vbeam.core import ElementGeometry, WaveData


@pytest.fixture
def array_bounds():
    """Create a simple array geometry for testing."""
    array_left = np.array([-5, 0, 0])
    array_right = np.array([5, 0, 0])
    array_bottom = np.array([0, -5, 0])
    array_top = np.array([0, 5, 0])
    return array_left, array_right, array_bottom, array_top


@pytest.fixture
def transmit_element() -> ElementGeometry:
    """Send from the center of the array."""
    return ElementGeometry(position=np.array([0, 0, 0]))


@pytest.mark.parametrize(
    "azimuth_rad,elevation_rad,test_point,expected_value",
    [
        (
            0,  # 0 degrees (straight ahead)
            0,  # 0 degrees (straight ahead)
            np.array([0, 0, 10]),  # Point directly in front
            1.0,
        ),
        (
            0,  # 0 degrees (straight ahead)
            0,  # 0 degrees (straight ahead)
            np.array([-10, 0, 10]),  # Point far left
            0.0,
        ),
        # Azimuth apodization
        (
            np.pi / 4,  # 45 degrees
            0,
            np.array([10, 0, 10]),  # Point that should be in beam at 45 deg
            1.0,
        ),
        # Elevation apodization
        (
            0,
            np.pi / 4,  # 45 degrees
            np.array([0, 10, 10]),  # Point that should be in beam at 45 deg
            1.0,
        ),
    ],
)
def test_plane_wave_transmit_apodization(
    array_bounds,
    transmit_element,
    azimuth_rad: float,
    elevation_rad: float,
    test_point,
    expected_value,
):
    # Create apodization instance
    apodization = PlaneWaveTransmitApodization(
        array_bounds=array_bounds,
        window=None,
    )

    # Create wave data with specified angle
    wave_data = WaveData(azimuth=azimuth_rad, elevation=elevation_rad)

    # Calculate apodization value
    result = apodization(
        sender=transmit_element,
        point_position=test_point,
        receiver=transmit_element,
        wave_data=wave_data,
    )

    assert result == pytest.approx(expected_value, abs=1e-6)


def test_plane_wave_transmit_apodization_with_window(array_bounds, transmit_element):
    # Create apodization instance with Hann window
    apodization = PlaneWaveTransmitApodization(
        array_bounds=array_bounds,
        window=Hanning(),
    )

    # Test point in the middle of the beam
    center_point = np.array([0, 0, 10])
    wave_data = WaveData(azimuth=0, elevation=0)

    result = apodization(
        sender=transmit_element,
        point_position=center_point,
        receiver=transmit_element,
        wave_data=wave_data,
    )

    # Center point should have maximum window value
    assert result == pytest.approx(1.0, abs=1e-6)

    # Test point near the edge of the beam
    edge_point = np.array([4.9, 0, 10])
    result = apodization(
        sender=transmit_element,
        point_position=edge_point,
        receiver=transmit_element,
        wave_data=wave_data,
    )

    # Edge point should have value between 0 and 1
    assert 0 < result < 1

    # Test point far outside the beam
    outside_point = np.array([10, 0, 10])  # Well beyond array bounds
    result = apodization(
        sender=transmit_element,
        point_position=outside_point,
        receiver=transmit_element,
        wave_data=wave_data,
    )

    # Point outside beam width should have zero apodization
    assert result == pytest.approx(0.0, abs=1e-6)


def test_invalid_direction(array_bounds, transmit_element):
    # Create apodization instance with invalid apodization_dimension
    with pytest.raises(ValueError, match="Expected either left/right.*"):
        PlaneWaveTransmitApodization(
            array_bounds=array_bounds[:1],
        )
