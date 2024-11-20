from vbeam.fastmath import numpy as np


def as_polar(cartesian_point: np.ndarray):
    """Convert cartesian coordinates (x,y,z) to spherical coordinates (r,θ,φ).

    Returns:
        np.ndarray: Array with [..., (r, theta, phi)] where:
            r = radial distance
            theta = polar angle (inclination from z-axis) [0,π]
            phi = azimuthal angle (from x-axis in x-y plane) [0,2π]

    Follows physics convention:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    (ISO 80000-2:2019)

    Note:
    - theta/phi are not the same as elevation/azimuth
    """
    x, y, z = cartesian_point[..., 0], cartesian_point[..., 1], cartesian_point[..., 2]

    # Radial distance
    radii = np.sqrt(x**2 + y**2 + z**2)

    # Polar angle (inclination from z-axis)
    theta = np.arccos(z / radii)

    # Azimuthal angle (from x-axis)
    phi = np.arctan2(y, x)

    return np.stack([radii, theta, phi], axis=-1)


def as_cartesian(polar_point: np.ndarray):
    """Convert spherical coordinates (r,θ,φ) to cartesian coordinates (x,y,z).

    Args:
        polar_point: Array with [..., (r, theta, phi)] where:
            r = radial distance
            theta = polar angle (inclination from z-axis)
            phi = azimuthal angle (from x-axis in x-y plane)

    Follows physics convention:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    (ISO 80000-2:2019)
    """
    r = polar_point[..., 0]
    polar_angles = polar_point[..., 1]  # theta
    azimuth_angles = polar_point[..., 2]  # phi

    return np.stack(
        [
            r * np.sin(polar_angles) * np.cos(azimuth_angles),  # x
            r * np.sin(polar_angles) * np.sin(azimuth_angles),  # y
            r * np.cos(polar_angles),  # z
        ],
        axis=-1,
    )


def applied_in_polar_coordinates(f):
    """Wrap a function that takes and returns a point in polar coordinates such that it
    takes and returns a point in cartesian coordinates instead.
    
    Usage:
    >>> import numpy as np
    >>> @applied_in_polar_coordinates
    ... def add_radius(polar_point, radius):
    ...   return polar_point + np.array([0, 0, radius])
    >>> cartesian_point = np.array([3, 0, 4])  # [x, y, z]
    >>> add_radius(cartesian_point, 1)
    array([3.6, 0. , 4.8])
    """
    def inner(p, *args, **kwargs):
        return as_cartesian(f(as_polar(p), *args, **kwargs))

    return inner


def applied_in_cartesian_coordinates(f):
    """Wrap a function that takes and returns a point in cartesian coordinates such that
    it takes and returns a point in polar coordinates instead.
    
    Usage:
    >>> import numpy as np
    >>> @applied_in_cartesian_coordinates
    ... def move_x(cartesian_point, dx):
    ...   return cartesian_point + np.array([dx, 0, 0])
    >>> polar_point = np.array([np.pi/2, 0, 1])  # [azimuth angle, polar angle, radius]
    >>> move_x(polar_point, 1)
    array([1.57079633, 0.        , 2.        ])
    """
    def inner(p, *args, **kwargs):
        return as_polar(f(as_cartesian(p), *args, **kwargs))

    return inner
