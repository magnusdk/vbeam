from vbeam.fastmath import numpy as np


def as_polar(cartesian_point: np.ndarray):
    """Return a point in cartesian coordinates in its polar coordinates representation.

    NOTE: All y-values must be 0. FIXME"""
    x, y, z = cartesian_point[..., 0], cartesian_point[..., 1], cartesian_point[..., 2]
    azimuth_angles = np.arctan2(x, z)
    radii = np.sqrt(x**2 + y**2 + z**2)
    return np.stack([azimuth_angles, np.zeros(radii.shape), radii], -1)


def as_cartesian(polar_point: np.ndarray):
    "Return a point in polar coordinates in its cartesian coordinates representation."
    azimuth_angles = polar_point[..., 0]
    polar_angles = polar_point[..., 1]
    r = polar_point[..., 2]
    return np.stack(
        [
            r * np.sin(azimuth_angles) * np.cos(polar_angles),
            r * np.sin(azimuth_angles) * np.sin(polar_angles),
            r * np.cos(azimuth_angles),
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
