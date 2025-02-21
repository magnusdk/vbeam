from spekk import ops

from vbeam.geometry.util import get_xyz


def rotate_xy(x: float, y: float, roll: float):
    sin_roll, cos_roll = ops.sin(roll), ops.cos(roll)
    x_new = x * cos_roll - y * sin_roll
    y_new = x * sin_roll + y * cos_roll
    return x_new, y_new


def rotate_xz(x: float, z: float, azimuth: float):
    sin_azimuth, cos_azimuth = ops.sin(azimuth), ops.cos(azimuth)
    x_new = x * cos_azimuth + z * sin_azimuth
    z_new = -x * sin_azimuth + z * cos_azimuth
    return x_new, z_new


def rotate_yz(y: float, z: float, elevation: float):
    sin_elevation, cos_elevation = ops.sin(elevation), ops.cos(elevation)
    y_new = y * cos_elevation + z * sin_elevation
    z_new = -y * sin_elevation + z * cos_elevation
    return y_new, z_new


def as_cartesian(point: ops.array) -> ops.array:
    azimuth, elevation, depth = get_xyz(point)
    x, y, z = 0, 0, depth
    y, z = rotate_yz(y, z, elevation)
    x, z = rotate_xz(x, z, azimuth)
    return ops.stack([x, y, z], axis="xyz")


def as_polar(point: ops.array) -> ops.array:
    x, y, z = get_xyz(point)
    azimuth = ops.atan2(x, z)
    elevation = ops.asin(y)
    depth = ops.linalg.vector_norm(point, axis="xyz")
    return ops.stack([azimuth, elevation, depth], axis="xyz")
