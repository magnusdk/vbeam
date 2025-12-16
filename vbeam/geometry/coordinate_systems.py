from spekk import ops


def get_xyz(v: ops.array):
    return v["xyz", 0], v["xyz", 1], v["xyz", 2]


def get_az_el_depth(v: ops.array):
    return v["az_el_depth", 0], v["az_el_depth", 1], v["az_el_depth", 2]


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


def as_cartesian(polar_point: ops.array) -> ops.array:
    azimuth, elevation, depth = get_az_el_depth(polar_point)
    x, y, z = 0, 0, depth
    y, z = rotate_yz(y, z, elevation)
    x, z = rotate_xz(x, z, azimuth)
    return ops.stack([x, y, z], axis="xyz")


def as_polar(cartesian_points: ops.array) -> ops.array:
    x, y, z = get_xyz(cartesian_points)
    depth = ops.linalg.vector_norm(cartesian_points, axis="xyz")
    azimuth = ops.atan2(x, z)
    elevation = ops.asin(y)
    return ops.stack([azimuth, elevation, depth], axis="az_el_depth")

####################################################################
def polar_to_cartesian(polar_point: ops.array) -> ops.array:
    azimuth, elevation, depth = get_az_el_depth(polar_point)
    x = depth * ops.sin(azimuth)
    z = depth * ops.cos(azimuth) * ops.cos(elevation)
    y = -depth * ops.cos(azimuth) * ops.sin(elevation)
    return ops.stack([x, y, z], axis="xyz")


def cartesian_to_polar(cartesian_points: ops.array) -> ops.array:
    x, y, z = get_xyz(cartesian_points)
    depth = ops.linalg.vector_norm(cartesian_points, axis="xyz")
    azimuth = ops.asin(x / (depth+1e-15))
    elevation = -ops.atan2(y, z)
    return ops.stack([azimuth, elevation, depth], axis="az_el_depth")

    