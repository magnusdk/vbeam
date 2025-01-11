from spekk import Dim, Module, ops

from vbeam.geometry.util import get_xyz


def _rotate_xy(x: float, y: float, roll: float):
    sin_roll, cos_roll = ops.sin(roll), ops.cos(roll)
    x_new = x * cos_roll - y * sin_roll
    y_new = x * sin_roll + y * cos_roll
    return x_new, y_new


def _rotate_xz(x: float, z: float, azimuth: float):
    sin_azimuth, cos_azimuth = ops.sin(azimuth), ops.cos(azimuth)
    x_new = x * cos_azimuth + z * sin_azimuth
    z_new = -x * sin_azimuth + z * cos_azimuth
    return x_new, z_new


def _rotate_yz(y: float, z: float, elevation: float):
    sin_elevation, cos_elevation = ops.sin(elevation), ops.cos(elevation)
    y_new = y * cos_elevation + z * sin_elevation
    z_new = -y * sin_elevation + z * cos_elevation
    return y_new, z_new


class Direction(Module):
    """A direction in 3D space, defined by `azimuth` (rotation of the xz-plane) and
    `elevation` (rotation of the yz-plane).

    NOTE: azimuth is applied first, then elevation. This is different from the standard
    spherical coordinate system which applies elevation first. The convention in
    ultrasound is to first rotate using azimuth, then elevation.
    """

    azimuth: float
    elevation: float

    @property
    def normalized_vector(self) -> ops.array:
        "The vector with magnitude one pointing in the direction."
        x, y, z = 0, 0, 1
        # NOTE: the order of rotations are reversed compared to what it says in the
        # docstring. If you believe that the docstring is wrong, please create an issue
        # or pull request on GitHub.
        y, z = _rotate_yz(y, z, self.elevation)
        x, z = _rotate_xz(x, z, self.azimuth)
        return ops.stack([x, y, z], axis="xyz")

    @staticmethod
    def from_array(v: ops.array) -> "Direction":
        """Construct a Direction object from an array.

        The array must be an instance of :class:`~spekk.ops.array_object.array`
        containing the dimension `'xyz'`. The dimension `'xyz'` must have size=3 and
        represent the x-, y-, and z-components of the vector.
        """
        if not isinstance(v, ops.array) or "xyz" not in v.dims:
            raise ValueError(
                "The array must be a spekk.ops.array containing the dimension 'xyz'. "
                "The 'xyz' dimension must have size=3 and represent the x-, y-, and "
                "z-components of the vector."
            )
        x, y, z = get_xyz(v)
        azimuth = ops.atan2(x, z)
        elevation = ops.asin(y)
        return Direction(azimuth, elevation)

    def __neg__(self) -> "Direction":
        return Direction(
            (self.azimuth + ops.pi) % (2 * ops.pi),
            (self.elevation + ops.pi) % (2 * ops.pi),
        )


class Orientation(Module):
    """An orientation, or rotation, in 3D space, defined by `azimuth` (rotation of the
    xz-plane), `elevation` (rotation of the yz-plane), and `roll` (rotation of the
    xy-plane).

    NOTE: Roll is applied first, followed by azimuth, then elevation. This is different
    from the standard spherical coordinate system which applies elevation before
    azimuth. The convention in ultrasound is to first rotate using azimuth, then
    elevation. Additionally, roll is applied before anything else because it represents
    the rotation of the ultrasound probe itself, which happens independently of
    focusing in azimuth and elevation. If you believe that this is incorrect, please
    create an issue or pull request on GitHub.
    """

    azimuth: float
    elevation: float
    roll: float

    def rotate(self, point: ops.array) -> ops.array:
        "Apply the rotation to the `point`."
        x, y, z = get_xyz(point)
        # NOTE: the order of rotations are reversed compared to what it says in the
        # docstring. If you believe that the docstring is wrong, please create an issue
        # or pull request on GitHub.
        y, z = _rotate_yz(y, z, self.elevation)
        x, z = _rotate_xz(x, z, self.azimuth)
        x, y = _rotate_xy(x, y, self.roll)
        return ops.stack([x, y, z], axis="xyz")

    def rotate_inverse(self, point: ops.array) -> ops.array:
        "Undo the rotation for the `point`."
        x, y, z = get_xyz(point)
        x, y = _rotate_xy(x, y, -self.roll)
        x, z = _rotate_xz(x, z, -self.azimuth)
        y, z = _rotate_yz(y, z, -self.elevation)
        return ops.stack([x, y, z], axis="xyz")

    @staticmethod
    def from_direction_and_roll(direction: Direction, roll: float) -> "Orientation":
        """Return a new Orientation object that points in the given Direction and with
        the given roll."""
        x, y, z = get_xyz(direction.normalized_vector)
        x, y = _rotate_xy(x, y, -roll)
        v = ops.stack([x, y, z], axis="xyz")
        direction = Direction.from_array(v)
        return Orientation(direction.azimuth, direction.elevation, roll)

    @property
    def direction(self) -> "Direction":
        "The direction of the unit vector (0,0,1) after being rotated."
        v = ops.array([0, 0, 1], ["xyz"])
        v = self.rotate(v)
        return Direction.from_array(v)


def average_directions(directions: Direction, *, axis: Dim) -> Direction:
    """Return the average direction, averaged over the given axis.

    NOTE: A single `Direction` object may represent multiple directions simultaneously
    because its attributes (e.g., `azimuth` and `elevation`) can be arrays. This is why
    it makes sense to average over a "single" `Direction` object across an
    axis/dimension.

    Examples:
        A single `Direction` object may represent multiple directions if `azimuth` or
        `elevation` are arrays:
        >>> azimuths = ops.array([-1, 1], ["azimuths"])  # <- An array of azimuths
        >>> direction = Direction(azimuths, 0)
        >>> average_direction = average_directions(direction, axis="azimuths")
        >>> assert average_direction.azimuth == 0
    """
    averaged_normalized_vectors = ops.mean(directions.normalized_vector, axis=axis)
    return Direction.from_array(averaged_normalized_vectors)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
