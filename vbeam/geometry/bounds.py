from spekk import Module, ops

from vbeam.geometry.plane import Plane


class RectangularBounds(Module):
    plane: Plane
    width: float
    height: float

    @property
    def center(self) -> ops.array:
        return self.plane.origin

    @property
    def upper_left(self) -> ops.array:
        return self.plane.from_plane_coordinates(-self.width / 2, self.height / 2)

    @property
    def center_top(self) -> ops.array:
        return self.plane.from_plane_coordinates(0, self.height / 2)

    @property
    def upper_right(self) -> ops.array:
        return self.plane.from_plane_coordinates(self.width / 2, self.height / 2)

    @property
    def center_right(self) -> ops.array:
        return self.plane.from_plane_coordinates(self.width / 2, 0)

    @property
    def lower_right(self) -> ops.array:
        return self.plane.from_plane_coordinates(self.width / 2, -self.height / 2)

    @property
    def center_bottom(self) -> ops.array:
        return self.plane.from_plane_coordinates(0, -self.height / 2)

    @property
    def lower_left(self) -> ops.array:
        return self.plane.from_plane_coordinates(-self.width / 2, -self.height / 2)

    @property
    def center_left(self) -> ops.array:
        return self.plane.from_plane_coordinates(-self.width / 2, 0)

    @property
    def corners(self) -> ops.array:
        """The corners of the bounds in the order of UL, UR, LR, LL.

        UL = Upper Left, UR = Upper Right, LR = Lower Right, LL = Lower Left.
        """
        corners = [self.upper_left, self.upper_right, self.lower_right, self.lower_left]
        return ops.stack(corners, axis="bounds_corners")
