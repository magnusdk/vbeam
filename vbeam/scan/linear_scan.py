from typing import Callable, Literal, Optional, Tuple, Union, overload

from spekk import ops

from vbeam.scan.base import CoordinateSystem, Scan
from vbeam.scan.util import parse_axes


class LinearScan(Scan):
    x: ops.array
    y: Optional[ops.array]
    z: ops.array

    def get_points(self) -> ops.array:
        y = (
            ops.array([0.0], ["y_axis"]) if self.y is None else self.y
        )  # If the scan is 2D
        points = ops.stack(ops.meshgrid(self.x, y, self.z, indexing="ij"), axis="xyz")
        return points

    def replace(
        self,
        x: Union[ops.array, Literal["unchanged"]] = "unchanged",
        y: Union[ops.array, Literal["unchanged"]] = "unchanged",
        z: Union[ops.array, Literal["unchanged"]] = "unchanged",
    ) -> "LinearScan":
        return LinearScan(
            x=x if x != "unchanged" else self.x,
            y=y if y != "unchanged" else self.y,
            z=z if z != "unchanged" else self.z,
        )

    def update(
        self,
        x: Optional[Callable[[ops.array], ops.array]] = None,
        y: Optional[Callable[[ops.array], ops.array]] = None,
        z: Optional[Callable[[ops.array], ops.array]] = None,
    ) -> "LinearScan":
        return self.replace(
            x(self.x) if x is not None else "unchanged",
            y(self.y) if y is not None else "unchanged",
            z(self.z) if z is not None else "unchanged",
        )

    def resize(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
    ) -> "LinearScan":
        if y is not None and self.y is None:
            raise ValueError("Cannot resize y because it is not defined on this scan")
        return self.replace(
            ops.linspace(self.x[0], self.x[-1], x) if x is not None else "unchanged",
            ops.linspace(self.y[0], self.y[-1], y) if y is not None else "unchanged",
            ops.linspace(self.z[0], self.z[-1], z) if z is not None else "unchanged",
        )

    @property
    def axes(self) -> Tuple[ops.array, ...]:
        if self.y is not None:
            return self.x, self.y, self.z
        else:
            return self.x, self.z

    @property
    def cartesian_bounds(self):
        return self.bounds  # LinearScan is already in cartesian coordinates :)

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.CARTESIAN

    def __repr__(self):
        return f"LinearScan(<shape={self.shape}>)"


@overload
def linear_scan(x: ops.array, z: ops.array) -> LinearScan: ...  # 2D scan


@overload
def linear_scan(x: ops.array, y: ops.array, z: ops.array) -> LinearScan: ...  # 3D scan


def linear_scan(*xyz: ops.array) -> LinearScan:
    "Construct a linear scan. See LinearScan documentation for more details."
    x, y, z = parse_axes(xyz)
    return LinearScan(x, y, z)
