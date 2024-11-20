from typing import Optional, Tuple

from fastmath import Array, field

from vbeam.fastmath import backend_manager
from vbeam.fastmath import numpy as api
from vbeam.scan.advanced.base import WrappedScan
from vbeam.scan.base import CoordinateSystem
from vbeam.scan.sector_scan import SectorScan
from vbeam.scan.util import scan_convert
from vbeam.util.vmap import vmap_all_except


def _get_scan_converted_points_and_indices(
    base_scan: SectorScan,
) -> Tuple[Array, Array]:
    # Scan convert a grid of ones to get a mask of the points that are mapped to the
    # cartesian grid.
    mask = scan_convert(api.ones(base_scan.shape), base_scan)
    mask = (mask == 1).reshape((base_scan.num_points,))

    # Calculate the scan converted points.
    points = base_scan.get_points(flatten=False)
    scan_converted_points = scan_convert(points, base_scan, 0, 1)
    scan_converted_points = scan_converted_points.reshape((base_scan.num_points, 3))

    # We must use Numpy for this because masking out values is hard on GPUs.
    with backend_manager.using_backend("numpy"):
        # Mask out the points and indices
        mask = api.array(mask)
        points = api.array(scan_converted_points)[mask]
        indices = api.arange(base_scan.num_points)[mask]

    # Wrap in array to ensure that they use the active backend.
    return api.array(points), api.array(indices)


class ScanConvertedSectorScan(WrappedScan):
    _base_scan: SectorScan = field(static=True)
    _points: Optional[Array] = None
    _indices: Optional[Array] = None

    def __post_init__(self):
        if self._points is None or self._indices is None:
            self._points, self._indices = _get_scan_converted_points_and_indices(
                self._base_scan
            )

    def get_points(self, flatten: bool = True) -> Array:
        if not flatten:
            raise ValueError(
                f"{self.__class__.__name__} only supports getting flattened points."
            )
        return self._points

    def unflatten(
        self, imaged_points: Array, points_axis: int = -1
    ) -> Array:
        image = api.zeros((self.base_scan.num_points,), dtype=imaged_points.dtype)

        def unflatten_1(imaged_points_1: Array):
            return api.add.at(image, self._indices, imaged_points_1)

        unflatten_all = vmap_all_except(unflatten_1, axis=points_axis)
        return self.base_scan.unflatten(unflatten_all(imaged_points), points_axis)

    @property
    def num_points(self) -> Tuple[int, ...]:
        return self._indices.shape[-1]

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return CoordinateSystem.CARTESIAN

    @WrappedScan.base_scan.setter
    def base_scan(self, new_base_scan):
        if not isinstance(new_base_scan, SectorScan):
            raise ValueError("base_scan must be a SectorScan.")
        WrappedScan.base_scan.fset(self, new_base_scan)
        self._points, self._indices = _get_scan_converted_points_and_indices(
            self.base_scan
        )

    def copy(self):
        return ScanConvertedSectorScan(self.base_scan, self._points, self._indices)
