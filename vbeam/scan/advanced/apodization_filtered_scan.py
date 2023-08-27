from typing import Optional, Sequence, Tuple

from spekk import Spec

from vbeam.apodization.util import get_apodization_values
from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import backend_manager
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.scan.advanced.base import ExtraDimsScanMixin, WrappedScan
from vbeam.scan.base import Scan
from vbeam.util.transformations import *
from vbeam.util.vmap import vmap_all_except


def get_point_indices_for_transmits(
    apodization_values: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the indices (and indices mask which is explained further
    down) for the points in an image that are to be imaged for a given transmit. Which
    points that are selected to be imaged are determined by the given apodization object
    and threshold.

    The shape of the returned arrays is (num_transmits, num_points_per_transmit).

    If the apodization value for a given point is greater than threshold then that point
    will be imaged for the given transmit.

    ## Explanation of indices_mask:
    It is important that the size of a dimension in an array is consistent in order to
    compile it using XLA. However, the number of points to be imaged may differ between
    transmits — some transmits may have fewer or more points in focus than other
    transmits.

    We add extra "padding"-points to transmits that have fewer points than the maximum
    number of points for a transmit. This padding has to be removed afterwards, so we
    also return indices_mask which is 0 for points that are added as padding.

    For example, if the number of points for 3 different transmits were 4, 6, and 5,
    then padding would be added so that 6 points were imaged for each transmit. The
    indices_mask would be [1,1,1,1,0,0], [1,1,1,1,1,1], , [1,1,1,1,1,0], respectively.
    Notice how the first 4 values in the mask for the first transmit are 1s while the
    two remaining are 0s.
    """

    # Apply thresholding to get the masks.
    masks = apodization_values >= threshold

    # Get the maximum number of points to image.
    num_focused_points = np.sum(masks, -1)
    max_num_focused_points = np.max(num_focused_points)

    # We use Numpy for performing the actual masking because that is hard to do on GPUs
    with backend_manager.using_backend("numpy"):

        def mask1(mask):
            masked_indices = np.arange(mask.size)[mask]
            num_masked_indices = np.sum(mask)
            # We set padding-indices to -1 (these will be ignored because of indices_mask)
            padding = np.ones((max_num_focused_points - num_masked_indices,)) * -1
            return np.concatenate([masked_indices, padding])

        m_dimensions = [f"dim{i}" for i in range(masks.ndim)]
        get_indices = compose(
            mask1,
            # Loop over all dimensions except the last one, which is expected to be the
            # points dimension.
            *[ForAll(dim) for dim in m_dimensions[:-1]],
        ).build(Spec({"mask": m_dimensions}))
        indices = get_indices(mask=masks)

    indices = indices.astype("int32")
    indices_mask = indices != -1
    return np.array(indices), np.array(indices_mask)


def recombine1(
    image: np.ndarray,
    imaged_points: np.ndarray,
    indices: np.ndarray,
    indices_mask: np.ndarray,
    points_axis: int,
):
    @vmap_all_except(points_axis)
    def recombine(imaged_points: np.ndarray) -> np.ndarray:
        return np.add.at(image, indices, imaged_points * indices_mask)

    return recombine(imaged_points)


@traceable_dataclass(
    ("_points", "_indices", "_indices_mask"),
    (
        "base_scan",
        "apodization",
        "sender",
        "receiver",
        "wave_data",
        "spec",
        "dimensions",
        "threshold",
    ),
)
class ApodizationFilteredScan(WrappedScan, ExtraDimsScanMixin):
    def __init__(
        self,
        base_scan: Scan,
        apodization: Apodization,
        sender: ElementGeometry,
        receiver: ElementGeometry,
        wave_data: WaveData,
        spec: Spec,
        dimensions: Sequence[str],
        threshold: float,
        _points: Optional[np.ndarray] = None,
        _indices: Optional[np.ndarray] = None,
        _indices_mask: Optional[np.ndarray] = None,
    ):
        # We need to store these values in case we need to recompute the indices.
        # For example, if a user resizes the scan then the indices must be recomputed.
        self._base_scan = base_scan
        self.apodization = apodization
        self.sender = sender
        self.receiver = receiver
        self.wave_data = wave_data
        self.spec = spec
        self.dimensions = dimensions
        self.threshold = threshold

        # If any of these values are None it means we are constructing a new instance
        # (contra just copying an existing one) and we need to compute the indices.
        self._points = _points
        self._indices = _indices
        self._indices_mask = _indices_mask
        if _points is None or _indices is None or _indices_mask is None:
            self._recompute_indices()

    def get_points(self, flatten: bool = True) -> np.ndarray:
        if not flatten:
            raise ValueError(
                f"{self.__class__.__name__} only supports getting flattened points."
            )
        return self._points

    def unflatten(
        self,
        imaged_points: np.ndarray,
        transmits_axis: int,
        points_axis: int,
    ) -> np.ndarray:
        recombine = np.vmap(
            recombine1,
            [None, transmits_axis, 0, 0, None],
        )
        image = np.zeros((self.base_scan.num_points,), dtype=imaged_points.dtype)
        recombined_points = recombine(
            image,
            imaged_points,
            self._indices,
            self._indices_mask,
            points_axis - 1 if points_axis > transmits_axis else points_axis,
        )
        recombined_points = np.sum(recombined_points, 0)
        if points_axis > transmits_axis:
            points_axis -= 1
        return self.base_scan.unflatten(recombined_points, points_axis)

    @property
    def num_points(self) -> Tuple[int, ...]:
        return self._indices.shape[-1]

    @WrappedScan.base_scan.setter
    def base_scan(self, new_base_scan):
        WrappedScan.base_scan.fset(self, new_base_scan)
        self._recompute_indices()

    def _recompute_indices(self):
        points = self.base_scan.get_points()
        apodization_values = get_apodization_values(
            self.apodization,
            self.sender,
            points,
            self.receiver,
            self.wave_data,
            self.spec,
            self.dimensions,
            sum_fn=np.mean,
        )

        points = self.base_scan.get_points()
        self._indices, self._indices_mask = get_point_indices_for_transmits(
            apodization_values,
            self.threshold,
        )
        self._points = points[self._indices]

    def copy(self):
        return ApodizationFilteredScan(
            self.base_scan,
            self.apodization,
            self.sender,
            self.receiver,
            self.wave_data,
            self.threshold,
            self._points,
            self._indices,
            self._indices_mask,
        )

    @staticmethod
    def from_setup(
        setup,
        dimensions: Sequence[str],
        threshold: float = 0.1,
        scan: Optional[Scan] = None,
    ) -> "ApodizationFilteredScan":
        from vbeam.data_importers import SignalForPointSetup

        if not isinstance(setup, SignalForPointSetup):
            raise TypeError(
                f"Expected a SignalForPointSetup, but type(setup) is {type(setup)!r}."
            )
        return ApodizationFilteredScan(
            scan if scan is not None else setup.scan,
            setup.apodization,
            setup.sender,
            setup.receiver,
            setup.wave_data,
            setup.spec,
            dimensions,
            threshold,
        )

    @property
    def flattened_points_dimensions(self) -> Sequence[str]:
        return self.dimensions

    @property
    def required_dimensions_for_unflatten(self) -> Sequence[str]:
        return self.dimensions
