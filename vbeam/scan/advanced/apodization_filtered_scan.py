from typing import Optional, Sequence, Tuple

from fastmath import Array, field, ops
from spekk import Spec

from vbeam.apodization.util import get_apodization_values
from vbeam.core import Apodization, ProbeGeometry, WaveData
from vbeam.fastmath import backend_manager
from vbeam.scan.advanced.base import ExtraDimsScanMixin, WrappedScan
from vbeam.scan.base import Scan
from vbeam.util.transformations import *
from vbeam.util.vmap import vmap_all_except


def get_point_indices_for_transmits(
    apodization_values: Array,
    threshold: float,
) -> Tuple[Array, Array]:
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
    num_focused_points = ops.sum(masks, -1)
    max_num_focused_points = ops.max(num_focused_points)

    # We use Numpy for performing the actual masking because that is hard to do on GPUs
    with backend_manager.using_backend("numpy"):

        def mask1(mask):
            masked_indices = ops.arange(mask.size)[mask]
            num_masked_indices = ops.sum(mask)
            # We set padding-indices to -1 (these will be ignored because of indices_mask)
            padding = ops.ones((max_num_focused_points - num_masked_indices,)) * -1
            return ops.concatenate([masked_indices, padding])

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
    return ops.array(indices), ops.array(indices_mask)


def recombine1(
    image: Array,
    imaged_points: Array,
    indices: Array,
    indices_mask: Array,
    points_axis: int,
):
    @vmap_all_except(points_axis)
    def recombine(imaged_points: Array) -> Array:
        return ops.add.at(image, indices, imaged_points * indices_mask)

    return recombine(imaged_points)


class ApodizationFilteredScan(WrappedScan, ExtraDimsScanMixin):
    apodization: Apodization = field(static=True)
    probe: ProbeGeometry = field(static=True)
    sender: Array = field(static=True)
    receiver: Array = field(static=True)
    wave_data: WaveData = field(static=True)
    spec: Spec = field(static=True)
    dimensions: Sequence[str] = field(static=True)
    threshold: float = field(static=True)
    _points: Optional[Array] = None
    _indices: Optional[Array] = None
    _indices_mask: Optional[Array] = None

    def __post_init__(self):
        # If any of these values are None it means we are constructing a new instance
        # (contra just copying an existing one) and we need to compute the indices.
        if self._points is None or self._indices is None or self._indices_mask is None:
            self._recompute_indices()

    def get_points(self, flatten: bool = True) -> Array:
        if not flatten:
            raise ValueError(
                f"{self.__class__.__name__} only supports getting flattened points."
            )
        return self._points

    def unflatten(
        self,
        imaged_points: Array,
        transmits_axis: int,
        points_axis: int,
    ) -> Array:
        recombine = ops.vmap(
            recombine1,
            [None, transmits_axis, 0, 0, None],
        )
        image = ops.zeros((self.base_scan.num_points,), dtype=imaged_points.dtype)
        recombined_points = recombine(
            image,
            imaged_points,
            self._indices,
            self._indices_mask,
            points_axis - 1 if points_axis > transmits_axis else points_axis,
        )
        recombined_points = ops.sum(recombined_points, 0)
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
            self.probe,
            self.sender,
            self.receiver,
            points,
            self.wave_data,
            self.spec,
            self.dimensions,
            average=True,
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
            self.probe,
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
            setup.probe,
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
