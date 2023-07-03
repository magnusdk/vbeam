import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from vbeam.core import Apodization, ElementGeometry, WaveData
from vbeam.fastmath import backend_manager
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass
from vbeam.scan import Scan
from vbeam.scan.points_optimizers.base import PointOptimizer, ShapeInfo
from vbeam.util import ensure_positive_index


def get_point_indices(
    apodization: Apodization,
    sender: ElementGeometry,
    point_pos: np.ndarray,
    receiver: ElementGeometry,
    wave_data: WaveData,
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
    vmapped_apodization = np.vmap(apodization, [None, 0, None, None])
    vmapped_apodization = np.vmap(vmapped_apodization, [None, None, None, 0])
    apodization_values = vmapped_apodization(sender, point_pos, receiver, wave_data)

    with backend_manager.using_backend("numpy"):
        masks = apodization_values > threshold
        # Get the maximum number of points to image for a transmit.
        all_axes_except_first = tuple(range(1, masks.ndim))
        num_focused_points = np.sum(masks, all_axes_except_first)
        max_num_focused_points = np.max(num_focused_points)

        def get_indices_1(mask: np.ndarray, max_focused_points: int) -> np.ndarray:
            """Return the indices of the points in focus for a single transmit, with
            padding such that its size equals max_num_focused_points."""
            masked_indices = np.arange(mask.size)[mask]
            num_masked_indices = np.sum(mask)
            # We set padding-indices to -1 (these will be ignored)
            padding = np.ones((max_focused_points - num_masked_indices,)) * -1
            return np.concatenate([masked_indices, padding])

        # TODO: get rid of for-loop.
        # Hint: `indices = np.full((masks.shape[0], max_num_focused_points), -1)`
        indices = np.zeros((masks.shape[0], max_num_focused_points))
        for i, mask in enumerate(masks):
            indices[i] = get_indices_1(mask, max_num_focused_points)

        indices = indices.astype("int32")
        indices_mask = indices != -1
        return indices, indices_mask


def recombine_points_by_indices(
    imaged_points: np.ndarray,
    indices: np.ndarray,
    indices_mask: np.ndarray,
    scan: Scan,
    *axes: int,
    image: Optional[np.ndarray] = None,
):
    axes = [ensure_positive_index(imaged_points.ndim, axis) for axis in axes]
    summed_axes = axes[:-1]
    points_axis = axes[-1]

    # Most of the logic in this function is in order to keep track of the axes. If
    # imaged_points has the same shape as indices then we could simply call
    # _recombine (step 1.) on them and be done. But since imaged_points may
    # have an arbitrary number of additional axes, we need to vectorize _recombine
    # over these additional axes (step 2.). The order of axes and indices may also
    # be different, so we need to transpose indices to match the order of axes
    # (step 3).

    # 1. Define _recombine which sums the imaged_points into a flattened array of
    # points that can be unflattened into the original scan shape.
    def _recombine(
        indices: np.ndarray, imaged_points: np.ndarray, indices_mask: np.ndarray
    ) -> np.ndarray:
        arr = (
            image
            if image is not None
            else np.zeros((scan.num_points,), dtype=imaged_points.dtype)
        )
        return np.add.at(arr, indices, imaged_points * indices_mask)

    # 2. Vectorize _recombine over the additional axes.
    num_axes_removed = 0
    for i in range(imaged_points.ndim):
        # An axis that is not the points_axis or any of the other indices-axes must
        # be vectorized over.
        if i not in axes:
            # The output axis after vmap should not change. Since summing over an
            # axis removes it from the output we also have to subtract the number of
            # axes that have been removed that came before the current one. Setting
            # out_axis to i means that the axis will be in the same position,
            # assuming no other axes have been removed. Setting it to
            # i - num_axes_removed takes the removed axes into account.
            out_axis = i - num_axes_removed
            _recombine = np.vmap(_recombine, [None, i, None], out_axes=out_axis)
        elif i in summed_axes:
            num_axes_removed += 1

    # 3. Transpose axes of indices to match the given imaged_points.
    # *axes specify what axes of imaged_points correspond to each axis in the
    # indices. However, these may not be in the same order, so we need to transpose
    # the indices first.
    transpose_axes = []
    for i in range(imaged_points.ndim):
        if i in axes:
            transpose_axes.append(axes.index(i))
    indices = np.transpose(indices, transpose_axes)
    indices_mask = np.transpose(indices_mask, transpose_axes)

    # 4. Actually perform the summation :)
    recombined_image = _recombine(indices, imaged_points, indices_mask)

    # 5. Unflatten the image to the original shape.
    # If an axis has been removed and it was in front of the points axis then the
    # points axis moves one step forward.
    for ax in summed_axes:
        if ax < points_axis:
            points_axis -= 1
    return scan.unflatten(recombined_image, points_axis)


@traceable_dataclass(("indices", "indices_mask"), ("indices_dimensions",))
class ApodizationThresholding(PointOptimizer):
    indices: np.ndarray
    indices_mask: np.ndarray
    indices_dimensions: Sequence[str]

    def reshape(self, point_pos: np.ndarray, scan: Scan):
        point_pos = point_pos.reshape(scan.num_points, 3)  # Ensure flat array
        return point_pos[self.indices]

    def recombine(self, imaged_points: np.ndarray, scan: Scan, *axes: int):
        if not isinstance(scan, Scan):
            raise TypeError(f"Expected a Scan, but type(scan) is {str(type(scan))}.")
        return recombine_points_by_indices(
            imaged_points, self.indices, self.indices_mask, scan, *axes
        )

    @property
    def shape_info(self) -> ShapeInfo:
        return ShapeInfo(
            after_reshape=[*self.indices_dimensions, "points"],
            required_for_recombine=[*self.indices_dimensions, "points"],
            after_recombine=["width", "height"],
        )

    @staticmethod
    def over_transmits(
        apodization: Apodization,
        sender: ElementGeometry,
        point_pos: Union[np.ndarray, Scan],
        receiver: ElementGeometry,
        wave_data: WaveData,
        threshold: float = 0.1,
    ) -> "ApodizationThresholding":
        """Return an ApodizationThresholding that selects only the points that have an
        apodization value above the threshold for each transmit"""
        if isinstance(point_pos, Scan):
            point_pos = point_pos.get_points()

        indices, indices_mask = get_point_indices(
            apodization, sender, point_pos, receiver, wave_data, threshold
        )
        return ApodizationThresholding(
            indices,
            indices_mask,
            ["transmits"],
        )
