from typing import Callable, Tuple

from spekk import ops


def get_window_indices(
    image_shape: Tuple[int, ...],
    window_radius: Tuple[int, ...],
) -> ops.array:
    indices = ops.stack(
        ops.meshgrid(*(ops.arange(0, s) for s in image_shape), indexing="ij"), -1
    )
    # Make it broadcastable
    indices = indices[(..., *(None for _ in range(len(window_radius))), slice(None))]
    indices_window_offset = ops.stack(
        ops.meshgrid(*(ops.arange(-r, r + 1) for r in window_radius), indexing="ij"), -1
    )
    return indices + indices_window_offset


def windowed(f: Callable, window_radius: Tuple[int, ...]) -> ops.array:
    def apply_windowed(image, *args, **kwargs):
        indices = get_window_indices(image.shape, window_radius)

        def apply1(indices):
            windowed_image = image[indices[..., 0], indices[..., 1]]
            return f(windowed_image, *args, **kwargs)

        # vmap over all axes in the image
        for _ in range(image.ndim):
            apply1 = ops.vmap(apply1, [0])
        return apply1(indices)

    return apply_windowed
