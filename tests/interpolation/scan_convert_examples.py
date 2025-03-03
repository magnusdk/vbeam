
import matplotlib.pyplot as plt

from spekk import ops
from vbeam.interpolation import (
    LinearCoordinates,
    IrregularSampledCoordinates,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ops.backend.set_backend("numpy")

def get_setup(use_irregular=False):
    # Create some image. Checkerboard pattern!
    if use_irregular:
        az = ops.array([-np.pi/4, -np.pi/6, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, np.pi/6, np.pi/4], dims=["azimuths"])
    else:
        az = ops.linspace(-np.pi / 4, np.pi / 4, 8, dim="azimuths")
    r = ops.linspace(0.1, 0.9, 40, dim="depths")
    data = ops.logical_xor(ops.cos(az * 8) > 0, ops.cos(r * ops.pi * 8) > 0)
    data = data * 0.75 + 0.25

    # Get the new grid (TODO: get the bounds from azimuths and depths)
    cartesian_grid = ops.stack(
        [
            ops.linspace(-0.7, 0.7, 100, dim="xs"),
            0,
            ops.linspace(0, 1, 100, dim="zs"),
        ],
        axis="xyz",
    )
    # Get the depths and azimuths that we want to index at
    depths_cartesian_grid = ops.linalg.vector_norm(cartesian_grid, axis="xyz")
    azimuths_cartesian_grid = ops.atan2(cartesian_grid["xyz", 0], cartesian_grid["xyz", 2])

    return az, r, data, azimuths_cartesian_grid, depths_cartesian_grid

def linear_coordinates_scan_convert():
    az, r, data, azimuths_cartesian_grid, depths_cartesian_grid = get_setup()

    data_coordinates = {
        "azimuths": LinearCoordinates(az[0], az[-1], az.size),
        "depths": LinearCoordinates(r[0], r[-1], r.size),
    }
    # interpolator = LinearNDInterpolator(data_coordinates, data, fill_value=0)
    interpolator = LinearNDInterpolator(data_coordinates, data, fill_value=None)

    interpolated_data = interpolator(
        {
            "azimuths": azimuths_cartesian_grid,
            "depths": depths_cartesian_grid,
        }
    )


    # Reference from RegularGridInterpolator
    interp = RegularGridInterpolator(
        (az.data, r.data),
        data.data,
        bounds_error=False,
        fill_value=0,
        method="linear",
    )
    ref_interpolated_data = interp((azimuths_cartesian_grid, depths_cartesian_grid))

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(interpolated_data.T, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(ref_interpolated_data.T, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("vbeam")
    axes[1].set_title("scipy.interpolate")
    plt.show()

    return


def irregular_coordinates_scan_convert():
    az, r, data, azimuths_cartesian_grid, depths_cartesian_grid = get_setup(use_irregular=True)

    data_coordinates = {
        "azimuths": IrregularSampledCoordinates(az[0], az[-1], az),
        "depths": LinearCoordinates(r[0], r[-1], r.size),
    }
    interpolator = LinearNDInterpolator(data_coordinates, data, fill_value=0)

    interpolated_data = interpolator(
        {
            "azimuths": azimuths_cartesian_grid,
            "depths": depths_cartesian_grid,
        }
    )

    # Reference from RegularGridInterpolator
    interp = RegularGridInterpolator(
        (az.data, r.data),
        data.data,
        bounds_error=False,
        fill_value=0,
        method="linear",
    )
    ref_interpolated_data = interp((azimuths_cartesian_grid, depths_cartesian_grid))

    fig, axes = plt.subplots(1,3)
    axes[0].imshow(interpolated_data.T, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(ref_interpolated_data.T, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(interpolated_data.T-ref_interpolated_data.T, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("vbeam")
    axes[1].set_title("scipy.interpolate")
    axes[2].set_title("diff: vbeam - scipy.interpolate")
    
    plt.show()

    return

if __name__== '__main__':

    # linear_coordinates_scan_convert()
    irregular_coordinates_scan_convert()

    a = 1
    