
import matplotlib.pyplot as plt

from vbeam.interpolation import IrregularSampledCoordinates, LinearCoordinates
from spekk import ops

ops.backend.set_backend("numpy")

def plot_linear_coordinates(index, n_samples):
    # For linear sampels
    x = ops.linspace(0, 9, 10, dim='depth')

    linear_coordinates = LinearCoordinates(ops.min(x), ops.max(x), x.size)
    # x_in = ops.linspace(0,9, 37, dim='depth')
    x_in = ops.linspace(-2, 9, 45, dim='depth')
    
    indices_info = linear_coordinates.get_nearest_indices(x_in, n_samples)

    # nearest_indices = indices_info.indices[index]
    nearest_indices_positions = indices_info.indices_positions[index]

    plt.figure()
    plt.plot(x.data, x.data, '.', color='k', label='new samples')
    plt.plot(x_in.data, x_in.data, 'o', label='new samples',  mfc='none')

    for ii in range(n_samples):
        plt.plot(x_in[index], float(nearest_indices_positions[ii].data), '.', color='g')
    plt.show()


def plot_irregular_sampled_coordinates(index, n_samples):
    # For linear sampels
    x = ops.array([0,0.5,1,2,2.4,3,4,5,6.6,9], dims=['depth_2'])

    irregular_sampled_coordinate = IrregularSampledCoordinates(ops.min(x), ops.max(x), x, dim='depth_2')
    # x_in = ops.linspace(0, 9, 37, dim='depth')
    x_in = ops.linspace(-2, 11, 53, dim='xs')

    indices_info = irregular_sampled_coordinate.get_nearest_indices2(x_in, n_samples)
    
    # nearest_indices = indices_info.indices[index]
    nearest_indices_positions = indices_info.indices_positions[index]
    
    plt.figure()
    plt.plot(x.data, x.data, '.', color='k', label='new samples')
    plt.plot(x_in.data, x_in.data, 'o', label='new samples',  mfc='none')

    for ii in range(n_samples):
        plt.plot(x_in[index], float(nearest_indices_positions[ii].data), '.', color='g')
    plt.title(f"{nearest_indices_positions.data}, \n {nearest_indices_positions.data-x_in[index:index+1]}")
    plt.show()

if __name__== '__main__':
    
    n_samples = 4
    index = 30

    # plot_linear_coordinates(index, n_samples)
    plot_irregular_sampled_coordinates(index, n_samples)

    a = 1
    