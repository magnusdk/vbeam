import matplotlib
print(matplotlib.get_backend())

import matplotlib.pyplot as plt

from vbeam.interpolation import IrregularSampledCoordinates, LinearCoordinates
from spekk import ops

# ops.backend.set_backend("numpy")

def plot_linear_coordinates(index, n_samples):
    # For linear sampels
    x = ops.linspace(0, 9, 10, dim='depth')

    linear_coordinates = LinearCoordinates(ops.min(x), ops.max(x), x.size)
    # x_in = ops.linspace(0,9, 37, dim='depth')
    # x_in = ops.linspace(-2, 9, 45, dim='depth')
    x_in = ops.linspace(-2, 11, 53, dim='xs')
    
    indices_info = linear_coordinates.get_nearest_indices(x_in, n_samples)

    nearest_indices_positions = indices_info.indices_positions[index]
    nearest_indices_index = indices_info.indices[index]
    
    plt.figure()
    plt.plot(x.data, x.data, '.', color='k', label='new samples')
    plt.plot(x_in.data, x_in.data, 'o', label='new samples',  mfc='none')

    for ii in range(n_samples):
        plt.plot(x_in[index], float(nearest_indices_positions[ii].data), '.', color='g')
    plt.title(f"indices_positions={nearest_indices_positions.data} | indices_index={nearest_indices_index.data}, \n {nearest_indices_positions.data-x_in[index:index+1]}")       
    plt.show()
    return indices_info


def plot_irregular_sampled_coordinates(index, n_samples):
    x = ops.array([0,0.5,1,2,2.4,3,4,5,6.6,9], dims=['depth_2'])

    irregular_sampled_coordinate = IrregularSampledCoordinates(ops.min(x), ops.max(x), x, dim='depth_2')
    x_in = ops.linspace(-2, 11, 53, dim='xs')

    indices_info = irregular_sampled_coordinate.get_nearest_indices(x_in, n_samples)
    
    nearest_indices_positions = indices_info.indices_positions[index]
    nearest_indices_index = indices_info.indices[index]
    
    plt.ion()
    plt.figure()
    plt.plot(x.data, x.data, '.', color='k', label='new samples')
    plt.plot(x_in.data, x_in.data, 'o', label='new samples',  mfc='none')

    for ii in range(n_samples):
        plt.plot(x_in[index], float(nearest_indices_positions[ii].data), '.', color='g')
    plt.title(f"indices_positions={nearest_indices_positions.data} | indices_index={nearest_indices_index.data}, \n {nearest_indices_positions.data-x_in[index:index+1]}")
    plt.show()
    return indices_info

if __name__== '__main__':

    n_samples = 2
    # index = 44
    # index = 20
    index = 45
    ops.backend.backend_name
    
    indices = plot_irregular_sampled_coordinates(index, n_samples)

    n_samples = 2
    index = 44
    # index = 40
    # index = 8
    # index = 45
    

    # indices = plot_linear_coordinates(index, n_samples)

    ops.sum(indices.offset_distances, axis=indices.dim_name, keepdims=True)

    a = 1
    
    