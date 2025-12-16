
# import matplotlib
# matplotlib.use('Qt5Agg')


# import matplotlib
# print(matplotlib.get_backend())

import matplotlib.pyplot as plt
from spekk import ops
from vbeam.geometry import as_cartesian, as_polar
from vbeam.geometry.coordinate_systems import polar_to_cartesian, cartesian_to_polar

# ax_range=60
# el_range=40

# azimuths = ops.linspace(-ax_range, ax_range, 30, dim="az") * ops.pi/180
# elevations = ops.linspace(-el_range, el_range, 5, dim="el") * ops.pi/180
# depths = ops.array([0, 10], dims=["depth"])

# p = ops.stack([azimuths, elevations, depths], axis="xyz")
# points = as_cartesian(p)

# print(points.dim_sizes)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(points["xyz", 0], points["xyz", 1], points["xyz", 2], marker=10)

# points = ops.merge_dims(points, ["az", "depth"], "points")
# n_points = points.dim_sizes["points"]
# n_el = points.dim_sizes["el"]
# origin = ops.zeros((n_points, n_el, 3), dims=["points","el","xyz"])

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# colors = colors*10

# for el_idx in range(n_el):
#     ax.plot([origin["xyz", 0, "el", el_idx], points["xyz", 0, "el", el_idx]],
#             [origin["xyz", 1, "el", el_idx], points["xyz", 1, "el", el_idx]],
#             [origin["xyz", 2, "el", el_idx], points["xyz", 2, "el", el_idx]],
#             color=colors[el_idx], linewidth=2)
    
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# ax.set_title('Multiple 3D Lines from Origin')
# plt.ion()
# plt.show()

# a = 1

def get_linspace_polar_coordinates():
    az_range=60
    el_range=40

    azimuths = ops.linspace(-az_range, az_range, 30, dim="az") * ops.pi/180
    elevations = ops.linspace(-el_range, el_range, 5, dim="el") * ops.pi/180
    depths = ops.array([0, 10], dims=["depth"])
    p = ops.stack([azimuths, elevations, depths], axis="az_el_depth")
    # return as_cartesian(p)
    return polar_to_cartesian(p)


def plot(points):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = colors*10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    p = ops.merge_dims(points, ["az", "el"], "points")

    for p_idx in range(p.dim_sizes["points"]):
        ax.plot([p["xyz", 0, "points", p_idx, "depth", 0], p["xyz", 0, "points", p_idx, "depth", 1]],
                [p["xyz", 1, "points", p_idx, "depth", 0], p["xyz", 1, "points", p_idx, "depth", 1]],
                [p["xyz", 2, "points", p_idx, "depth", 0], p["xyz", 2, "points", p_idx, "depth", 1]],
                # color=colors[p_idx], linewidth=2)
                color="b", linewidth=2)

    # for el_idx in range(n_el):
    #     ax.plot([origin["xyz", 0, "el", el_idx], points["xyz", 0, "el", el_idx]],
    #             [origin["xyz", 1, "el", el_idx], points["xyz", 1, "el", el_idx]],
    #             [origin["xyz", 2, "el", el_idx], points["xyz", 2, "el", el_idx]],
    #             color=colors[el_idx], linewidth=2)
        
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Multiple 3D Lines from Origin')
    plt.ion()
    plt.show()
    a = 1

if __name__ == "__main__":

    points = get_linspace_polar_coordinates()
    # plot(points)

########################################

    # Test the conversion functions
    azimuth = 60 * ops.pi / 180
    elevation = 40 * ops.pi / 180
    depth = 2.0
    polar_points = ops.stack([azimuth, elevation, depth], axis="az_el_depth")

    cartesian_points = as_cartesian(polar_points)
    # cartesian_points = polar_to_cartesian(polar_points)
    print("Cartesian Points:")
    print(cartesian_points)

    # recovered_polar_points = cartesian_to_polar(cartesian_points)
    recovered_polar_points = as_polar(cartesian_points)
    print("Recovered Polar Points:")
    print(recovered_polar_points)


    print("Original Polar Points:")
    print(ops.array([azimuth, elevation, depth], dims=["az_el_depth"]))

