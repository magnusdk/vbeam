# Raxterize is a library for rasterizing polygons in JAX
Raxterize was created so that we could have a general way of scan-converting an image to cartesian coordinates. At the moment it is really slow and should probably not be used.

There are 2 steps to rasterization in Raxterize:
1. Convert the beamformed image to polygons
2. Interpolate points into those polygons

For each point in the new grid we attempt to interpolate into every polygon, so the time complexity is O(n^2).