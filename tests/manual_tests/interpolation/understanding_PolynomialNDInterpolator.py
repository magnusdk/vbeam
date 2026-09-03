"""Step through PolynomialNDInterpolator in a debugger.

Creates a small 2D grid and interpolates a single query point so every
intermediate value is easy to inspect.

Set a breakpoint inside PolynomialNDInterpolator.__call__ (in
nd_interpolator.py) or anywhere in this file and run with the debugger.
"""

from spekk import ops

ops.backend.set_backend("numpy")

from vbeam.interpolation import PolynomialNDInterpolator, LinearCoordinate

# --- Build a tiny 4x5 grid with known values ---
# data[z, x] = z * 10 + x  so values are easy to verify by hand.
data = ops.reshape(
    ops.arange(20, dtype="float32"), (4, 5), dims=["z", "x"]
)
print("Data (4 z-rows × 5 x-cols):")
print(data)
print(f"  dim_sizes: {data.dim_sizes}")

coords = {
    "z": LinearCoordinate(start=0.0, stop=3.0, size=4),  # indices 0..3
    "x": LinearCoordinate(start=0.0, stop=4.0, size=5),  # indices 0..4
}

interp = PolynomialNDInterpolator(
    coordinates=coords, data=data, fill_value=0.0, n_samples=2
)

# --- Single scalar query ---
# Query at z=1.5, x=2.5  →  should land exactly between 4 grid points:
#   data[1,2]=12, data[1,3]=13, data[2,2]=17, data[2,3]=18
# Linear interp: 0.25*(12+13+17+18) = 15.0
xi = {"z": ops.array(1.5), "x": ops.array(2.5)}

# Put a breakpoint on the next line and step into interp(xi)
result = interp(xi)

print(f"\nQuery: z=1.5, x=2.5")
print(f"Result: {result}  (expected 15.0)")

# --- Small batch query ---
xi_batch = {
    "z": ops.linspace(0.0, 3.0, 4, dim="qz"),
    "x": ops.linspace(0.0, 4.0, 5, dim="qx"),
}

result_batch = interp(xi_batch)
print(f"\nBatch query (4 qz × 5 qx):")
print(result_batch)
print(f"  dim_sizes: {result_batch.dim_sizes}")
print(f"  (should exactly reproduce the original data)")
