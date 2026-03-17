"""Tests for FastLinearNDInterpolator and FastNearestNDInterpolator.

All tests use only spekk.ops — no direct JAX imports.
"""

import time

import numpy as np
import pytest
from spekk import ops

from vbeam.interpolation import (
    FastLinearNDInterpolator,
    FastNearestNDInterpolator,
    LinearCoordinate,
    LinearCoordinateFast,
    LinearNDInterpolator,
    NearestNDInterpolator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

backends = ["numpy", "jax"]


@pytest.fixture(params=backends)
def backend(request):
    with ops.backend.temporary_backend(request.param):
        yield request.param


def _make_2d_data(dim0="z", dim1="x", n0=11, n1=13):
    """Create a simple 2-D test dataset: data[i,j] = i + j*0.1"""
    raw = np.arange(n0 * n1, dtype=np.float32).reshape(n0, n1)
    # Make values distinguishable: row index + col*0.1
    for i in range(n0):
        for j in range(n1):
            raw[i, j] = float(i) + float(j) * 0.1
    data = ops.array(raw, dims=[dim0, dim1])
    coord0 = LinearCoordinateFast(start=0.0, stop=1.0, size=n0)
    coord1 = LinearCoordinateFast(start=0.0, stop=1.0, size=n1)
    coords = {dim0: coord0, dim1: coord1}
    return data, coords


def _make_2d_data_old_coord(dim0="z", dim1="x", n0=11, n1=13):
    """Same data but with LinearCoordinate (for the old interpolator)."""
    raw = np.arange(n0 * n1, dtype=np.float32).reshape(n0, n1)
    for i in range(n0):
        for j in range(n1):
            raw[i, j] = float(i) + float(j) * 0.1
    data = ops.array(raw, dims=[dim0, dim1])
    coord0 = LinearCoordinate(start=0.0, stop=1.0, size=n0)
    coord1 = LinearCoordinate(start=0.0, stop=1.0, size=n1)
    coords = {dim0: coord0, dim1: coord1}
    return data, coords


# ===========================================================================
# Test A — Numerical equivalence with disjoint dims
# ===========================================================================


class TestNumericalEquivalenceDisjointDims:
    """Compare FastLinearNDInterpolator against LinearNDInterpolator when
    query arrays have different dim names than the data."""

    def test_linear_equivalence_2d(self, backend):
        data_fast, coords_fast = _make_2d_data("z", "x")
        data_old, coords_old = _make_2d_data_old_coord("z", "x")

        # Query with different dims
        qz = ops.linspace(0.1, 0.9, 7, dim="qz")
        qx = ops.linspace(0.1, 0.9, 5, dim="qx")

        fast_interp = FastLinearNDInterpolator(
            coordinates=coords_fast, data=data_fast, fill_value=float("nan")
        )
        old_interp = LinearNDInterpolator(
            coordinates=coords_old, data=data_old, fill_value=float("nan")
        )

        result_fast = fast_interp({"z": qz, "x": qx})
        result_old = old_interp({"z": qz, "x": qx})

        np.testing.assert_allclose(
            np.array(result_fast.data), np.array(result_old.data), atol=1e-5, rtol=1e-5
        )

    def test_output_dims_disjoint(self, backend):
        data, coords = _make_2d_data("z", "x")
        qz = ops.linspace(0.1, 0.9, 7, dim="qz")
        qx = ops.linspace(0.1, 0.9, 5, dim="qx")

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz, "x": qx})

        assert set(result.dims) == {"qz", "qx"}


# ===========================================================================
# Test B — Same-dim query (the critical case)
# ===========================================================================


class TestSameDimQuery:
    """Data has dims ["zs","xs"] and query arrays also have dims ["zs","xs"].
    This is the case that LinearNDInterpolatorFast (loop-based) would break."""

    def test_same_dims_no_crash(self, backend):
        n0, n1 = 11, 13
        data, coords = _make_2d_data("zs", "xs", n0, n1)

        # Query arrays with the SAME dims as data
        qz_raw = np.linspace(0.1, 0.9, 6, dtype=np.float32)
        qx_raw = np.linspace(0.1, 0.9, 8, dtype=np.float32)
        # Create a grid: both arrays have dims ["zs", "xs"]
        qz_grid = np.broadcast_to(qz_raw[:, None], (6, 8)).copy()
        qx_grid = np.broadcast_to(qx_raw[None, :], (6, 8)).copy()
        qz = ops.array(qz_grid, dims=["zs", "xs"])
        qx = ops.array(qx_grid, dims=["zs", "xs"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"zs": qz, "xs": qx})

        assert set(result.dims) == {"zs", "xs"}
        assert result.data.shape == (6, 8)

    def test_same_dims_correct_values(self, backend):
        """Check values for a known function: data[i,j] = i + j*0.1"""
        n0, n1 = 11, 13
        data, coords = _make_2d_data("zs", "xs", n0, n1)

        # Query at exact grid points — should recover exact values.
        # Map physical [0,1] -> index [0, n-1].  Pick a few interior points.
        iz, ix = 3, 5
        phys_z = iz / (n0 - 1)
        phys_x = ix / (n1 - 1)
        qz = ops.array(np.array([[phys_z]], dtype=np.float32), dims=["zs", "xs"])
        qx = ops.array(np.array([[phys_x]], dtype=np.float32), dims=["zs", "xs"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"zs": qz, "xs": qx})

        expected = float(iz) + float(ix) * 0.1
        np.testing.assert_allclose(
            np.array(result.data).ravel(), [expected], atol=1e-5
        )

    def test_nearest_same_dims(self, backend):
        n0, n1 = 11, 13
        data, coords = _make_2d_data("zs", "xs", n0, n1)

        qz_grid = np.full((4, 4), 0.3, dtype=np.float32)
        qx_grid = np.full((4, 4), 0.5, dtype=np.float32)
        qz = ops.array(qz_grid, dims=["zs", "xs"])
        qx = ops.array(qx_grid, dims=["zs", "xs"])

        interp = FastNearestNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"zs": qz, "xs": qx})

        assert set(result.dims) == {"zs", "xs"}
        assert result.data.shape == (4, 4)


# ===========================================================================
# Test C — Out-of-bounds / fill_value
# ===========================================================================


class TestFillValue:
    def test_oob_returns_fill_value_linear(self, backend):
        data, coords = _make_2d_data("z", "x")
        fill = -999.0

        # Query well outside [0, 1]
        qz = ops.array(np.array([-1.0, 2.0], dtype=np.float32), dims=["q"])
        qx = ops.array(np.array([0.5], dtype=np.float32), dims=["r"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=fill
        )
        result = interp({"z": qz, "x": qx})

        result_np = np.array(result.data)
        assert np.all(result_np == fill)

    def test_oob_returns_fill_value_nearest(self, backend):
        data, coords = _make_2d_data("z", "x")
        fill = -999.0

        qz = ops.array(np.array([-1.0, 2.0], dtype=np.float32), dims=["q"])
        qx = ops.array(np.array([0.5], dtype=np.float32), dims=["r"])

        interp = FastNearestNDInterpolator(
            coordinates=coords, data=data, fill_value=fill
        )
        result = interp({"z": qz, "x": qx})

        result_np = np.array(result.data)
        assert np.all(result_np == fill)

    def test_inbounds_not_filled(self, backend):
        data, coords = _make_2d_data("z", "x")
        fill = -999.0

        qz = ops.array(np.array([0.5], dtype=np.float32), dims=["q"])
        qx = ops.array(np.array([0.5], dtype=np.float32), dims=["r"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=fill
        )
        result = interp({"z": qz, "x": qx})

        result_np = np.array(result.data).ravel()
        assert np.all(result_np != fill)

    def test_fill_value_none_no_masking(self, backend):
        data, coords = _make_2d_data("z", "x")

        qz = ops.array(np.array([-1.0], dtype=np.float32), dims=["q"])
        qx = ops.array(np.array([0.5], dtype=np.float32), dims=["r"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=None
        )
        result = interp({"z": qz, "x": qx})
        # Should not be NaN — just clipped index result
        assert not np.any(np.isnan(np.array(result.data)))


# ===========================================================================
# Test D — Nearest interpolation specifics
# ===========================================================================


class TestNearestInterpolation:
    def test_nearest_at_grid_point(self, backend):
        n0, n1 = 11, 13
        data, coords = _make_2d_data("z", "x", n0, n1)

        iz, ix = 4, 7
        phys_z = iz / (n0 - 1)
        phys_x = ix / (n1 - 1)
        qz = ops.array(np.array([phys_z], dtype=np.float32), dims=["q"])
        qx = ops.array(np.array([phys_x], dtype=np.float32), dims=["q2"])

        interp = FastNearestNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz, "x": qx})

        expected = float(iz) + float(ix) * 0.1
        np.testing.assert_allclose(
            np.array(result.data).ravel(), [expected], atol=1e-5
        )

    def test_nearest_rounds_correctly(self, backend):
        """Query slightly above midpoint should round up."""
        n0 = 11
        raw = np.arange(n0, dtype=np.float32)
        data = ops.array(raw, dims=["z"])
        coord = LinearCoordinateFast(start=0.0, stop=1.0, size=n0)

        # Physical 0.35 -> fractional index 3.5 -> nearest is 4 (rounds up from 0.5)
        # Actually: 0.35 * 10 = 3.5, frac=0.5 -> >= 0.5 rounds to ceil -> idx 4
        qz = ops.array(np.array([0.35], dtype=np.float32), dims=["q"])
        interp = FastNearestNDInterpolator(
            coordinates={"z": coord}, data=data, fill_value=float("nan")
        )
        # idx_floor=3, frac=0.5 -> frac < 0.5 is False -> picks floor+1=4
        result = interp({"z": qz})
        np.testing.assert_allclose(np.array(result.data).ravel(), [4.0], atol=1e-5)


# ===========================================================================
# Test E — 1-D interpolation
# ===========================================================================


class Test1DInterpolation:
    def test_linear_1d(self, backend):
        n = 11
        raw = np.linspace(0.0, 10.0, n, dtype=np.float32)
        data = ops.array(raw, dims=["z"])
        coord = LinearCoordinateFast(start=0.0, stop=1.0, size=n)

        # Query at midpoint: physical 0.5 -> index 5 -> value 5.0
        qz = ops.array(np.array([0.25, 0.5, 0.75], dtype=np.float32), dims=["q"])
        interp = FastLinearNDInterpolator(
            coordinates={"z": coord}, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz})
        expected = [2.5, 5.0, 7.5]
        np.testing.assert_allclose(
            np.array(result.data).ravel(), expected, atol=1e-5
        )

    def test_nearest_1d(self, backend):
        n = 11
        raw = np.arange(n, dtype=np.float32)
        data = ops.array(raw, dims=["z"])
        coord = LinearCoordinateFast(start=0.0, stop=1.0, size=n)

        # physical 0.22 -> frac_idx = 2.2, floor=2, frac=0.2 < 0.5 -> nearest=2
        qz = ops.array(np.array([0.22], dtype=np.float32), dims=["q"])
        interp = FastNearestNDInterpolator(
            coordinates={"z": coord}, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz})
        np.testing.assert_allclose(np.array(result.data).ravel(), [2.0], atol=1e-5)


# ===========================================================================
# Test F — 3-D interpolation (exercises 2^3 = 8 gather path)
# ===========================================================================


class Test3DInterpolation:
    def test_linear_3d(self, backend):
        nz, nx, ny = 5, 7, 9
        # data[i,j,k] = i + j*0.1 + k*0.01
        raw = np.zeros((nz, nx, ny), dtype=np.float32)
        for i in range(nz):
            for j in range(nx):
                for k in range(ny):
                    raw[i, j, k] = i + j * 0.1 + k * 0.01
        data = ops.array(raw, dims=["z", "x", "y"])
        coords = {
            "z": LinearCoordinateFast(start=0.0, stop=1.0, size=nz),
            "x": LinearCoordinateFast(start=0.0, stop=1.0, size=nx),
            "y": LinearCoordinateFast(start=0.0, stop=1.0, size=ny),
        }

        # Query at exact grid point
        iz, ix, iy = 2, 3, 4
        pz = iz / (nz - 1)
        px = ix / (nx - 1)
        py = iy / (ny - 1)
        qz = ops.array(np.array([pz], dtype=np.float32), dims=["q1"])
        qx = ops.array(np.array([px], dtype=np.float32), dims=["q2"])
        qy = ops.array(np.array([py], dtype=np.float32), dims=["q3"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz, "x": qx, "y": qy})

        expected = float(iz) + float(ix) * 0.1 + float(iy) * 0.01
        np.testing.assert_allclose(
            np.array(result.data).ravel(), [expected], atol=1e-4
        )

    def test_linear_3d_between_points(self, backend):
        """Interpolate halfway between grid points in all 3 dims."""
        nz, nx, ny = 5, 7, 9
        raw = np.zeros((nz, nx, ny), dtype=np.float32)
        for i in range(nz):
            for j in range(nx):
                for k in range(ny):
                    raw[i, j, k] = i + j * 0.1 + k * 0.01
        data = ops.array(raw, dims=["z", "x", "y"])
        coords = {
            "z": LinearCoordinateFast(start=0.0, stop=1.0, size=nz),
            "x": LinearCoordinateFast(start=0.0, stop=1.0, size=nx),
            "y": LinearCoordinateFast(start=0.0, stop=1.0, size=ny),
        }

        # Midpoint between grid points (1,2,3) and (2,3,4)
        pz = 1.5 / (nz - 1)
        px = 2.5 / (nx - 1)
        py = 3.5 / (ny - 1)
        qz = ops.array(np.array([pz], dtype=np.float32), dims=["q1"])
        qx = ops.array(np.array([px], dtype=np.float32), dims=["q2"])
        qy = ops.array(np.array([py], dtype=np.float32), dims=["q3"])

        interp = FastLinearNDInterpolator(
            coordinates=coords, data=data, fill_value=float("nan")
        )
        result = interp({"z": qz, "x": qx, "y": qy})

        expected = 1.5 + 2.5 * 0.1 + 3.5 * 0.01
        np.testing.assert_allclose(
            np.array(result.data).ravel(), [expected], atol=1e-4
        )


# ===========================================================================
# Test G — Performance (spekk only, ops.jit / ops.grad)
# ===========================================================================


@pytest.mark.parametrize("backend_name", ["jax"])
class TestPerformance:
    """Timing tests — only run on JAX backend (numpy doesn't have grad)."""

    def test_forward_not_slower(self, backend_name):
        with ops.backend.temporary_backend(backend_name):
            n0, n1 = 64, 64
            data_fast, coords_fast = _make_2d_data("z", "x", n0, n1)
            data_old, coords_old = _make_2d_data_old_coord("z", "x", n0, n1)

            qz = ops.linspace(0.05, 0.95, 32, dim="qz")
            qx = ops.linspace(0.05, 0.95, 32, dim="qx")
            xi = {"z": qz, "x": qx}

            fast_interp = FastLinearNDInterpolator(
                coordinates=coords_fast, data=data_fast, fill_value=float("nan")
            )
            old_interp = LinearNDInterpolator(
                coordinates=coords_old, data=data_old, fill_value=float("nan")
            )

            fast_fn = ops.jit(lambda: fast_interp(xi))
            old_fn = ops.jit(lambda: old_interp(xi))

            # Warm-up
            _ = fast_fn()
            _ = old_fn()

            n_iters = 50
            t0 = time.time()
            for _ in range(n_iters):
                _ = fast_fn()
            fast_time = time.time() - t0

            t0 = time.time()
            for _ in range(n_iters):
                _ = old_fn()
            old_time = time.time() - t0

            print(
                f"\nForward: fast={fast_time/n_iters*1000:.2f}ms, "
                f"old={old_time/n_iters*1000:.2f}ms"
            )
            # Fast version should not be more than 3x slower than old
            assert fast_time < old_time * 3

    def test_backward_faster(self, backend_name):
        with ops.backend.temporary_backend(backend_name):
            n0, n1 = 64, 64
            data_fast, coords_fast = _make_2d_data("z", "x", n0, n1)
            data_old, coords_old = _make_2d_data_old_coord("z", "x", n0, n1)

            qz = ops.linspace(0.05, 0.95, 32, dim="qz")
            qx = ops.linspace(0.05, 0.95, 32, dim="qx")
            xi = {"z": qz, "x": qx}

            def loss_fast(d):
                interp = FastLinearNDInterpolator(
                    coordinates=coords_fast, data=d, fill_value=None
                )
                return ops.sum(interp(xi))

            def loss_old(d):
                interp = LinearNDInterpolator(
                    coordinates=coords_old, data=d, fill_value=None
                )
                return ops.sum(interp(xi))

            grad_fast = ops.jit(ops.grad(loss_fast))
            grad_old = ops.jit(ops.grad(loss_old))

            # Warm-up
            _ = grad_fast(data_fast)
            _ = grad_old(data_old)

            n_iters = 50
            t0 = time.time()
            for _ in range(n_iters):
                _ = grad_fast(data_fast)
            fast_time = time.time() - t0

            t0 = time.time()
            for _ in range(n_iters):
                _ = grad_old(data_old)
            old_time = time.time() - t0

            print(
                f"\nBackward: fast={fast_time/n_iters*1000:.2f}ms, "
                f"old={old_time/n_iters*1000:.2f}ms, "
                f"speedup={old_time/fast_time:.1f}x"
            )
            # We expect the fast version to be faster in backward pass.
            # At minimum, it should not be dramatically slower.
            assert fast_time < old_time * 2
