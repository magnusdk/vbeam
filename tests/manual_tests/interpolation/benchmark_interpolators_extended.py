"""Extended interpolator benchmarks: backward ray tracing, compilation time,
quadratic (n_samples=3), memory usage, query density scaling, and complex data.

Complements benchmark_interpolators.py with additional angles to evaluate
interpolator performance.

Usage:
    python benchmark_interpolators_extended.py
"""

import os

bench_device = os.environ.get("BENCH_DEVICE", "gpu").lower()
if bench_device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time

import jax
import numpy as np
from spekk import ops, replace

from vbeam.delay_models.speed_of_sound import SpeedOfSoundRayTracing
from vbeam.interpolation import (
    FastLinearNDInterpolator,
    GeneralFastLinearNDInterpolator,
    LinearCoordinate,
    LinearCoordinateFast,
    LinearNDInterpolator,
    LinearNDInterpolatorFast,
)
from vbeam.interpolation.nd_interpolator import OptimizedLinearNDInterpolator
from vbeam_sim.beamformer_importers.research_replication.path_based_beamformer_importer import (
    PathBasedBeamformerImporter,
)

ops.backend.set_backend("jax")

SEP = "-" * 100
WARMUP = 3
N_ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_until_ready(value):
    if value is None:
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if hasattr(value, "data"):
        _block_until_ready(value.data)
        return
    if isinstance(value, dict):
        for v in value.values():
            _block_until_ready(v)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _block_until_ready(v)


def _time_fn(fn, n_warmup=WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        out = fn()
        _block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn()
        _block_until_ready(out)
    return (time.perf_counter() - t0) / n_iters * 1000  # ms


def _make_data(sizes, dim_names, *, dtype="float32", seed=42):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        real = rng.standard_normal([int(s) for s in sizes]).astype("float32")
        imag = rng.standard_normal([int(s) for s in sizes]).astype("float32")
        raw = (real + 1j * imag).astype(dtype)
    else:
        raw = rng.standard_normal([int(s) for s in sizes]).astype(dtype)
    return ops.array(raw, dims=list(dim_names))


def _coord(size):
    return LinearCoordinate(start=0.0, stop=1.0, size=size)


def _coord_fast(size):
    return LinearCoordinateFast(start=0.0, stop=1.0, size=size)


def _print_header(title):
    print(f"\n{title:^100}")
    print(SEP)


def _print_table(rows, ref_idx=0):
    """Print a table of (name, ms) tuples with speedup relative to ref_idx."""
    ref_ms = rows[ref_idx][1]
    print(f"  {'Interpolator':<26} | {'Time (ms)':>10} | {'Speedup':>10}")
    print("  " + "-" * 54)
    for name, ms in rows:
        speedup = ref_ms / ms if ms > 0 else float("inf")
        print(f"  {name:<26} | {ms:>8.3f}ms | {speedup:>8.2f}x")


# All linear interpolators we test (name, class, needs_fast_coords).
LINEAR_INTERPOLATORS = [
    ("LinearND (old)", LinearNDInterpolator, False),
    ("OptimizedLinearND", OptimizedLinearNDInterpolator, False),
    ("LinearNDFast", LinearNDInterpolatorFast, True),
    ("FastLinearND", FastLinearNDInterpolator, True),
    ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, False),
]


# ---------------------------------------------------------------------------
# 1. Backward pass through ray tracing
# ---------------------------------------------------------------------------


def run_ray_tracing_backward():
    _print_header("1. RAY TRACING — FORWARD + BACKWARD (gradient through SoS map)")

    importer = PathBasedBeamformerImporter()
    scan = importer.get_points_in_linear_for_optimization()

    n_elements = 128
    pitch = 0.0003
    x_pos = np.arange(n_elements) * pitch - (n_elements - 1) * pitch / 2
    ep = np.zeros((3, n_elements), dtype=np.float32)
    ep[0] = x_pos
    point1 = ops.array(ep, dims=["xyz", "tx"])
    point2 = scan.points

    n_xs, n_zs = 100, 100
    rng = np.random.default_rng(42)
    sos_np = 1540.0 + 20.0 * rng.standard_normal((n_xs, n_zs)).astype(np.float32)
    sos_map = ops.array(sos_np, dims=["xs", "zs"])

    ray_tracer_base = SpeedOfSoundRayTracing.from_scan(scan, sos_map, n_steps=64)
    coords_fast = {
        dim: LinearCoordinateFast(c.start, c.stop, c.size)
        for dim, c in ray_tracer_base.coordinates.items()
    }
    ray_tracer_fast = replace(ray_tracer_base, coordinates=coords_fast)

    test_cases = [
        ("LinearND (old)", LinearNDInterpolator, ray_tracer_base),
        ("OptimizedLinearND", OptimizedLinearNDInterpolator, ray_tracer_base),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, ray_tracer_base),
        ("LinearNDFast", LinearNDInterpolatorFast, ray_tracer_fast),
        ("FastLinearND", FastLinearNDInterpolator, ray_tracer_fast),
    ]

    for unroll_value in [1, True]:
        unroll_label = "unroll=True (full)" if unroll_value is True else f"unroll={unroll_value}"
        print(f"\n  n_steps: 64  |  SoS map: {n_xs}x{n_zs}  |  128 elements → 100x100 scan  |  {unroll_label}")
        print()

        # Forward
        print(f"  --- Forward ({unroll_label}) ---")
        fwd_results = []
        for name, interp_type, base_rt in test_cases:
            rt = replace(base_rt, interpolator_type=interp_type, unroll=unroll_value)

            @ops.jit
            def fwd(p1, p2, _rt=rt):
                return _rt.get_delay_between(p1, p2)

            ms = _time_fn(lambda: fwd(point1, point2), n_iters=20)
            fwd_results.append((name, ms))
        _print_table(fwd_results)

        # Backward (gradient of sum of delays w.r.t. SoS map)
        print(f"\n  --- Backward ({unroll_label}, grad w.r.t. SoS map) ---")
        bwd_results = []
        for name, interp_type, base_rt in test_cases:
            rt = replace(base_rt, interpolator_type=interp_type, unroll=unroll_value)

            def loss(sos, _rt=rt, _p1=point1, _p2=point2):
                rt_local = replace(_rt, speed_of_sound=sos)
                return ops.sum(rt_local.get_delay_between(_p1, _p2))

            try:
                grad_fn = ops.jit(ops.grad(loss))
                ms = _time_fn(lambda: grad_fn(sos_map), n_iters=20)
                bwd_results.append((name, ms))
            except Exception as e:
                print(f"  {name:<26} | FAILED: {e}")
                bwd_results.append((name, float("inf")))

        _print_table(bwd_results)

    print(SEP)


# ---------------------------------------------------------------------------
# 2. Compilation time
# ---------------------------------------------------------------------------


def run_compilation_time():
    _print_header("2. JIT COMPILATION TIME")

    sizes_2d = (256, 256)
    sizes_3d = (64, 64, 64)
    n_query = 64

    configs = [
        ("2D (256²)", 2, sizes_2d, ["z", "x"], ["qz", "qx"]),
        ("3D (64³)", 3, sizes_3d, ["z", "x", "y"], ["qz", "qx", "qy"]),
    ]

    for label, ndim, sizes, dim_names, q_dim_names in configs:
        print(f"\n  Config: {label}")
        data = _make_data(sizes, dim_names)
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {
            d: ops.linspace(0.05, 0.95, n_query, dim=qd)
            for d, qd in zip(dim_names, q_dim_names)
        }

        print(f"  {'Interpolator':<30} | {'Compile fwd (ms)':>16} | {'Compile bwd (ms)':>16}")
        print("  " + "-" * 70)

        for name, cls, needs_fast in LINEAR_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            # Clear JAX caches to force recompilation.
            jax.clear_caches()

            # Forward compilation
            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            t0 = time.perf_counter()
            out = fwd(data, xi)
            _block_until_ready(out)
            fwd_compile_ms = (time.perf_counter() - t0) * 1000

            # Backward compilation
            jax.clear_caches()

            def make_loss(_cls, _c):
                def loss(d, x):
                    return ops.sum(_cls(coordinates=_c, data=d, fill_value=None)(x))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                t0 = time.perf_counter()
                out = grad_fn(data, xi)
                _block_until_ready(out)
                bwd_compile_ms = (time.perf_counter() - t0) * 1000
                bwd_str = f"{bwd_compile_ms:>14.1f}ms"
            except Exception:
                bwd_str = f"{'FAILED':>16}"

            print(f"  {name:<30} | {fwd_compile_ms:>14.1f}ms | {bwd_str}")

        # Also measure GeneralFast with n_samples=3 (quadratic).
        jax.clear_caches()
        cls = GeneralFastLinearNDInterpolator

        @ops.jit
        def fwd_q(d, x):
            return cls(coordinates=coords, data=d, fill_value=None, n_samples=3)(x)

        t0 = time.perf_counter()
        out = fwd_q(data, xi)
        _block_until_ready(out)
        fwd_q_ms = (time.perf_counter() - t0) * 1000

        jax.clear_caches()

        def loss_q(d, x):
            return ops.sum(
                cls(coordinates=coords, data=d, fill_value=None, n_samples=3)(x)
            )

        try:
            grad_fn_q = ops.jit(ops.grad(loss_q))
            t0 = time.perf_counter()
            out = grad_fn_q(data, xi)
            _block_until_ready(out)
            bwd_q_ms = (time.perf_counter() - t0) * 1000
            bwd_q_str = f"{bwd_q_ms:>14.1f}ms"
        except Exception:
            bwd_q_str = f"{'FAILED':>16}"

        print(f"  {'GeneralFast n=3 (quadratic)':<30} | {fwd_q_ms:>14.1f}ms | {bwd_q_str}")

    print(SEP)


# ---------------------------------------------------------------------------
# 3. Quadratic (n_samples=3) vs linear (n_samples=2)
# ---------------------------------------------------------------------------


def run_quadratic_benchmark():
    _print_header("3. QUADRATIC (n_samples=3) vs LINEAR (n_samples=2) — GeneralFastLinearND")

    configs = [
        ("2D (512²)", (512, 512), ["z", "x"], ["qz", "qx"], 256),
        ("3D (128³)", (128, 128, 128), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
    ]

    for label, sizes, dim_names, q_dim_names, n_query in configs:
        print(f"\n  Config: {label}  |  n_query: {n_query}")
        data = _make_data(sizes, dim_names)
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        xi = {
            d: ops.linspace(0.05, 0.95, n_query, dim=qd)
            for d, qd in zip(dim_names, q_dim_names)
        }

        results_fwd = []
        results_bwd = []

        for n_samples, mode_label in [(1, "nearest (n=1)"), (2, "linear (n=2)"), (3, "quadratic (n=3)")]:
            cls = GeneralFastLinearNDInterpolator

            @ops.jit
            def fwd(d, x, _n=n_samples):
                return cls(coordinates=coords, data=d, fill_value=None, n_samples=_n)(x)

            fwd_ms = _time_fn(lambda: fwd(data, xi))
            results_fwd.append((f"GeneralFast {mode_label}", fwd_ms))

            def make_loss(_n):
                def loss(d, x):
                    return ops.sum(
                        cls(coordinates=coords, data=d, fill_value=None, n_samples=_n)(x)
                    )
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(n_samples)))
                bwd_ms = _time_fn(lambda: grad_fn(data, xi))
                results_bwd.append((f"GeneralFast {mode_label}", bwd_ms))
            except Exception:
                results_bwd.append((f"GeneralFast {mode_label}", float("inf")))

        # Add FastLinearND (n=2 only) as reference.
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}

        @ops.jit
        def fwd_fast(d, x):
            return FastLinearNDInterpolator(
                coordinates=coords_fast, data=d, fill_value=None
            )(x)

        fwd_ms = _time_fn(lambda: fwd_fast(data, xi))
        results_fwd.append(("FastLinearND (n=2 ref)", fwd_ms))

        def loss_fast(d, x):
            return ops.sum(
                FastLinearNDInterpolator(
                    coordinates=coords_fast, data=d, fill_value=None
                )(x)
            )

        try:
            grad_fn_fast = ops.jit(ops.grad(loss_fast))
            bwd_ms = _time_fn(lambda: grad_fn_fast(data, xi))
            results_bwd.append(("FastLinearND (n=2 ref)", bwd_ms))
        except Exception:
            results_bwd.append(("FastLinearND (n=2 ref)", float("inf")))

        print("\n  --- Forward ---")
        _print_table(results_fwd, ref_idx=1)  # ref = linear (n=2)
        print("\n  --- Backward ---")
        _print_table(results_bwd, ref_idx=1)

    print(SEP)


# ---------------------------------------------------------------------------
# 4. Memory usage (via XLA compiled memory_analysis)
# ---------------------------------------------------------------------------


def _compile_and_analyze(fn, *args):
    """Lower+compile a function with jax.jit and return memory_analysis."""
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return compiled.memory_analysis()


def run_memory_benchmark():
    _print_header("4. XLA COMPILED MEMORY ANALYSIS (forward + backward)")

    import jax.numpy as jnp

    configs = [
        ("3D (128³)", (128, 128, 128), ["z", "x", "y"], 64),
        ("3D (256³)", (256, 256, 256), ["z", "x", "y"], 128),
    ]

    for label, sizes, dim_names, n_query in configs:
        print(f"\n  Config: {label}  |  n_query: {n_query}")
        data_jax = jnp.array(
            np.random.default_rng(42).standard_normal([int(s) for s in sizes]).astype("float32")
        )
        xi_vals = [jnp.linspace(0.05, 0.95, n_query) for _ in dim_names]
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}

        print(
            f"  {'Interpolator':<26} "
            f"| {'Fwd temp':>10} | {'Fwd arg':>10} | {'Fwd out':>10} "
            f"| {'Bwd temp':>10} | {'Bwd out':>10}"
        )
        print("  " + "-" * 86)

        for name, cls, needs_fast in LINEAR_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            def make_fwd(_cls, _c):
                def fwd(data_raw, *xis):
                    d = ops.array(data_raw, dims=dim_names)
                    x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
                    return _cls(coordinates=_c, data=d, fill_value=None)(x).data
                return fwd

            def make_loss(_cls, _c):
                def loss(data_raw, *xis):
                    d = ops.array(data_raw, dims=dim_names)
                    x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
                    return jnp.sum(_cls(coordinates=_c, data=d, fill_value=None)(x).data)
                return loss

            try:
                fwd_mem = _compile_and_analyze(make_fwd(cls, c), data_jax, *xi_vals)
                fwd_temp = f"{fwd_mem.temp_size_in_bytes / 1e6:>8.2f}MB"
                fwd_arg = f"{fwd_mem.argument_size_in_bytes / 1e6:>8.2f}MB"
                fwd_out = f"{fwd_mem.output_size_in_bytes / 1e6:>8.2f}MB"
            except Exception:
                fwd_temp = fwd_arg = fwd_out = f"{'ERR':>10}"

            try:
                bwd_fn = jax.grad(make_loss(cls, c))
                bwd_mem = _compile_and_analyze(bwd_fn, data_jax, *xi_vals)
                bwd_temp = f"{bwd_mem.temp_size_in_bytes / 1e6:>8.2f}MB"
                bwd_out = f"{bwd_mem.output_size_in_bytes / 1e6:>8.2f}MB"
            except Exception:
                bwd_temp = bwd_out = f"{'ERR':>10}"

            print(
                f"  {name:<26} "
                f"| {fwd_temp} | {fwd_arg} | {fwd_out} "
                f"| {bwd_temp} | {bwd_out}"
            )

        # Also test GeneralFast n=3 (quadratic)
        name = "GeneralFast n=3 (quad)"
        cls = GeneralFastLinearNDInterpolator

        def make_fwd_q(data_raw, *xis):
            d = ops.array(data_raw, dims=dim_names)
            x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
            return cls(coordinates=coords, data=d, fill_value=None, n_samples=3)(x).data

        def make_loss_q(data_raw, *xis):
            d = ops.array(data_raw, dims=dim_names)
            x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
            return jnp.sum(cls(coordinates=coords, data=d, fill_value=None, n_samples=3)(x).data)

        try:
            fwd_mem = _compile_and_analyze(make_fwd_q, data_jax, *xi_vals)
            fwd_temp = f"{fwd_mem.temp_size_in_bytes / 1e6:>8.2f}MB"
            fwd_arg = f"{fwd_mem.argument_size_in_bytes / 1e6:>8.2f}MB"
            fwd_out = f"{fwd_mem.output_size_in_bytes / 1e6:>8.2f}MB"
        except Exception:
            fwd_temp = fwd_arg = fwd_out = f"{'ERR':>10}"

        try:
            bwd_mem = _compile_and_analyze(jax.grad(make_loss_q), data_jax, *xi_vals)
            bwd_temp = f"{bwd_mem.temp_size_in_bytes / 1e6:>8.2f}MB"
            bwd_out = f"{bwd_mem.output_size_in_bytes / 1e6:>8.2f}MB"
        except Exception:
            bwd_temp = bwd_out = f"{'ERR':>10}"

        print(
            f"  {name:<26} "
            f"| {fwd_temp} | {fwd_arg} | {fwd_out} "
            f"| {bwd_temp} | {bwd_out}"
        )

    print(SEP)


# ---------------------------------------------------------------------------
# 5. Query density scaling
# ---------------------------------------------------------------------------


def run_query_density_benchmark():
    _print_header("5. QUERY DENSITY SCALING (2D, data=512², varying n_query)")

    sizes = (512, 512)
    dim_names = ["z", "x"]
    q_dim_names = ["qz", "qx"]
    query_sizes = [16, 64, 256, 512, 1024]

    data = _make_data(sizes, dim_names)
    coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
    coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}

    test_interps = [
        ("LinearND (old)", LinearNDInterpolator, False),
        ("FastLinearND", FastLinearNDInterpolator, True),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, False),
    ]

    # Forward
    print("\n  --- Forward (ms) ---")
    header = f"  {'Interpolator':<22}"
    for nq in query_sizes:
        header += f" | {'nq=' + str(nq):>10}"
    print(header)
    print("  " + "-" * (24 + 13 * len(query_sizes)))

    for name, cls, needs_fast in test_interps:
        c = coords_fast if needs_fast else coords
        row = f"  {name:<22}"

        for nq in query_sizes:
            xi = {
                d: ops.linspace(0.05, 0.95, nq, dim=qd)
                for d, qd in zip(dim_names, q_dim_names)
            }

            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            ms = _time_fn(lambda: fwd(data, xi), n_iters=50)
            row += f" | {ms:>8.3f}ms"
        print(row)

    # Backward
    print("\n  --- Backward (ms) ---")
    print(header)
    print("  " + "-" * (24 + 13 * len(query_sizes)))

    for name, cls, needs_fast in test_interps:
        c = coords_fast if needs_fast else coords
        row = f"  {name:<22}"

        for nq in query_sizes:
            xi = {
                d: ops.linspace(0.05, 0.95, nq, dim=qd)
                for d, qd in zip(dim_names, q_dim_names)
            }

            def make_loss(_cls, _c):
                def loss(d, x):
                    return ops.sum(_cls(coordinates=_c, data=d, fill_value=None)(x))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                ms = _time_fn(lambda: grad_fn(data, xi), n_iters=50)
                row += f" | {ms:>8.3f}ms"
            except Exception:
                row += f" | {'FAILED':>10}"
        print(row)

    print(SEP)


# ---------------------------------------------------------------------------
# 6. Complex-valued data
# ---------------------------------------------------------------------------


def run_complex_benchmark():
    _print_header("6. COMPLEX-VALUED DATA (complex64)")

    configs = [
        ("2D (512²)", (512, 512), ["z", "x"], ["qz", "qx"], 256),
        ("3D (128³)", (128, 128, 128), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
    ]

    for label, sizes, dim_names, q_dim_names, n_query in configs:
        print(f"\n  Config: {label}  |  n_query: {n_query}  |  dtype: complex64")
        data = _make_data(sizes, dim_names, dtype="complex64")
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {
            d: ops.linspace(0.05, 0.95, n_query, dim=qd)
            for d, qd in zip(dim_names, q_dim_names)
        }

        fwd_results = []
        bwd_results = []

        for name, cls, needs_fast in LINEAR_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            fwd_ms = _time_fn(lambda: fwd(data, xi))
            fwd_results.append((name, fwd_ms))

            def make_loss(_cls, _c):
                def loss(d, x):
                    # For complex, sum the real part to get a real scalar loss.
                    return ops.sum(ops.real(_cls(coordinates=_c, data=d, fill_value=None)(x)))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                bwd_ms = _time_fn(lambda: grad_fn(data, xi))
                bwd_results.append((name, bwd_ms))
            except Exception:
                bwd_results.append((name, float("inf")))

        print("\n  --- Forward ---")
        _print_table(fwd_results)
        print("\n  --- Backward ---")
        _print_table(bwd_results)

    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all():
    print(f"\n{'EXTENDED INTERPOLATOR BENCHMARKS':^100}")
    print(f"{'=' * 100}")
    print(
        f"  Device: {bench_device}  |  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    print(f"  Backend: {ops.backend.backend_name}")
    print(f"{'=' * 100}")

    run_ray_tracing_backward()
    run_compilation_time()
    run_quadratic_benchmark()
    run_memory_benchmark()
    run_query_density_benchmark()
    run_complex_benchmark()

    print(f"\n{'=' * 100}")
    print("  All extended benchmarks done.")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    run_all()
