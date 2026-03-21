"""Benchmark FastLinearNDInterpolator, FastNearestNDInterpolator,
LinearNDInterpolator, LinearNDInterpolatorFast, and NearestNDInterpolator.

Tests forward and backward pass performance for 1-D, 2-D, and 3-D cases
at various array sizes.  Uses only spekk.ops (no direct JAX imports).

Usage:
    python benchmark_interpolators.py
"""

import os

# Device selection:
# - BENCH_DEVICE=gpu (default): request GPU (CUDA_VISIBLE_DEVICES=0)
# - BENCH_DEVICE=cpu: force CPU (CUDA_VISIBLE_DEVICES="")
bench_device = os.environ.get("BENCH_DEVICE", "gpu").lower()
if bench_device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import time

import jax
import numpy as np
import optax
from spekk import ops

from vbeam.interpolation import (
    FastLinearNDInterpolator,
    FastNearestNDInterpolator,
    GeneralFastLinearNDInterpolator,
    LinearCoordinate,
    LinearCoordinateFast,
    LinearNDInterpolator,
    LinearNDInterpolatorFast,
    NearestNDInterpolator,
)
from vbeam.interpolation.nd_interpolator import OptimizedLinearNDInterpolator
from vbeam.delay_models.speed_of_sound import SpeedOfSoundRayTracing
from spekk import replace
from vbeam_ntnu.experimental.aberration_corrections.losses import total_variation
from vbeam_ntnu.experimental.aberration_corrections.path_based_aberration_model import (
    coherence,
)
from vbeam_sim.beamformer_importers.research_replication.path_based_beamformer_importer import (
    PathBasedBeamformerImporter,
)

ops.backend.set_backend("jax")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP = 3
N_ITERS = 100
ESTIMATE_H_UPDATES = 200


def _coord_fast(size):
    return LinearCoordinateFast(start=0.0, stop=1.0, size=size)


def _coord_old(size):
    return LinearCoordinate(start=0.0, stop=1.0, size=size)


def _make_data(sizes, dim_names):
    raw = np.random.default_rng(42).standard_normal([int(s) for s in sizes]).astype(
        np.float32
    )
    return ops.array(raw, dims=list(dim_names))


def _make_query(n_query, dim_names_query):
    """Create query arrays with given dim names, each of length n_query."""
    arrs = {}
    for i, dim in enumerate(dim_names_query):
        arrs[dim] = ops.linspace(0.05, 0.95, n_query, dim=dim)
    return arrs


def _block_until_ready(value):
    """Synchronize async backend work (important for JAX timing accuracy)."""
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


def _time_forward(fn, n_warmup=WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        out = fn()
        _block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn()
        _block_until_ready(out)
    return (time.perf_counter() - t0) / n_iters * 1000  # ms


def _time_backward(grad_fn, *args, n_warmup=WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        out = grad_fn(*args)
        _block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = grad_fn(*args)
        _block_until_ready(out)
    return (time.perf_counter() - t0) / n_iters * 1000  # ms


# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

# Each config: (label, ndim, data_sizes, n_query)
# Sizes must be large enough that GPU compute exceeds the ~4.4ms kernel launch
# overhead floor; otherwise all timings collapse to the same value.
CONFIGS = [
    # 1-D
    ("1D-small",  1, (1024,),          512),
    ("1D-med",    1, (8192,),         4096),
    ("1D-large",  1, (65536,),       32768),
    # 2-D
    ("2D-small",  2, (256, 256),       128),
    ("2D-med",    2, (512, 512),       256),
    ("2D-large",  2, (2048, 2048),     512),
    # 3-D
    ("3D-small",  3, (32, 32, 32),      16),
    ("3D-med",    3, (64, 64, 64),      32),
    ("3D-large",  3, (128, 128, 128),   64),
]

DATA_DIMS = {1: ["z"], 2: ["z", "x"], 3: ["z", "x", "y"]}
QUERY_DIMS = {1: ["qz"], 2: ["qz", "qx"], 3: ["qz", "qx", "qy"]}

# Coordinate micro-benchmark configs: (label, size, n_query)
COORD_CONFIGS = [
    ("coord-small", 1024, 512),
    ("coord-med", 8192, 4096),
    ("coord-large", 65536, 32768),
]


# ---------------------------------------------------------------------------
# Interpolator factories
# ---------------------------------------------------------------------------


def _build_interpolators(ndim, data_sizes, n_query):
    dim_names = DATA_DIMS[ndim]
    q_dim_names = QUERY_DIMS[ndim]
    data_fast = _make_data(data_sizes, dim_names)
    data_old = _make_data(data_sizes, dim_names)

    coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, data_sizes)}
    coords_old = {d: _coord_old(s) for d, s in zip(dim_names, data_sizes)}

    xi = {d: ops.linspace(0.05, 0.95, n_query, dim=qd) for d, qd in zip(dim_names, q_dim_names)}

    interpolators = {
        "FastLinearND": FastLinearNDInterpolator(
            coordinates=coords_fast, data=data_fast, fill_value=None
        ),
        "FastNearestND": FastNearestNDInterpolator(
            coordinates=coords_fast, data=data_fast, fill_value=None
        ),
        "LinearND (old)": LinearNDInterpolator(
            coordinates=coords_old, data=data_old, fill_value=None
        ),
        "LinearNDFast": LinearNDInterpolatorFast(
            coordinates=coords_fast, data=data_fast, fill_value=None
        ),
        "NearestND (old)": NearestNDInterpolator(
            coordinates=coords_old, data=data_old, fill_value=None
        ),
        "OptimizedLinearND": OptimizedLinearNDInterpolator(
            coordinates=coords_old, data=data_old, fill_value=None
        ),
        "GeneralFastLinearND": GeneralFastLinearNDInterpolator(
            coordinates=coords_old, data=data_old, fill_value=None
        ),
    }

    return interpolators, xi, data_fast, data_old, coords_fast, coords_old, dim_names


def _run_coordinate_micro_benchmark():
    sep = "-" * 100
    print(f"\n{'COORDINATE MICRO-BENCHMARK (get_nearest_indices)':^100}")
    print(sep)

    for label, size, n_query in COORD_CONFIGS:
        print(f"  Config: {label}  |  size: {size}  |  n_query: {n_query}")

        coord_fast = LinearCoordinateFast(start=0.0, stop=1.0, size=size)
        coord_old = LinearCoordinate(start=0.0, stop=1.0, size=size)
        x = ops.linspace(0.05, 0.95, n_query, dim="q")

        entries = []

        # Pass x as argument to prevent XLA constant folding.
        # Old coordinate API
        @ops.jit
        def fn_old_1(x): return coord_old.get_nearest_indices(x, 1)
        @ops.jit
        def fn_old_2(x): return coord_old.get_nearest_indices(x, 2)
        entries.append(("LinearCoordinate n=1", _time_forward(lambda: fn_old_1(x))))
        entries.append(("LinearCoordinate n=2", _time_forward(lambda: fn_old_2(x))))

        # New fast API
        @ops.jit
        def fn_fast_1(x): return coord_fast.get_nearest_indices(x, 1)
        @ops.jit
        def fn_fast_2(x): return coord_fast.get_nearest_indices(x, 2)
        entries.append(("LinearCoordinateFast n=1", _time_forward(lambda: fn_fast_1(x))))
        entries.append(("LinearCoordinateFast n=2", _time_forward(lambda: fn_fast_2(x))))

        # Backward compatibility helper
        @ops.jit
        def fn_fast_legacy(x): return coord_fast.get_index_and_frac(x)
        entries.append(
            ("LinearCoordinateFast get_index_and_frac", _time_forward(lambda: fn_fast_legacy(x)))
        )

        ref_old_2 = None
        for name, ms in entries:
            if name == "LinearCoordinate n=2":
                ref_old_2 = ms
                break

        print(f"\n  {'Method':<38} | {'Time (ms)':>10} | {'x vs old n=2':>12}")
        print("  " + "-" * 68)
        for name, ms in entries:
            if ref_old_2 and ref_old_2 > 0:
                rel = ref_old_2 / ms
                rel_str = f"{rel:>10.2f}x"
            else:
                rel_str = f"{'n/a':>12}"
            print(f"  {name:<38} | {ms:>8.3f}ms | {rel_str}")
        print()

    print(sep)


def _set_dummy_channel_data(bf_setup, seed=0):
    """Replace channel data by deterministic dummy data with same shape/dims/dtype."""
    data = bf_setup.channel_data.data
    rng = np.random.default_rng(seed)
    np_dtype = np.asarray(data.data).dtype

    if np.issubdtype(np_dtype, np.complexfloating):
        real = rng.standard_normal(data.shape)
        imag = rng.standard_normal(data.shape)
        dummy_np = (real + 1j * imag).astype(np_dtype)
    else:
        dummy_np = rng.standard_normal(data.shape).astype(np_dtype)

    bf_setup.channel_data.data = ops.array(dummy_np, dims=data.dims)


def _run_estimate_h_like_benchmark():
    """Benchmark an estimate_h-like optimization loop with n_updates=200.

    This is closer to the beamforming workload than pure interpolation kernels.
    """
    sep = "-" * 100
    print(f"\n{'ESTIMATE_H-LIKE BENCHMARK (n_updates=200)':^100}")
    print(sep)

    importer = PathBasedBeamformerImporter()
    scan_opt = importer.get_points_in_linear_for_optimization()

    test_interpolators = [
        ("LinearND (old)", LinearNDInterpolator),
        ("LinearNDFast", LinearNDInterpolatorFast),
        ("OptimizedLinearND", OptimizedLinearNDInterpolator),
        ("FastLinearND", FastLinearNDInterpolator),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator),
    ]

    print(
        f"  {'Interpolator':<20} | {'Total (s)':>10} | {'ms/update':>10} | {'updates/s':>10} | {'Final loss':>12}"
    )
    print("  " + "-" * 81)

    def _estimate_setup_parameter_timed_after_jit(
        bf_setup,
        loss_fn,
        parameters_getter,
        parameters_setter,
        *,
        learning_rate,
        n_updates,
        optimizer_fn,
    ):
        """Run estimate_setup_parameter-like loop, timing only after JIT warmup.

        This excludes compilation overhead by executing one warmup `step` first,
        then starting the timer for the update loop.
        """

        def set_parameters_and_compute_loss(params):
            updated_bf_setup = parameters_setter(bf_setup, params)
            return loss_fn(updated_bf_setup)

        value_and_grad = ops.value_and_grad(set_parameters_and_compute_loss)
        optimizer = optimizer_fn(learning_rate)
        params = parameters_getter(bf_setup)
        optimizer_state = optimizer.init(params.data)
        dims = params.dims

        @jax.jit
        def step(params_jax, optimizer_state):
            params_local = ops.array(params_jax, dims=dims)
            loss, grad = value_and_grad(params_local)
            updates, optimizer_state_local = optimizer.update(
                grad.data, optimizer_state, params_local.data
            )
            params_jax_out = optax.apply_updates(params_local.data, updates)
            return params_jax_out, optimizer_state_local, loss.data

        # Warmup/compile step (excluded from measured time)
        params_jax, optimizer_state, loss = step(params.data, optimizer_state)
        _block_until_ready((params_jax, optimizer_state, loss))

        losses = []
        t0 = time.perf_counter()
        for _ in range(n_updates):
            params_jax, optimizer_state, loss = step(params_jax, optimizer_state)
            losses.append(loss)
        _block_until_ready(loss)
        total_s = time.perf_counter() - t0

        params_final = ops.array(params_jax, dims=dims)
        bf_setup = parameters_setter(bf_setup, params_final)
        losses = ops.array(losses, dims=["iteration"])
        return bf_setup, losses, total_s

    for i, (name, interp_type) in enumerate(test_interpolators):
        # Build a fresh setup for each run.
        bf_setup = importer.get_bf_setup(
            flatten=True,
            tx_rx_first=True,
            linear_interpolation=False,
        )
        bf_setup.scan = scan_opt
        bf_setup.interpolator_type = interp_type
        _set_dummy_channel_data(bf_setup, seed=100 + i)

        def get_h_in_setup(setup):
            return setup.delay_model.h

        def set_h_into_setup(setup, h):
            setup.delay_model.h = h
            return setup

        def loss_fn(setup):
            h = setup.delay_model.h
            coh = ops.mean(coherence(setup))
            tv = total_variation(h["time_phase", 0], power=2)
            return -coh + 5e11 * tv

        _, losses, total_s = _estimate_setup_parameter_timed_after_jit(
            bf_setup,
            loss_fn=loss_fn,
            parameters_getter=get_h_in_setup,
            parameters_setter=set_h_into_setup,
            learning_rate=ops.reshape(ops.array([2e-13, 1e-13]), (-1, 1, 1)).data,
            n_updates=ESTIMATE_H_UPDATES,
            optimizer_fn=optax.sgd,
        )

        final_loss = float(np.asarray(losses.data)[-1])
        ms_per_update = total_s / ESTIMATE_H_UPDATES * 1000.0
        updates_per_s = ESTIMATE_H_UPDATES / total_s
        print(
            f"  {name:<20} | {total_s:>8.3f}s | {ms_per_update:>8.3f}ms | {updates_per_s:>8.2f} | {final_loss:>12.4e}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Ray tracing benchmark
# ---------------------------------------------------------------------------

RAY_TRACING_N_ITERS = 20
RAY_TRACING_WARMUP = 2


def _run_ray_tracing_benchmark():
    """Benchmark SpeedOfSoundRayTracing.get_delay_between with different interpolators.

    Uses the same scan data as estimate_h. Creates a 2D speed-of-sound map and
    measures how long get_delay_between takes from probe elements to scan points.
    """
    sep = "-" * 100
    print(f"\n{'RAY TRACING BENCHMARK (get_delay_between)':^100}")
    print(sep)

    importer = PathBasedBeamformerImporter()
    scan = importer.get_points_in_linear_for_optimization()

    # Probe element positions as point1 (shape: [xyz=3, tx=128]).
    # Construct from probe pitch — 128 elements spaced 0.3mm apart at z=0.
    n_elements = 128
    pitch = 0.0003
    x_pos = np.arange(n_elements) * pitch - (n_elements - 1) * pitch / 2
    ep = np.zeros((3, n_elements), dtype=np.float32)
    ep[0] = x_pos
    point1 = ops.array(ep, dims=["xyz", "tx"])
    # Scan points as point2 (shape: [xyz=3, xs=100, zs=100])
    point2 = scan.points

    print(f"  point1 dims: {point1.dim_sizes}")
    print(f"  point2 dims: {point2.dim_sizes}")

    # Create a realistic SoS map with some variation.
    n_xs, n_zs = 100, 100
    rng = np.random.default_rng(42)
    sos_np = 1540.0 + 20.0 * rng.standard_normal((n_xs, n_zs)).astype(np.float32)
    sos_map = ops.array(sos_np, dims=["xs", "zs"])

    ray_tracer_base = SpeedOfSoundRayTracing.from_scan(
        scan, sos_map, n_steps=64
    )

    # Build a variant with LinearCoordinateFast for FastLinearNDInterpolator.
    coords_fast = {
        dim: LinearCoordinateFast(c.start, c.stop, c.size)
        for dim, c in ray_tracer_base.coordinates.items()
    }
    ray_tracer_fast_coords = replace(ray_tracer_base, coordinates=coords_fast)

    # Interpolators to test.
    test_interpolators = [
        ("LinearND (old)", LinearNDInterpolator, ray_tracer_base),
        ("OptimizedLinearND", OptimizedLinearNDInterpolator, ray_tracer_base),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, ray_tracer_base),
        ("FastLinearND", FastLinearNDInterpolator, ray_tracer_fast_coords),
    ]

    print(
        f"  n_steps: {ray_tracer_base.n_steps}  |  SoS map: {n_xs}x{n_zs}"
    )
    print(
        f"  Warmup: {RAY_TRACING_WARMUP}  |  Iterations: {RAY_TRACING_N_ITERS}"
    )
    print()
    print(
        f"  {'Interpolator':<22} | {'Total (ms)':>10} | {'Speedup':>10}"
    )
    print("  " + "-" * 50)

    results = []
    for name, interp_type, base_rt in test_interpolators:
        ray_tracer = replace(base_rt, interpolator_type=interp_type)

        @ops.jit
        def run(p1, p2, _rt=ray_tracer):
            return _rt.get_delay_between(p1, p2)

        # Warmup
        for _ in range(RAY_TRACING_WARMUP):
            out = run(point1, point2)
            _block_until_ready(out)

        t0 = time.perf_counter()
        for _ in range(RAY_TRACING_N_ITERS):
            out = run(point1, point2)
            _block_until_ready(out)
        elapsed_ms = (time.perf_counter() - t0) / RAY_TRACING_N_ITERS * 1000
        results.append((name, elapsed_ms))

    ref_ms = results[0][1]  # LinearND (old) as reference
    for name, ms in results:
        speedup = ref_ms / ms if ms > 0 else float("inf")
        print(f"  {name:<22} | {ms:>8.2f}ms | {speedup:>8.2f}x")

    print(sep)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmarks():
    sep = "-" * 100
    print(f"\n{'INTERPOLATOR BENCHMARK':^100}")
    print(f"{'=' * 100}")
    print(
        f"  Requested device: {bench_device}  |  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    print(f"  Warmup: {WARMUP}  |  Iterations: {N_ITERS}  |  Backend: {ops.backend.backend_name}")
    print(f"{'=' * 100}\n")

    _run_coordinate_micro_benchmark()
    _run_estimate_h_like_benchmark()
    _run_ray_tracing_benchmark()

    for label, ndim, data_sizes, n_query in CONFIGS:
        print(sep)
        print(f"  Config: {label}  |  data shape: {data_sizes}  |  n_query: {n_query}  |  ndim: {ndim}")
        print(sep)

        interpolators, xi, data_fast, data_old, coords_fast, coords_old, dim_names = (
            _build_interpolators(ndim, data_sizes, n_query)
        )

        rows = []

        for name, interp in interpolators.items():
            # Pass data and xi as arguments to prevent XLA constant folding,
            # matching how estimate_h uses interpolators with dynamic inputs.
            is_fast_coord = name in ("FastLinearND", "FastNearestND", "LinearNDFast")
            if is_fast_coord:
                d, c = data_fast, coords_fast
            else:
                d, c = data_old, coords_old
            interp_cls = type(interp)

            # Forward
            @ops.jit
            def fwd_fn(data_arg, xi_arg, _cls=interp_cls, _c=c):
                _interp = _cls(coordinates=_c, data=data_arg, fill_value=None)
                return _interp(xi_arg)

            fwd_ms = _time_forward(lambda: fwd_fn(d, xi))

            # Backward
            def make_loss(_cls, coords):
                def loss(data_arg, xi_arg):
                    interp_inner = _cls(
                        coordinates=coords, data=data_arg, fill_value=None
                    )
                    return ops.sum(interp_inner(xi_arg))
                return loss

            loss_fn = make_loss(interp_cls, c)

            try:
                grad_fn = ops.jit(ops.grad(loss_fn))
                bwd_ms = _time_backward(grad_fn, d, xi)
            except Exception:
                bwd_ms = None

            rows.append((name, fwd_ms, bwd_ms))

        # Build reference for relative speed columns.
        ref_fwd = None
        ref_bwd = None
        for name, fwd_ms, bwd_ms in rows:
            if name == "LinearND (old)":
                ref_fwd = fwd_ms
                ref_bwd = bwd_ms
                break

        print(
            f"\n  {'Interpolator':<18} | {'Forward (ms)':>12} | {'Fwd x vs old':>12} | {'Backward (ms)':>13} | {'Bwd x vs old':>12}"
        )
        print("  " + "-" * 85)

        for name, fwd_ms, bwd_ms in rows:
            if ref_fwd and ref_fwd > 0:
                fwd_rel = ref_fwd / fwd_ms
                fwd_rel_str = f"{fwd_rel:>10.2f}x"
            else:
                fwd_rel_str = f"{'n/a':>12}"

            if bwd_ms is None:
                bwd_str = f"{'FAILED':>13}"
                bwd_rel_str = f"{'n/a':>12}"
            else:
                bwd_str = f"{bwd_ms:>11.3f}ms"
                if ref_bwd and ref_bwd > 0:
                    bwd_rel = ref_bwd / bwd_ms
                    bwd_rel_str = f"{bwd_rel:>10.2f}x"
                else:
                    bwd_rel_str = f"{'n/a':>12}"

            print(
                f"  {name:<18} | {fwd_ms:>10.3f}ms | {fwd_rel_str} | {bwd_str} | {bwd_rel_str}"
            )

        print()

    print(f"{'=' * 100}")
    print("  Done.")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    run_benchmarks()
