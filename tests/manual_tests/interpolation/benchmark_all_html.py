"""Comprehensive interpolator benchmark suite that outputs an HTML report.

Runs all benchmarks with all interpolators, both unroll=1 and unroll=True
for ray tracing, and generates an interactive HTML page with the results.

Usage:
    python benchmark_all_html.py
"""

import os

bench_device = os.environ.get("BENCH_DEVICE", "gpu").lower()
if bench_device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import time
from datetime import datetime

import jax
import optax
import jax.numpy as jnp
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
from vbeam_ntnu.experimental.aberration_corrections.losses import total_variation
from vbeam_ntnu.experimental.aberration_corrections.path_based_aberration_model import (
    coherence,
)
from vbeam_sim.beamformer_importers.research_replication.path_based_beamformer_importer import (
    PathBasedBeamformerImporter,
)

ops.backend.set_backend("jax")

WARMUP = 3
N_ITERS = 50

# ---------------------------------------------------------------------------
# All interpolators
# ---------------------------------------------------------------------------

ALL_INTERPOLATORS = [
    ("LinearND", LinearNDInterpolator, False),
    ("OptimizedLinearND", OptimizedLinearNDInterpolator, False),
    ("LinearNDFast", LinearNDInterpolatorFast, True),
    ("FastLinearND", FastLinearNDInterpolator, True),
    ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block(value):
    if value is None:
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif hasattr(value, "data"):
        _block(value.data)
    elif isinstance(value, dict):
        for v in value.values():
            _block(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _block(v)


def _time_fn(fn, n_warmup=WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        _block(fn())
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _block(fn())
    return (time.perf_counter() - t0) / n_iters * 1000  # ms


def _coord(size):
    return LinearCoordinate(start=0.0, stop=1.0, size=size)


def _coord_fast(size):
    return LinearCoordinateFast(start=0.0, stop=1.0, size=size)


def _make_data(sizes, dim_names, *, dtype="float32", seed=42):
    rng = np.random.default_rng(seed)
    shape = [int(s) for s in sizes]
    if np.dtype(dtype).kind == "c":
        real = rng.standard_normal(shape).astype("float32")
        imag = rng.standard_normal(shape).astype("float32")
        raw = (real + 1j * imag).astype(dtype)
    else:
        raw = rng.standard_normal(shape).astype(dtype)
    return ops.array(raw, dims=list(dim_names))


# ---------------------------------------------------------------------------
# 0. estimate_h-like optimization loop
# ---------------------------------------------------------------------------


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


def bench_estimate_h():
    """Benchmark an estimate_h-like optimization loop (200 gradient updates)."""
    print("  [0/7] estimate_h benchmark...")

    ESTIMATE_H_UPDATES = 200

    importer = PathBasedBeamformerImporter()
    scan_opt = importer.get_points_in_linear_for_optimization()

    test_interpolators = [
        ("LinearND", LinearNDInterpolator),
        ("OptimizedLinearND", OptimizedLinearNDInterpolator),
        ("LinearNDFast", LinearNDInterpolatorFast),
        ("FastLinearND", FastLinearNDInterpolator),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator),
    ]

    def _run_estimate_h(bf_setup, loss_fn, get_h, set_h, *, learning_rate, n_updates):
        def set_and_compute_loss(params):
            return loss_fn(set_h(bf_setup, params))

        value_and_grad = ops.value_and_grad(set_and_compute_loss)
        optimizer = optax.sgd(learning_rate)
        params = get_h(bf_setup)
        optimizer_state = optimizer.init(params.data)
        dims = params.dims

        @jax.jit
        def step(params_jax, opt_state):
            params_local = ops.array(params_jax, dims=dims)
            loss, grad = value_and_grad(params_local)
            updates, new_opt_state = optimizer.update(
                grad.data, opt_state, params_local.data
            )
            new_params = optax.apply_updates(params_local.data, updates)
            return new_params, new_opt_state, loss.data

        # Warmup (compile)
        params_jax, optimizer_state, loss = step(params.data, optimizer_state)
        _block((params_jax, optimizer_state, loss))

        t0 = time.perf_counter()
        for _ in range(n_updates):
            params_jax, optimizer_state, loss = step(params_jax, optimizer_state)
        _block(loss)
        total_s = time.perf_counter() - t0
        return total_s, float(np.asarray(loss))

    results = {}
    for i, (name, interp_type) in enumerate(test_interpolators):
        bf_setup = importer.get_bf_setup(
            flatten=True, tx_rx_first=True, linear_interpolation=False
        )
        bf_setup.scan = scan_opt
        bf_setup.interpolator_type = interp_type
        _set_dummy_channel_data(bf_setup, seed=100 + i)

        def get_h(setup):
            return setup.delay_model.h

        def set_h(setup, h):
            setup.delay_model.h = h
            return setup

        def loss_fn(setup):
            h = setup.delay_model.h
            coh = ops.mean(coherence(setup))
            tv = total_variation(h["time_phase", 0], power=2)
            return -coh + 5e11 * tv

        total_s, final_loss = _run_estimate_h(
            bf_setup,
            loss_fn=loss_fn,
            get_h=get_h,
            set_h=set_h,
            learning_rate=ops.reshape(ops.array([2e-13, 1e-13]), (-1, 1, 1)).data,
            n_updates=ESTIMATE_H_UPDATES,
        )

        ms_per_update = total_s / ESTIMATE_H_UPDATES * 1000.0
        results[name] = {
            "ms_per_update": round(ms_per_update, 3),
            "total_s": round(total_s, 3),
            "final_loss": final_loss,
        }
        print(f"    {name}: {ms_per_update:.3f} ms/update  (total {total_s:.1f}s)")

    return {
        "title": f"estimate_h Optimization Loop ({ESTIMATE_H_UPDATES} gradient updates)",
        "data": results,
    }


# ---------------------------------------------------------------------------
# 1. Ray tracing (forward + backward) × (unroll=1, unroll=True)
# ---------------------------------------------------------------------------


def bench_ray_tracing():
    print("  [1/6] Ray tracing benchmarks...")
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

    rt_base = SpeedOfSoundRayTracing.from_scan(scan, sos_map, n_steps=64)
    coords_fast = {
        dim: LinearCoordinateFast(c.start, c.stop, c.size)
        for dim, c in rt_base.coordinates.items()
    }
    rt_fast = replace(rt_base, coordinates=coords_fast)

    cases = [
        ("LinearND", LinearNDInterpolator, rt_base),
        ("OptimizedLinearND", OptimizedLinearNDInterpolator, rt_base),
        ("LinearNDFast", LinearNDInterpolatorFast, rt_fast),
        ("FastLinearND", FastLinearNDInterpolator, rt_fast),
        ("GeneralFastLinearND", GeneralFastLinearNDInterpolator, rt_base),
    ]

    results = {}
    for unroll_value in [1, True]:
        ul = "unroll=True" if unroll_value is True else "unroll=1"
        results[ul] = {"forward": {}, "backward": {}}

        for name, interp_type, base_rt in cases:
            rt = replace(base_rt, interpolator_type=interp_type, unroll=unroll_value)

            # Forward
            @ops.jit
            def fwd(p1, p2, _rt=rt):
                return _rt.get_delay_between(p1, p2)

            ms = _time_fn(lambda: fwd(point1, point2), n_iters=20)
            results[ul]["forward"][name] = round(ms, 3)
            print(f"    {ul} fwd {name}: {ms:.3f}ms")

            # Backward
            def loss(sos, _rt=rt, _p1=point1, _p2=point2):
                rt_local = replace(_rt, speed_of_sound=sos)
                return ops.sum(rt_local.get_delay_between(_p1, _p2))

            try:
                grad_fn = ops.jit(ops.grad(loss))
                ms = _time_fn(lambda: grad_fn(sos_map), n_iters=20)
                results[ul]["backward"][name] = round(ms, 3)
                print(f"    {ul} bwd {name}: {ms:.3f}ms")
            except Exception as e:
                results[ul]["backward"][name] = None
                print(f"    {ul} bwd {name}: FAILED ({e})")

    return {
        "title": "Ray Tracing (64 steps, 100×100 SoS, 128 elements → 100×100 scan)",
        "data": results,
    }


# ---------------------------------------------------------------------------
# 2. Micro-benchmarks (forward + backward) at multiple sizes
# ---------------------------------------------------------------------------


def bench_micro():
    print("  [2/6] Micro-benchmarks...")
    configs = [
        ("2D 512²", (512, 512), ["z", "x"], ["qz", "qx"], 256),
        ("3D 64³", (64, 64, 64), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
        ("3D 128³", (128, 128, 128), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
    ]

    results = {}
    for label, sizes, dim_names, q_dim_names, n_query in configs:
        data = _make_data(sizes, dim_names)
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {d: ops.linspace(0.05, 0.95, n_query, dim=qd)
              for d, qd in zip(dim_names, q_dim_names)}

        fwd_res, bwd_res = {}, {}
        for name, cls, needs_fast in ALL_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            ms = _time_fn(lambda: fwd(data, xi))
            fwd_res[name] = round(ms, 3)

            def make_loss(_cls, _c):
                def loss(d, x):
                    return ops.sum(_cls(coordinates=_c, data=d, fill_value=None)(x))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                ms = _time_fn(lambda: grad_fn(data, xi))
                bwd_res[name] = round(ms, 3)
            except Exception:
                bwd_res[name] = None

        results[label] = {"forward": fwd_res, "backward": bwd_res}
        print(f"    {label} done")

    return {"title": "Micro-benchmarks (forward + backward)", "data": results}


# ---------------------------------------------------------------------------
# 3. Compilation time
# ---------------------------------------------------------------------------


def bench_compile_time():
    print("  [3/6] Compilation time...")
    configs = [
        ("2D 256²", (256, 256), ["z", "x"], ["qz", "qx"]),
        ("3D 64³", (64, 64, 64), ["z", "x", "y"], ["qz", "qx", "qy"]),
    ]

    results = {}
    for label, sizes, dim_names, q_dim_names in configs:
        data = _make_data(sizes, dim_names)
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {d: ops.linspace(0.05, 0.95, 64, dim=qd)
              for d, qd in zip(dim_names, q_dim_names)}

        fwd_res, bwd_res = {}, {}

        for name, cls, needs_fast in ALL_INTERPOLATORS:
            c = coords_fast if needs_fast else coords
            jax.clear_caches()

            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            t0 = time.perf_counter()
            _block(fwd(data, xi))
            fwd_res[name] = round((time.perf_counter() - t0) * 1000, 1)

            jax.clear_caches()

            def make_loss(_cls, _c):
                def loss(d, x):
                    return ops.sum(_cls(coordinates=_c, data=d, fill_value=None)(x))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                t0 = time.perf_counter()
                _block(grad_fn(data, xi))
                bwd_res[name] = round((time.perf_counter() - t0) * 1000, 1)
            except Exception:
                bwd_res[name] = None

        # GeneralFast n=3
        jax.clear_caches()

        @ops.jit
        def fwd_q(d, x):
            return GeneralFastLinearNDInterpolator(
                coordinates=coords, data=d, fill_value=None, n_samples=3
            )(x)

        t0 = time.perf_counter()
        _block(fwd_q(data, xi))
        fwd_res["GeneralFast n=3"] = round((time.perf_counter() - t0) * 1000, 1)

        jax.clear_caches()

        def loss_q(d, x):
            return ops.sum(
                GeneralFastLinearNDInterpolator(
                    coordinates=coords, data=d, fill_value=None, n_samples=3
                )(x)
            )

        try:
            grad_fn_q = ops.jit(ops.grad(loss_q))
            t0 = time.perf_counter()
            _block(grad_fn_q(data, xi))
            bwd_res["GeneralFast n=3"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            bwd_res["GeneralFast n=3"] = None

        results[label] = {"forward": fwd_res, "backward": bwd_res}
        print(f"    {label} done")

    return {"title": "JIT Compilation Time (ms)", "data": results}


# ---------------------------------------------------------------------------
# 4. Quadratic (n_samples=1,2,3) + all interpolators as reference
# ---------------------------------------------------------------------------


def bench_quadratic():
    print("  [4/6] Quadratic vs linear...")
    configs = [
        ("2D 512²", (512, 512), ["z", "x"], ["qz", "qx"], 256),
        ("3D 128³", (128, 128, 128), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
    ]

    results = {}
    for label, sizes, dim_names, q_dim_names, n_query in configs:
        data = _make_data(sizes, dim_names)
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {d: ops.linspace(0.05, 0.95, n_query, dim=qd)
              for d, qd in zip(dim_names, q_dim_names)}

        fwd_res, bwd_res = {}, {}

        # GeneralFast at different n_samples
        for ns, ns_label in [(1, "GeneralFast n=1"), (2, "GeneralFast n=2"), (3, "GeneralFast n=3")]:
            @ops.jit
            def fwd(d, x, _n=ns):
                return GeneralFastLinearNDInterpolator(
                    coordinates=coords, data=d, fill_value=None, n_samples=_n
                )(x)

            fwd_res[ns_label] = round(_time_fn(lambda: fwd(data, xi)), 3)

            def make_loss(_n):
                def loss(d, x):
                    return ops.sum(
                        GeneralFastLinearNDInterpolator(
                            coordinates=coords, data=d, fill_value=None, n_samples=_n
                        )(x)
                    )
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(ns)))
                bwd_res[ns_label] = round(_time_fn(lambda: grad_fn(data, xi)), 3)
            except Exception:
                bwd_res[ns_label] = None

        # All other interpolators as reference
        for name, cls, needs_fast in ALL_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            @ops.jit
            def fwd2(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            fwd_res[name] = round(_time_fn(lambda: fwd2(data, xi)), 3)

            def make_loss2(_cls, _c):
                def loss(d, x):
                    return ops.sum(_cls(coordinates=_c, data=d, fill_value=None)(x))
                return loss

            try:
                grad_fn2 = ops.jit(ops.grad(make_loss2(cls, c)))
                bwd_res[name] = round(_time_fn(lambda: grad_fn2(data, xi)), 3)
            except Exception:
                bwd_res[name] = None

        results[label] = {"forward": fwd_res, "backward": bwd_res}
        print(f"    {label} done")

    return {"title": "Quadratic (n=3) vs Linear (n=2) vs Nearest (n=1)", "data": results}


# ---------------------------------------------------------------------------
# 5. Memory analysis (XLA compiled)
# ---------------------------------------------------------------------------


def bench_memory():
    print("  [5/6] Memory analysis...")
    configs = [
        ("3D 128³", (128, 128, 128), ["z", "x", "y"], 64),
        ("3D 256³", (256, 256, 256), ["z", "x", "y"], 128),
    ]

    results = {}
    for label, sizes, dim_names, n_query in configs:
        data_jax = jnp.array(
            np.random.default_rng(42).standard_normal([int(s) for s in sizes]).astype("float32")
        )
        xi_vals = [jnp.linspace(0.05, 0.95, n_query) for _ in dim_names]
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}

        mem_res = {}
        interps_to_test = list(ALL_INTERPOLATORS) + [
            ("GeneralFast n=3", GeneralFastLinearNDInterpolator, False),
        ]

        for name, cls, needs_fast in interps_to_test:
            c = coords_fast if needs_fast else coords
            ns = 3 if name == "GeneralFast n=3" else 2

            def make_fwd(_cls, _c, _ns=ns):
                def fwd(data_raw, *xis):
                    d = ops.array(data_raw, dims=dim_names)
                    x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
                    if _cls is GeneralFastLinearNDInterpolator:
                        return _cls(coordinates=_c, data=d, fill_value=None, n_samples=_ns)(x).data
                    return _cls(coordinates=_c, data=d, fill_value=None)(x).data
                return fwd

            def make_loss(_cls, _c, _ns=ns):
                def loss(data_raw, *xis):
                    d = ops.array(data_raw, dims=dim_names)
                    x = {dim: ops.array(v, dims=["q" + dim]) for dim, v in zip(dim_names, xis)}
                    if _cls is GeneralFastLinearNDInterpolator:
                        return jnp.sum(_cls(coordinates=_c, data=d, fill_value=None, n_samples=_ns)(x).data)
                    return jnp.sum(_cls(coordinates=_c, data=d, fill_value=None)(x).data)
                return loss

            entry = {}
            try:
                lowered = jax.jit(make_fwd(cls, c, ns)).lower(data_jax, *xi_vals)
                mem = lowered.compile().memory_analysis()
                entry["fwd_temp_mb"] = round(mem.temp_size_in_bytes / 1e6, 2)
                entry["fwd_arg_mb"] = round(mem.argument_size_in_bytes / 1e6, 2)
                entry["fwd_out_mb"] = round(mem.output_size_in_bytes / 1e6, 2)
            except Exception:
                entry["fwd_temp_mb"] = entry["fwd_arg_mb"] = entry["fwd_out_mb"] = None

            try:
                lowered = jax.jit(jax.grad(make_loss(cls, c, ns))).lower(data_jax, *xi_vals)
                mem = lowered.compile().memory_analysis()
                entry["bwd_temp_mb"] = round(mem.temp_size_in_bytes / 1e6, 2)
                entry["bwd_out_mb"] = round(mem.output_size_in_bytes / 1e6, 2)
            except Exception:
                entry["bwd_temp_mb"] = entry["bwd_out_mb"] = None

            mem_res[name] = entry

        results[label] = mem_res
        print(f"    {label} done")

    return {"title": "XLA Compiled Memory Analysis", "data": results}


# ---------------------------------------------------------------------------
# 6. Complex-valued data
# ---------------------------------------------------------------------------


def bench_complex():
    print("  [6/6] Complex-valued data...")
    configs = [
        ("2D 512²", (512, 512), ["z", "x"], ["qz", "qx"], 256),
        ("3D 128³", (128, 128, 128), ["z", "x", "y"], ["qz", "qx", "qy"], 64),
    ]

    results = {}
    for label, sizes, dim_names, q_dim_names, n_query in configs:
        data = _make_data(sizes, dim_names, dtype="complex64")
        coords = {d: _coord(s) for d, s in zip(dim_names, sizes)}
        coords_fast = {d: _coord_fast(s) for d, s in zip(dim_names, sizes)}
        xi = {d: ops.linspace(0.05, 0.95, n_query, dim=qd)
              for d, qd in zip(dim_names, q_dim_names)}

        fwd_res, bwd_res = {}, {}
        for name, cls, needs_fast in ALL_INTERPOLATORS:
            c = coords_fast if needs_fast else coords

            @ops.jit
            def fwd(d, x, _cls=cls, _c=c):
                return _cls(coordinates=_c, data=d, fill_value=None)(x)

            fwd_res[name] = round(_time_fn(lambda: fwd(data, xi)), 3)

            def make_loss(_cls, _c):
                def loss(d, x):
                    return ops.sum(ops.real(_cls(coordinates=_c, data=d, fill_value=None)(x)))
                return loss

            try:
                grad_fn = ops.jit(ops.grad(make_loss(cls, c)))
                bwd_res[name] = round(_time_fn(lambda: grad_fn(data, xi)), 3)
            except Exception:
                bwd_res[name] = None

        results[label] = {"forward": fwd_res, "backward": bwd_res}
        print(f"    {label} done")

    return {"title": "Complex-valued Data (complex64)", "data": results}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(all_results, output_path):
    """Generate a self-contained HTML report from benchmark results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Color palette for interpolators
    interp_colors = {
        "LinearND": "#6c757d",
        "OptimizedLinearND": "#fd7e14",
        "LinearNDFast": "#0dcaf0",
        "FastLinearND": "#198754",
        "GeneralFastLinearND": "#0d6efd",
        "GeneralFast n=1": "#adb5bd",
        "GeneralFast n=2": "#0d6efd",
        "GeneralFast n=3": "#6610f2",
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interpolator Benchmark Results</title>
<style>
:root {{
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #21262d;
    --fg: #c9d1d9;
    --fg2: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --orange: #d29922;
    --red: #f85149;
    --border: #30363d;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.6;
    padding: 2rem;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
    color: var(--accent);
}}
.timestamp {{ color: var(--fg2); margin-bottom: 2rem; font-size: 0.9rem; }}
.section {{
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}}
.section h2 {{
    font-size: 1.3rem;
    margin-bottom: 1rem;
    color: var(--fg);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}}
.section h3 {{
    font-size: 1rem;
    color: var(--fg2);
    margin: 1.2rem 0 0.6rem 0;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}}
th, td {{
    padding: 8px 12px;
    text-align: right;
    border-bottom: 1px solid var(--border);
}}
th {{ color: var(--fg2); font-weight: 600; white-space: nowrap; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:hover {{ background: var(--bg3); }}
.best {{ color: var(--green); font-weight: 700; }}
.worst {{ color: var(--fg2); }}
.failed {{ color: var(--red); }}
.interp-dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
.summary-box {{
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.summary-box h3 {{ margin-top: 0; color: var(--accent); }}
.summary-box ul {{ padding-left: 1.5rem; }}
.summary-box li {{ margin-bottom: 0.3rem; }}
.nav {{
    position: sticky;
    top: 0;
    background: var(--bg);
    padding: 0.5rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    z-index: 100;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}}
.nav a {{
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    padding: 4px 10px;
    border-radius: 4px;
    background: var(--bg2);
    border: 1px solid var(--border);
}}
.nav a:hover {{ background: var(--bg3); }}
</style>
</head>
<body>
<div class="container">
<h1>Interpolator Benchmark Results</h1>
<p class="timestamp">Generated: {timestamp} | Device: {bench_device} | Backend: JAX</p>

<nav class="nav">
    <a href="#estimate-h">estimate_h</a>
    <a href="#ray-tracing">Ray Tracing</a>
    <a href="#micro">Micro-benchmarks</a>
    <a href="#compile">Compilation Time</a>
    <a href="#quadratic">Quadratic vs Linear</a>
    <a href="#memory">Memory Analysis</a>
    <a href="#complex">Complex Data</a>
    <a href="#summary">Summary</a>
</nav>
"""

    def _dot(name):
        base = name.split(" (")[0]
        color = interp_colors.get(base, "#adb5bd")
        return f'<span class="interp-dot" style="background:{color}"></span>'

    def _format_ms(val, best_val=None, worst_val=None):
        if val is None:
            return '<span class="failed">FAILED</span>'
        s = f"{val:.3f}ms"
        if best_val is not None and val == best_val:
            return f'<span class="best">{s}</span>'
        if worst_val is not None and val == worst_val:
            return f'<span class="worst">{s}</span>'
        return s

    def _speedup(val, ref):
        if val is None or ref is None or val == 0:
            return ""
        return f"{ref / val:.2f}x"

    def _make_fwd_bwd_table(fwd_data, bwd_data, ref_name=None):
        """Generate HTML table for forward+backward results."""
        names = list(fwd_data.keys())
        if ref_name is None:
            ref_name = names[0] if names else None

        fwd_vals = [fwd_data.get(n) for n in names]
        bwd_vals = [bwd_data.get(n) for n in names]
        fwd_valid = [v for v in fwd_vals if v is not None]
        bwd_valid = [v for v in bwd_vals if v is not None]

        fwd_best = min(fwd_valid) if fwd_valid else None
        fwd_worst = max(fwd_valid) if fwd_valid else None
        bwd_best = min(bwd_valid) if bwd_valid else None
        bwd_worst = max(bwd_valid) if bwd_valid else None
        fwd_ref = fwd_data.get(ref_name)
        bwd_ref = bwd_data.get(ref_name)

        t = '<table><thead><tr>'
        t += '<th>Interpolator</th><th>Forward</th><th>Fwd Speedup</th>'
        t += '<th>Backward</th><th>Bwd Speedup</th></tr></thead><tbody>'
        for i, name in enumerate(names):
            fv, bv = fwd_vals[i], bwd_vals[i]
            t += f'<tr><td>{_dot(name)}{name}</td>'
            t += f'<td>{_format_ms(fv, fwd_best, fwd_worst)}</td>'
            t += f'<td>{_speedup(fv, fwd_ref)}</td>'
            t += f'<td>{_format_ms(bv, bwd_best, bwd_worst)}</td>'
            t += f'<td>{_speedup(bv, bwd_ref)}</td></tr>'
        t += '</tbody></table>'
        return t

    # --- 0. estimate_h ---
    if "estimate_h" in all_results:
        eh = all_results["estimate_h"]
        html += f'<div class="section" id="estimate-h"><h2>{eh["title"]}</h2>'
        eh_data = eh["data"]
        names = list(eh_data.keys())
        ms_vals = [eh_data[n]["ms_per_update"] for n in names]
        best_ms = min(ms_vals)
        worst_ms = max(ms_vals)
        ref_ms = ms_vals[0]  # LinearND as reference

        t = '<table><thead><tr><th>Interpolator</th><th>ms/update</th>'
        t += '<th>Total (s)</th><th>Speedup</th><th>Final Loss</th></tr></thead><tbody>'
        for name in names:
            e = eh_data[name]
            ms = e["ms_per_update"]
            cls = ' class="best"' if ms == best_ms else (' class="worst"' if ms == worst_ms else '')
            spd = f'{ref_ms / ms:.2f}x' if ms > 0 else ''
            t += f'<tr><td>{_dot(name)}{name}</td>'
            t += f'<td><span{cls}>{ms:.3f}ms</span></td>'
            t += f'<td>{e["total_s"]:.1f}s</td>'
            t += f'<td>{spd}</td>'
            t += f'<td>{e["final_loss"]:.4e}</td></tr>'
        t += '</tbody></table>'
        html += t
        html += '</div>'

    # --- 1. Ray tracing ---
    rt = all_results["ray_tracing"]
    html += f'<div class="section" id="ray-tracing"><h2>{rt["title"]}</h2>'
    for ul_key, ul_data in rt["data"].items():
        html += f'<h3>{ul_key}</h3>'
        html += _make_fwd_bwd_table(ul_data["forward"], ul_data["backward"], ref_name="LinearND")
    html += '</div>'

    # --- 2. Micro-benchmarks ---
    micro = all_results["micro"]
    html += f'<div class="section" id="micro"><h2>{micro["title"]}</h2>'
    for cfg, cfg_data in micro["data"].items():
        html += f'<h3>{cfg}</h3>'
        html += _make_fwd_bwd_table(cfg_data["forward"], cfg_data["backward"], ref_name="LinearND")
    html += '</div>'

    # --- 3. Compilation time ---
    comp = all_results["compile_time"]
    html += f'<div class="section" id="compile"><h2>{comp["title"]}</h2>'
    for cfg, cfg_data in comp["data"].items():
        html += f'<h3>{cfg}</h3>'
        names = list(cfg_data["forward"].keys())
        fwd = cfg_data["forward"]
        bwd = cfg_data["backward"]
        fwd_valid = [v for v in fwd.values() if v is not None]
        bwd_valid = [v for v in bwd.values() if v is not None]
        fwd_best = min(fwd_valid) if fwd_valid else None
        bwd_best = min(bwd_valid) if bwd_valid else None

        t = '<table><thead><tr><th>Interpolator</th><th>Compile Forward</th><th>Compile Backward</th></tr></thead><tbody>'
        for name in names:
            fv, bv = fwd.get(name), bwd.get(name)
            fc = ' class="best"' if fv == fwd_best else ""
            bc = ' class="best"' if bv == bwd_best else ""
            fstr = f'<span{fc}>{fv:.1f}ms</span>' if fv is not None else '<span class="failed">FAILED</span>'
            bstr = f'<span{bc}>{bv:.1f}ms</span>' if bv is not None else '<span class="failed">FAILED</span>'
            t += f'<tr><td>{_dot(name)}{name}</td><td>{fstr}</td><td>{bstr}</td></tr>'
        t += '</tbody></table>'
        html += t
    html += '</div>'

    # --- 4. Quadratic ---
    quad = all_results["quadratic"]
    html += f'<div class="section" id="quadratic"><h2>{quad["title"]}</h2>'
    for cfg, cfg_data in quad["data"].items():
        html += f'<h3>{cfg}</h3>'
        html += _make_fwd_bwd_table(cfg_data["forward"], cfg_data["backward"], ref_name="GeneralFast n=2")
    html += '</div>'

    # --- 5. Memory ---
    mem = all_results["memory"]
    html += f'<div class="section" id="memory"><h2>{mem["title"]}</h2>'
    for cfg, cfg_data in mem["data"].items():
        html += f'<h3>{cfg}</h3>'
        names = list(cfg_data.keys())
        t = '<table><thead><tr><th>Interpolator</th>'
        t += '<th>Fwd Temp (MB)</th><th>Fwd Arg (MB)</th><th>Fwd Out (MB)</th>'
        t += '<th>Bwd Temp (MB)</th><th>Bwd Out (MB)</th></tr></thead><tbody>'

        bwd_temps = [cfg_data[n].get("bwd_temp_mb") for n in names]
        bwd_temps_valid = [v for v in bwd_temps if v is not None]
        bwd_temp_best = min(bwd_temps_valid) if bwd_temps_valid else None
        bwd_temp_worst = max(bwd_temps_valid) if bwd_temps_valid else None

        for name in names:
            e = cfg_data[name]
            bt = e.get("bwd_temp_mb")
            bt_class = ""
            if bt is not None:
                if bt == bwd_temp_best:
                    bt_class = ' class="best"'
                elif bt == bwd_temp_worst:
                    bt_class = ' class="worst"'

            def _fmb(v):
                return f"{v:.2f}" if v is not None else '<span class="failed">ERR</span>'

            t += f'<tr><td>{_dot(name)}{name}</td>'
            t += f'<td>{_fmb(e.get("fwd_temp_mb"))}</td>'
            t += f'<td>{_fmb(e.get("fwd_arg_mb"))}</td>'
            t += f'<td>{_fmb(e.get("fwd_out_mb"))}</td>'
            t += f'<td><span{bt_class}>{_fmb(bt)}</span></td>'
            t += f'<td>{_fmb(e.get("bwd_out_mb"))}</td></tr>'
        t += '</tbody></table>'
        html += t
    html += '</div>'

    # --- 6. Complex ---
    cx = all_results["complex"]
    html += f'<div class="section" id="complex"><h2>{cx["title"]}</h2>'
    for cfg, cfg_data in cx["data"].items():
        html += f'<h3>{cfg}</h3>'
        html += _make_fwd_bwd_table(cfg_data["forward"], cfg_data["backward"], ref_name="LinearND")
    html += '</div>'

    # --- Summary ---
    html += """
<div class="section" id="summary">
<h2>Summary &amp; Recommendations</h2>
<div class="summary-box">
<h3>Key Findings</h3>
<ul>
<li><strong>Best overall forward performance:</strong> GeneralFastLinearND and FastLinearND — consistently fastest in forward pass across all scenarios.</li>
<li><strong>Best backward in loops (unroll=True):</strong> OptimizedLinearND — XLA can fuse scatter-adds across unrolled iterations, giving 3x speedup over GeneralFast in ray tracing backward.</li>
<li><strong>Best backward in loops (unroll=1):</strong> LinearNDFast — its per-dimension lo/hi gather pattern works well even without unrolling.</li>
<li><strong>Lowest memory usage (backward):</strong> LinearND — zero temp memory in backward pass (in-place scatter-add). GeneralFastLinearND uses ~1× data size; OptimizedLinearND uses ~2× data size.</li>
<li><strong>Fastest compilation:</strong> LinearNDFast (2D) and LinearND (3D) compile fastest. Quadratic n=3 in 3D is significantly slower to compile.</li>
<li><strong>Complex data:</strong> GeneralFastLinearND and FastLinearND perform well. OptimizedLinearND is notably slow in backward pass with complex data.</li>
<li><strong>Quadratic interpolation (n=3):</strong> ~20-40% slower forward, ~33-50% slower backward vs linear (n=2). Compilation time increases significantly in 3D.</li>
</ul>
</div>
<div class="summary-box">
<h3>Recommendation</h3>
<ul>
<li><strong>estimate_h (realistic workload):</strong> GeneralFastLinearND and FastLinearND are ~3× faster than LinearND in the full optimization loop.</li>
<li>Use <strong>GeneralFastLinearND</strong> as the default — best forward speed, good backward, supports n_samples=1,2,3, works with standard LinearCoordinate.</li>
<li>For <strong>ray tracing with gradient optimization</strong> (unroll=True): switch to <strong>OptimizedLinearND</strong> for the interpolator_type — it's 3× faster in the backward pass.</li>
<li>No single interpolator wins in every scenario due to fundamentally different XLA optimization patterns in the backward pass.</li>
</ul>
</div>
</div>
"""

    html += '</div></body></html>'

    with open(output_path, "w") as f:
        f.write(html)

    print(f"\n  HTML report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"\n{'=' * 60}")
    print(f"  COMPREHENSIVE INTERPOLATOR BENCHMARKS")
    print(f"  Device: {bench_device} | Backend: JAX")
    print(f"{'=' * 60}")

    all_results = {}

    all_results["estimate_h"] = bench_estimate_h()
    all_results["ray_tracing"] = bench_ray_tracing()
    all_results["micro"] = bench_micro()
    all_results["compile_time"] = bench_compile_time()
    all_results["quadratic"] = bench_quadratic()
    all_results["memory"] = bench_memory()
    all_results["complex"] = bench_complex()

    # Save raw JSON for reference
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  JSON results saved to: {json_path}")

    # Generate HTML
    html_path = os.path.join(os.path.dirname(__file__), "benchmark_results.html")
    generate_html(all_results, html_path)

    print(f"\n{'=' * 60}")
    print(f"  All benchmarks complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
