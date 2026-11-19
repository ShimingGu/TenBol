#!/usr/bin/env python
"""Tuned parallel-CPU CAMB baseline on this allocation.

Measures the complete reference pipeline (background, thermal history, transfer,
linear spectrum, sigma(R), HMcode, result assembly) for batches of *distinct*
cosmologies with warmed worker processes.  Cold start (interpreter + BBN table
load + first CAMB call) is measured separately and reported, not hidden.

Run: PYTHONPATH=. python benchmarks/tenbol/baseline_cpu.py --batch 32 --workers 16 --cuda
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from tenbol.adapters.camb_reference import (
    Cosmology,
    ReferenceSettings,
    reference_run,
)

SETTINGS = ReferenceSettings()
REDSHIFTS = (0.0, 0.5, 1.0, 2.0)
K_H = np.geomspace(1e-4, 10.0, 256)


def make_batch(n: int, seed: int = 20261106):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(
            Cosmology(
                h=float(rng.uniform(0.60, 0.80)),
                omega_b=float(rng.uniform(0.020, 0.0245)),
                omega_c=float(rng.uniform(0.10, 0.16)),
                A_s=float(rng.uniform(1.5e-9, 2.7e-9)),
                n_s=float(rng.uniform(0.92, 1.02)),
                m_nu_sum=float(rng.uniform(0.06, 0.30)),
            )
        )
    return out


def reference_power(cosmology):
    return reference_run(cosmology, SETTINGS, redshifts=REDSHIFTS,
                         minkh=K_H[0], maxkh=K_H[-1], npoints=K_H.size)["pk_nlin"].T


def timed_batch(batch, pool=None):
    t0 = time.perf_counter()
    if pool is not None:
        values = pool.map(reference_power, batch)
    else:
        values = [reference_power(cosmology) for cosmology in batch]
    return time.perf_counter() - t0, np.stack(values)


def benchmark_cuda(batch, reference, repeats):
    from tenbol.physics.background import CosmologyBatch
    from tenbol.power import clear_cache, matter_power_nl

    params = CosmologyBatch.from_columns(
        h=[c.h for c in batch], omega_b=[c.omega_b for c in batch],
        omega_c=[c.omega_c for c in batch], A_s=[c.A_s for c in batch],
        n_s=[c.n_s for c in batch], m_nu=[c.m_nu_sum for c in batch],
    )
    # Warm the CUDA context and compiled kernels. Each measured batch rebuilds
    # the background and thermal history for its input parameters.
    matter_power_nl(params, K_H, REDSHIFTS, backend="cuda")
    times = []
    for _ in range(repeats):
        clear_cache()
        t0 = time.perf_counter()
        result = matter_power_nl(params, K_H, REDSHIFTS, backend="cuda")
        times.append(time.perf_counter() - t0)
    difference = np.abs(result / reference - 1.0)
    ci, ki, zi = np.unravel_index(np.argmax(difference), difference.shape)
    return {
        "times": times, "median": float(np.median(times)),
        "max_relative_error": float(difference[ci, ki, zi]),
        "worst_point": {"cosmology": int(ci), "k_h": float(K_H[ki]), "z": REDSHIFTS[zi]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profiles/tenbol/evidence")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cuda", action="store_true", help="also time and validate Tenbol's CUDA pipeline")
    args = ap.parse_args()
    if args.batch < 1 or args.repeats < 1 or not any(1 <= w <= args.batch for w in args.workers):
        ap.error("provide positive batch and repeat counts and at least one usable worker count")
    os.makedirs(args.out, exist_ok=True)

    results = {"settings": SETTINGS.__dict__, "batch_size": args.batch, "node_cpus": os.cpu_count()}

    # cold import/first-call cost
    t0 = time.perf_counter()
    reference_power(Cosmology(0.67, 0.0224, 0.120, 2.1e-9, 0.965, 0.06))
    results["cold_first_call_s"] = time.perf_counter() - t0

    batch = make_batch(args.batch)
    timings = {}
    for workers in args.workers:
        if not 1 <= workers <= args.batch:
            continue
        if workers == 1:
            ts = []
            for _ in range(args.repeats):
                elapsed, reference = timed_batch(batch)
                ts.append(elapsed)
        else:
            # CAMB's OpenMP runtime cannot safely be inherited from the parent.
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                timed_batch(batch[:workers], pool)
                ts = []
                for _ in range(args.repeats):
                    elapsed, reference = timed_batch(batch, pool)
                    ts.append(elapsed)
        timings[str(workers)] = {
            "times": ts,
            "median": float(np.median(ts)),
            "min": float(np.min(ts)),
            "evaluations_per_s": float(len(batch) / np.median(ts)),
        }
        print(f"workers={workers:2d}  median={np.median(ts):.3f}s  "
              f"({len(batch)/np.median(ts):.1f} evals/s)  times={[round(t,3) for t in ts]}")

    results["timings"] = timings
    if args.cuda:
        results["cuda"] = benchmark_cuda(batch, reference, args.repeats)
        best_cpu = min(run["median"] for run in timings.values())
        results["cuda"]["speedup_vs_best_cpu"] = best_cpu / results["cuda"]["median"]
        print("CUDA", json.dumps(results["cuda"]))
    path = os.path.join(args.out, "baseline_cpu.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
