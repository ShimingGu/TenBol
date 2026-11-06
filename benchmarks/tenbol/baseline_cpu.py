#!/usr/bin/env python
"""Tuned parallel-CPU CAMB baseline on this allocation.

Measures the complete reference pipeline (background, thermal history, transfer,
linear spectrum, sigma(R), HMcode, result assembly) for batches of *distinct*
cosmologies with warmed worker processes.  Cold start (interpreter + BBN table
load + first CAMB call) is measured separately and reported, not hidden.

Run: PYTHONPATH=. python benchmarks/tenbol/baseline_cpu.py --batch 32 --workers 16
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
    _init_worker,
    _worker,
)

SETTINGS = ReferenceSettings()


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


def timed_batch(batch, workers, pool=None):
    args = [(c, (0.0, 0.5, 1.0, 2.0)) for c in batch]
    t0 = time.perf_counter()
    if pool is not None:
        pool.map(_worker, args)
    else:
        for a in args:
            _worker(a)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profiles/tenbol/evidence")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {"settings": SETTINGS.__dict__, "batch_size": args.batch, "node_cpus": os.cpu_count()}

    # cold import/first-call cost
    t0 = time.perf_counter()
    reference_run(Cosmology(0.67, 0.0224, 0.120, 2.1e-9, 0.965, 0.06), SETTINGS)
    results["cold_first_call_s"] = time.perf_counter() - t0

    batch = make_batch(args.batch)
    timings = {}
    _init_worker(SETTINGS)
    for workers in args.workers:
        if workers > args.batch:
            continue
        if workers == 1:
            _init_worker(SETTINGS)
            timed_batch(batch[:2], 1)                # warm-up
            ts = []
            for _ in range(args.repeats):
                ts.append(timed_batch(batch, 1))
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(SETTINGS,)) as pool:
                timed_batch(batch[: min(workers, args.batch)], workers, pool)   # warm-up
                ts = []
                for _ in range(args.repeats):
                    ts.append(timed_batch(batch, workers, pool))
        timings[str(workers)] = {
            "times": ts,
            "median": float(np.median(ts)),
            "min": float(np.min(ts)),
            "evaluations_per_s": float(len(batch) / np.median(ts)),
        }
        print(f"workers={workers:2d}  median={np.median(ts):.3f}s  "
              f"({len(batch)/np.median(ts):.1f} evals/s)  times={[round(t,3) for t in ts]}")

    results["timings"] = timings
    path = os.path.join(args.out, "baseline_cpu.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
