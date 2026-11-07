"""Accuracy suite over the workload domain.

Draws ``N`` cosmologies from the frozen workload range (seed 20261106), runs
them as a single batch through :func:`tenbol.power.matter_power`, and compares
the k-resolved growth against CAMB 2.0.5.

Comparison is on the *growth ratio* ``delta(z) / delta(z=1000)`` rather than the
absolute spectrum: our delta is normalised to the primordial curvature
perturbation with a different convention from CAMB's transfer columns (the two
differ by a constant ~1e-4 factor), and the ratio is independent of it.  The
absolute normalisation is a separate, still-open item.

Usage::

    PYTHONPATH=. python benchmarks/tenbol/accuracy_suite.py --n 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tenbol.adapters.camb_reference import Cosmology, ReferenceSettings, reference_run
from tenbol.physics.background import CosmologyBatch
from tenbol.power import matter_power

SEED = 20261106
# Bounds of profiles/tenbol/workload_matter_v1.json
BOUNDS = {
    "omega_b": (0.0200, 0.0245),
    "omega_c": (0.1000, 0.1600),
    "h": (0.600, 0.800),
    "A_s": (1.5e-9, 2.7e-9),
    "n_s": (0.92, 1.02),
    "m_nu": (0.06, 0.30),
}


def draw(n: int, seed: int = SEED) -> CosmologyBatch:
    rng = np.random.default_rng(seed)

    def u(key):
        lo, hi = BOUNDS[key]
        return rng.uniform(lo, hi, n)

    return CosmologyBatch.from_columns(
        h=u("h"), omega_b=u("omega_b"), omega_c=u("omega_c"),
        A_s=u("A_s"), n_s=u("n_s"),
        m_nu=u("m_nu")[:, None], N_eff=[3.046] * n)


def camb_growth(cosmology: Cosmology, kh, z_ref=1000.0, z_out=0.0):
    ref = reference_run(cosmology, ReferenceSettings(),
                        redshifts=[z_ref, z_out], npoints=512, want_transfer=True)
    q = np.asarray(ref["transfer_q"]) / cosmology.h
    zt = [float(x) for x in np.asarray(ref["transfer_z"])]
    tr = ref["transfer"]
    i_ref = int(np.argmin(np.abs(np.asarray(zt) - z_ref)))
    i_out = int(np.argmin(np.abs(np.asarray(zt) - z_out)))
    out = []
    for k in kh:
        a = np.interp(np.log(k), np.log(q), tr["delta_tot"][:, i_ref])
        b = np.interp(np.log(k), np.log(q), tr["delta_tot"][:, i_out])
        # out is delta(z_out)/delta(z_ref), the same orientation as
        # PowerResult.growth.  Writing a/b here made the comparison ratio the
        # square of the real one, which is what produced the 1.5e8 % readings.
        out.append(b / a)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    k_h = np.geomspace(1e-3, 0.1, args.k)
    cosmo = draw(args.n)

    t0 = time.time()
    res = matter_power(cosmo, k_h, [1000.0, 0.0], chunk_size=args.chunk)
    ours = res.growth(1000.0)[:, :, 1]           # (n_cosmo, n_k)
    t_ours = time.time() - t0

    ratios = np.empty_like(ours)
    for i in range(len(cosmo.h)):
        c = Cosmology(h=float(cosmo.h[i]), omega_b=float(cosmo.omega_b[i]),
                      omega_c=float(cosmo.omega_c[i]), A_s=float(cosmo.A_s[i]),
                      n_s=float(cosmo.n_s[i]), m_nu_sum=float(cosmo.m_nu[i][0]))
        ratios[i] = ours[i] / camb_growth(c, k_h)

    dev = np.abs(ratios - 1.0)
    print(f"n_cosmo={args.n}  k={k_h.size}  steps={res.n_steps}  "
          f"tenbol {t_ours:.1f}s")
    print(f"{'k [h/Mpc]':>11} {'median':>9} {'p95':>9} {'max':>9}")
    for j, k in enumerate(k_h):
        col = dev[:, j]
        print(f"{k:11.4f} {np.median(col):9.4%} {np.percentile(col, 95):9.4%} {col.max():9.4%}")
    print(f"worst overall: {dev.max():.4%}   mean: {dev.mean():.4%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "seed": SEED, "n_cosmo": args.n, "k_h": k_h.tolist(),
            "growth_ratio_vs_camb": ratios.tolist(),
            "max_dev": float(dev.max()), "median_dev": float(np.median(dev)),
            "steps": int(res.n_steps), "wall_s": t_ours,
        }, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
