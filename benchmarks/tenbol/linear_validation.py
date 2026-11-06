#!/usr/bin/env python
"""Linear-spectrum validation harness (candidate vs CAMB reference).

Records, on the manifest's k/z grid for a batch of cosmologies:

* the maximum relative difference in linear total-matter power;
* the same after normalising both spectra at one reference scale, which
  separates a *shape* error from a *normalisation* error;
* the redshift-by-redshift breakdown.

The result file is machine readable and is the evidence referenced by
``Tenbol_progress.md``.  Run with::

    PYTHONPATH=. python benchmarks/tenbol/linear_validation.py --out profiles/tenbol/evidence
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from tenbol.adapters.camb_reference import Cosmology, ReferenceSettings, reference_run
from tenbol.physics.background import Background, CosmologyBatch
from tenbol.physics.perturbations import PerturbationBatch, PerturbationSettings, evolve
from tenbol.physics.thermal import Reionization, ThermalHistory

Z_OUT = (0.0, 0.5, 1.0, 2.0)
KH = np.geomspace(1.0e-4, 10.0, 64)
REF_NORM_KH = 1.0e-2


def candidate_pk(cosmo: Cosmology, kh, z_out, settings: PerturbationSettings):
    """Candidate linear P(k, z) in (Mpc/h)^3 on the requested h/Mpc grid."""
    batch = CosmologyBatch.from_columns(
        h=[cosmo.h], omega_b=[cosmo.omega_b], omega_c=[cosmo.omega_c],
        A_s=[cosmo.A_s], n_s=[cosmo.n_s], m_nu=[cosmo.m_nu_sum], N_eff=[cosmo.N_eff],
        T_cmb=cosmo.T_cmb,
    )
    bg = Background(batch)
    th = ThermalHistory(bg, Reionization(tau=0.0544))
    k_phys = kh * cosmo.h
    pb = PerturbationBatch(bg, th, k_phys, np.zeros(kh.size, dtype=int), settings)
    t0 = time.perf_counter()
    rec = evolve(pb, z_out=list(z_out), a_end=1.0)
    wall = time.perf_counter() - t0
    dm = rec["dm"]
    pref = (2.0 * np.pi**2 / k_phys**3) * cosmo.A_s * (k_phys / cosmo.pivot_k) ** (cosmo.n_s - 1.0)
    pk = pref[:, None] * dm**2
    pk = pk * cosmo.h**3                      # (Mpc)^3 -> (Mpc/h)^3
    return pk, wall, rec["n_steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profiles/tenbol/evidence")
    ap.add_argument("--n-cosmologies", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(20261106)
    cosmologies = [Cosmology(h=0.67, omega_b=0.0224, omega_c=0.120, A_s=2.1e-9,
                             n_s=0.965, m_nu_sum=0.06)]
    for _ in range(args.n_cosmologies - 1):
        cosmologies.append(Cosmology(
            h=float(rng.uniform(0.60, 0.80)), omega_b=float(rng.uniform(0.020, 0.0245)),
            omega_c=float(rng.uniform(0.10, 0.16)), A_s=float(rng.uniform(1.5e-9, 2.7e-9)),
            n_s=float(rng.uniform(0.92, 1.02)), m_nu_sum=float(rng.uniform(0.06, 0.30))))

    settings = PerturbationSettings()
    ref_settings = ReferenceSettings()
    results = {"kh": KH.tolist(), "z": list(Z_OUT), "cosmologies": [],
               "solver_settings": settings.__dict__}
    for cosmo in cosmologies:
        ref = reference_run(cosmo, ref_settings, redshifts=Z_OUT, npoints=256)
        pk_ref = np.array([np.exp(np.interp(np.log(KH), np.log(ref["kh"]), np.log(row)))
                           for row in ref["pk_lin"]]).T     # -> (n_k, n_z)
        pk_ours, wall, steps = candidate_pk(cosmo, KH, Z_OUT, settings)
        rel = pk_ours / pk_ref - 1.0
        i_norm = int(np.argmin(np.abs(np.log(KH / REF_NORM_KH))))
        ratio = pk_ours / pk_ref
        shape = ratio / ratio[i_norm, :][None, :] - 1.0
        i_flat = int(np.argmax(np.abs(rel)))
        i_k, i_z = np.unravel_index(i_flat, rel.shape)
        entry = {
            "params": cosmo.as_dict(),
            "wall_time_s": wall,
            "n_steps": int(steps),
            "max_rel": float(rel[i_k, i_z]),
            "argmax": {"z": float(Z_OUT[i_z]), "kh": float(KH[i_k])},
            "max_rel_per_z": [float(np.max(np.abs(rel[:, i]))) for i in range(len(Z_OUT))],
            "max_shape_rel": float(np.max(np.abs(shape))),
            "shape_per_z": [float(np.max(np.abs(shape[:, i]))) for i in range(len(Z_OUT))],
            "ratio_profile_z0": ratio[:, 0].tolist(),
        }
        results["cosmologies"].append(entry)
        print(json.dumps({k: entry[k] for k in ("max_rel", "max_shape_rel", "wall_time_s", "n_steps")}))

    path = os.path.join(args.out, "linear_validation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
