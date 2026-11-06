#!/usr/bin/env python
"""Reference convergence and physics-sensitivity experiments (CAMB only).

Establishes, before any Tenbol solver code is trusted:
  * how converged the CAMB reference is w.r.t. accuracy settings and internal
    k sampling, so that a 0.5% candidate-vs-reference criterion is meaningful;
  * how sensitive final nonlinear power is to choices Tenbol may implement
    differently (recombination parameter set, N_eff rounding, T_CMB);
  * the tuned parallel-CPU CAMB baseline on this cluster.

Run:  python benchmarks/tenbol/reference_convergence.py --out profiles/tenbol/evidence
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np

from tenbol.adapters.camb_reference import (
    Cosmology,
    ReferenceSettings,
    reference_run,
)

FID = Cosmology(h=0.67, omega_b=0.0224, omega_c=0.120, A_s=2.1e-9, n_s=0.965, m_nu_sum=0.06)
GRID_KH = np.geomspace(1.0e-4, 10.0, 256)
BASE = ReferenceSettings()


def compare(pk_a, pk_b):
    """Max relative difference of two [z,k] power arrays and its location."""
    rel = pk_a / pk_b - 1.0
    i = int(np.argmax(np.abs(rel)))
    iz, ik = np.unravel_index(i, rel.shape)
    return float(rel[iz, ik]), int(iz), int(ik)


def run_one(cosmo, settings, npoints=256, minkh=1e-4, maxkh=None, redshifts=(0.0, 0.5, 1.0, 2.0)):
    t0 = time.perf_counter()
    r = reference_run(
        cosmo,
        settings,
        redshifts=redshifts,
        minkh=minkh,
        maxkh=maxkh,
        npoints=npoints,
    )
    r["wall_time"] = time.perf_counter() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="profiles/tenbol/evidence")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {"fiducial": FID.as_dict(), "base_settings": asdict(BASE)}
    timings = {}

    # ---- reference baseline -------------------------------------------------
    t0 = time.perf_counter()
    base = run_one(FID, BASE)
    timings["single_cold"] = time.perf_counter() - t0
    base2 = run_one(FID, BASE)
    timings["single_warm"] = base2["wall_time"]
    results["timings"] = timings
    results["sigma8_base"] = base["sigma8"].tolist()
    # k grid: the returned kh is CAMB's own interpolation grid
    results["kh"] = base["kh"].tolist()
    results["z"] = base["z"].tolist()

    def rel(name, **kw):
        r = run_one(FID, BASE.replace(**kw))
        mx, iz, ik = compare(r["pk_nlin"], base["pk_nlin"])
        mxl, _, _ = compare(r["pk_lin"], base["pk_lin"])
        out = {
            "settings": kw,
            "max_rel_nlin": mx,
            "max_rel_lin": mxl,
            "argmax": {"z": float(base["z"][iz]), "kh": float(base["kh"][ik])},
            "wall_time": r["wall_time"],
        }
        results.setdefault("sensitivity", {})[name] = out
        print(f"{name:40s} nlin={mx:+.2e} lin={mxl:+.2e} at z={base['z'][iz]:g} k={base['kh'][ik]:.3g}")
        return out

    print("== accuracy (AccurBoost / TimeStep / lAccuracy / k sampling) ==")
    rel("accuracy_boost_1", accuracy_boost=1.0, l_accuracy_boost=1.0)
    rel("accuracy_boost_1p5", accuracy_boost=1.5, l_accuracy_boost=1.5, time_step_boost=1.5)
    rel("accuracy_boost_3", accuracy_boost=3.0, l_accuracy_boost=3.0, time_step_boost=3.0)
    rel("time_step_boost_1", time_step_boost=1.0)
    rel("l_accuracy_boost_1", l_accuracy_boost=1.0)
    rel("kperlogint_0", k_per_logint=0)
    rel("kperlogint_20", k_per_logint=20)
    rel("kperlogint_80", k_per_logint=80)
    rel("kperlogint_160", k_per_logint=160)

    # interpolation of the reference onto the user grid
    r128 = run_one(FID, BASE, npoints=128)
    r512 = run_one(FID, BASE, npoints=512)
    base_interp = run_one(FID, BASE, npoints=1024)
    for name, r in (("npoints128", r128), ("npoints512", r512), ("npoints256", base)):
        # compare all on the common 256 grid via interpolation
        pk = np.exp(np.interp(np.log(GRID_KH), np.log(r["kh"]), np.log(r["pk_nlin"][0])))
        pk_ref = np.exp(np.interp(np.log(GRID_KH), np.log(base_interp["kh"]), np.log(base_interp["pk_nlin"][0])))
        mx = float(np.max(np.abs(pk / pk_ref - 1.0)))
        results.setdefault("sensitivity", {})[f"interp_{name}"] = {"max_rel_nlin_z0": mx}
        print(f"interp {name:12s} max rel at z=0 on user grid: {mx:.2e}")

    print("== internal kmax (HMcode integration range) ==")
    # evaluate at k<=10 while CAMB internally covers kmax_h
    r10 = run_one(FID, BASE.replace(kmax_h=10.0), maxkh=10.0)
    r40 = run_one(FID, BASE.replace(kmax_h=40.0), maxkh=10.0)
    for name, r in (("kmax10", r10), ("kmax40", r40)):
        # same kh grid? they differ; resample onto user grid
        pk = np.array([np.exp(np.interp(np.log(GRID_KH), np.log(r["kh"]), np.log(pk_row))) for pk_row in r["pk_nlin"]])
        pk_ref = np.array([np.exp(np.interp(np.log(GRID_KH), np.log(base["kh"]), np.log(pk_row))) for pk_row in base["pk_nlin"]])
        mx = float(np.max(np.abs(pk / pk_ref - 1.0)))
        results.setdefault("sensitivity", {})[name] = {"max_rel_nlin": mx, "wall_time": r["wall_time"]}
        print(f"{name:12s} max rel vs kmax20: {mx:.2e}  (t={r['wall_time']:.2f}s)")

    print("== recombination settings ==")
    rel("recomb_planck", recomb_fudge=1.125, recomb_fudge_he=0.86,
        recomb_he_rate_correction=False, recomb_hswitch=True)
    rel("recomb_no_he_rate_corr", recomb_he_rate_correction=False)
    rel("recomb_hswitch_off", recomb_hswitch=False)
    # large-ish deliberate perturbation to calibrate x_e sensitivity
    rel("recomb_fudge_+1pct", recomb_fudge=1.125 * 1.01)
    rel("recomb_fudge_he_+1pct", recomb_fudge_he=0.8472367977 * 1.01)

    print("== background/thermal inputs ==")
    for name, c in (
        ("Neff_3.044", replace(FID, N_eff=3.044)),
        ("Neff_3.046", FID),
        ("Tcmb_2.725", replace(FID, T_cmb=2.725)),
        ("Tcmb_2.726", replace(FID, T_cmb=2.726)),
    ):
        r = run_one(c, BASE)
        mx, iz, ik = compare(r["pk_nlin"], base["pk_nlin"])
        results.setdefault("sensitivity", {})[name] = {
            "max_rel_nlin": mx, "argmax": {"z": float(base["z"][iz]), "kh": float(base["kh"][ik])}}
        print(f"{name:20s} nlin={mx:+.2e}")

    # (The tuned parallel-CPU timing baseline lives in
    # benchmarks/tenbol/baseline_cpu.py so that a stalled pool cannot block
    # the convergence record.)
    with open(os.path.join(args.out, "reference_convergence.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print("wrote", os.path.join(args.out, "reference_convergence.json"))


if __name__ == "__main__":
    main()
