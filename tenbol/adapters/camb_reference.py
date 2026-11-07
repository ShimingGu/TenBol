"""Reference adapter: CAMB 2.0.5 used for *validation only*.

Nothing in the Tenbol production path imports this module.  It is the
independently implemented reference required by §15.1 of the design, pinned to
an exact package version and an explicit set of settings so that candidate and
reference agree on physics, gauge, units, matter definition and HMcode
configuration.

The settings recorded here are the ones frozen in
``profiles/tenbol/workload_matter_v1.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

# Parallel worker pools must not oversubscribe the node: force single-threaded
# BLAS/OpenMP in the reference adapter (the outer process pool provides the
# parallelism).  Left as an override, not a default, so that a deliberate
# threaded single-process run can be requested explicitly.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if os.environ.get("TENBOL_ALLOW_THREADED_BLAS") != "1":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import camb  # noqa: E402


# --------------------------------------------------------------------------
# Workload definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM + thermal massive-neutrino eigenstates (degenerate masses)."""

    h: float
    omega_b: float
    omega_c: float
    A_s: float
    n_s: float
    m_nu_sum: float      # eV, total mass in massive eigenstates
    N_eff: float = 3.046
    T_cmb: float = 2.7255
    num_massive: int = 1
    pivot_k: float = 0.05   # 1/Mpc

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceSettings:
    """Pinned CAMB settings.  ``kmax_h`` is the internal spectrum maximum."""

    accuracy_boost: float = 2.0
    l_accuracy_boost: float = 2.0
    time_step_boost: float = 2.0
    l_sample_boost: float = 1.0
    k_per_logint: int = 40
    kmax_h: float = 20.0
    halofit: str = "mead2020"
    hmcode_a_baryon: float = 3.13
    hmcode_eta_baryon: float = 0.603
    hmcode_log_t_agn: float = 7.8
    hmcode_wiggle_max_fnu: float = 0.01
    recomb_fudge: float | None = None      # None -> CAMB default parameter set
    recomb_fudge_he: float | None = None
    recomb_he_rate_correction: bool | None = None
    recomb_hswitch: bool | None = None
    tau_reio: float = 0.0544
    dark_energy_w: float = -1.0

    def camb_params(self, cosmo: Cosmology, redshifts: Sequence[float]):
        p = camb.CAMBparams()
        p.set_cosmology(
            H0=cosmo.h * 100.0,
            ombh2=cosmo.omega_b,
            omch2=cosmo.omega_c,
            mnu=cosmo.m_nu_sum,
            num_massive_neutrinos=cosmo.num_massive,
            nnu=cosmo.N_eff,
            TCMB=cosmo.T_cmb,
            tau=self.tau_reio,
        )
        p.set_dark_energy(w=self.dark_energy_w)
        p.InitPower.set_params(As=cosmo.A_s, ns=cosmo.n_s, pivot_scalar=cosmo.pivot_k)
        p.set_matter_power(
            redshifts=list(redshifts),
            kmax=self.kmax_h,
            k_per_logint=self.k_per_logint,
        )
        p.NonLinearModel.set_params(
            halofit_version=self.halofit,
            HMCode_A_baryon=self.hmcode_a_baryon,
            HMCode_eta_baryon=self.hmcode_eta_baryon,
            HMCode_logT_AGN=self.hmcode_log_t_agn,
            HMCode_wiggle_max_fnu=self.hmcode_wiggle_max_fnu,
        )
        p.set_accuracy(
            AccuracyBoost=self.accuracy_boost,
            lAccuracyBoost=self.l_accuracy_boost,
            lSampleBoost=self.l_sample_boost,
        )
        p.Accuracy.TimeStepBoost = self.time_step_boost
        if self.recomb_fudge is not None:
            p.Recomb.RECFAST_fudge = self.recomb_fudge
        if self.recomb_fudge_he is not None:
            p.Recomb.RECFAST_fudge_He = self.recomb_fudge_he
        if self.recomb_he_rate_correction is not None:
            p.Recomb.RECFAST_He_rate_correction = self.recomb_he_rate_correction
        if self.recomb_hswitch is not None:
            p.Recomb.RECFAST_Hswitch = self.recomb_hswitch
        return p

    def replace(self, **kwargs) -> "ReferenceSettings":
        d = asdict(self)
        d.update(kwargs)
        return ReferenceSettings(**d)


def reference_run(
    cosmo: Cosmology,
    settings: ReferenceSettings,
    redshifts: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    minkh: float = 1.0e-4,
    maxkh: float | None = None,
    npoints: int = 256,
    want_transfer: bool = False,
) -> dict:
    """Evaluate CAMB and return linear/nonlinear matter power on a log k grid.

    Power is CAMB's ``delta_tot`` (total matter, synchronous gauge) in
    (Mpc/h)^3, k in h/Mpc.  No extrapolation beyond the reference internal k
    range is performed.
    """
    p = settings.camb_params(cosmo, redshifts)
    res = camb.get_results(p)
    if maxkh is None:
        maxkh = settings.kmax_h
    if maxkh > settings.kmax_h * (1 + 1e-12):
        raise ValueError("requested k exceeds reference internal kmax")

    kh_raw, z_raw, pk_nl_raw = res.get_nonlinear_matter_power_spectrum(
        var1="delta_tot", var2="delta_tot"
    )
    kh_raw_l, z_raw_l, pk_lin_raw = res.get_linear_matter_power_spectrum(
        var1="delta_tot", var2="delta_tot"
    )
    if not np.allclose(kh_raw, kh_raw_l):
        raise RuntimeError("CAMB linear/nonlinear k grids disagree")

    order = np.argsort(z_raw)
    z_sorted = np.asarray(z_raw)[order]
    pk_nl_raw = pk_nl_raw[order]
    pk_lin_raw = pk_lin_raw[np.argsort(z_raw_l)]

    # Resample with a cubic spline in (log k, log P).  The reference internal
    # grid is dense (k_per_logint), so this adds no measurable error; the
    # magnitude is checked by ``reference_convergence.py``.
    from scipy.interpolate import CubicSpline

    kh = np.geomspace(minkh, maxkh, npoints)
    logk_raw = np.log(np.asarray(kh_raw))
    out_nl = np.array(
        [np.exp(CubicSpline(logk_raw, np.log(row))(np.log(kh))) for row in pk_nl_raw]
    )
    out_lin = np.array(
        [np.exp(CubicSpline(logk_raw, np.log(row))(np.log(kh))) for row in pk_lin_raw]
    )
    out = {
        "kh": kh,
        "z": z_sorted,
        "pk_nlin": out_nl,
        "pk_lin": out_lin,
        "sigma8": np.asarray(res.get_sigma8())[order],
        "params": cosmo.as_dict(),
        "settings": asdict(settings),
    }
    if want_transfer:
        tr = res.get_matter_transfer_data()
        names = ["delta_cdm", "delta_baryon", "delta_photon", "delta_neutrino",
                 "delta_tot", "delta_nonu", "v_baryon_cdm", "v_newtonian_baryon"]
        zcols = np.asarray(p.Transfer.PK_redshifts)[: tr.transfer_data.shape[2]]
        zorder = np.argsort(zcols)[::-1]        # transfer columns are reversed
        zcols = zcols[zorder]
        order_z = np.argsort(zcols)             # ascending, matches `out["z"]`
        out["transfer"] = {
            name: np.array(tr.transfer_data[camb.model.transfer_names.index(name), :, :])[:, order_z]
            for name in names
        }
        out["transfer_q"] = np.asarray(tr.q)
        out["transfer_z"] = zcols[order_z]      # ascending, same order as out["z"]
    return out
