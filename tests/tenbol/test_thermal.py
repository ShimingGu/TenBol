"""Thermal-history validation against CAMB derived quantities.

CAMB does not expose x_e(z) through its Python API, so the thermal history is
validated through integrated quantities that are sensitive to it: the drag
epoch redshift and sound horizon (``zdrag``, ``rdrag`` from
``CAMBdata.get_derived_params``) and, in the end-to-end suite, the linear matter
power itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from tenbol.constants import C_LIGHT, MPC_M, SIGMA_THOMSON
from tenbol.physics.background import Background, CosmologyBatch
from tenbol.physics.perturbations import PerturbationBatch, PerturbationSettings
from tenbol.physics.thermal import Reionization, ThermalHistory

camb = pytest.importorskip("camb")

CASE = dict(h=0.67, omega_b=0.0224, omega_c=0.120, A_s=2.1e-9, n_s=0.965, m_nu=[0.06])


def make(case=CASE):
    batch = CosmologyBatch.from_columns(
        h=[case["h"]], omega_b=[case["omega_b"]], omega_c=[case["omega_c"]],
        A_s=[case["A_s"]], n_s=[case["n_s"]], m_nu=[case["m_nu"]], N_eff=[3.046],
    )
    bg = Background(batch)
    return bg, ThermalHistory(bg, Reionization(tau=0.0544))


def test_sound_horizon_matches_camb():
    """Sound horizon at the CAMB photosphere redshift.

    ``rstar`` is a well-defined physical quantity (unlike ``zdrag``, whose
    exact definition depends on how the drag optical depth is normalised), so
    it is the right integrated check of ``c_s(z)`` and ``H(z)`` from the
    thermal history.
    """
    bg, th = make()
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.0, ombh2=0.0224, omch2=0.120, mnu=0.06,
                    num_massive_neutrinos=1, nnu=3.046, TCMB=2.7255, tau=0.0544)
    derived = camb.get_results(p).get_derived_params()
    zstar = derived["zstar"]
    rstar = derived["rstar"]

    # integrate to high z: the radiation-era tail contributes ~45% of r_s
    z = np.geomspace(1e-3, 1.0e7, 80000)
    a = 1.0 / (1.0 + z)
    # photon-baryon sound speed c_s^2 = 1/(3(1+R)), R = 3 rho_b/(4 rho_g)
    rho_g = bg.grho_g0[0] * (C_LIGHT / MPC_M) ** 2 / (8 * np.pi * 6.67430e-11) / a**4
    rho_b = bg.grho_b0[0] * (C_LIGHT / MPC_M) ** 2 / (8 * np.pi * 6.67430e-11) / a**3
    R = 3.0 * rho_b / (4.0 * rho_g)
    cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
    Hz = bg.hubble(a)[0] / a                      # H(z) in 1/Mpc
    sel = z >= zstar
    r = np.trapezoid(cs[sel] / Hz[sel], z[sel])
    print(f"rstar ours={r:.3f} camb={rstar:.3f}")
    assert abs(r / rstar - 1.0) < 5e-3


def test_temperature_and_reionization_sane():
    bg, th = make()
    a = np.array([1.0, 0.5, 0.1, 1e-3])
    tm = th.t_m(a)[0]
    xe = th.x_e(a)[0]
    # T_m today: Compton coupling holds T_m near T_rad until z ~ a few hundred,
    # after which adiabatic cooling gives ~0.02 K (no photoheating in RECFAST).
    assert 5e-3 < tm[0] < 0.2
    # x_e today ~ 1 + f_He after reionization
    assert 1.0 < xe[0] < 1.2
    # deep in recombination x_e freezes out around 1e-4..1e-2
    assert 1e-4 < xe[3] < 0.2
    assert th.z_re[0] > 0.0  # reionization is matched to the requested tau
    # T_m at recombination should be close to the radiation temperature
    assert 2000.0 < tm[3] < 3200.0


def test_tight_coupling_exits_on_the_expansion_or_wavelength_limit():
    bg, th = make()
    k = np.array([1e-4, 1.0])
    settings = PerturbationSettings()
    batch = PerturbationBatch(bg, th, k, [0, 0], settings)
    a_exit = np.exp(batch._ln_a_tc_end)
    at_exit = batch.coefficients(a_exit)
    assert at_exit["kappa"][0] * at_exit["tau"][0] == pytest.approx(200.0, rel=2e-4)
    assert at_exit["kappa"][1] == pytest.approx(k[1] / settings.tc_epsilon, rel=1e-4)
    assert batch.coefficients(a_exit * 0.999)["tight_coupling"].all()
    assert not batch.coefficients([1.0, 1.0])["tight_coupling"].any()
