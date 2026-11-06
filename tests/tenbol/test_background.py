"""Background validation against CAMB (validation adapter only)."""
from __future__ import annotations

import numpy as np
import pytest

from tenbol.constants import GRHO_UNIT, H100_MPC
from tenbol.physics.background import Background, CosmologyBatch

camb = pytest.importorskip("camb")

CASES = [
    dict(h=0.67, omega_b=0.0224, omega_c=0.120, A_s=2.1e-9, n_s=0.965, m_nu=[0.06]),
    dict(h=0.60, omega_b=0.0200, omega_c=0.160, A_s=1.5e-9, n_s=0.920, m_nu=[0.30]),
    dict(h=0.80, omega_b=0.0245, omega_c=0.100, A_s=2.7e-9, n_s=1.020, m_nu=[0.10]),
]


def camb_background(case, zs):
    p = camb.CAMBparams()
    p.set_cosmology(
        H0=case["h"] * 100,
        ombh2=case["omega_b"],
        omch2=case["omega_c"],
        mnu=float(np.sum(case["m_nu"])),
        num_massive_neutrinos=len(case["m_nu"]),
        nnu=3.046,
        TCMB=2.7255,
    )
    res = camb.get_results(p)
    Hz = np.array([float(res.hubble_parameter(z)) for z in zs])  # km/s/Mpc
    chi = np.array([res.comoving_radial_distance(z) for z in zs])  # Mpc
    rho_g = res.get_background_densities(1.0)
    return Hz, chi, rho_g


def test_background_matches_camb():
    for case in CASES:
        batch = CosmologyBatch.from_columns(
            h=[case["h"]],
            omega_b=[case["omega_b"]],
            omega_c=[case["omega_c"]],
            A_s=[case["A_s"]],
            n_s=[case["n_s"]],
            m_nu=[case["m_nu"]],
            N_eff=[3.046],
        )
        bg = Background(batch)
        zs = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0, 3000.0])
        a = 1.0 / (1.0 + zs)
        Hz_camb, chi_camb, _ = camb_background(case, zs)
        H_a = (bg.hubble(a) / a)[0]  # 1/Mpc
        Hz_ours = H_a * 2.99792458e5  # km/s/Mpc
        assert np.max(np.abs(Hz_ours / Hz_camb - 1.0)) < 3e-4
        tau = bg.tau(a)[0]
        # chi(z) = tau(a=1) - tau(a); skip z = 0 where both are identical
        tau0 = np.squeeze(bg.tau(np.array([1.0])))[()]
        chi_ours = tau0 - tau
        mask = zs > 0
        assert np.max(np.abs(chi_ours[mask] / chi_camb[mask] - 1.0)) < 1e-3


def test_neutrino_density_matches_camb_units():
    """Massless-limit and nonrelativistic-limit checks of nu_rho_p."""
    from tenbol.physics.background import fd_density_pressure as _fd_density_pressure
    rho, p = _fd_density_pressure(np.array([1e-6, 1e-3]))
    assert np.allclose(rho, 1.0, atol=1e-9)
    assert np.allclose(p, 1.0 / 3.0, atol=1e-9)
    mu = np.array([1e3])
    rho, p = _fd_density_pressure(mu)
    # nonrelativistic: rho = 0.3174 mu, p = rho/(3) * <v^2>/... -> p ~ (5 zeta5/(2 zeta3))/mu^2 * rho
    expected = 3 * 1.2020569031595942 / (2 * 7 * np.pi**4 / 120) * 1e3
    assert abs(rho[0] / expected - 1) < 2e-3
