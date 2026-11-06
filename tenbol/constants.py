"""Physical constants and unit conversions.

Values are aligned with CAMB 2.0.5 ``fortran/constants.f90`` so that a
candidate/ reference comparison is not polluted by constant mismatches:
distances are Mpc, k is 1/Mpc, conformal time is Mpc (c = 1), and
``density_unit`` is the CAMB ``grho`` convention

    8 pi G rho / c^2  [1/Mpc^2]

which makes the flat Friedmann equation ``H_hat^2 = grho / 3`` with
``H_hat = a H``.
"""

from __future__ import annotations

import numpy as np

# --- fundamental constants (SI, CODATA 2018) --------------------------------
C_LIGHT = 2.99792458e8            # m s^-1
G_NEWTON = 6.67430e-11           # m^3 kg^-1 s^-2
SIGMA_THOMSON = 6.6524587321e-29  # m^2
SIGMA_BOLTZ = 5.670374419e-8      # W m^-2 K^-4
K_BOLTZ = 1.380649e-23            # J K^-1
H_PLANCK = 6.62607015e-34         # J s
HBAR = H_PLANCK / (2 * np.pi)
EV_J = 1.602176634e-19            # J
M_P = 1.67262192369e-27           # kg
M_E = 9.1093837015e-31            # kg
M_H = 1.673575e-27                # kg  (average H atom, CAMB convention)
M_HE4 = 6.646479073e-27           # kg
MASS_RATIO_HE_H = M_HE4 / M_H

MPC_M = 3.085677581e22            # m
MPC_IN_SEC = MPC_M / C_LIGHT
C_KM_S = 2.99792458e5             # km s^-1

T_CMB0_DEFAULT = 2.7255           # K (Fixsen 2009, CAMB default)
N_EFF_STANDARD = 3.044            # CAMB default_nnu used for BBN delta_neff

ZETA3 = 1.2020569031595942853997
ZETA5 = 1.0369277551433699263313
ZETA7 = 1.0083492773819228268397

KAPPA = 8.0 * np.pi * G_NEWTON
A_RAD = 8.0 * np.pi**5 * K_BOLTZ**4 / (15.0 * C_LIGHT**3 * H_PLANCK**3)

# --- Mpc-unit conversions ---------------------------------------------------
# H in 1/Mpc from H0 = 100 h km/s/Mpc
H100_MPC = 100.0 / C_KM_S                     # 3.3356...e-4 1/Mpc
# 8 pi G rho / c^2  for Omega_i at a=1 with rho in units of rho_crit
GRHO_UNIT = 3.0 * H100_MPC**2                 # 1/Mpc^2 per (Omega h^2)

# 8 pi G rho_gamma / c^2 today, for T_CMB0
GRHO_GAMMA_UNIT = KAPPA * A_RAD / C_LIGHT**4 * MPC_M**2
# per massless neutrino eigenstate: 7/8 (4/11)^(4/3) rho_gamma
GRHO_NU_PER_SPECIES = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)
T_NU_OVER_T_CMB = (4.0 / 11.0) ** (1.0 / 3.0)

# --- neutrino momentum quadrature ------------------------------------------
# Integral of q^3 f(q) dq for Fermi-Dirac statistics: 7 pi^4 / 120
FD_MOMENT3 = 7.0 * np.pi**4 / 120.0
# 3 zeta(3) = integral of q^2 f(q) dq
FD_MOMENT2 = 3.0 * ZETA3


def grho_gamma(T_cmb: float = T_CMB0_DEFAULT) -> float:
    """8 pi G rho_gamma / c^2 at a = 1 [1/Mpc^2]."""
    return GRHO_GAMMA_UNIT * T_cmb**4


def nu_temperature_eV(T_cmb: float = T_CMB0_DEFAULT) -> float:
    """Today's neutrino temperature in eV."""
    return K_BOLTZ * T_cmb / EV_J * T_NU_OVER_T_CMB


def compton_ct() -> float:
    """CAMB's Compton_CT prefactor for the matter-temperature equation [Mpc]."""
    return MPC_IN_SEC * (8.0 / 3.0) * (SIGMA_THOMSON / (M_E * C_LIGHT)) * A_RAD


def helium_fraction_fhe(y_he: float) -> float:
    """n_He / n_H for helium mass fraction ``y_he``."""
    return y_he / (MASS_RATIO_HE_H * (1.0 - y_he))


def hydrogen_number_density(omega_b: float, y_he: float, h: float) -> float:
    """Mean hydrogen number density today [Mpc^-3].

    ``omega_b = Omega_b h^2``; rho_crit = 3 H0^2 / (8 pi G).
    """
    rho_crit = 3.0 * (H100_MPC * h) ** 2 * C_LIGHT**2 * MPC_M**2 / KAPPA  # kg/m^3
    rho_b = rho_crit * omega_b / h**2
    return rho_b * (1.0 - y_he) / M_H * MPC_M**3
