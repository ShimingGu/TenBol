"""Recombination / thermal history (RECFAST-level), batched.

The reference (CAMB 2.0.5) uses RECFAST with the ``cosmorec`` fudge parameter
set.  The measurement in ``profiles/tenbol/evidence/reference_convergence.json``
shows final nonlinear power is insensitive to thermal-history details at the
<4e-4 level (1% fudge change: 5e-5; Hswitch off: 3.7e-4), so a faithful
RECFAST implementation of hydrogen and helium recombination plus a tanh
reionization history is sufficient for the 0.5% acceptance budget.

Equations follow Seager, Sasselov & Scott (1999) with CAMB's atomic data and
fudge factors (``fortran/recfast.f90``):

* hydrogen: Saha while ``x_H > 0.99``, then the Peebles equation with Sobolev
  escape, the two-photon rate ``Lambda = 8.2245809 /s`` and the
  CosmoRec-calibrated double-Gaussian correction to ``K``;
* helium: Saha while ``x_He > 0.98``, then the singlet-Sobolev Peebles
  equation with the helium fudge;
* matter temperature: Compton coupling to the radiation field, integrated with
  an exponential (unconditionally stable) update in ``ln a``.

The Peebles rate is stiff around ``z ~ 1000-2000`` (``|dx/dz| ~ 3`` while the
grid spacing is ``|dz| ~ 3``), so the ionisation equations are integrated with
*local sub-stepping*: the step is subdivided until the predicted change of each
fraction is below 0.02.  This is explicit RK4 on a self-refined grid; the cost
is confined to the recombination window.  ``x_H`` and ``x_He`` are initialised
from Saha at the switch, matching RECFAST.

Only the background thermal history is provided; perturbed recombination is
outside the supported model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log, log1p, sqrt

import numpy as np

from ..constants import (
    A_RAD,
    C_LIGHT,
    H_PLANCK,
    K_BOLTZ,
    M_E,
    M_H,
    MPC_M,
    SIGMA_THOMSON,
)

# --- atomic data and RECFAST constants (CAMB values) ------------------------
LAMBDA_2S = 8.2245809                     # 2s-1s two-photon rate [1/s]
L_H_ALPHA = 8.225916453e6                 # m^-1
L_H_ION = 1.096787737e7                   # m^-1
L_HE1_ION = 1.98310772e7                  # m^-1
L_HE_2S = 1.66277434e7                    # m^-1
L_HE_2P = 1.71134891e7                    # m^-1

A_PPB, B_PPB, C_PPB, D_PPB = 4.309, -0.6166, 0.6703, 0.5300
A_VF, B_VF = 10.0**-16.744, 0.711
T_0_VF, T_1_VF = 10.0**0.477121, 10.0**5.114
LAMBDA_HE = 51.3

LALPHA = 1.0 / L_H_ALPHA                  # m
LALPHA_HE = 1.0 / L_HE_2P
CL = H_PLANCK * C_LIGHT * L_H_ALPHA / K_BOLTZ      # K
CDB = H_PLANCK * C_LIGHT * (L_H_ION - L_H_ALPHA) / K_BOLTZ
CB1_H = H_PLANCK * C_LIGHT * L_H_ION / K_BOLTZ
CB1_HE1 = H_PLANCK * C_LIGHT * L_HE1_ION / K_BOLTZ
CDB_HE = H_PLANCK * C_LIGHT * (L_HE1_ION - L_HE_2S) / K_BOLTZ
BFACT = H_PLANCK * C_LIGHT * (L_HE_2P - L_HE_2S) / K_BOLTZ
CR = 2.0 * np.pi * (M_E / H_PLANCK) * (K_BOLTZ / H_PLANCK)
CK = LALPHA**3 / (8.0 * np.pi)             # m^3
CK_HE = LALPHA_HE**3 / (8.0 * np.pi)

# CosmoRec-calibrated fudge parameter set (CAMB ``recfast_cosmorec``)
FUDGE = 1.125
FUDGE_HE = 0.8472367977
AG1, AG2 = -0.1395272483, 0.0729891952
ZG1, ZG2 = 7.2813061282, 6.7667038679
WG1, WG2 = 0.1638966410, 0.2785834127
HE_RATE_CORRECTION = True


@dataclass(frozen=True)
class Reionization:
    """Tanh reionization history with the optical depth as input."""

    tau: float = 0.0544
    delta_z: float = 0.5
    z_max: float = 40.0

    def ionized_fraction(self, z, z_re):
        return 0.5 * (1.0 + np.tanh((z_re - np.asarray(z)) / self.delta_z))


def saha_hydrogen(n, Tm):
    """Saha ionized fraction of hydrogen."""
    expo = exp(min(CB1_H / Tm, 700.0))
    saha = (2.0 * np.pi * M_E * K_BOLTZ * Tm / H_PLANCK**2) ** (-1.5) * expo
    b = n * saha
    return 2.0 / (1.0 + sqrt(1.0 + 4.0 * b))


def saha_helium(n_he, Tm):
    """Saha He II fraction."""
    expo = exp(min(CB1_HE1 / Tm, 700.0))
    saha = 4.0 * (2.0 * np.pi * M_E * K_BOLTZ * Tm / H_PLANCK**2) ** (-1.5) * expo
    b = n_he * saha
    return 2.0 / (1.0 + sqrt(1.0 + 4.0 * b))


def _he_rate_factor(z):
    if not HE_RATE_CORRECTION:
        return 1.0
    u = (z - 1968.1134791219417) / 774.0312440582285
    u2 = u * u
    corr = 1.0 + (0.07805480599148856 + 0.899368710079757 * u - 3.063896868095767 * u2) / (
        1.0 + 3.752520187028219 * u2 + 18.126758791787946 * u2 * u2
    )
    return corr if (1500.0 < z < 3000.0) else 1.0


class _SampleThermo:
    """RECFAST ionisation equations for one cosmology."""

    def __init__(self, T_now, f_he, n_H0_m3):
        self.T_now = T_now
        self.f_he = f_he
        self.n_H0 = n_H0_m3

    def n_H(self, z):
        return self.n_H0 * (1.0 + z) ** 3

    def derivs(self, z, x_H, x_He, Tm, H_sec):
        """(dx_H/dz, dx_He/dz)."""
        Tm = max(Tm, 1.0)
        x_H = min(max(x_H, 0.0), 1.0)
        x_He = min(max(x_He, 0.0), 1.0)
        theta = Tm * 1.0e-4
        log_theta = log(theta)
        Rdown = 1.0e-19 * A_PPB * exp(B_PPB * log_theta) / (
            1.0 + C_PPB * exp(D_PPB * log_theta)
        )
        cr_t32 = (CR * Tm) ** 1.5
        Rup = Rdown * cr_t32 * exp(-min(CDB / Tm, 700.0))
        RupE = Rup * exp(-min(CL / Tm, 700.0))

        x = x_H + self.f_he * x_He          # electron fraction per H nucleus
        n = self.n_H(z)
        n_He = self.f_he * n

        log_1pz = log1p(z)
        K = CK / H_sec * (
            1.0
            + AG1 * exp(-((log_1pz - ZG1) / WG1) ** 2)
            + AG2 * exp(-((log_1pz - ZG2) / WG2) ** 2)
        )

        A_H = x * x_H * n * Rdown - RupE * (1.0 - x_H)
        if x_H > 0.985:
            dxH_dz = A_H / (H_sec * (1.0 + z))
        else:
            B_H = 1.0 + K * LAMBDA_2S * n * (1.0 - x_H)
            C_H = (
                1.0 / FUDGE
                + K * LAMBDA_2S * n * (1.0 - x_H) / FUDGE
                + K * Rup * n * (1.0 - x_H)
            )
            dxH_dz = A_H * B_H / (H_sec * (1.0 + z) * C_H)

        if x_He <= 1.0e-10:
            dxHe_dz = 0.0
        else:
            sq0 = sqrt(Tm / T_0_VF)
            sq1 = sqrt(Tm / T_1_VF)
            l0 = log1p(sq0)
            l1 = log1p(sq1)
            Rdown_He = A_VF / sq0 * exp(-((1.0 - B_VF) * l0 + (1.0 + B_VF) * l1))
            Rup_He = 4.0 * Rdown_He * cr_t32 * exp(-min(CDB_HE / Tm, 700.0))
            RupHeE = Rup_He * exp(-min(CL / Tm, 700.0))
            he_boltz = exp(min(BFACT / Tm, 680.0))
            K_He = CK_HE / H_sec
            A_He = x * x_He * n * Rdown_He - RupHeE * (1.0 - x_He)
            LHe = LAMBDA_HE * n_He * (1.0 - x_He) * he_boltz
            MHe = (LAMBDA_HE + Rup_He) * n_He * (1.0 - x_He) * he_boltz
            BHe = 1.0 + K_He * LHe
            CHe = 1.0 + K_He * MHe
            dxHe_dz = _he_rate_factor(z) * A_He * BHe / (H_sec * (1.0 + z) * CHe)
        return dxH_dz, dxHe_dz


class ThermalHistory:
    """Tabulated ionization fraction and matter temperature for a batch."""

    def __init__(self, background, reionization: Reionization = Reionization(),
                 z_start: float = 1.0e7, n_table: int = 6000,
                 max_substeps: int = 64, max_change: float = 0.02):
        self.bg = background
        self.reion = reionization
        self.z_start = z_start
        self.max_substeps = max_substeps
        self.max_change = max_change
        B = background.B
        self.ln_a_table = np.linspace(np.log(1.0 / (1.0 + z_start)), 0.0, n_table)
        self.xe_table = np.empty((B, n_table))
        self.tm_table = np.empty((B, n_table))
        self.z_re = np.empty(B)
        self.tau_total = np.empty(B)

        a = np.exp(self.ln_a_table)
        z = np.exp(-self.ln_a_table) - 1.0
        z_mid = 1.0 / np.exp(0.5 * (self.ln_a_table[:-1] + self.ln_a_table[1:])) - 1.0

        def hubble_grid(scale_factors):
            # Bound the temporary storage used by the neutrino quadrature.
            columns = []
            for start in range(0, scale_factors.size, 256):
                segment = scale_factors[start:start + 256]
                grid = np.broadcast_to(segment, (B, segment.size))
                columns.append(background.hubbles_per_s(grid))
            return np.concatenate(columns, axis=1)

        h = hubble_grid(a)
        h_mid = hubble_grid(1.0 / (1.0 + z_mid))
        for i in range(B):
            self._solve_one(i, a, z, z_mid, h[i], h_mid[i])
        self._build_splines()

    # ------------------------------------------------------------------
    def _solve_one(self, i, a, z, z_mid, H_sec, H_mid):
        from scipy.optimize import brentq

        bg = self.bg

        th = _SampleThermo(float(bg.T_cmb[i]), float(bg.f_he[i]), float(bg.N_H0[i]))

        n_a = len(z)
        x_H = np.empty(n_a)
        x_He = np.empty(n_a)
        Tm = np.empty(n_a)
        x_H[0] = 1.0
        x_He[0] = 1.0
        Tm[0] = th.T_now * (1.0 + z[0])

        for k in range(n_a - 1):
            zk, zk1 = z[k], z[k + 1]
            dlna = self.ln_a_table[k + 1] - self.ln_a_table[k]
            xh, xhe, tm = x_H[k], x_He[k], Tm[k]

            # ---- hydrogen ------------------------------------------------
            sh = saha_hydrogen(th.n_H(zk), tm)
            if sh > 0.99:
                xh_new = sh
            else:
                if xh > 0.99:
                    xh = sh             # continuity at the Saha -> ODE switch
                xh_new = self._rk4_substep(th, zk, zk1, xh, xhe, tm, H_sec[k], H_sec[k + 1], 0)

            # ---- helium --------------------------------------------------
            she = saha_helium(th.f_he * th.n_H(zk), tm)
            if she > 0.98:
                xhe_new = she
            else:
                if xhe > 0.98:
                    xhe = she
                xhe_new = self._rk4_substep(th, zk, zk1, xh, xhe, tm, H_sec[k], H_sec[k + 1], 1)

            x_H[k + 1] = min(max(xh_new, 0.0), 1.0)
            x_He[k + 1] = min(max(xhe_new, 0.0), 1.0)

            # ---- matter temperature -------------------------------------
            T_rad_m = th.T_now * (1.0 + z_mid[k])
            xe = x_H[k + 1] + th.f_he * x_He[k + 1]
            gamma = (
                (8.0 * SIGMA_THOMSON * A_RAD * T_rad_m**4) / (3.0 * M_E * C_LIGHT)
            ) * (xe / (1.0 + xe + th.f_he))
            r = gamma / H_mid[k]
            T_eq = r * T_rad_m / (2.0 + r)
            decay = exp(-(2.0 + r) * dlna)
            Tm[k + 1] = T_eq + (tm - T_eq) * decay

        # ---- reionization: tanh matching the requested optical depth -----
        z_lo = np.asarray(z) <= self.reion.z_max

        def optical_depth(z_re):
            """Optical depth from z = 0 to ``z_max`` (the reionization window).

            The requested ``tau`` follows CAMB's convention: the optical depth
            added between today and the end of reionization.  Integrating past
            the recombination photosphere would diverge (the pre-recombination
            plasma is optically thick), which is not what the parameter means.
            """
            xre = self.reion.ionized_fraction(z[z_lo], z_re)
            xe_tot = np.maximum((x_H + th.f_he * x_He)[z_lo], xre * (1.0 + th.f_he))
            ne = th.n_H0 * xe_tot / a[z_lo] ** 3
            integrand = SIGMA_THOMSON * ne * C_LIGHT / ((1.0 + z[z_lo]) * H_sec[z_lo])
            if getattr(self, '_debug', False):
                print('   DEBUG integrand', integrand.min(), integrand.max(), 'ne', ne.min(), ne.max(), 'Hs', H_sec.min(), H_sec.max(), 'xe_tot', xe_tot.min(), xe_tot.max())
            # z decreases along the table: integrate from high to low z
            return np.trapezoid(integrand, z[z_lo][::-1])

        z_hi = self.reion.z_max
        tau_min = optical_depth(0.0)
        tau_max = optical_depth(z_hi)
        if getattr(self, '_debug', False):
            print('DEBUG tau_min/tau_max', tau_min, tau_max, 'n_lo', z_lo.sum(), 'xe_lo_max', self.reion.ionized_fraction(z[z_lo], z_hi).max())
        if tau_max <= self.reion.tau:
            # reionization cannot supply the requested depth
            z_re = z_hi
        elif tau_min >= self.reion.tau:
            # recombination alone exceeds the requested depth: no reionization
            z_re = 0.0
        else:
            z_re = brentq(lambda zr: optical_depth(zr) - self.reion.tau, 0.0, z_hi,
                          xtol=1e-5, rtol=1e-10)
        self.z_re[i] = z_re
        self.tau_total[i] = optical_depth(z_re)
        xre = self.reion.ionized_fraction(z, z_re)
        self.xe_table[i] = np.maximum(x_H + th.f_he * x_He, xre * (1.0 + th.f_he))
        self.tm_table[i] = Tm

    def _rk4_substep(self, th, z0, z1, xh, xhe, tm, H0, H1, component):
        """RK4 advance of one species with adaptive sub-stepping."""
        h = z1 - z0
        d = th.derivs(z0, xh, xhe, tm, H0)
        rate = abs(d[component])
        n_sub = ceil(rate * abs(h) / self.max_change) if rate > 0 else 1
        n_sub = min(max(n_sub, 1), self.max_substeps)
        hs = h / n_sub
        val = xh if component == 0 else xhe
        for j in range(n_sub):
            zz = z0 + j * hs
            Hs = H0 if j < n_sub - 1 else H1
            k1 = th.derivs(zz, xh, xhe, tm, Hs)
            k2 = th.derivs(zz + hs / 2, xh + hs * k1[0] / 2, xhe + hs * k1[1] / 2, tm, Hs)
            k3 = th.derivs(zz + hs / 2, xh + hs * k2[0] / 2, xhe + hs * k2[1] / 2, tm, Hs)
            k4 = th.derivs(zz + hs, xh + hs * k3[0], xhe + hs * k3[1], tm, Hs)
            new = (val + hs * (k1[component] + 2 * k2[component]
                              + 2 * k3[component] + k4[component]) / 6.0)
            val = min(max(new, 0.0), 1.0)
            if component == 0:
                xh = val
            else:
                xhe = val
        return val

    def _build_splines(self):
        from scipy.interpolate import CubicSpline

        self._ln_xe = np.log(np.maximum(self.xe_table, 1e-30))
        self._spl_xe = CubicSpline(self.ln_a_table, self._ln_xe, axis=1)
        self._spl_tm = CubicSpline(self.ln_a_table, np.log(self.tm_table), axis=1)

    # ------------------------------------------------------------------
    def x_e(self, a):
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        return np.exp(self._spl_xe(np.log(a)))

    def t_m(self, a):
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        return np.exp(self._spl_tm(np.log(a)))

    def dln_tm_dlna(self, a):
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        return self._spl_tm(np.log(a), 1)

    def opacity(self, a):
        """Thomson opacity ``a n_e sigma_T`` [1/Mpc]."""
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        xe = self.x_e(a)
        # Conformal Thomson opacity: kappa = a * n_e * sigma_T with the physical
        # electron density n_e = N_H0 * xe / a^3, so kappa = sigma_T * N_H0 *
        # xe * a * ... i.e. it scales as a^-2.  SIGMA_THOMSON is in m^2 and
        # N_H0 in m^-3, hence the single MPC_M factor converting 1/m to 1/Mpc.
        ne_m3 = self.bg.N_H0[:, None] * xe / a[None, :] ** 3
        return SIGMA_THOMSON * MPC_M * ne_m3 * a[None, :]

    def cs2_b(self, a):
        """Baryon adiabatic sound speed squared (c = 1)."""
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        tm = self.t_m(a)
        dln = self.dln_tm_dlna(a)
        return (K_BOLTZ * tm / M_H / C_LIGHT**2) * (1.0 - dln / 3.0)
