"""Homogeneous background for flat LCDM + thermal massive neutrinos.

Everything is computed in CAMB's ``grho`` convention

    grho_i(a) = 8 pi G a^2 rho_i(a) / c^2      [1/Mpc^2]

so that ``H_hat(a) = a H(a) = sqrt(grho_total(a) / 3)`` and conformal time
``tau(a) = int_0^a da' / H_hat(a')``.  Arrays are batch-first: parameter arrays
have shape ``(B,)`` and every method returns arrays whose trailing dimensions
broadcast against a state array of shape ``(B, ...)``.

Neutrino degeneracies follow CAMB's default ``share_delta_neff`` convention:
with ``N_eff`` and ``S`` degenerate massive eigenstates, every neutrino gets
``neff_i = N_eff / (floor(N_eff - S) + S)`` and a massive eigenstate has
temperature ``neff_i^(1/4) T_nu``.  This reproduces CAMB's density split; the
test suite compares against ``CAMBdata.get_background_densities``.

Massive-neutrino thermodynamics uses explicit Fermi-Dirac quadrature instead of
CAMB's fitted polynomials; both agree at the 1e-5 level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import (
    C_LIGHT,
    EV_J,
    FD_MOMENT3,
    GRHO_GAMMA_UNIT,
    GRHO_NU_PER_SPECIES,
    GRHO_UNIT,
    H100_MPC,
    KAPPA,
    K_BOLTZ,
    M_H,
    MPC_M,
    T_NU_OVER_T_CMB,
)
from .bbn import helium_mass_fraction

# Gauss-Legendre quadrature for the background Fermi-Dirac integrals,
# q in [0, 32] (the occupation number is < 1e-14 beyond q = 32).
_GL_Q, _GL_W = np.polynomial.legendre.leggauss(160)
_Q_NODES = 0.5 * 32.0 * (_GL_Q + 1.0)
_Q_WEIGHTS = 0.5 * 32.0 * _GL_W
_FD = 1.0 / (np.exp(_Q_NODES) + 1.0)
_Q2F = _Q_WEIGHTS * _Q_NODES**2 * _FD

# Gauss-Laguerre nodes for the *perturbation* momentum quadrature:
#   int_0^inf q^2 f(q) X(q) dq  ~=  sum_i NU_W[i] X(NU_Q[i])
# with the factor 1/(1+e^-q) retained.  12 nodes are converged far below the
# accuracy needed for P(k).
_LAG_Q, _LAG_W = np.polynomial.laguerre.laggauss(12)
NU_Q = _LAG_Q
NU_W = _LAG_W * _LAG_Q**2 / (1.0 + np.exp(-_LAG_Q))
NU_W_SUM = float(np.sum(NU_W))
assert abs(NU_W_SUM / 1.8030853544 - 1.0) < 1e-6, NU_W_SUM


def fd_density_pressure(mu):
    """``(rho, p)`` of a thermal massive eigenstate in units of one massless
    eigenstate with the same temperature.

    ``mu = a m / T_eff`` (dimensionless).  ``rho -> 1``, ``p -> 1/3`` for
    ``mu -> 0`` and ``rho -> 0.31735 mu`` for large ``mu``.
    """
    mu = np.asarray(mu)
    mu_ = mu[..., None]
    eps = np.sqrt(_Q_NODES**2 + mu_**2)
    rho = np.sum(_Q2F * eps, axis=-1) / FD_MOMENT3
    p = np.sum(_Q2F * _Q_NODES**2 / eps, axis=-1) / (3.0 * FD_MOMENT3)
    return rho, p


@dataclass(frozen=True)
class CosmologyBatch:
    """Batch of flat LCDM cosmologies with thermal massive neutrinos.

    ``m_nu`` has shape (B, S) and holds the eigenstate masses in eV; ``N_eff``
    is the total effective number of relativistic species (including the
    relativistic limit of the massive eigenstates).
    """

    h: np.ndarray
    omega_b: np.ndarray
    omega_c: np.ndarray
    A_s: np.ndarray
    n_s: np.ndarray
    m_nu: np.ndarray
    N_eff: np.ndarray
    T_cmb: np.ndarray
    y_he: np.ndarray | None = None
    pivot_k: float = 0.05

    @classmethod
    def from_columns(cls, *, sample_id=None, h, omega_b, omega_c, A_s, n_s, m_nu,
                     N_eff=3.046, T_cmb=2.7255, num_massive=None, pivot_k=0.05):
        h = np.atleast_1d(np.asarray(h, dtype=np.float64))
        B = h.shape[0]

        def _col(v):
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 0:
                return np.full(B, float(v))
            return np.broadcast_to(v, (B,)).astype(np.float64)

        m_nu_arr = np.asarray(m_nu, dtype=np.float64)
        if m_nu_arr.ndim == 0:
            m_nu_arr = np.full((B, 1), float(m_nu_arr))
        elif m_nu_arr.ndim == 1:
            if m_nu_arr.shape[0] == B:
                m_nu_arr = m_nu_arr[:, None]
            else:
                m_nu_arr = np.broadcast_to(m_nu_arr[None, :], (B, m_nu_arr.shape[0]))
        if m_nu_arr.shape[0] != B:
            raise ValueError("m_nu must have leading dimension B")
        if m_nu_arr.shape[1] == 1 and num_massive and num_massive > 1:
            m_nu_arr = np.repeat(m_nu_arr / num_massive, num_massive, axis=1)
        if sample_id is not None and len(sample_id) != B:
            raise ValueError("sample_id length must match batch size")
        return cls(
            h=h,
            omega_b=_col(omega_b),
            omega_c=_col(omega_c),
            A_s=_col(A_s),
            n_s=_col(n_s),
            m_nu=m_nu_arr,
            N_eff=_col(N_eff),
            T_cmb=_col(T_cmb),
            y_he=None,
            pivot_k=pivot_k,
        )


class Background:
    """Background quantities for a batch of cosmologies."""

    def __init__(self, cosmo: CosmologyBatch, y_he=None):
        self.cosmo = cosmo
        self.h = np.asarray(cosmo.h, dtype=np.float64)
        self.omega_b = np.asarray(cosmo.omega_b, dtype=np.float64)
        self.omega_c = np.asarray(cosmo.omega_c, dtype=np.float64)
        self.T_cmb = np.asarray(cosmo.T_cmb, dtype=np.float64)
        self.N_eff = np.asarray(cosmo.N_eff, dtype=np.float64)
        m_nu = np.atleast_2d(np.asarray(cosmo.m_nu, dtype=np.float64))
        B = self.h.shape[0]
        if m_nu.shape[0] == 1 and B > 1:
            m_nu = np.broadcast_to(m_nu, (B, m_nu.shape[1])).copy()
        self.m_nu = m_nu
        self.B, self.n_species = m_nu.shape
        if np.any(self.N_eff < self.n_species - 1e-9):
            raise ValueError("N_eff smaller than the number of massive species")

        self.y_he = (
            np.full(B, float(y_he))
            if y_he is not None
            else helium_mass_fraction(self.omega_b, self.T_cmb, self.N_eff)
        )
        self.f_he = self.y_he / (3.9715 * (1.0 - self.y_he))

        # ---- CAMB share_delta_neff degeneracies --------------------------
        n_fractional_massless = self.N_eff - self.n_species
        actual_massless = np.floor(n_fractional_massless + 1e-6)
        self.neff_i = self.N_eff / (actual_massless + self.n_species)
        self.massless_degeneracy = self.neff_i * actual_massless
        self.massive_degeneracy = np.repeat(self.neff_i[:, None], self.n_species, axis=1)
        self.nu_T_scale = self.massive_degeneracy ** 0.25

        # ---- densities at a = 1 -----------------------------------------
        self.grho_g0 = GRHO_GAMMA_UNIT * self.T_cmb**4
        self.T_nu_eV = K_BOLTZ * self.T_cmb / EV_J * T_NU_OVER_T_CMB
        self._mu0 = self.m_nu / (self.T_nu_eV[:, None] * self.nu_T_scale)
        rho0, _ = fd_density_pressure(self._mu0)
        self.grho_nu_species_unit = GRHO_NU_PER_SPECIES * self.grho_g0
        self.grho_nu_mass0 = (
            self.grho_nu_species_unit[:, None] * self.massive_degeneracy * rho0
        )
        self.grho_nu_massless0 = self.grho_nu_species_unit * self.massless_degeneracy

        H0 = H100_MPC * self.h
        self.grho_c0 = GRHO_UNIT * self.omega_c
        self.grho_b0 = GRHO_UNIT * self.omega_b
        self.grho_nu0 = self.grho_nu_massless0 + np.sum(self.grho_nu_mass0, axis=1)
        self.grho_r0 = self.grho_g0 + self.grho_nu0
        self.grho_v0 = 3.0 * H0**2 - (self.grho_c0 + self.grho_b0 + self.grho_r0)
        # hydrogen number density today [m^-3], and H0 in s^-1 for the thermal ODE
        rho_crit = 3.0 * H0**2 * C_LIGHT**2 / (MPC_M**2 * KAPPA)   # kg/m^3
        self.N_H0 = rho_crit * self.omega_b / self.h**2 * (1.0 - self.y_he) / M_H   # m^-3
        self._build_tau_table()

    # ------------------------------------------------------------------
    # shape handling
    #
    # ``a`` is the scale factor with the batch on the leading axis: a scalar,
    # shape (B,), or shape (B, N, ...).  Returned arrays have that same shape
    # (or (B,) for scalar input), so callers can always broadcast against a
    # state of shape (B, ...).
    # ------------------------------------------------------------------
    def _a(self, a):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 0:
            return a.reshape(1), 1
        if a.shape[0] == self.B:
            return a, a.ndim
        # a grid shared by every sample: broadcast to (B, ...)
        return np.broadcast_to(a, (self.B,) + a.shape), a.ndim + 1

    def _col(self, x, ndim):
        if ndim <= 1:
            return np.asarray(x)
        return np.asarray(x).reshape(np.shape(x) + (1,) * (ndim - 1))

    # ------------------------------------------------------------------
    # densities
    # ------------------------------------------------------------------
    def nu_rho_p(self, a):
        """Massive-neutrino ``(rho, p)`` relative to a massless eigenstate."""
        a, ndim = self._a(a)
        if ndim == 1:
            mu = a[:, None] * self._mu0
        else:
            mu = a[..., None] * self._mu0.reshape(
                (self.B,) + (1,) * (ndim - 1) + (self.n_species,)
            )
        return fd_density_pressure(mu)

    def _nu_mass_term(self, a, quantity: str):
        rho, pnu = self.nu_rho_p(a)
        q = rho if quantity == "rho" else pnu
        return np.sum(
            self.grho_nu_species_unit.reshape(
                (self.B,) + (1,) * (q.ndim - 2) + (1,)
            ) * self.massive_degeneracy.reshape(
                (self.B,) + (1,) * (q.ndim - 2) + (self.n_species,)
            ) * q,
            axis=-1,
        )

    def grho_parts(self, a):
        a, ndim = self._a(a)
        inv_a = 1.0 / a
        return {
            "c": self._col(self.grho_c0, ndim) * inv_a,
            "b": self._col(self.grho_b0, ndim) * inv_a,
            "g": self._col(self.grho_g0, ndim) * inv_a**2,
            "nu_massless": self._col(self.grho_nu_massless0, ndim) * inv_a**2,
            "nu_mass": self._nu_mass_term(a, "rho") * inv_a**2,
            "v": self._col(self.grho_v0, ndim) * a**2,
        }

    def grho(self, a):
        parts = self.grho_parts(a)
        return np.sum(np.stack(list(parts.values())), axis=0)

    def pressure(self, a):
        a, ndim = self._a(a)
        inv_a = 1.0 / a
        return (
            self._col(self.grho_g0 + self.grho_nu_massless0, ndim) * inv_a**2 / 3.0
            + self._nu_mass_term(a, "p") * inv_a**2
            - self._col(self.grho_v0, ndim) * a**2
        )

    def hubble(self, a):
        """``H_hat = a H`` [1/Mpc]."""
        return np.sqrt(self.grho(a) / 3.0)

    def hubble_prime(self, a):
        """``dH_hat/dtau`` [1/Mpc^2] = -H_hat^2 (1 + 3 w_tot) / 2."""
        H = self.hubble(a)
        w_tot = self.pressure(a) / self.grho(a)
        return -0.5 * H**2 * (1.0 + 3.0 * w_tot)

    def hubbles_per_s(self, a):
        """H(a) in s^-1."""
        a, _ = self._a(a)
        return self.hubble(a) / a * C_LIGHT / MPC_M

    # ------------------------------------------------------------------
    # conformal time
    # ------------------------------------------------------------------
    def _build_tau_table(self):
        from scipy.integrate import cumulative_simpson

        ln_a = np.linspace(np.log(1.0e-9), np.log(2.0), 6000)
        a = np.exp(ln_a)[None, :]                # (1, N) -> broadcast to (B, N)
        dtau_dlna = 1.0 / self.hubble(a)          # dtau/dln a = 1/H_hat
        tau = cumulative_simpson(dtau_dlna, x=ln_a, axis=-1, initial=0.0)
        tau0 = np.exp(2 * ln_a[0]) / (2.0 * np.sqrt(self.grho_r0))
        self._tau_ln_a = ln_a
        self._tau_table = tau + tau0[:, None]
        self._tau_slope = np.gradient(self._tau_table, ln_a, axis=1)

    def tau(self, a):
        """Conformal time [Mpc]; cubic Hermite in ln a.

        Below the first table node the exact radiation-era quadratic
        ``tau = a^2 / (2 sqrt(grho_r0))`` is used; above the last node this
        raises rather than extrapolating silently.
        """
        a, ndim = self._a(a)
        shape = a.shape[1:]
        ln_a = np.log(a).ravel()
        lo, hi = self._tau_ln_a[0], self._tau_ln_a[-1]
        if np.any(a > np.exp(hi)):
            raise ValueError("tau(a) requested beyond the background table")
        ln_clip = np.clip(ln_a, lo, hi)
        tab = self._tau_ln_a
        i = np.clip(np.searchsorted(tab, ln_clip) - 1, 0, len(tab) - 2)
        x0 = tab[i]
        h = tab[i + 1] - x0
        t = (ln_clip - x0) / h
        y0 = self._tau_table[:, i]
        y1 = self._tau_table[:, i + 1]
        m0 = self._tau_slope[:, i] * h
        m1 = self._tau_slope[:, i + 1] * h
        out = (
            y0 * (2 * t**3 - 3 * t**2 + 1)
            + y1 * (-2 * t**3 + 3 * t**2)
            + m0 * (t**3 - 2 * t**2 + t)
            + m1 * (t**3 - t**2)
        )
        if not shape:
            return out.reshape(self.B)
        return out.reshape((self.B,) + shape)

    def dtau_dlna(self, a):
        a, _ = self._a(a)
        return 1.0 / self.hubble(a)

    # ------------------------------------------------------------------
    # interpolation tables for the neutrino thermodynamics
    # ------------------------------------------------------------------
    def _nu_tables(self):
        if getattr(self, "_nu_spl", None) is not None:
            return self._nu_spl
        from scipy.interpolate import CubicSpline

        mu = np.exp(np.linspace(np.log(1e-6), np.log(1e6), 800))
        rho, p = fd_density_pressure(mu)
        lnmu = np.log(mu)
        self._nu_spl = (
            CubicSpline(lnmu, np.log(rho)),
            CubicSpline(lnmu, np.log(p)),
        )
        return self._nu_spl

    def nu_rho_p_interp(self, mu):
        """log-log interpolated ``(rho, p)``; accurate to ~1e-6 over 1e-6..1e6."""
        spl_rho, spl_p = self._nu_tables()
        mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
        return np.exp(spl_rho(np.log(mu))), np.exp(spl_p(np.log(mu)))
