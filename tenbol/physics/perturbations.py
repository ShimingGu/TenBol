"""Batched synchronous-gauge scalar perturbation solver.

Conventions follow Ma & Bertschinger (1995) with CAMB's variable definitions
(``fortran/equations.f90``), which removes ambiguity in the metric convention:

* ``dc, db``: CDM and baryon density perturbations (CDM rest frame, so the CDM
  velocity vanishes identically);
* ``vb = theta_b/k``; ``dg``, ``qg = 4 theta_g/(3k)``, ``pig = F_{g,2}``;
* ``dr, qr, pir`` for massless neutrinos; ``G_l`` photon E-mode polarization;
* ``Fg_l`` photon moments (l>=3); ``Fr_l`` massless-neutrino moments (l>=3);
* ``Fnu_{q,l}`` massive-neutrino moments per momentum bin;
* ``etak = k*eta``.

The metric is reconstructed algebraically from the two constraint equations
(``z = h'/(2k)`` from (0,0), ``etak' = dgq/2`` from (0,i)); the (i,j) equation
follows from the Bianchi identity.

Controlled approximations, all profile parameters validated by convergence
against the reference:

* tight coupling with first-order slip while ``kappa/k > 1/epsilon``.  During
  tight coupling the photon quadrupole, polarization and higher moments are
  frozen (they are O(kappa^-1) there);
* radiation streaming approximation for photons and massless neutrinos once
  ``k*tau > rsa_ktau`` (arXiv:1104.2933), where their moments are obtained
  algebraically from the matter sources;
* hierarchy truncation with the free-streaming closure
  ``F_L' = k F_{L-1} - (L+1) F_L/tau``.

The integrator is an explicit embedded RK45 (Dormand-Prince) in ``x = ln a``
with a global adaptive step.  Per-trajectory stepping and k-bucketed ``lmax``
are separate scheduling optimizations (see the numerical-method experiments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import NumericalFailure
from .background import NU_Q, NU_W
from .state import SCALARS, StateLayout

# Dormand-Prince 5(4) coefficients
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    np.array([]),
    np.array([1 / 5]),
    np.array([3 / 40, 9 / 40]),
    np.array([44 / 45, -56 / 15, 32 / 9]),
    np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
    np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
    np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


@dataclass(frozen=True)
class PerturbationSettings:
    a_start: float = 1.0e-7
    lmax_g: int = 24
    lmax_pol: int = 20
    lmax_r: int = 24
    lmax_nu: int = 12
    rtol: float = 1.0e-5
    atol: float = 1.0e-12
    tc_epsilon: float = 0.1
    rsa_ktau: float = 30.0
    rsa_ktau_nu: float = 30.0
    rsa_ktau_r: float = 30.0
    # CAMB applies the late radiation truncation only for k > 0.03 /Mpc
    # (DoLateRadTruncation with latePhotSwitchBoost); below that the photon
    # hierarchy is integrated to z = 0.
    rsa_kmin: float = 0.0
    rsa_kmin_r: float = 0.0
    max_steps: int = 400000
    h_start: float = 1.0e-4
    dlna_max: float = 0.005
    dlna_min: float = 1.0e-6
    dlna_acoustic: float = 0.02
    stiff_factor: float = 0.2
    dlna_superhorizon: float = 1.0e-3
    h_max: float = 0.05
    h_min: float = 1.0e-8


class PerturbationBatch:
    """Coefficient evaluation for a batch of trajectories ``(sample, k)``."""

    def __init__(self, bg, thermal, k, sample, settings: PerturbationSettings):
        self.bg = bg
        self.th = thermal
        self.k = np.asarray(k, dtype=np.float64)
        self.sample = np.asarray(sample, dtype=np.int64)
        self.s = settings
        self.T = self.k.shape[0]
        self.n_species = bg.n_species
        self.layout = StateLayout(
            lmax_g=settings.lmax_g, lmax_pol=settings.lmax_pol,
            lmax_r=settings.lmax_r, lmax_nu=settings.lmax_nu,
            n_species=self.n_species, n_q=NU_Q.size)
        s = self.sample
        self.grho_c0 = bg.grho_c0[s]
        self.grho_b0 = bg.grho_b0[s]
        self.grho_g0 = bg.grho_g0[s]
        self.grho_r0 = bg.grho_nu_massless0[s]
        self.grho_v0 = bg.grho_v0[s]
        self.grho_nu_unit = bg.grho_nu_species_unit[s][:, None]
        self.degeneracy = bg.massive_degeneracy[s]
        self.mu0 = bg._mu0[s]
        self._ln_a = thermal.ln_a_table
        self._kap_tab = thermal.opacity(np.exp(self._ln_a))
        # d(kappa)/dln a, same grid; kappa varies by many orders of magnitude
        # through recombination, and the tight-coupling slip needs its
        # logarithmic derivative (CAMB: dopacity/opacity).
        self._dkap_tab = np.gradient(self._kap_tab, self._ln_a, axis=1)
        self._tau_grid = bg.tau(np.exp(self._ln_a))                    # (B, N)
        self._cs2_tab = thermal.cs2_b(np.exp(self._ln_a))
        self._tau_tab = bg._tau_table
        self._rho_spl, self._p_spl = bg._nu_tables()

    # ------------------------------------------------------------------
    def _interp(self, table, ln_a):
        tab = self._ln_a
        i = np.clip(np.searchsorted(tab, ln_a) - 1, 0, len(tab) - 2)
        t = (ln_a - tab[i]) / (tab[i + 1] - tab[i])
        return table[self.sample, i] * (1.0 - t) + table[self.sample, i + 1] * t

    def coefficients(self, a):
        a = float(a) if np.ndim(a) == 0 else np.asarray(a, dtype=np.float64)
        inv_a = 1.0 / a
        # self.mu0 is already per-trajectory (T,), so it needs its own axis;
        # a[:, None] * self.mu0 would broadcast to (T, T) for T > 1.
        mu = np.maximum(a[:, None] * self.mu0[:, None], 1e-12)
        rho = np.exp(self._rho_spl(np.log(mu)))
        pnu = np.exp(self._p_spl(np.log(mu)))
        grho_nu_mass = np.sum(self.grho_nu_unit * self.degeneracy * rho, axis=1) * inv_a**2
        p_nu_mass = np.sum(self.grho_nu_unit * self.degeneracy * pnu, axis=1) * inv_a**2
        grho_c = self.grho_c0 * inv_a
        grho_b = self.grho_b0 * inv_a
        grho_g = self.grho_g0 * inv_a**2
        grho_r = self.grho_r0 * inv_a**2
        grho_v = self.grho_v0 * a**2
        grho = grho_c + grho_b + grho_g + grho_r + grho_nu_mass + grho_v
        h = np.sqrt(grho / 3.0)
        gpres = (grho_g + grho_r) / 3.0 + p_nu_mass - grho_v
        hprime = -0.5 * h**2 * (1.0 + 3.0 * gpres / grho)
        ln_a = np.log(a)
        # NOTE: _tau_grid is evaluated on the *thermal* ln-a grid, which is
        # the grid `_interp` indexes; using bg._tau_table here silently
        # mixes two different grids and corrupts 1/tau.
        tau = self._interp(self._tau_grid, ln_a)
        # The pre-recombination opacity reaches ~1e100 /Mpc; above ~1e30 the
        # coupling is exactly the tight-coupling limit, and capping avoids
        # overflow in branch values that are discarded by the regime mask.
        kappa = np.minimum(self._interp(self._kap_tab, ln_a), 1.0e30)
        dopacity = self._interp(self._dkap_tab, ln_a) * h   # dkappa/dtau
        cs2 = self._interp(self._cs2_tab, ln_a)
        am = np.maximum(a[:, None] * self.mu0[:, None], 1e-30)
        vnu = 1.0 / np.sqrt(1.0 + (am[:, :, None] / NU_Q[None, None, :]) ** 2)
        return {
            "h": h, "hprime": hprime, "gpres": gpres, "tau": tau,
            "kappa": kappa, "dopacity": dopacity, "cs2": cs2, "grho_c": grho_c, "grho_b": grho_b,
            "grho_g": grho_g, "grho_r": grho_r, "grho_nu_mass": grho_nu_mass,
            "grho_nu_species": (self.grho_nu_unit * self.degeneracy * rho)
            * (inv_a ** 2)[:, None],
            "k": self.k, "vnu": vnu,
        }


# ---------------------------------------------------------------------------

def initial_state(batch: "PerturbationBatch", a0: float):
    """Superhorizon adiabatic series (CAMB ``InitialScalarPerturbations``).

    Returns the dense state tensor ``Y[D, T]``.
    """
    bg = batch.bg
    L = batch.layout
    s = batch.sample
    k = batch.k
    T = batch.T
    tau0 = np.broadcast_to(bg.tau(np.array([a0])).reshape(-1)[s], (T,))
    grho_b = batch.grho_b0 / a0
    grho_c = batch.grho_c0 / a0
    grho_g = batch.grho_g0 / a0 ** 2
    grho_nu = (batch.grho_r0 + np.sum(batch.grho_nu_unit * batch.degeneracy, axis=1)) / a0 ** 2

    x = k * tau0
    om = (grho_b + grho_c) / np.sqrt(3.0 * (grho_g + grho_nu))
    omtau = om * tau0
    Rv = grho_nu / (grho_nu + grho_g)
    Rp15 = 4.0 * Rv + 15.0

    Y = L.allocate(T)
    dg = x ** 2 / 3.0 * (1.0 - omtau / 5.0)
    dr = dg.copy()
    db = 0.75 * dg
    qg = dg * x / 9.0
    qr = -(4.0 * Rv + 23.0) / Rp15 * x ** 3 / 27.0
    pir = 4.0 / 3.0 * x ** 2 / Rp15 * (
        1.0 + omtau / 4.0 * (4.0 * Rv - 5.0) / (2.0 * Rv + 15.0))
    eta = 2.0 * (1.0 - x ** 2 / 12.0 * (-10.0 / Rp15 + 1.0))

    Y[L.scalar("dc")] = db
    Y[L.scalar("db")] = db
    Y[L.scalar("vb")] = 0.75 * qg
    Y[L.scalar("dg")] = dg
    Y[L.scalar("qg")] = qg
    Y[L.scalar("dr")] = dr
    Y[L.scalar("qr")] = qr
    Y[L.scalar("pir")] = pir
    Y[L.scalar("etak")] = -eta * k / 2.0
    Y[L.fr(3)] = 4.0 / 21.0 / Rp15 * x ** 3
    return Y


def _l_coeff(k, l):
    return k * l / (2.0 * l + 1.0), k * (l + 1.0) / (2.0 * l + 1.0)


def rhs(Y, a, batch: "PerturbationBatch", c=None):
    """Right-hand side on the dense state tensor.

    Returns ``dY`` with the same ``(D, T)`` shape, holding ``dY/dln a``.
    Every operation is a row-wise or block-wise tensor expression with
    per-trajectory scalar coefficients: no host callbacks, no per-trajectory
    control flow, no dynamic allocation beyond the output tensor.  This is the
    sequence a generated device kernel executes.
    """
    if c is None:
        c = batch.coefficients(a)
    s = batch.s
    L = batch.layout
    k = c["k"]
    k2 = k ** 2
    h = c["h"]
    kappa = c["kappa"]
    cs2 = c["cs2"]
    tau = c["tau"]
    grho_g, grho_r = c["grho_g"], c["grho_r"]
    grho_b, grho_c = c["grho_b"], c["grho_c"]
    grho_spT = c["grho_nu_species"].T                     # (S, T)
    T = batch.T
    n_s, n_q = L.n_species, L.n_q
    w = NU_W

    dc = Y[L.scalar("dc")]
    db = Y[L.scalar("db")]
    vb = Y[L.scalar("vb")]
    dg = Y[L.scalar("dg")]
    qg = Y[L.scalar("qg")]
    pig = Y[L.scalar("pig")]
    dr = Y[L.scalar("dr")]
    qr = Y[L.scalar("qr")]
    pir = Y[L.scalar("pir")]
    etak = Y[L.scalar("etak")]
    Fnu = Y[L.offsets["Fnu"]].reshape(n_s, n_q, L.lmax_nu + 1, T)

    # ---- massive-neutrino projections ---------------------------------
    epsT = np.transpose(1.0 / c["vnu"], (1, 2, 0))        # (S, nq, T)
    wq = w[None, :, None]
    den = np.sum(wq * epsT, axis=1)                       # (S, T)
    clxnu = np.sum(wq * epsT * Fnu[..., 0, :], axis=1) / den
    # CAMB normalises the massive-neutrino heat flux by the same weighted
    # energy sum as the density (Nu_Integrate_L01: fnu/rhonu), i.e. by
    # sum(w * eps), not by sum(w); the two differ whenever the modes are
    # relativistic.
    qnu = np.sum(wq * Fnu[..., 1, :], axis=1) / np.sum(wq * epsT, axis=1)
    dgrho_nu = np.sum(grho_spT * clxnu, axis=0)
    dgq_nu = np.sum(grho_spT * qnu, axis=0)

    dgrho_matter = grho_b * db + grho_c * dc + dgrho_nu
    dgq_matter = grho_b * vb + dgq_nu

    tcu = kappa > (k / s.tc_epsilon)
    ktau = k * tau
    rsa_g = (ktau > s.rsa_ktau) & (k > s.rsa_kmin)
    rsa_r = (ktau > s.rsa_ktau_nu) & (k > s.rsa_kmin_r)

    # ---- radiation streaming approximation ----------------------------
    zm = (0.5 * dgrho_matter / k + etak) / h
    dzm = -h * zm - 0.5 * dgrho_matter / k
    k_rsa = np.where(rsa_g, k, 1.0)
    kap_rsa = np.where(rsa_g, kappa, 0.0)
    dg_rsa = -4.0 * dzm / k_rsa - 4.0 / k_rsa * kap_rsa * (vb + zm)
    qg_rsa = -4.0 / 3.0 * zm
    dr_rsa = -4.0 * dzm / k
    qr_rsa = -4.0 / 3.0 * zm

    dg_e = np.where(rsa_g, dg_rsa, dg)
    qg_e = np.where(rsa_g, qg_rsa, qg)
    dr_e = np.where(rsa_r, dr_rsa, dr)
    qr_e = np.where(rsa_r, qr_rsa, qr)

    dgrho = dgrho_matter + grho_g * dg_e + grho_r * dr_e
    dgq = dgq_matter + grho_g * qg_e + grho_r * qr_e

    inv_tau = np.minimum(1.0 / np.maximum(tau, 1e-300), k)
    z = (0.5 * dgrho / k + etak) / h
    sigma = z + 1.5 * dgq / k2

    dY = np.zeros_like(Y)
    dY[L.scalar("etak")] = 0.5 * dgq
    dY[L.scalar("dc")] = -k * z
    db_dot = -k * (z + vb)
    dY[L.scalar("db")] = db_dot
    delta_p_b = cs2 * db

    # ---- baryon velocity: tight coupling or full equation -------------
    pig_tc = 32.0 / 45.0 / np.maximum(kappa, 1e-300) * k * (sigma + vb)
    dg_dot = -k * (4.0 / 3.0 * z + qg_e)
    adotdota = (h ** 2 - c["gpres"]) / 2.0
    pb43 = 4.0 / 3.0 * grho_g / grho_b
    photbar = grho_g / grho_b
    slip = (
        -(2.0 * h / (1.0 + pb43)
          + np.where(tcu, c.get('dopacity', 0.0) / np.maximum(kappa, 1e-300), 0.0))
        * (vb - 0.75 * qg_e)
        + (-adotdota * vb - k / 2.0 * h * dg_e + k * (cs2 * db_dot - dg_dot / 4.0))
        / (kappa * (1.0 + pb43))
    )
    vb_tc = (
        (-h * vb + cs2 * k * db + k / 4.0 * pb43 * (dg_e - 2.0 * pig_tc)) / (1.0 + pb43)
        + pb43 / (1.0 + pb43) * slip
    )
    kap_notc = np.where(tcu, 0.0, kappa)
    vb_full = -h * vb + k * delta_p_b - photbar * kap_notc * (4.0 / 3.0 * vb - qg_e)
    dY[L.scalar("vb")] = np.where(tcu, vb_tc, vb_full)
    vb_e = dY[L.scalar("vb")]

    qg_tc = (4.0 / 3.0 * (-vb_e - h * vb + delta_p_b * k) / pb43
             + k / 3.0 * dg_e - 2.0 / 3.0 * k * pig_tc)
    qg_full = k / 3.0 * dg_e - 4.0 / 3.0 * k * pig + 4.0 / 3.0 * kap_notc * (vb - 0.75 * qg_e)
    dY[L.scalar("qg")] = np.where(rsa_g, 0.0, np.where(tcu, qg_tc, qg_full))

    polter = pig / 10.0 + 0.6 * Y[L.g(2)]
    pig_dot = (k * 2.0 / 5.0 * qg_e - k * 3.0 / 5.0 * Y[L.fg(3)]
               - np.where(tcu | rsa_g, 0.0, kappa) * (pig - polter)
               + (8.0 / 15.0) * k * sigma)
    dY[L.scalar("pig")] = np.where(tcu | rsa_g, 0.0, pig_dot)
    dY[L.scalar("dg")] = np.where(rsa_g, 0.0, dg_dot)

    # ---- photon moments (rows), tight coupling / RSA freeze them -------
    clamp = tcu | rsa_g
    for l in range(3, L.lmax_g):
        kk, kk2 = _l_coeff(k, l)
        dY[L.fg(l)] = np.where(
            clamp, 0.0,
            kk * Y[L.fg(l - 1)] - kk2 * Y[L.fg(l + 1)] - kappa * Y[L.fg(l)])
    kk, _ = _l_coeff(k, L.lmax_g)
    dY[L.fg(L.lmax_g)] = np.where(
        clamp, 0.0,
        kk * Y[L.fg(L.lmax_g - 1)] - (L.lmax_g + 1) * Y[L.fg(L.lmax_g)] * inv_tau
        - kappa * Y[L.fg(L.lmax_g)])

    # ---- polarization --------------------------------------------------
    dY[L.g(2)] = np.where(clamp, 0.0,
                          -kappa * (Y[L.g(2)] - polter) - k / 3.0 * Y[L.g(3)])
    for l in range(3, L.lmax_pol):
        kk, _ = _l_coeff(k, l)
        polfack = (l + 3.0) * (l - 1.0) / (l + 1.0) * k
        dY[L.g(l)] = np.where(
            clamp, 0.0,
            -kappa * Y[L.g(l)] + kk * Y[L.g(l - 1)] - polfack * Y[L.g(l + 1)])
    poltrunc = L.lmax_pol / max(1, L.lmax_pol - 2)
    kk, _ = _l_coeff(k, L.lmax_pol)
    dY[L.g(L.lmax_pol)] = np.where(
        clamp, 0.0,
        -kappa * Y[L.g(L.lmax_pol)] + k * poltrunc * Y[L.g(L.lmax_pol - 1)]
        - (L.lmax_pol + 3) * Y[L.g(L.lmax_pol)] * inv_tau)

    # ---- massless neutrinos -------------------------------------------
    dY[L.scalar("dr")] = np.where(rsa_r, 0.0, -k * (4.0 / 3.0 * z + qr_e))
    kk1, kk12 = _l_coeff(k, 1)
    dY[L.scalar("qr")] = np.where(rsa_r, 0.0, kk1 * dr_e - kk12 * pir)
    kk2_, kk22 = _l_coeff(k, 2)
    dY[L.scalar("pir")] = np.where(
        rsa_r, 0.0, kk2_ * qr_e - kk22 * Y[L.fr(3)] + (8.0 / 15.0) * k * sigma)
    for l in range(3, L.lmax_r):
        kk, kk2 = _l_coeff(k, l)
        dY[L.fr(l)] = np.where(rsa_r, 0.0,
                               kk * Y[L.fr(l - 1)] - kk2 * Y[L.fr(l + 1)])
    kk, _ = _l_coeff(k, L.lmax_r)
    dY[L.fr(L.lmax_r)] = np.where(
        rsa_r, 0.0,
        kk * Y[L.fr(L.lmax_r - 1)] - (L.lmax_r + 1) * Y[L.fr(L.lmax_r)] * inv_tau)

    # ---- massive neutrinos (block layout (S, nq, L+1, T)) --------------
    vT = np.transpose(c["vnu"], (1, 2, 0))                # (S, nq, T)
    kb = k[None, None, :]
    zb = z[None, None, :]
    dFnu = np.zeros_like(Fnu)
    kk = np.array([k * l / (2.0 * l + 1.0) for l in range(L.lmax_nu + 2)])
    kk2 = np.array([k * (l + 1.0) / (2.0 * l + 1.0) for l in range(L.lmax_nu + 2)])
    dFnu[..., 0, :] = -kb * (4.0 / 3.0 * zb + vT * Fnu[..., 1, :])
    dFnu[..., 1, :] = vT * (kk[1][None, None, :] * Fnu[..., 0, :]
                            - kk2[1][None, None, :] * Fnu[..., 2, :])
    dFnu[..., 2, :] = (vT * (kk[2][None, None, :] * Fnu[..., 1, :]
                             - kk2[2][None, None, :] * Fnu[..., 3, :])
                       + (8.0 / 15.0) * kb * sigma[None, None, :])
    for l in range(3, L.lmax_nu):
        dFnu[..., l, :] = vT * (kk[l][None, None, :] * Fnu[..., l - 1, :]
                                - kk2[l][None, None, :] * Fnu[..., l + 1, :])
    l = L.lmax_nu
    dFnu[..., l, :] = (kb * vT * Fnu[..., l - 1, :]
                       - (l + 1) * Fnu[..., l, :] * inv_tau[None, None, :])
    dY[L.offsets["Fnu"]] = dFnu.reshape(L.offsets["Fnu"].stop - L.offsets["Fnu"].start, T)

    # d/dtau -> d/dln a
    dY *= (1.0 / h)[None, :]
    return dY


# ---------------------------------------------------------------------------
_ERROR_KEYS = ("dc", "db", "dg", "qg", "vb", "dr", "qr", "etak")


def _event_times(batch: "PerturbationBatch", a0, a1):
    """Per-trajectory scale factors of the regime switches.

    Tight coupling ends when ``kappa = k/epsilon``; radiation streaming starts
    when ``k*tau`` reaches its threshold.  Crossings are located on the
    tabulated background grid and linearly interpolated in ``ln a``.  Each
    trajectory clamps its own step to land exactly on its switch, so no RK
    stage straddles a discontinuity (see Tenbol_decisions.md D8).
    """
    s = batch.s
    ln_a = batch._ln_a
    kap = batch._kap_tab[batch.sample]
    tau = batch._tau_grid[batch.sample]
    k = batch.k[:, None]
    out = {}
    for name, ref, extra in (
        ("tc", kap, k / s.tc_epsilon),
        ("rsa_g", k * tau, np.where(k > s.rsa_kmin, s.rsa_ktau, np.inf)),
        ("rsa_r", k * tau, np.where(k > s.rsa_kmin_r, s.rsa_ktau_r, np.inf)),
    ):
        diff = ref - extra
        change = np.sign(diff[:, :-1]) != np.sign(diff[:, 1:])
        arr = np.full((batch.T, 2), np.inf)
        has = change.any(axis=1)
        first = np.argmax(change, axis=1)
        i = np.arange(batch.T)
        f = diff[i, first] / (diff[i, first] - diff[i, first + 1])
        arr[:, 0] = np.where(has, np.exp(ln_a[first] + f * (ln_a[first + 1] - ln_a[first])), np.inf)
        change2 = change.copy()
        change2[i, first] = False
        has2 = change2.any(axis=1)
        second = np.argmax(change2, axis=1)
        f2 = diff[i, second] / (diff[i, second] - diff[i, second + 1])
        arr[:, 1] = np.where(has2, np.exp(ln_a[second] + f2 * (ln_a[second + 1] - ln_a[second])),
                             np.inf)
        out[name] = arr
    return out


def _take(c, idx, n_total):
    """Select trajectory `idx` out of a coefficient dict of length ``n_total``."""
    out = {}
    for key, val in c.items():
        arr = np.asarray(val)
        out[key] = arr[idx] if arr.ndim >= 1 and arr.shape[0] == n_total else val
    return out


def _record_rows(rec, Y, c, rec_rows, idx, layout: StateLayout):
    """Record a (possibly interpolated) state slice into the output arrays.

    ``Y`` has shape ``(D, n)`` and ``c`` is already sliced to those ``n``
    trajectories; ``rec_rows`` are the output rows they map to.
    """
    for key in ("dc", "db", "dg", "dr", "vb", "qg", "qr", "etak"):
        rec[key][rec_rows, idx] = Y[layout.scalar(key)]
    grho_b, grho_c = c["grho_b"], c["grho_c"]
    grho = c["grho_nu_species"]                          # (n, S)
    n_s, n_q, n_l = layout.n_species, layout.n_q, layout.lmax_nu + 1
    Fnu = Y[layout.offsets["Fnu"]].reshape(n_s, n_q, n_l, Y.shape[1])
    epsT = np.transpose(1.0 / c["vnu"], (1, 2, 0))
    wq = NU_W[None, :, None]
    den = np.sum(wq * epsT, axis=1)
    clxnu = np.sum(wq * epsT * Fnu[..., 0, :], axis=1) / den
    dgrho_nu = np.sum(np.asarray(grho).T * clxnu, axis=0)
    rec["dm"][rec_rows, idx] = (grho_c * Y[layout.scalar("dc")]
                                + grho_b * Y[layout.scalar("db")]
                                + dgrho_nu) / (grho_c + grho_b + np.sum(grho, axis=1))


def _hermite(Y0, Y1, f0, f1, delta, theta):
    """Cubic Hermite interpolation inside one step (dense output)."""
    t = theta[None, :]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    d = delta[None, :]
    return h00 * Y0 + h10 * d * f0 + h01 * Y1 + h11 * d * f1


def evolve(batch: "PerturbationBatch", z_out, a_end: float = 1.0,
           settings: "PerturbationSettings | None" = None):
    """Integrate the dense state tensor with a fixed physics-based RK4 schedule.

    The schedule combines the angular CFL (``k*dtau``), the Thomson stiffness
    (``kappa*dtau``) and a super-horizon refinement, plus exact landing on
    regime switches and requested output scale factors.  A fixed schedule
    (instead of an embedded error controller) keeps every step branch-free in
    the state and reproducible, which is what the generated device kernels
    need; step-size convergence is validated by halving ``dlna_max``.
    """
    s = settings or batch.s
    L = batch.layout
    T = batch.T
    a = np.full(T, s.a_start)
    Y = initial_state(batch, s.a_start)
    z_out = np.asarray(z_out, dtype=np.float64)
    order = np.argsort(1.0 / (1.0 + z_out))
    a_out = np.sort(1.0 / (1.0 + z_out))
    n_out = len(a_out)
    rec = {k: np.full((T, n_out), np.nan) for k in
           ("dc", "db", "dg", "dr", "vb", "qg", "qr", "etak", "dm")}

    events = _event_times(batch, s.a_start, a_end)
    targets = np.sort(np.vstack([np.where(np.isfinite(events[n][:, 0]), events[n][:, 0], np.inf)
                                 for n in ("tc", "rsa_g", "rsa_r")]), axis=0).T   # (T, 3)
    next_ev = np.zeros(T, dtype=np.int64)
    pending = targets[:, 0].copy()
    next_out = np.zeros(T, dtype=np.int64)
    done = np.zeros(T, dtype=bool)
    c = batch.coefficients(a)
    total_steps = 0
    while not np.all(done):
        total_steps += 1
        if total_steps > s.max_steps:
            raise NumericalFailure("perturbation integration exceeded max_steps")

        # Only regime switches and the end time constrain the step; requested
        # output scale factors are handled by dense output (see _hermite), so
        # the solution does not depend on which redshifts the caller asks for.
        target = np.minimum(np.full(T, a_end), np.where(pending > a, pending, a_end))
        dln_target = np.where(target > a, np.log(np.maximum(target, 1e-300) / a), 0.0)
        tcu = c["kappa"] > (batch.k / s.tc_epsilon)
        dtau = 1.0 / c["h"]
        lim = np.full(T, s.dlna_max)
        lim = np.where((batch.k * c["tau"]) < s.rsa_ktau,
                       np.minimum(lim, s.dlna_acoustic / np.maximum(batch.k * dtau, 1e-30)), lim)
        lim = np.where((batch.k * c["tau"]) < 1.0,
                       np.minimum(lim, s.dlna_superhorizon), lim)
        lim = np.where(~tcu,
                       np.minimum(lim, s.stiff_factor / np.maximum(c["kappa"] * dtau, 1e-30)), lim)
        h = np.minimum.reduce([lim, dln_target, np.full(T, s.h_max)])
        h = np.maximum(h, s.dlna_min)
        h = np.where(done, 0.0, h)

        hn = h[None, :]
        a_prev = a.copy()
        Y_prev = Y
        k1 = rhs(Y, a, batch, c)
        k2 = rhs(Y + 0.5 * hn * k1, a * np.exp(0.5 * h), batch)
        k3 = rhs(Y + 0.5 * hn * k2, a * np.exp(0.5 * h), batch)
        k4 = rhs(Y + hn * k3, a * np.exp(h), batch)
        Y_new = Y + (hn / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(Y_new)):
            bad = int(np.argmax(~np.isfinite(Y_new).all(axis=0)))
            raise NumericalFailure(
                f"non-finite perturbation state at a={a[bad]:.3e}, k={batch.k[bad]:.3e}")
        Y = Y_new
        a = np.where(done, a, np.minimum(a * np.exp(h), a_end))
        c = batch.coefficients(a)
        if getattr(batch, "_probe_list", None):
            for tgt in list(batch._probe_list):
                crossed = (a_prev < tgt) & (a >= tgt)
                if np.any(crossed):
                    j = int(np.argmax(crossed))
                    dc = float(Y[L.scalar("dc")][j]); dcp = float(k1[L.scalar("dc")][j])
                    hh = float(c["h"][j]); kk = float(batch.k[j])
                    dm = float(Y[L.scalar("dc")][j])
                    zmet = -dcp / kk                      # from delta_c' = -k z
                    dzk = float(k1[L.scalar("etak")][j])      # d(etak)/dln a
                    zmet2 = zmet
                    sigma = zmet2 + 3.0 * hh * dzk / kk ** 2
                    vnewt = -kk * (float(Y[L.scalar("vb")][j]) + sigma) / hh
                    print(f"PROBE a={float(a[j]):.3e} z={1/float(a[j])-1:8.1f} "
                          f"dc={dc: .4e} dc'={dcp: .4e} dm={dm: .4e} "
                          f"zmet={zmet: .4e} z_poisson={zmet:.3e} "
                          f"ratio_zdot={(-dcp)/(dc*hh):.4f} "
                          f"etak={float(Y[L.scalar('etak')][j]): .4e} "
                          f"qg={float(Y[L.scalar('qg')][j]): .4e} "
                          f"db={float(Y[L.scalar('db')][j]): .4e} "
                          f"dbp={float(k1[L.scalar('db')][j]): .4e} "
                          f"dcp={float(k1[L.scalar('dc')][j]): .4e} "
                          f"vb={float(Y[L.scalar('vb')][j]): .4e} "
                          f"zmet={zmet2: .4e} sigma={sigma: .4e} vnewt={vnewt: .4e}", flush=True)
                    batch._probe_list.remove(tgt)

        # regime switches: install the CAMB tight-coupling exit values
        for j, name in enumerate(("tc", "rsa_g", "rsa_r")):
            # only trajectories whose *current* pending event is this switch
            hit = (~done) & (next_ev == j) & (a >= pending)
            if name == "tc" and np.any(hit):
                kap = np.maximum(c["kappa"][hit], 1e-300)
                pig_tc = Y[L.scalar("pig")][hit]
                f3 = 3.0 / 7.0 * pig_tc * batch.k[hit] / kap
                Y[L.fg(3), hit] = f3
                Y[L.g(2), hit] = pig_tc / 4.0
                Y[L.g(3), hit] = f3 / 4.0
        advanced = (~done) & (a >= pending)
        if np.any(advanced):
            next_ev = np.where(advanced, next_ev + 1, next_ev)
            pending = np.where(advanced,
                               np.where(next_ev < 3, targets[np.arange(T), np.minimum(next_ev, 2)],
                                        np.inf), pending)

        while True:
            ready = (next_out < n_out) & (a >= a_out[np.minimum(next_out, n_out - 1)] - 1e-14)
            if not np.any(ready):
                break
            rows = np.nonzero(ready)[0]
            idx = next_out[rows]
            a_tgt = a_out[idx]
            # interpolate from the step just taken: a_prev -> a
            theta = np.clip((np.log(a_tgt) - np.log(a_prev[rows])) / h[rows], 0.0, 1.0)
            Ymid = _hermite(Y_prev[:, rows], Y[:, rows], k1[:, rows], k4[:, rows],
                            h[rows], theta)
            cm = _take(c, rows, T)
            for j in np.unique(idx):
                sel_j = np.nonzero(idx == j)[0]
                _record_rows(rec, Ymid[:, sel_j], _take(cm, sel_j, len(rows)),
                             rows[sel_j], int(j), L)
            next_out[rows] += 1
        done = a >= a_end - 1e-14
    for key in rec:
        rec[key] = rec[key][:, order]
    rec["n_steps"] = total_steps
    return rec
