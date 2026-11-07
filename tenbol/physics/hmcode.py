"""HMcode-2020 halo model (CAMB ``halofit_version="mead2020"``).

Port of ``halofit.f90``'s ``imead = 3`` branch: the 2020 halo model of
Mead et al. (arXiv:2009.01858), without the optional baryon-feedback response
(``mead2020_feedback``), which is a separate three-pass variant.

Units follow CAMB's HMcode internals: masses in ``Msun/h``, radii in
``Mpc/h``, ``k`` in ``h/Mpc``, ``P`` in ``(Mpc/h)^3``.

The implementation is vectorised over ``k`` and over the mass grid so the whole
model can be lifted to a tensor backend later; only the per-redshift table
setup uses loops, and that runs once per cosmology.

Reference: ``camb-2.0.5/fortran/halofit.f90``, subroutines ``HMcode_redshift``,
``halomod_tables``, ``fill_conc``, ``halomod``, ``p_1h``, ``p_2h`` and the
parameter functions ``eta``, ``kstar``, ``As``, ``fdamp``, ``alpha``,
``delta_c``, ``Delta_v``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import sici

__all__ = ["HMcode2020", "HMParameters"]

# Fixed HMcode constants (halofit.f90 module data)
PI = np.pi
M_MIN = 1.0            # Msun/h
M_MAX = 1.0e18
N_MASS = 256
RHO_CRIT = 2.775e11    # Msun/Mpc^3 (h^2 units folded in, as in CAMB)
F_BULLOCK = 0.01 ** (1.0 / 3.0)
FDAMP_MIN = 0.0
FDAMP_MAX = 1.0
ALPHA_MIN = 0.0
ALPHA_MAX = 10.0


@dataclass
class HMParameters:
    """Values of the redshift-dependent HMcode-2020 parameters, kept for tests."""

    delta_c: float
    delta_v: float
    eta: float
    k_star: float
    a_conc: float
    fdamp: float
    alpha: float
    k_damp: float
    n_eff: float
    r_nl: float
    sigma8_cold: float


@dataclass
class HMcode2020:
    """HMcode-2020 for one cosmology at one redshift.

    Parameters
    ----------
    omega_m, omega_b, omega_c, omega_nu : float
        Present-day density fractions (``Omega``, not ``Omega h^2``).
    h : float
        ``H0 / 100``.
    n_s : float
        Scalar spectral index (used to clip the effective spectral index).
    k_lin, p_lin : ndarray
        Linear power at the *tabulation* redshift (usually z=0), ``k`` in
        h/Mpc and ``P`` in ``(Mpc/h)^3``, increasing and strictly positive.
    growth : callable
        ``growth(a) -> D(a)`` normalised to ``D(1) = 1``.
    z : float
        Redshift at which to evaluate the model.
    """

    omega_m: float
    omega_b: float
    omega_c: float
    omega_nu: float
    h: float
    n_s: float
    k_lin: np.ndarray
    p_lin: np.ndarray
    growth: object
    z: float = 0.0
    terms: object = None
    params: HMParameters = field(init=False)

    def __post_init__(self) -> None:
        self.omega_m = float(self.omega_m)
        self.omega_b = float(self.omega_b)
        self.omega_c = float(self.omega_c)
        self.omega_nu = float(self.omega_nu)
        self.h = float(self.h)
        self.n_s = float(self.n_s)
        self.z = float(self.z)
        self.f_nu = self.omega_nu / self.omega_m

        self.k_lin = np.asarray(self.k_lin, dtype=np.float64)
        self.p_lin = np.asarray(self.p_lin, dtype=np.float64)
        order = np.argsort(self.k_lin)
        self.k_lin = self.k_lin[order]
        self.p_lin = self.p_lin[order]

        # Linear spectrum interpolator in (ln k, ln P); the linear input is the
        # cold-matter spectrum scaled by T_cb/T_cbnu**2 when neutrinos are present.
        self._lnk = np.log(self.k_lin)
        self._spl = CubicSpline(self._lnk, np.log(self.p_lin), extrapolate=True)

        # Cold-matter linear spectrum (used for sigma_cold); with massless
        # neutrinos only it equals the total one.
        self._lnp_cold = np.log(self.p_lin)
        self._spl_cold = CubicSpline(self._lnk, self._lnp_cold, extrapolate=True)

        self._build_tables()
        self._build_wiggle()

    # -- linear spectrum -------------------------------------------------
    def _delta2(self, k, cold=True):
        """Dimensionless linear power ``k^3 P(k) / (2 pi^2)``."""
        spl = self._spl_cold if cold else self._spl
        return np.exp(spl(np.log(k))) * k ** 3 / (2.0 * PI ** 2)

    def _pk(self, k, cold=True):
        spl = self._spl_cold if cold else self._spl
        return np.exp(spl(np.log(k)))

    # -- sigma(R) --------------------------------------------------------
    def _sigma_r(self, r, cold=True):
        """``sigma(R)`` at the tabulation redshift, from the linear spectrum."""
        lnk = self._lnk
        integrand = self._delta2(self.k_lin, cold=cold)      # = k^3 P / 2pi^2
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        # Broadcast over (n_r, n_k) so the whole radius grid is one tensor
        # operation: this is the shape the GPU backend will consume.
        x = r[:, None] * self.k_lin[None, :]
        small = np.abs(x) < 1e-3
        w = np.empty_like(x)
        w[small] = 1.0 - x[small] ** 2 / 10.0
        xb = x[~small]
        w[~small] = 3.0 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
        out = np.trapezoid(integrand[None, :] * w * w, lnk, axis=1)
        return np.sqrt(np.maximum(out, 0.0))

    # -- parameter fits --------------------------------------------------
    def _mead_terms(self):
        """Growth terms ``(x, y)``; EdS (1, 1) if no provider was supplied."""
        if self.terms is None:
            return 1.0, 1.0
        x, y = self.terms(self.z)[:2]
        return float(x), float(y)

    @staticmethod
    def _delta_c(z, omega_m, f_nu, x, y):
        """Mead (2017) collapse density, ``dc_Mead``."""
        om = _omega_m_z(z, omega_m)
        p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
        p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
        dc0 = (3.0 / 20.0) * (12.0 * PI) ** (2.0 / 3.0)
        dc = 1.0
        dc += _f_mead(x, y, p10, p11, p12, p13) * np.log10(om) ** 1
        dc += _f_mead(x, y, p20, p21, p22, p23)
        return dc * dc0 * (1.0 - 0.041 * f_nu)

    @staticmethod
    def _delta_v(z, omega_m, f_nu, x, y):
        """Mead (2017) virial overdensity, ``Dv_Mead``."""
        om = _omega_m_z(z, omega_m)
        p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
        p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
        dv0 = 18.0 * PI ** 2
        dv = 1.0
        dv += _f_mead(x, y, p30, p31, p32, p33) * np.log10(om) ** 1
        dv += _f_mead(x, y, p40, p41, p42, p43) * np.log10(om) ** 2
        return dv * dv0 * (1.0 + 0.763 * f_nu)

    # -- tables ----------------------------------------------------------
    def _build_tables(self):
        m = np.exp(np.linspace(np.log(M_MIN), np.log(M_MAX), N_MASS))
        rho = RHO_CRIT * self.omega_m
        r = (3.0 * m / (4.0 * PI * rho)) ** (1.0 / 3.0)

        d = float(np.ravel(self.growth(1.0 / (1.0 + self.z)))[0])
        sig8_cold = float(self._sigma_r(np.array([8.0]), cold=True)[0]) * d
        self._d = d
        self._sig8_cold = sig8_cold

        x_t, y_t = self._mead_terms()
        dc = self._delta_c(self.z, self.omega_m, self.f_nu, x_t, y_t)
        dv = self._delta_v(self.z, self.omega_m, self.f_nu, x_t, y_t)

        sig = self._sigma_r(r, cold=True) * d
        sigf = self._sigma_r(r * F_BULLOCK, cold=True) * d
        nu = dc / sig

        # Non-linear scale: nu(r_nl) = 1
        if nu[0] > 1.0:
            r_nl = float(r[0])
        else:
            r_nl = float(np.exp(np.interp(np.log(1.0), np.log(nu), np.log(r))))
        # Effective spectral index at r_nl from the local slope of sigma(R)
        lnr = np.log(r)
        slope = np.gradient(np.log(np.maximum(sig, 1e-300)), lnr)
        n_eff = float(np.interp(np.log(r_nl), lnr, -3.0 - 2.0 * slope))
        n_eff = min(max(n_eff, self.n_s - 4.0), self.n_s)

        eta = 0.1281 * sig8_cold ** (-0.3644)
        k_star = 0.05618 * sig8_cold ** (-1.013)
        a_conc = 5.196
        fdamp = min(max(0.2696 * sig8_cold ** 0.9403, FDAMP_MIN), FDAMP_MAX)
        alpha = min(max(1.875 * 1.603 ** n_eff, ALPHA_MIN), ALPHA_MAX)
        k_damp = 0.05699 * sig8_cold ** (-1.089)

        # Bullock formation redshifts: g(a_f) = dc * D(z) / sigma(M * f_Bullock)
        # Note halofit.f90 uses dc*growz/sigf with sigf already growth-scaled,
        # i.e. the *tabulation-redshift* sigma of the scaled radius.
        target = dc * d / sigf
        a_grid = np.linspace(1e-3, 1.0, 512)
        # ``growth`` must accept an array: evaluating it per point would mean
        # hundreds of ODE integrations per redshift.
        g_grid = np.atleast_1d(np.asarray(self.growth(a_grid), dtype=np.float64)).ravel()
        a_f = np.interp(target, g_grid, a_grid, left=a_grid[0], right=a_grid[-1])
        a_f = np.where(target <= g_grid[0], a_grid[0], a_f)
        z_f = 1.0 / a_f - 1.0
        z_f = np.where(target > d, self.z, z_f)      # halo forms in the future

        # Dolag corrections are unity for a cosmological constant
        conc = a_conc * (1.0 + z_f) / (1.0 + self.z)
        rv = r / dv ** (1.0 / 3.0)
        rs = rv / conc
        nfw_norm = 1.0 / (np.log(1.0 + conc) - conc / (1.0 + conc))
        nfw_moment2 = (0.5 * conc ** 2 - 2.0 * conc
                       + 3.0 * np.log(1.0 + conc) - conc / (1.0 + conc))

        # Sheth & Tormen mass function, integrated with the trapezium rule
        gst = 0.21616 * (1.0 + (0.707 * nu ** 2) ** (-0.3)) * np.exp(-0.707 * nu ** 2 / 2.0)
        w = np.empty_like(nu)
        w[0] = 0.5 * gst[0] * m[0] * (nu[1] - nu[0])
        w[-1] = 0.5 * gst[-1] * m[-1] * (nu[-1] - nu[-2])
        w[1:-1] = 0.5 * gst[1:-1] * m[1:-1] * (nu[2:] - nu[:-2])

        self._m = m
        self._r = r
        self._nu_eta = nu ** eta
        self._rs = rs
        self._conc = conc
        self._nfw_norm = nfw_norm
        self._nfw_moment2 = nfw_moment2
        self._p1h_weight = w
        self._rho_m = rho
        self.params = HMParameters(
            delta_c=dc, delta_v=dv, eta=eta, k_star=k_star, a_conc=a_conc,
            fdamp=fdamp, alpha=alpha, k_damp=k_damp, n_eff=n_eff, r_nl=r_nl,
            sigma8_cold=sig8_cold)

    # -- halo model pieces ----------------------------------------------
    def _window(self, k):
        """NFW window for a broadcast ``(n_mass, n_k)`` array of ``k * r_s`` arguments.

        The caller passes ``k * nu**eta`` already expanded over the mass axis.
        """
        k = np.atleast_2d(np.asarray(k, dtype=np.float64))
        ks = k * self._rs[:, None]
        c = self._conc[:, None]
        norm = self._nfw_norm[:, None]
        m2 = self._nfw_moment2[:, None]
        ks2 = (1.0 + c) * ks

        with np.errstate(invalid="ignore", divide="ignore"):
            small = c * ks < 0.2
            out_small = 1.0 - ks ** 2 * m2 * norm / 6.0

            si1, ci1 = sici(ks)
            si2, ci2 = sici(ks2)
            s1, c1 = np.sin(ks), np.cos(ks)
            s2, c2 = np.sin(ks2), np.cos(ks2)

            # For large arguments cos(ks)*(ci2 - ci1) suffers catastrophic
            # cancellation because ci oscillates; halofit.f90 evaluates
            # sin(c*ks) = sin(ks2 - ks) by angle subtraction instead.
            wide = ks2 > 4.0
            out_wide = (c1 * (ci2 - ci1) + s1 * (si2 - si1)
                        - (s2 * c1 - c2 * s1) / ks2) * norm
            out_near = (c1 * (ci2 - ci1) + s1 * (si2 - si1)
                        - np.sin(c * ks) / ks2) * norm
            out_big = np.where(wide, out_wide, out_near)
            out = np.where(small, out_small, out_big)
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    def _build_wiggle(self):
        """Isolate the BAO wiggle as P(k) minus its Gaussian-smoothed version.

        halofit.f90's calculate_psmooth smooths P/P_nowiggle with a Gaussian of
        width ``wiggle_sigma = 0.25`` in log k; smoothing log P directly over the
        same width gives the same broadband component to the accuracy that
        matters here, and avoids porting the Eisenstein & Hu no-wiggle fit.
        """
        lnk = self._lnk
        dlnk = float(np.median(np.diff(lnk)))
        if dlnk <= 0.0:
            self._p_wiggle = np.zeros_like(self.p_lin)
            self._sigv = 0.0
            return
        lnp = np.log(self.p_lin)
        # Gaussian smoothing in ln k with sigma = 0.25 (pixels = sigma / dlnk)
        n = lnp.size
        half = max(1, int(np.ceil(4.0 * 0.25 / dlnk)))
        x = np.arange(-half, half + 1) * dlnk
        ker = np.exp(-0.5 * (x / 0.25) ** 2)
        ker /= ker.sum()
        lnp_s = np.convolve(np.pad(lnp, half, mode="edge"), ker, mode="same")[half:half + n]
        self._p_wiggle = np.exp(lnp) - np.exp(lnp_s)

        # Velocity dispersion sigma_v^2 = (1/6pi^2) int dk P(k), R -> 0 limit of
        # CAMB's sigmaV; sets how fast the wiggle is damped away.
        self._sigv = float(np.sqrt(max(np.trapezoid(self.p_lin, self.k_lin) / (6.0 * PI ** 2), 0.0)))

    def _dewiggle(self, k, d2_lin):
        """Linear Delta^2 with the BAO wiggle damped by exp(-(k sigma_v)^2)."""
        pw = np.interp(np.log(k), self._lnk, self._p_wiggle, left=0.0, right=0.0)
        f = np.exp(-(k * self._sigv) ** 2)
        d2_w = pw * k ** 3 / (2.0 * PI ** 2) * self._d ** 2
        return d2_lin + (f - 1.0) * d2_w

    def _p_1h(self, k):
        k = np.atleast_1d(np.asarray(k, dtype=np.float64))
        w = self._window(k * self._nu_eta[:, None])
        s = np.sum(self._p1h_weight[:, None] * w * w, axis=0) / self._rho_m
        s = s * (1.0 - self.f_nu) ** 2
        p1h = s * k ** 3 / (2.0 * PI ** 2)
        x = (k / self.params.k_star) ** 4
        return p1h * x / (1.0 + x)

    def _p_2h(self, k, d2_lin):
        """Two-halo term in Delta^2 units; ``d2_lin`` is the linear Delta^2(k)."""
        k = np.atleast_1d(np.asarray(k, dtype=np.float64))
        frac = self.params.fdamp
        x = (k / self.params.k_damp) ** 2.85
        return self._dewiggle(k, d2_lin) * (1.0 - frac * x / (1.0 + x))

    def power(self, k):
        """Non-linear matter power ``P(k)`` at this redshift, ``(Mpc/h)³``."""
        k = np.atleast_1d(np.asarray(k, dtype=np.float64))
        # HMcode works internally in Delta^2(k) = k^3 P(k) / (2 pi^2), exactly as
        # CAMB's fill_plintab does; the conversion back to P(k) happens once, here.
        d2_lin = self._delta2(k, cold=False) * self._d ** 2
        p1h = self._p_1h(k)
        p2h = self._p_2h(k, d2_lin)
        a = self.params.alpha
        d2_nl = (np.maximum(p2h, 0.0) ** a + np.maximum(p1h, 0.0) ** a) ** (1.0 / a)
        return d2_nl * 2.0 * PI ** 2 / k ** 3


# -- Mead (2017) growth terms -------------------------------------------

def _omega_m_z(z, omega_m):
    """``Omega_m(z)`` for a flat universe with a cosmological constant."""
    e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    return omega_m * (1.0 + z) ** 3 / e2


def _f_mead(x, y, p0, p1, p2, p3):
    return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2 + p3 * (1.0 - y)
