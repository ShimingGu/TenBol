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
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.special import hyp2f1, sici

from tenbol.constants import T_CMB0_DEFAULT

__all__ = ["HMcode2020", "HMParameters"]

# Fixed HMcode constants (halofit.f90 module data)
PI = np.pi
# CAMB's HMcode builds sigma(R) on a 64-point table in ln R over [1e-4, 1e3]
# Mpc/h and interpolates with a cubic (halofit.f90 n_sigma_interpolation).
# Reproducing that grid matters: our exact integral and CAMB's interpolated
# table differ at the 0.1-1% level, which is the size of the residual we see.
R_SIG_MIN = 1.0e-4
R_SIG_MAX = 1.0e3
N_SIG = 64
M_MIN = 1.0            # Msun/h
M_MAX = 1.0e18
N_MASS = 256
RHO_CRIT = 2.775e11    # Msun/Mpc^3 (h^2 units folded in, as in CAMB)
F_BULLOCK = 0.01 ** (1.0 / 3.0)
FDAMP_MIN = 1.0e-3
FDAMP_MAX = 0.99
ALPHA_MIN = 0.5
ALPHA_MAX = 2.0
# Wiggle-isolation table, from halofit.f90:197-202
KMIN_WIGGLE = 5.0e-3      # h/Mpc
KMAX_WIGGLE = 5.0
NK_WIGGLE = 512
WIGGLE_SIGMA = 0.25       # Gaussian width in ln k
KNORM_NOWIGGLE = 0.03     # no-wiggle spectrum is pinned to the linear one here
SIGMA_K_MAX = 100.0     # h/Mpc, upper end of the extrapolated variance grid


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
        Wavenumbers in h/Mpc and total linear power in (Mpc/h)^3, divided by
        the squared growth factor at the target redshift.
    p_cold : ndarray or None
        Cold linear power on the same grid and with the same normalization.
        Defaults to the supplied total spectrum.
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
    p_cold: np.ndarray | None = None
    t_cmb: float = T_CMB0_DEFAULT
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

        self._lnk = np.log(self.k_lin)
        self._spl = CubicSpline(self._lnk, np.log(self.p_lin), extrapolate=True)
        cold = self.p_lin if self.p_cold is None else np.asarray(self.p_cold)[order]
        self._spl_cold = CubicSpline(self._lnk, np.log(cold), extrapolate=True)

        # Halo formation uses sigma(R) at radii much smaller than the output
        # wavelengths. Continue the short-wavelength power law for this
        # integral on the host, without extrapolating a cubic spline or
        # evolving more Boltzmann modes.
        self._sigma_lnk = self._lnk
        if self.k_lin[-1] < SIGMA_K_MAX:
            tail = np.linspace(self._lnk[-1], np.log(SIGMA_K_MAX), 42)[1:]
            self._sigma_lnk = np.concatenate((self._lnk, tail))
        self._sigma_k = np.exp(self._sigma_lnk)

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
        spl = self._spl_cold if cold else self._spl
        log_p = spl(self._lnk)
        if self._sigma_lnk.size > self._lnk.size:
            anchor = max(0, log_p.size - 5)
            slope = (log_p[-1] - log_p[anchor]) / (self._lnk[-1] - self._lnk[anchor])
            tail = self._sigma_lnk[log_p.size:] - self._lnk[-1]
            log_p = np.concatenate((log_p, log_p[-1] + min(slope, self.n_s - 3.0) * tail))
        integrand = np.exp(log_p + 3.0 * self._sigma_lnk) / (2.0 * PI ** 2)
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        # Broadcast over (n_r, n_k) so the whole radius grid is one tensor
        # operation: this is the shape the GPU backend will consume.
        x = r[:, None] * self._sigma_k[None, :]
        small = np.abs(x) < 1e-3
        w = np.empty_like(x)
        w[small] = 1.0 - x[small] ** 2 / 10.0
        xb = x[~small]
        w[~small] = 3.0 * (np.sin(xb) - xb * np.cos(xb)) / xb ** 3
        out = np.trapezoid(integrand[None, :] * w * w, self._sigma_lnk, axis=1)
        return np.sqrt(np.maximum(out, 0.0))

    # -- parameter fits --------------------------------------------------
    def _mead_terms(self):
        """Growth terms ``(x, y)`` for Mead (2017) ``dc_Mead`` / ``Dv_Mead``.

        These are **not** optional: for LCDM at z = 0 they are (0.788, 0.931),
        nowhere near 1, and both fits are written in terms of them.  Returning
        (1, 1) - the Einstein-de Sitter value - silently runs the whole halo model
        in the wrong cosmology and costs 28% of the nonlinear power at high k
        (measured: max |dev| 28.4% with the fallback against 1.1% with the real
        terms).  So fail loudly instead.
        """
        if self.terms is None:
            raise ValueError(
                "HMcode2020 needs growth terms: pass terms=lambda z: "
                "sigma.growth_terms(bg, z). Without them Mead's dc/Dv fits run in "
                "the EdS limit and the high-k nonlinear power is 28% low."
            )
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

        # sigma(R) through CAMB's 64-point ln R table, as its sigma_lut does.
        # sig8_cold below stays an exact integral because CAMB computes that
        # one with sigma_integral(8, 0, itype, cosm), not through the table.
        r_tab = np.geomspace(R_SIG_MIN, R_SIG_MAX, N_SIG)
        sig_tab = self._sigma_r(r_tab, cold=True)
        lut = CubicSpline(np.log(r_tab), np.log(np.maximum(sig_tab, 1e-300)),
                          extrapolate=True)
        sig = np.exp(lut(np.log(r))) * d
        sigf = np.exp(lut(np.log(r * F_BULLOCK))) * d
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
        # CAMB inverts the growth table with interp_cubic_sorted (zcoll_bull);
        # linear interpolation here shifts the formation redshifts and through
        # them the concentrations.  g(a) is strictly increasing, so a cubic in
        # (g -> a) is well defined.
        spl_af = CubicSpline(g_grid, a_grid, extrapolate=True)
        a_f = spl_af(np.clip(target, g_grid[0], g_grid[-1]))
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
        """Isolate the BAO wiggle the way CAMB's ``calculate_psmooth`` does.

        halofit.f90:1853-1873 divides by an Eisenstein & Hu *no-wiggle* spectrum,
        smooths that ratio with a Gaussian of width ``wiggle_sigma = 0.25`` in log
        k, then multiplies the no-wiggle spectrum back.  Smoothing log P directly -
        what this method used to do - folds the ~k^-3 broadband slope into the
        smooth part.

        Everything here is computed in ``Delta^2 = k^3 P / (2 pi^2)`` because that
        is CAMB's internal convention (``Pk_nowiggle`` is built as
        ``k**(n_s+3) * T^2``).  Mixing conventions puts a k^3 between the ratio's
        numerator and denominator; measured, ``P/P_nw`` then runs from 3.41 at
        k = 0.02 down to 2.2e-4 at k = 0.5 instead of hovering near 1, and the
        smoother swallows that slope as signal - which is exactly how the earlier
        attempt lost 18 percentage points.
        """
        lnk = np.log(np.geomspace(KMIN_WIGGLE, KMAX_WIGGLE, NK_WIGGLE))
        dx = float(lnk[1] - lnk[0])
        kk = np.exp(lnk)
        p_lin = self._pk(np.clip(kk, self.k_lin[0], self.k_lin[-1]), cold=False)
        d2_lin = p_lin * kk ** 3 / (2.0 * PI ** 2)
        d2_nw = _eisenstein_hu_nowiggle(
            kk, omega_m=self.omega_m, omega_b=self.omega_b,
            h=self.h, n_s=self.n_s, t_cmb=self.t_cmb)
        # Pin the no-wiggle spectrum to the linear one at the normalisation
        # wavenumber, as calculate_nowiggle does with knorm = 0.03.
        j = int(np.argmin(np.abs(kk - KNORM_NOWIGGLE)))
        if d2_nw[j] > 0.0:
            d2_nw = d2_nw * (d2_lin[j] / d2_nw[j])
        ratio = d2_lin / np.maximum(d2_nw, 1e-300)

        # CAMB leaves points within nsig = 3 sigma of either end unsmoothed, and
        # normalises by the kernel weight actually used.
        offsets = np.arange(1 - ratio.size, ratio.size)
        kernel = np.exp(-0.5 * (offsets * dx / WIGGLE_SIGMA) ** 2)
        # Convolve both the data and the support, so interior points retain the
        # same finite-grid normalization as the reference.
        center = slice(ratio.size - 1, 2 * ratio.size - 1)
        smoothed = (np.convolve(ratio, kernel, "full")[center]
                    / np.convolve(np.ones_like(ratio), kernel, "full")[center])
        edge = 3.0 * WIGGLE_SIGMA
        at_edge = (lnk - lnk[0] < edge) | (lnk[-1] - lnk < edge)
        smoothed[at_edge] = ratio[at_edge]

        self._lnk_wiggle = lnk
        # Back to P units: _dewiggle applies the k^3/(2 pi^2) conversion itself.
        self._p_wiggle = (d2_lin - smoothed * d2_nw) * 2.0 * PI ** 2 / kk ** 3

        # Velocity dispersion sigma_v^2 = (1/6pi^2) int dk P(k), the R -> 0 limit
        # of CAMB's sigmaV; sets how fast the wiggle is damped away.
        variance = np.trapezoid(self.p_lin, self.k_lin) / (6.0 * PI ** 2)
        self._sigv = self._d * float(np.sqrt(max(variance, 0.0)))

    def _dewiggle(self, k, d2_lin):
        """Linear Delta^2 with the BAO wiggle damped by exp(-(k sigma_v)^2)."""
        pw = np.interp(np.log(k), self._lnk_wiggle, self._p_wiggle, left=0.0, right=0.0)
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

def halo_growth(omega_m):
    """Scale-independent growth in the flat matter–Lambda halo background."""
    a = np.linspace(0.0, 1.0, 512)
    growth_over_a = hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -(1.0 - omega_m) / omega_m * a ** 3)
    normalization = float(growth_over_a[-1])
    growth = CubicSpline(a, a * growth_over_a / normalization)
    integrated = CubicSpline(a, cumulative_simpson(growth_over_a, x=a, initial=0.0))
    return growth, integrated, normalization


def cold_transfer_ratio(k, *, omega_m, f_nu, n_massive, h, t_cmb, growth):
    """Eisenstein–Hu cold-to-total transfer ratio used by CAMB's HMcode."""
    if f_nu == 0.0 or n_massive == 0:
        return np.ones_like(k)
    exponent = 6.0 * f_nu / (5.0 + np.sqrt(25.0 - 24.0 * f_nu))
    theta = t_cmb / 2.7
    z_eq = 2.5e4 * omega_m * h ** 2 / theta ** 4
    q = k * theta ** 2 / (omega_m * h)
    y_fs = 17.2 * f_nu * (1.0 + 0.488 * f_nu ** (-7.0 / 6.0)) * (n_massive * q / f_nu) ** 2
    free_growth = ((1.0 + z_eq) * growth / (1.0 + y_fs)) ** 0.7
    return ((1.0 + free_growth) / ((1.0 - f_nu) ** (0.7 / exponent) + free_growth)) ** (exponent / 0.7)


def _eisenstein_hu_nowiggle(k, omega_m, omega_b, h, n_s, t_cmb=T_CMB0_DEFAULT):
    """Eisenstein & Hu (1998) no-wiggle spectrum, **in Delta^2 units**.

    Port of halofit.f90's ``Tk_nw_init`` / ``Tk_nw`` / ``Pk_nowiggle``
    (astro-ph/9709112, equations 26 and 28-31).  The result is
    ``k**(n_s+3) * T^2``, which is ``Delta^2 = k^3 P / (2 pi^2)`` and not ``P`` -
    CAMB's HMcode works in Delta^2 throughout, and mixing the two conventions puts
    a factor k^3 between the ratio's numerator and denominator.  Measured, that
    makes ``P/P_nw`` run from 3.41 at k = 0.02 to 2.2e-4 at k = 0.5 instead of
    hovering near 1, and the dewiggle then mistakes the whole slope for signal.
    """
    k = np.asarray(k, dtype=np.float64)
    wm = omega_m * h ** 2
    wb = omega_b * h ** 2
    rb = omega_b / omega_m
    s = 44.5 * np.log(9.83 / wm) / np.sqrt(1.0 + 10.0 * wb ** 0.75)
    alpha = (1.0 - 0.328 * np.log(431.0 * wm) * rb
             + 0.38 * np.log(22.3 * wm) * rb ** 2)
    gamma = omega_m * h * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s * h) ** 4))
    q = k * (t_cmb / 2.7) ** 2 / gamma
    big_l = np.log(2.0 * np.e + 1.8 * q)
    big_c = 14.2 + 731.0 / (1.0 + 62.5 * q)
    tk = big_l / (big_l + big_c * q ** 2)
    return k ** (n_s + 3.0) * tk ** 2


def _omega_m_z(z, omega_m):
    """``Omega_m(z)`` for a flat universe with a cosmological constant."""
    e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    return omega_m * (1.0 + z) ** 3 / e2


def _f_mead(x, y, p0, p1, p2, p3):
    return p0 + p1 * (1.0 - x) + p2 * (1.0 - x) ** 2 + p3 * (1.0 - y)
