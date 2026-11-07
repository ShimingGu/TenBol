"""Linear growth factor and sigma(R) for the Tenbol halo-model stage.

Both are needed by the HMcode-2020 port: ``sigma(R)`` supplies the mass
function and the non-linear scale, and the growth factor carries the linear
spectrum from its tabulation redshift to any other.

Everything here is written on the same conventions as the rest of Tenbol:

* ``k`` in ``h/Mpc``, ``P(k)`` in ``(Mpc/h)^3`` (CAMB's transfer units),
* ``R`` in ``Mpc/h``,
* ``a`` is the scale factor and derivatives are with respect to ``ln a``.

The growth factor is obtained by integrating the standard linear growth
equation rather than by a fitting formula, so it stays correct for massive
neutrinos and for any dark-energy history the background模块 supports.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["linear_growth", "growth_terms", "sigma_r", "sigma8", "top_hat_window"]


def _omega_m(bg, a):
    """``8 pi G rho_m / (3 H^2)`` from the background's own density split."""
    parts = bg.grho_parts(a)
    total = bg.grho(a)
    matter = parts["c"] + parts["b"] + parts["nu_mass"]
    return float(np.ravel(matter / total)[0])


def linear_growth(bg, a_out, a_start: float = 3.0e-2):
    """Scale-independent linear growth factor ``D(a)``, normalised to ``D(1) = 1``.

    Solves

    .. math::

        D'' + (2 + d\\ln\\mathcal{H}/d\\ln a)\\,D' = \\tfrac{3}{2}\\Omega_m(a)\\,D

    with primes denoting ``d/d ln a``.  The initial condition is the
    matter-era solution ``D ∝ a``.  ``a_start`` must sit where radiation is
    negligible: at a = 1e-3 (z ~ 1000) radiation is still 31% of the density,
    D ∝ a is not the growing solution there, and imposing it biases the
    normalisation gnorm = g(a=1) low by 7% (0.726 against 0.783).  HMcode's
    Mead(2017) fits read that normalisation directly as their ``x`` term, so
    the error propagates into delta_c and Delta_v.

    Parameters
    ----------
    bg : Background
        Background cosmology (one or many, leading axis is the batch).
    a_out : array_like
        Scale factors at which to return the growth factor.

    Returns
    -------
    ndarray
        Growth factor with shape ``(batch, len(a_out))``.
    """
    a_out = np.atleast_1d(np.asarray(a_out, dtype=np.float64))

    def rhs(ln_a, y):
        a = np.exp(ln_a)
        d = y[0]
        dd = y[1]
        h = float(np.ravel(bg.hubble(a))[0])
        # ``hubble_prime`` is d(H_hat)/d(tau), so d ln H_hat / d ln a = hp/H_hat**2.
        # The growth equation wants d ln H / d ln a for the *physical* H = H_hat/a,
        # which is that minus one: for EdS this gives -1.5, the value that makes
        # D ~ a an exact solution.
        hp = float(np.ravel(bg.hubble_prime(a))[0])
        dlnh = hp / h ** 2 - 1.0
        om = _omega_m(bg, a)
        return np.array([dd, -(2.0 + dlnh) * dd + 1.5 * om * d])

    if np.any(a_out > 1.0):
        raise ValueError("a_out must not exceed 1: the growth factor is normalised at a=1")
    y0 = np.array([a_start, a_start])       # D = a, dD/dln a = a in the matter era
    # Always integrate up to a = 1 so the normalisation is taken there, whatever
    # the caller asked for; then interpolate inside the solved range.
    sol = solve_ivp(rhs, (np.log(a_start), 0.0), y0, method="RK45",
                    rtol=1e-9, atol=1e-12, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"growth ODE failed: {sol.message}")
    d1 = float(sol.sol(0.0)[0])
    a_clipped = np.maximum(a_out, a_start)
    d = sol.sol(np.log(a_clipped))[0]
    # Below a_start the solution cannot be integrated, but that region is deep
    # in matter domination (radiation is 1% at a_start = 0.03) where D ~ a is
    # accurate.  Without this, callers that tabulate from a = 1e-3 -- CAMB's
    # amin_growth_interpolation -- would get a constant extrapolation and a
    # non-monotonic table.
    d_start = float(sol.sol(np.log(a_start))[0])
    d = np.where(a_out < a_start, d_start * (a_out / a_start), d)
    # The batch axis: Background currently broadcasts a single cosmology, so the
    # solution is one trajectory; keep the (batch, len(a_out)) contract explicit.
    return np.atleast_2d(d / d1)


def top_hat_window(x):
    """Fourier transform of a unit top-hat, ``3 (sin x - x cos x) / x^3``."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-3
    out = np.empty_like(x)
    safe = np.where(small, 1.0, x)
    out[~small] = 3.0 * (np.sin(safe[~small]) - safe[~small] * np.cos(safe[~small])) / safe[~small] ** 3
    xs = x[small]
    out[small] = 1.0 - xs ** 2 / 10.0
    return out


def sigma_r(k, pk, r):
    """RMS density fluctuation in a sphere of comoving radius ``R``.

    ``sigma^2(R) = (1/2pi^2) \\int dk k^2 P(k) |W(kR)|^2`` with ``k`` in
    ``h/Mpc``, ``P`` in ``(Mpc/h)^3`` and ``R`` in ``Mpc/h``.

    Parameters
    ----------
    k, pk : ndarray
        Linear spectrum, ``(n_k,)`` each, ``k`` increasing and strictly positive.
    r : float or ndarray
        Radius or radii in ``Mpc/h``.
    """
    k = np.asarray(k, dtype=np.float64)
    pk = np.asarray(pk, dtype=np.float64)
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    lnk = np.log(k)
    integrand = k ** 3 * pk / (2.0 * np.pi ** 2)      # d ln k integrand, without W^2
    out = np.empty(r.shape)
    for i, ri in enumerate(r):
        w = top_hat_window(k * ri)
        out[i] = np.trapezoid(integrand * w * w, lnk)
    return np.sqrt(np.maximum(out, 0.0))


def sigma8(k, pk):
    """``sigma(R = 8 Mpc/h)``."""
    return float(sigma_r(k, pk, 8.0)[0])


def growth_terms(bg, z, a_start: float = 3.0e-2, n_tab: int = 512):
    """Mead (2017) growth terms ``x`` and ``y`` at redshift ``z``.

    CAMB's HMcode fits (``dc_Mead``, ``Dv_Mead``) are written in terms of

        x = g(z) / a,   y = G(z) / a,
        g(a) = growth normalised so that g = a deep in the matter era,
        G(a) = integral_0^a g(a') / a' da',

    both of which equal 1 for an Einstein-de Sitter universe.  ``g`` is built
    from the same normalised growth factor ``linear_growth`` returns, so the two
    services cannot drift apart: ``g(a) = gnorm * D(a)`` with ``gnorm`` fixed by
    the matter-era condition ``g(a_start) = a_start``.
    """
    a_tab = np.linspace(a_start, 1.0, n_tab)
    d_tab = linear_growth(bg, a_tab)[0]
    gnorm = a_start / d_tab[0]                  # g(a_start) = a_start
    g = gnorm * d_tab

    # G(a) = int_0^a g/a' da'; below a_tab[0] the ratio g/a' -> 1 and contributes g[0].
    over_a = g / a_tab
    g_acc = np.empty(n_tab)
    g_acc[0] = g[0]
    g_acc[1:] = g[0] + np.cumsum(
        0.5 * (over_a[1:] + over_a[:-1]) * np.diff(a_tab))

    a = 1.0 / (1.0 + float(z))
    g_z = float(np.interp(a, a_tab, g))
    G_z = float(np.interp(a, a_tab, g_acc))
    return g_z / a, G_z / a, gnorm
