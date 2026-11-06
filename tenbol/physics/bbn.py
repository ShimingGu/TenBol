"""Primordial helium abundance.

CAMB 2.0.5 predicts Y_He from the PRIMAT table through a bivariate spline.
Y_He enters Tenbol only through the number densities of baryons/electrons
(``n_H = rho_b (1-Y_He)/m_H``) and through the recombination rates, so a
documented fit is sufficient provided it reproduces the reference to well below
the sensitivity of the final power spectrum.

The cubic below is a least-squares fit to ``camb.bbn.get_predictor().Y_He``
over ``ombh2 in [0.018, 0.027]`` at the workload's fixed
``delta_neff = N_eff - 3.044 = 0.002``.  Measured residuals:
``max|dY| = 1.6e-6``, ``max relative = 6.4e-6``; the residual T_CMB dependence
over [2.7250, 2.7258] K is < 1e-5 (it enters only through
``ombh2 (2.7255/T_CMB)^3``).
"""

from __future__ import annotations

import numpy as np

from ..constants import N_EFF_STANDARD, T_CMB0_DEFAULT

# np.polyfit(x = ombh2 - 0.0224, Y_He, 3), highest power first
_YHE_COEF = (394.5199373912534, -11.448662802239443, 0.4288293685873558, 0.24569714540118287)
_YHE_PIVOT = 0.0224


def helium_mass_fraction(ombh2, T_cmb=T_CMB0_DEFAULT, n_eff: float = 3.046):
    """BBN helium mass fraction Y_He, matching CAMB's default predictor.

    ``ombh2`` may be an array (batch dimension); the return shape follows it.
    """
    om_scaled = np.asarray(ombh2) * (T_CMB0_DEFAULT / np.asarray(T_cmb)) ** 3
    x = om_scaled - _YHE_PIVOT
    # delta_neff only enters through the (tiny) table dependence; the fit is
    # for delta_neff = n_eff - 3.044.  We correct linearly using the measured
    # table slope (dY/dDeltaN ~ 0.0132 at the pivot, from the PRIMAT table).
    delta = np.asarray(n_eff) - N_EFF_STANDARD
    base = np.polyval(_YHE_COEF, x)
    return base + 0.0132 * (delta - 0.002)
