import numpy as np

from tenbol.physics.hmcode import HMcode2020, halo_growth


def test_halo_variance_extends_the_cold_spectrum():
    k = np.geomspace(1e-4, 20.0, 128)
    extended = np.concatenate((k, np.geomspace(20.0, 1e3, 42)[1:]))
    growth, accumulated, normalization = halo_growth(0.31)
    a = 1.0 / 3.0
    terms = lambda _: (growth(a) * normalization / a, accumulated(a) / a)

    def model(grid):
        p_total = 1e3 * grid ** -2.5
        return HMcode2020(
            omega_m=0.31, omega_b=0.05, omega_c=0.25, omega_nu=0.01,
            h=0.67, n_s=0.965, k_lin=grid, p_lin=p_total,
            p_cold=1.21 * p_total, growth=growth, z=2.0, terms=terms,
        )

    truncated = model(k)
    reference = model(extended)
    radii = [0.02, 0.1, 8.0]
    np.testing.assert_allclose(truncated._sigma_r(radii), reference._sigma_r(radii), rtol=1e-12)
    np.testing.assert_allclose(
        truncated._sigma_r(radii) / truncated._sigma_r(radii, cold=False), 1.1, rtol=1e-12
    )
