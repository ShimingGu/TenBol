"""Growth factor and sigma(R) against the CAMB reference."""

import numpy as np
import pytest

from tenbol.adapters.camb_reference import Cosmology, ReferenceSettings, reference_run
from tenbol.physics.background import Background, CosmologyBatch
from tenbol.physics.sigma import growth_terms, linear_growth, sigma8

COSMO = dict(h=0.67, omega_b=0.0224, omega_c=0.120, A_s=2.1e-9, n_s=0.965)


@pytest.fixture(scope="module")
def background():
    batch = CosmologyBatch.from_columns(
        h=[COSMO["h"]], omega_b=[COSMO["omega_b"]], omega_c=[COSMO["omega_c"]],
        A_s=[COSMO["A_s"]], n_s=[COSMO["n_s"]], m_nu=[[0.06]], N_eff=[3.046])
    return Background(batch)


@pytest.fixture(scope="module")
def reference():
    cosmo = Cosmology(m_nu_sum=0.06, **COSMO)
    return reference_run(cosmo, ReferenceSettings(), redshifts=[0.0], npoints=2048)


def test_growth_is_normalised_and_monotonic(background):
    a = np.array([1.0, 0.8, 0.5, 0.25, 0.1])
    d = linear_growth(background, a)[0]
    assert d[0] == pytest.approx(1.0, abs=1e-12)
    # a decreases along the array, so a monotonic D(a) must decrease too
    assert np.all(np.diff(d) < 0.0)


def test_batched_growth_terms_match_individual_cosmologies(background):
    second = dict(COSMO, h=0.72, omega_c=0.14, m_nu=0.30)
    other = Background(CosmologyBatch.from_columns(**second))
    batch = Background(CosmologyBatch.from_columns(
        **dict(COSMO, h=[COSMO["h"], second["h"]],
               omega_c=[COSMO["omega_c"], second["omega_c"]], m_nu=[0.06, 0.30])
    ))
    z = [0.0, 1.0, 2.0]
    x, y, norm = growth_terms(batch, z)
    assert x.shape == y.shape == (2, 3)
    assert not np.allclose(x[0], x[1])
    for i, single in enumerate((background, other)):
        sx, sy, snorm = growth_terms(single, z)
        np.testing.assert_allclose(x[i], sx[0], rtol=1e-7)
        np.testing.assert_allclose(y[i], sy[0], rtol=1e-7)
        np.testing.assert_allclose(norm[i], snorm[0], rtol=1e-7)


def test_sigma8_matches_camb(background, reference):
    kh = np.asarray(reference["kh"])
    pk = np.asarray(reference["pk_lin"][0])
    ours = sigma8(kh, pk)
    camb = float(np.asarray(reference["sigma8"])[0])
    assert ours == pytest.approx(camb, rel=1e-4)
