from dataclasses import replace

import numpy as np
import pytest

from tenbol.errors import NumericalFailure
from tenbol.physics.background import CosmologyBatch
from tenbol.physics.perturbations import PerturbationSettings
from tenbol.power import matter_power


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA is unavailable: {exc}")
    if count == 0:
        pytest.skip("No CUDA device")


def test_cuda_matches_numpy_and_preserves_batch_order(cuda):
    batch = CosmologyBatch.from_columns(
        h=[0.67, 0.72], omega_b=[0.0224, 0.022], omega_c=[0.12, 0.13],
        A_s=2.1e-9, n_s=0.965, m_nu=[0.06, 0.30],
    )
    settings = PerturbationSettings(dlna_max=0.01, dlna_superhorizon=0.004)
    k = [0.2, 1e-4, 0.08]
    reference = matter_power(batch, k, [0.0, 1.0, 30.0], settings, bucket_by_k=False)
    candidate = matter_power(batch, k, [1.0, 30.0, 0.0, 1.0], settings, backend="cuda")
    for component in ("delta_m", "delta_cdm", "delta_b"):
        expected = getattr(reference, component)[:, :, [1, 2, 0, 1]]
        np.testing.assert_allclose(getattr(candidate, component), expected, rtol=2.5e-3)
    with pytest.raises(NumericalFailure, match="max_steps"):
        matter_power(batch, k, 0.0, replace(settings, max_steps=1), backend="cuda")
