"""Batched matter-power API.

This is the surface the MCMC sampler and the GPU backends both drive: hand it
a batch of cosmologies plus a k grid and it returns one dense tensor of linear
matter perturbations, ``(n_cosmo, n_k, n_z)``.

The batch is flattened into Tenbol's trajectory axis exactly as the solver
expects -- ``T = n_cosmo * n_k`` -- so every cosmology/wavenumber pair is one
column of ``Y[D, T]`` and the whole thing is a single batched integration.
Two cosmologies with the same parameters share a ``Background`` and
``ThermalHistory`` through the cache instead of being recomputed.

Units: ``k`` in h/Mpc, outputs are the gauge-invariant density perturbation
``delta_m(k, z)`` normalised to the primordial curvature perturbation, i.e.
directly comparable with CAMB's ``delta_tot`` transfer columns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from tenbol.physics.background import Background, CosmologyBatch
from tenbol.physics.perturbations import (PerturbationBatch, PerturbationSettings,
                                          evolve)
from tenbol.physics.thermal import Reionization, ThermalHistory

__all__ = ["PowerResult", "matter_power", "clear_cache", "cache_info"]


@dataclass
class PowerResult:
    """One batched evaluation of the linear matter spectrum."""

    k_h: np.ndarray                 # (n_k,)   [h/Mpc]
    z: np.ndarray                   # (n_z,)
    delta_m: np.ndarray             # (n_cosmo, n_k, n_z)
    delta_cdm: np.ndarray           # (n_cosmo, n_k, n_z)
    delta_b: np.ndarray             # (n_cosmo, n_k, n_z)
    n_steps: int = 0

    def __post_init__(self):
        n_c, n_k = self.delta_m.shape[0], self.delta_m.shape[1]
        assert self.delta_m.shape == (n_c, n_k, self.z.size)

    def growth(self, z_ref: float = 1000.0) -> np.ndarray:
        """``delta(z) / delta(z_ref)``, the k-resolved growth, ``(n_c, n_k, n_z)``."""
        i = int(np.argmin(np.abs(self.z - z_ref)))
        return self.delta_m / self.delta_m[:, :, [i]]


# --- caching -----------------------------------------------------------
# Keyed on the cosmological parameters only.  Background/ThermalHistory are the
# expensive part per cosmology and are identical across wavenumbers, so a batch
# of N cosmologies pays for them N times rather than N * n_k.
_CACHE: dict[str, tuple] = {}
_MAX_CACHE = 256


def _cosmo_key(cosmologies: CosmologyBatch) -> str:
    h = cosmologies.h.tobytes() + cosmologies.omega_b.tobytes() + cosmologies.omega_c.tobytes()
    h += np.asarray(cosmologies.m_nu, dtype=np.float64).tobytes()
    h += np.asarray(cosmologies.N_eff, dtype=np.float64).tobytes()
    return hashlib.blake2b(h, digest_size=16).hexdigest()


def clear_cache() -> None:
    _CACHE.clear()


def cache_info() -> dict:
    return {"entries": len(_CACHE), "max_entries": _MAX_CACHE}


def _background_pair(cosmologies: CosmologyBatch, reionization: Reionization):
    key = _cosmo_key(cosmologies)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit, True
    bg = Background(cosmologies)
    th = ThermalHistory(bg, reionization)
    if len(_CACHE) >= _MAX_CACHE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (bg, th)
    return (bg, th), False


def matter_power(cosmologies: CosmologyBatch, k_h, redshifts,
                 settings: PerturbationSettings | None = None,
                 reionization: Reionization | None = None,
                 use_cache: bool = True) -> PowerResult:
    """Linear matter perturbations for every (cosmology, k, z) in one batch.

    Parameters
    ----------
    cosmologies : CosmologyBatch
        ``n_cosmo`` cosmologies.
    k_h : array_like
        Wavenumbers in ``h/Mpc``, shared by all cosmologies.  They are converted
        to 1/Mpc per cosmology, since ``k = k_h * h`` differs across the batch.
    redshifts : array_like
        Output redshifts, ascending or descending (order is preserved in the
        returned tensor's last axis).
    """
    k_h = np.atleast_1d(np.asarray(k_h, dtype=np.float64))
    z = np.atleast_1d(np.asarray(redshifts, dtype=np.float64))
    s = settings or PerturbationSettings()
    reion = reionization or Reionization(tau=0.0544)

    n_c, n_k = len(cosmologies.h), k_h.size
    sample = np.repeat(np.arange(n_c, dtype=np.int64), n_k)      # cosmology per trajectory
    k_mpc = (k_h[None, :] * cosmologies.h[:, None]).ravel()      # h/Mpc -> 1/Mpc

    if use_cache:
        (bg, th), _ = _background_pair(cosmologies, reion)
    else:
        bg = Background(cosmologies)
        th = ThermalHistory(bg, reion)

    batch = PerturbationBatch(bg, th, k_mpc, sample, s)
    rec = evolve(batch, z_out=z, a_end=1.0)

    shape = (n_c, n_k, z.size)
    return PowerResult(
        k_h=k_h, z=z,
        delta_m=np.asarray(rec["dm"]).reshape(shape),
        delta_cdm=np.asarray(rec["dc"]).reshape(shape),
        delta_b=np.asarray(rec["db"]).reshape(shape),
        n_steps=int(rec["n_steps"]),
    )
