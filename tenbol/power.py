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

    def growth_callable(self, k_index: int = 0):
        """``growth(a) -> D(a)`` normalised to ``D(1) = 1``, for HMcode.

        HMcode takes a callable, and the single-fluid ODE in ``sigma.py`` reads
        1.86% high at z = 0 against CAMB's ``delta_tot`` while this solver reads
        0.28%.  So hand it the solver: at a small ``k`` the matter delta *is*
        the scale-independent growth factor, which is the quantity HMcode wants.

        The first cosmology in the batch is used, matching HMcode's one-cosmology
        construction.
        """
        zs = np.asarray(self.z, dtype=np.float64)
        a_all = 1.0 / (1.0 + zs)
        # np.interp requires an ascending x axis, and a = 1/(1+z) runs the other
        # way from z: sorting by z would hand it a descending array and silently
        # interpolate against the wrong end (measured: solver/CAMB read 0.0128 at
        # z = 0 and 1938 at z = 500 before this was fixed).
        order = np.argsort(a_all)
        a = a_all[order]
        d = np.asarray(self.delta_m[0, k_index, :], dtype=np.float64)[order]
        if a[-1] < 1.0:
            # The grid stops short of today: hold the last value rather than
            # inventing a slope, and record that in the docstring above.
            a = np.concatenate([a, [1.0]])
            d = np.concatenate([d, d[-1:]])
        d = d / d[-1]

        def growth(a_query) -> np.ndarray:
            aq = np.atleast_1d(np.asarray(a_query, dtype=np.float64))
            out = np.interp(aq, a, d)
            # Below the tabulated range, continue the matter-era growing mode
            # (D proportional to a) instead of clamping. HMcode inverts a cubic
            # spline of D(a) over a = 1e-3..1, so a flat segment there violates
            # the strictly-increasing requirement and raises inside CubicSpline.
            lo = aq < a[0]
            if np.any(lo):
                out = np.where(lo, d[0] * aq / a[0], out)
            return out

        return growth


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
                 use_cache: bool = True,
                 bucket_by_k: bool = True,
                 chunk_size: int | None = None) -> PowerResult:
    """Linear matter perturbations for every (cosmology, k, z) in one batch.

    ``bucket_by_k`` splits the integration into one batch per wavenumber.  The
    solver advances every trajectory in a batch with a *shared* step, chosen as
    the minimum over all of them, so a single stiff trajectory slows the whole
    batch down: at 128 cosmologies x 5 k the flat batch did not finish in ten
    minutes, while n=16 and n=32 took 94 s and 138 s.  Per-k buckets restore
    roughly linear scaling at the cost of a few extra integrations.  This is a
    crude version of the step-size bucketing described in
    docs/gpu_tensor_prior_art.md.

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
    if chunk_size and n_c > chunk_size:
        # Split the cosmology axis.  The solver takes one shared step for the
        # whole batch, so a large batch inherits the worst trajectory's step;
        # chunking keeps each batch small enough to stay tractable (and mirrors
        # the memory-driven batching a device backend needs anyway).
        parts = []
        for start in range(0, n_c, chunk_size):
            stop = min(start + chunk_size, n_c)
            sub = CosmologyBatch.from_columns(
                h=cosmologies.h[start:stop], omega_b=cosmologies.omega_b[start:stop],
                omega_c=cosmologies.omega_c[start:stop], A_s=cosmologies.A_s[start:stop],
                n_s=cosmologies.n_s[start:stop],
                m_nu=np.atleast_2d(cosmologies.m_nu)[start:stop],
                N_eff=cosmologies.N_eff[start:stop])
            parts.append(matter_power(sub, k_h, redshifts, settings, reionization,
                                      use_cache, bucket_by_k, chunk_size=None))
        return PowerResult(
            k_h=k_h, z=parts[0].z,
            delta_m=np.concatenate([p.delta_m for p in parts], axis=0),
            delta_cdm=np.concatenate([p.delta_cdm for p in parts], axis=0),
            delta_b=np.concatenate([p.delta_b for p in parts], axis=0),
            n_steps=sum(p.n_steps for p in parts),
        )
    if bucket_by_k and n_k > 1:
        parts = [matter_power(cosmologies, [kk], redshifts, settings, reionization,
                              use_cache, bucket_by_k=False) for kk in k_h]
        return PowerResult(
            k_h=k_h, z=parts[0].z,
            delta_m=np.concatenate([p.delta_m for p in parts], axis=1),
            delta_cdm=np.concatenate([p.delta_cdm for p in parts], axis=1),
            delta_b=np.concatenate([p.delta_b for p in parts], axis=1),
            n_steps=sum(p.n_steps for p in parts),
        )
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
