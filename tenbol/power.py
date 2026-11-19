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

    def growth_callable(self, k_index: int = 0, cosmo_index: int = 0):
        """``growth(a) -> D(a)`` normalised to ``D(1) = 1``, for HMcode.

        HMcode takes a callable, and the single-fluid ODE in ``sigma.py`` reads
        1.86% high at z = 0 against CAMB's ``delta_tot`` while this solver reads
        0.28%.  So hand it the solver: at a small ``k`` the matter delta *is*
        the scale-independent growth factor, which is the quantity HMcode wants.

        ``cosmo_index`` selects which trajectory in the batch to use, since
        HMcode is built one cosmology at a time.
        """
        zs = np.asarray(self.z, dtype=np.float64)
        a_all = 1.0 / (1.0 + zs)
        # np.interp requires an ascending x axis, and a = 1/(1+z) runs the other
        # way from z: sorting by z would hand it a descending array and silently
        # interpolate against the wrong end (measured: solver/CAMB read 0.0128 at
        # z = 0 and 1938 at z = 500 before this was fixed).
        order = np.argsort(a_all)
        a = a_all[order]
        d = np.asarray(self.delta_m[cosmo_index, k_index, :], dtype=np.float64)[order]
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


def matter_power_nl(cosmologies: CosmologyBatch, k_h, redshifts,
                    settings: PerturbationSettings | None = None,
                    reionization: Reionization | None = None,
                    use_cache: bool = True,
                    k_grid_size: int = 128,
                    k_range: tuple[float, float] = (1.0e-4, 20.0),
                    *, backend: str = "numpy", device: int | None = None) -> np.ndarray:
    """Nonlinear matter power from HMcode-2020, ``(n_c, n_k, n_z)`` in (Mpc/h)^3.

    The Boltzmann solver evaluates total linear power at each requested
    redshift on an internal grid of ``k_grid_size`` points over ``k_range``.
    HMcode derives cold power for halo variance and uses the total spectrum
    for the two-halo term. It extends the variance quadrature on the host
    and interpolates the combined nonlinear power back to ``k_h``.
    ``backend`` and ``device`` select the linear solver used by HMcode.
    """
    from scipy.interpolate import CubicSpline

    from tenbol.physics.hmcode import HMcode2020, cold_transfer_ratio, halo_growth

    k_out = np.atleast_1d(np.asarray(k_h, dtype=np.float64))
    z_out = np.atleast_1d(np.asarray(redshifts, dtype=np.float64))
    n_c = len(cosmologies.h)
    if not z_out.size:
        return np.empty((n_c, k_out.size, 0))
    reion = reionization or Reionization()
    k_in = np.geomspace(float(k_range[0]), float(k_range[1]), int(k_grid_size))
    log_k_out = np.log(k_out)
    z_grid, restore_order = np.unique(z_out, return_inverse=True)
    lin = matter_power(cosmologies, k_in, z_grid, settings=settings,
                       reionization=reion, use_cache=use_cache,
                       backend=backend, device=device)
    if use_cache:
        (bg, _th), _ = _background_pair(cosmologies, reion)
    else:
        bg = Background(cosmologies)

    parts = bg.grho_parts(1.0)
    tot = np.sum(np.stack(list(parts.values())), axis=0)
    om_b = parts["b"] / tot
    om_c = parts["c"] / tot
    massive = bg.m_nu > 0.0
    om_nu = np.sum(np.where(massive, bg.grho_nu_mass0, 0.0), axis=1) / tot
    om_m = om_b + om_c + om_nu

    a = 1.0 / (1.0 + z_grid)
    out = np.empty((n_c, k_out.size, z_grid.size), dtype=np.float64)
    for ci in range(n_c):
        h = cosmologies.h[ci]
        k_mpc = k_in * h
        n_s = cosmologies.n_s[ci]
        t_cmb = cosmologies.T_cmb[ci]
        primordial = ((2.0 * np.pi ** 2 / k_mpc ** 3) * cosmologies.A_s[ci]
                      * (k_mpc / cosmologies.pivot_k) ** (n_s - 1.0) * h ** 3)
        growth, integrated, normalization = halo_growth(om_m[ci])
        d = growth(a)
        unnormalized = d * normalization
        x = unnormalized / a
        y = integrated(a) / a
        n_massive = np.count_nonzero(massive[ci])
        for jz, z in enumerate(z_grid):
            # Each input spectrum is stored divided by D(z)^2. Restoring D(z)
            # inside HMcode then preserves the scale dependence at this z.
            p_tot = primordial * (lin.delta_m[ci, :, jz] / d[jz]) ** 2
            ratio = cold_transfer_ratio(
                k_in, omega_m=om_m[ci], f_nu=om_nu[ci] / om_m[ci],
                n_massive=n_massive, h=h, t_cmb=t_cmb, growth=unnormalized[jz])
            ok = np.isfinite(p_tot) & (p_tot > 0.0)
            terms = x[jz], y[jz]
            model = HMcode2020(
                omega_m=om_m[ci], omega_b=om_b[ci], omega_c=om_c[ci], omega_nu=om_nu[ci],
                h=h, n_s=n_s, t_cmb=t_cmb, k_lin=k_in[ok], p_lin=p_tot[ok],
                p_cold=p_tot[ok] * ratio[ok] ** 2, growth=growth,
                terms=lambda _, xy=terms: xy, z=z)
            log_k_in = np.log(model.k_lin)
            log_p = CubicSpline(log_k_in, np.log(model.power(model.k_lin)))
            out[ci, :, jz] = np.exp(log_p(np.clip(log_k_out, log_k_in[0], log_k_in[-1])))
    return out[:, :, restore_order]


# --- caching -----------------------------------------------------------
# Keyed on the cosmological parameters only.  Background/ThermalHistory are the
# expensive part per cosmology and are identical across wavenumbers, so a batch
# of N cosmologies pays for them N times rather than N * n_k.
_CACHE: dict[tuple[str, Reionization], tuple] = {}
_MAX_CACHE = 256


def _cosmo_key(cosmologies: CosmologyBatch) -> str:
    h = cosmologies.h.tobytes() + cosmologies.omega_b.tobytes() + cosmologies.omega_c.tobytes()
    h += np.asarray(cosmologies.m_nu, dtype=np.float64).tobytes()
    h += np.asarray(cosmologies.N_eff, dtype=np.float64).tobytes()
    h += np.asarray(cosmologies.T_cmb, dtype=np.float64).tobytes()
    return hashlib.blake2b(h, digest_size=16).hexdigest()


def clear_cache() -> None:
    _CACHE.clear()


def cache_info() -> dict:
    return {"entries": len(_CACHE), "max_entries": _MAX_CACHE}


def _background_pair(cosmologies: CosmologyBatch, reionization: Reionization):
    key = _cosmo_key(cosmologies), reionization
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
                 bucket_by_k: bool | None = None,
                 chunk_size: int | None = None, *,
                 backend: str = "numpy", device: int | None = None) -> PowerResult:
    """Linear matter perturbations for every (cosmology, k, z) in one batch.

    ``bucket_by_k`` splits the integration into one batch per wavenumber.
    By default NumPy uses these buckets and CUDA launches all wavenumbers
    together, with each trajectory taking its own steps on the device.

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
    backend : {"numpy", "cuda"}
        Integrator used for the perturbations.
    device : int or None
        CUDA device ordinal. The default uses the current CUDA device.
    """
    k_h = np.atleast_1d(np.asarray(k_h, dtype=np.float64))
    z = np.atleast_1d(np.asarray(redshifts, dtype=np.float64))
    if backend not in ("numpy", "cuda"):
        raise ValueError(f"Unknown matter power backend: {backend!r}")
    # A CUDA convergence run found the wider acoustic step changed linear
    # power by less than 1e-5 across its k grid. Explicit settings take priority.
    acoustic_step = 0.24 if backend == "cuda" else PerturbationSettings.dlna_acoustic
    s = settings or PerturbationSettings(dlna_acoustic=acoustic_step)
    reion = reionization or Reionization()
    if bucket_by_k is None:
        bucket_by_k = backend == "numpy"

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
                N_eff=cosmologies.N_eff[start:stop], T_cmb=cosmologies.T_cmb[start:stop],
                pivot_k=cosmologies.pivot_k)
            parts.append(matter_power(sub, k_h, redshifts, s, reionization,
                                      use_cache, bucket_by_k, chunk_size=None,
                                      backend=backend, device=device))
        return PowerResult(
            k_h=k_h, z=parts[0].z,
            delta_m=np.concatenate([p.delta_m for p in parts], axis=0),
            delta_cdm=np.concatenate([p.delta_cdm for p in parts], axis=0),
            delta_b=np.concatenate([p.delta_b for p in parts], axis=0),
            n_steps=sum(p.n_steps for p in parts),
        )
    if bucket_by_k and n_k > 1:
        parts = [matter_power(cosmologies, [kk], redshifts, s, reionization,
                              use_cache, bucket_by_k=False, backend=backend,
                              device=device) for kk in k_h]
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
    if backend == "cuda":
        from tenbol.backends.cuda_solver import evolve_cuda

        rec = evolve_cuda(batch, z, device=device)
    else:
        rec = evolve(batch, z_out=z, a_end=1.0)

    shape = (n_c, n_k, z.size)
    return PowerResult(
        k_h=k_h, z=z,
        delta_m=np.asarray(rec["dm"]).reshape(shape),
        delta_cdm=np.asarray(rec["dc"]).reshape(shape),
        delta_b=np.asarray(rec["db"]).reshape(shape),
        n_steps=int(rec["n_steps"]),
    )
