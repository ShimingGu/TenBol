"""Compile and launch the native, independently stepped CUDA solver."""

from functools import lru_cache
from importlib.resources import files

import numpy as np

from tenbol.errors import ExecutionFailure, NumericalFailure, UnsupportedConfiguration
from tenbol.physics.background import NU_Q, NU_W
from tenbol.physics.perturbations import initial_state


@lru_cache(maxsize=32)
def _kernel(layout):
    import cupy as cp

    constants = {name.upper(): row for name, row in layout.describe()["scalars"].items()}
    constants.update(
        FG=layout.fg(3), POL=layout.g(2), FR=layout.fr(3),
        FNU=layout.offsets["Fnu"].start, DIM=layout.size, NS=layout.n_species,
        NQ=layout.n_q, LG=layout.lmax_g, LP=layout.lmax_pol,
        LR=layout.lmax_r, LNU=layout.lmax_nu,
    )
    source = files(__package__).joinpath("trajectory.cu").read_text(encoding="utf-8")
    options = tuple(f"-D{name}={value}" for name, value in constants.items())
    return cp.RawKernel(source, "integrate", options=options)


def _coefficient_tables(batch):
    """Pack one coefficient grid per cosmology, shared by all its modes."""
    bg = batch.bg
    a = np.exp(batch._ln_a)
    mu = np.maximum(a[None, :, None] * bg._mu0[:, None, :], 1e-12)
    weight = bg.grho_nu_species_unit[:, None, None] * bg.massive_degeneracy[:, None, :]
    nu_density = weight * np.exp(batch._rho_spl(np.log(mu)))
    nu_pressure = np.sum(weight * np.exp(batch._p_spl(np.log(mu))), axis=2)
    table = np.concatenate((
        np.stack((batch._tau_grid, batch._kap_tab, batch._dkap_tab,
                  batch._cs2_tab, nu_pressure), axis=2),
        nu_density,
    ), axis=2)
    background = np.column_stack((bg.grho_c0, bg.grho_b0, bg.grho_g0,
                                  bg.grho_nu_massless0, bg.grho_v0, bg._mu0))
    return table, background


def evolve_cuda(batch, z_out, device=None):
    """Return total, cold and baryon transfers in trajectory order."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise UnsupportedConfiguration(
            'The CUDA backend requires CuPy. Install the "cuda" extra.'
        ) from exc

    settings = batch.s
    layout = batch.layout
    if layout.lmax_pol < 3 or (layout.n_species and
            min(layout.lmax_nu, settings.lmax_nu_high_ktau) < 3):
        raise UnsupportedConfiguration("CUDA requires lmax >= 3 for polarization and massive neutrinos")

    z = np.asarray(z_out, dtype=np.float64)
    if z.ndim != 1 or not z.size or not np.all(np.isfinite(z) & (z >= 0.0)):
        raise ValueError("CUDA redshifts must be a nonempty vector of finite, nonnegative values")
    x, z_index = np.unique(-np.log1p(z), return_inverse=True)
    x_start = np.log(settings.a_start)
    if not np.isfinite(x_start) or x[0] < x_start:
        raise ValueError("CUDA output must lie at or after the starting scale factor")
    if not batch.T or not np.all(np.isfinite(batch.k) & (batch.k > 0.0)):
        raise ValueError("CUDA requires a nonempty batch of finite, positive wavenumbers")

    ln_a = batch._ln_a
    grid_spacing = (ln_a[-1] - ln_a[0]) / (ln_a.size - 1)
    if not np.allclose(np.diff(ln_a), grid_spacing, rtol=1e-10, atol=0.0):
        raise UnsupportedConfiguration("CUDA requires a uniformly spaced log-scale-factor grid")

    table, background = _coefficient_tables(batch)
    # Similar wavenumbers do similar amounts of work, so put them in the same warp.
    order = np.argsort(batch.k, kind="stable")
    initial = np.ascontiguousarray(initial_state(batch, settings.a_start)[:, order])
    switches = np.ascontiguousarray(np.stack((batch._taurend[order], batch._taurend_nu[order])))
    k = batch.k[order]
    n = batch.T
    threads = 32
    try:
        with cp.cuda.Device(device):
            result = cp.empty((x.size, n, 3), dtype=cp.float64)
            steps = cp.empty(n, dtype=cp.int32)
            last_x = cp.empty(n, dtype=cp.float64)
            _kernel(layout)(((n + threads - 1) // threads,), (threads,), (
                cp.asarray(initial), cp.asarray(k), cp.asarray(batch.sample[order], dtype=cp.int32),
                cp.asarray(background), cp.asarray(table), cp.asarray(switches),
                cp.asarray(NU_Q), cp.asarray(NU_W), cp.asarray(x),
                result, steps, last_x,
                np.int32(n), np.int32(x.size), np.int32(ln_a.size),
                np.float64(ln_a[0]), np.float64(grid_spacing), np.float64(x_start),
                np.int32(settings.max_steps),
                np.int32(min(settings.lmax_nu_high_ktau, layout.lmax_nu)),
                np.int32(max(0, min(settings.nu_freeze_from_l, layout.lmax_nu + 1))),
                np.float64(settings.tc_epsilon),
                np.float64(settings.rsa_ktau), np.float64(settings.rsa_ktau_r),
                np.float64(settings.rsa_ktau_nu),
                np.float64(settings.rsa_kmin), np.float64(settings.rsa_kmin_r),
                np.float64(settings.dlna_min), np.float64(min(settings.dlna_max, settings.h_max)),
                np.float64(settings.dlna_acoustic), np.float64(settings.dlna_nu),
                np.float64(settings.dlna_superhorizon), np.float64(settings.stiff_factor),
            ))
            counts = steps.get()
            failed = np.flatnonzero(counts <= 0)
            if failed.size:
                j = failed[0]
                reason = "non-finite state" if counts[j] < 0 else "exceeded max_steps"
                a = np.exp(last_x.get()[j])
                raise NumericalFailure(f"CUDA {reason} at a={a:.3e}, k={k[j]:.3e} /Mpc")
            values = result.get()
    except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError,
            cp.cuda.compiler.CompileException) as exc:
        raise ExecutionFailure(f"CUDA perturbation integration failed: {exc}") from exc

    values = values[z_index][:, np.argsort(order), :].transpose(1, 0, 2)
    return {"dc": values[:, :, 0], "db": values[:, :, 1], "dm": values[:, :, 2],
            "n_steps": int(counts.max())}
