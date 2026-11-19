"""Torch backend for the perturbation solver.

The RHS is a stack of element-wise tensor operations over ``Y[D, T]``, so the
numpy implementation in ``tenbol.physics.perturbations`` translates to torch
almost line for line.  Keeping the device as a parameter means the same code
runs on CPU (for bit-comparison against the numpy path) and on any CUDA/MPS
device.

The cosmological coefficients are still produced by the numpy ``coefficients``
routine -- they are cheap (one evaluation per step, a few hundred floats) and
independent of the state -- and uploaded to the device each step.  That
CPU->GPU hop is the obvious thing to remove once the device path is trusted.

Design notes tied to the GPU goal:

* ``Y`` is ``(D, T)`` and every operation is over a whole row or a whole block.
  There is no per-trajectory Python loop, only the structural loops over
  multipole index ``l``, which act on whole rows.
* The layout offsets come from ``StateLayout``, so the same indexing serves
  numpy and torch.
* No dynamic control flow depends on the state; regime masks (tight coupling,
  RSA) are precomputed tensors.
"""

from __future__ import annotations

import numpy as np
import torch

from tenbol.physics.background import NU_PERT_W
from tenbol.physics.perturbations import _l_coeff, _neutrino_response

__all__ = ["rhs_torch", "evolve_torch", "to_device"]


def to_device(c: dict, layout, device, dtype=torch.float64) -> dict:
    """Upload the numpy coefficient dict to ``device`` as torch tensors."""
    out = {}
    for key, val in c.items():
        arr = np.asarray(val, dtype=np.float64)
        out[key] = torch.as_tensor(arr, dtype=dtype, device=device)
    # scalars used repeatedly
    out["k2"] = out["k"] ** 2
    return out


def rhs_torch(Y, a, layout, c, settings):
    """Torch port of ``tenbol.physics.perturbations.rhs``.

    ``Y`` is ``(D, T)`` on the target device; ``c`` is the dict from
    ``to_device``; returns ``dY`` with the same shape (derivatives w.r.t. ln a).
    """
    L = layout
    s = settings
    k = c["k"]
    k2 = c["k2"]
    # The eight _l_coeff call sites below each followed with torch.as_tensor on
    # the numpy result, so every rhs re-uploaded the same constants: with four
    # rhs per RK4 step that is dozens of host->device transfers per step, and on
    # a T=1..128 batch it dominated everything (75 ms/step measured).  The
    # coefficients depend only on k and l, so build them once on device.
    _lmax_any = max(L.lmax_g, L.lmax_r, L.lmax_pol) + 1
    _ls = torch.arange(_lmax_any + 1, dtype=Y.dtype, device=Y.device)
    _den = 2.0 * _ls + 1.0
    kk_tab = (k[None, :] * _ls[:, None]) / _den[:, None]
    kk2_tab = (k[None, :] * (_ls[:, None] + 1.0)) / _den[:, None]
    h = c["h"]
    kappa = c["kappa"]
    cs2 = c["cs2"]
    tau = c["tau"]
    grho_g, grho_r = c["grho_g"], c["grho_r"]
    grho_b, grho_c = c["grho_b"], c["grho_c"]
    grho_rel = c["grho_nu_rel_species"].transpose(0, 1)
    T = Y.shape[1]
    n_s, n_q = L.n_species, L.n_q
    w = torch.as_tensor(NU_PERT_W, dtype=Y.dtype, device=Y.device)

    def sc(name):
        return Y[L.scalar(name)]

    dc, db, vb, dg, qg, pig = sc("dc"), sc("db"), sc("vb"), sc("dg"), sc("qg"), sc("pig")
    dr, qr, pir, etak = sc("dr"), sc("qr"), sc("pir"), sc("etak")
    Fnu = Y[L.offsets["Fnu"]].reshape(n_s, n_q, L.lmax_nu + 1, T)

    # ---- massive-neutrino projections ---------------------------------
    epsT = c["vnu"].reciprocal().permute(1, 2, 0)          # (S, nq, T)
    wq = w[None, :, None]
    dgrho_nu = (grho_rel * (wq * epsT * Fnu[..., 0, :]).sum(dim=1)).sum(dim=0)
    dgq_nu = (grho_rel * (wq * Fnu[..., 1, :]).sum(dim=1)).sum(dim=0)

    tcu = c["tight_coupling"] != 0
    ktau = k * tau
    # CAMB switches to the algebraic streaming form at tau > max(15/k, taurend),
    # i.e. k*tau > max(15, k*taurend) (equations.f90:313-323) - NOT at a constant
    # k*tau.  A constant threshold fires early, and measured against numpy this
    # was the whole of a 1.4% error in dc at k=0.3: the two agreed to the last bit
    # for a >= 0.05, where both are fully in the RSA branch, and diverged only
    # across the a ~ 1e-3..1e-2 transition, where the constant threshold fired
    # first.  The numpy side carries this fix already; the port had never been
    # brought across.
    rsa_g = (ktau > torch.maximum(torch.as_tensor(float(s.rsa_ktau), dtype=Y.dtype, device=Y.device),
                                  k * c["taurend"])) & (k > s.rsa_kmin)
    rsa_r = (ktau > torch.maximum(torch.as_tensor(float(s.rsa_ktau_nu), dtype=Y.dtype, device=Y.device),
                                  k * c["taurend_nu"])) & (k > s.rsa_kmin_r)
    streaming = torch.zeros_like(k)
    if s.nu_freeze_from_l <= 0 and n_s:
        cb = grho_c + grho_b
        delta_cb = (grho_c * dc + grho_b * db) / cb
        clustered, derivative, free_streaming = _neutrino_response(
            k, cb, epsT.reciprocal(), grho_rel, w)
        clustered = clustered.sum(dim=0)
        dgrho_nu = torch.where(rsa_r, clustered * delta_cb, dgrho_nu)
        late_flux = clustered * grho_b / cb * vb - derivative.sum(dim=0) * h * delta_cb / k
        dgq_nu = torch.where(rsa_r, late_flux, dgq_nu)
        streaming = torch.where(rsa_r, free_streaming.sum(dim=0), streaming)

    dgrho_matter = grho_b * db + grho_c * dc + dgrho_nu
    dgq_matter = grho_b * vb + dgq_nu

    # ---- radiation streaming approximation ----------------------------
    zm = (0.5 * dgrho_matter / k + etak) / h
    dzm = -h * zm - 0.5 * dgrho_matter / k
    k_rsa = torch.where(rsa_g, k, torch.ones_like(k))
    kap_rsa = torch.where(rsa_g, kappa, torch.zeros_like(kappa))
    dg_rsa = -4.0 * dzm / k_rsa - 4.0 / k_rsa * kap_rsa * (vb + zm)
    qg_rsa = -4.0 / 3.0 * zm
    dr_rsa = -4.0 * dzm / k
    qr_rsa = -4.0 / 3.0 * zm

    dg_e = torch.where(rsa_g, dg_rsa, dg)
    qg_e = torch.where(rsa_g, qg_rsa, qg)
    dr_e = torch.where(rsa_r, dr_rsa, dr)
    qr_e = torch.where(rsa_r, qr_rsa, qr)

    dgrho = dgrho_matter + grho_g * dg_e + grho_r * dr_e

    inv_tau = torch.minimum(tau.clamp_min(1e-300).reciprocal(), k)
    z = (0.5 * dgrho / k + etak) / h
    dgq = dgq_matter + grho_g * qg_e + grho_r * qr_e - streaming * z
    sigma = z + 1.5 * dgq / k2

    dY = torch.zeros_like(Y)
    dY[L.scalar("etak")] = 0.5 * dgq
    dY[L.scalar("dc")] = -k * z
    db_dot = -k * (z + vb)
    dY[L.scalar("db")] = db_dot
    delta_p_b = cs2 * db

    # ---- baryon velocity ------------------------------------------------
    pig_tc = 32.0 / 45.0 / kappa.clamp_min(1e-300) * k * (sigma + vb)
    dg_dot = -k * (4.0 / 3.0 * z + qg_e)
    adotdota = (h ** 2 - c["gpres"]) / 2.0
    pb43 = 4.0 / 3.0 * grho_g / grho_b
    photbar = grho_g / grho_b
    dopacity = c.get("dopacity")
    slip_damp = 2.0 * h / (1.0 + pb43)
    if dopacity is not None:
        slip_damp = slip_damp + torch.where(
            tcu, dopacity / kappa.clamp_min(1e-300), torch.zeros_like(kappa))
    slip = (
        -slip_damp * (vb - 0.75 * qg_e)
        + (-adotdota * vb - k / 2.0 * h * dg_e + k * (cs2 * db_dot - dg_dot / 4.0))
        / (kappa * (1.0 + pb43))
    )
    # ---- second-order tight coupling --------------------------------
    # CAMB hard-codes second_order_tightcoupling = .true. (equations.f90:52) and
    # uses it to correct the quadrupole (equations.f90:2496-2512).  First-order
    # tight coupling does not produce Silk damping; this correction is what damps
    # the acoustic oscillations below the diffusion length.
    #
    # The numpy solver has carried this for a while; the port never did.  That is
    # exactly the observed signature at k_Mpc = 4: numpy and torch agree to
    # 1e-16 outside tight coupling and differ by 1.4% inside it, because this
    # block only has an effect where tcu is true.
    kap_safe = kappa.clamp_min(1e-300)
    _zero_k = torch.zeros_like(kappa)
    _dop = torch.where(tcu, dopacity, _zero_k) if dopacity is not None else _zero_k
    dgs = grho_g * pig_tc + grho_r * pir
    sigmadot = -2.0 * h * sigma - dgs / k + etak
    qgdot_tc = k * (dg_e / 4.0 - pig_tc / 2.0) + kap_safe * slip
    _r = _dop * (11.0 / 6.0) / kap_safe ** 2
    pig_tc = (32.0 / 45.0 / kap_safe * k * (sigma + 0.75 * qg_e) * (1.0 + _r)
              + (32.0 / 45.0 / kap_safe ** 2) * k * (sigmadot + 0.75 * qgdot_tc)
              * (-11.0 / 6.0))

    vb_tc = (
        (-h * vb + cs2 * k * db + k / 4.0 * pb43 * (dg_e - 2.0 * pig_tc)) / (1.0 + pb43)
        + pb43 / (1.0 + pb43) * slip
    )
    kap_notc = torch.where(tcu, torch.zeros_like(kappa), kappa)
    vb_full = -h * vb + k * delta_p_b - photbar * kap_notc * (4.0 / 3.0 * vb - qg_e)
    dY[L.scalar("vb")] = torch.where(tcu, vb_tc, vb_full)
    vb_e = dY[L.scalar("vb")]

    qg_tc = (4.0 / 3.0 * (-vb_e - h * vb + delta_p_b * k) / pb43
             + k / 3.0 * dg_e - 2.0 / 3.0 * k * pig_tc)
    qg_full = (k / 3.0 * dg_e - 4.0 / 3.0 * k * pig
               + 4.0 / 3.0 * kap_notc * (vb - 0.75 * qg_e))
    dY[L.scalar("qg")] = torch.where(
        rsa_g, torch.zeros_like(qg_e), torch.where(tcu, qg_tc, qg_full))

    polter = pig / 10.0 + 0.6 * Y[L.g(2)]
    pig_dot = (k * 2.0 / 5.0 * qg_e - k * 3.0 / 5.0 * Y[L.fg(3)]
               - torch.where(tcu | rsa_g, torch.zeros_like(kappa), kappa)
               * (pig - polter)
               + (8.0 / 15.0) * k * sigma)
    dY[L.scalar("pig")] = torch.where(tcu | rsa_g, torch.zeros_like(pig), pig_dot)
    dY[L.scalar("dg")] = torch.where(rsa_g, torch.zeros_like(dg), dg_dot)

    # ---- photon moments -------------------------------------------------
    clamp = tcu | rsa_g
    zero = torch.zeros_like(k)
    for l in range(3, L.lmax_g):
        kk, kk2 = kk_tab[l], kk2_tab[l]
        dY[L.fg(l)] = torch.where(
            clamp, torch.zeros_like(k),
            kk * Y[L.fg(l - 1)] - kk2 * Y[L.fg(l + 1)] - kappa * Y[L.fg(l)])
    kk = kk_tab[L.lmax_g]
    dY[L.fg(L.lmax_g)] = torch.where(
        clamp, torch.zeros_like(k),
        kk * Y[L.fg(L.lmax_g - 1)]
        - (L.lmax_g + 1) * Y[L.fg(L.lmax_g)] * inv_tau
        - kappa * Y[L.fg(L.lmax_g)])

    # ---- polarization ---------------------------------------------------
    dY[L.g(2)] = torch.where(clamp, torch.zeros_like(k),
                             -kappa * (Y[L.g(2)] - polter) - k / 3.0 * Y[L.g(3)])
    for l in range(3, L.lmax_pol):
        kk = kk_tab[l]
        polfack = (l + 3.0) * (l - 1.0) / (l + 1.0) * k
        dY[L.g(l)] = torch.where(
            clamp, torch.zeros_like(k),
            -kappa * Y[L.g(l)] + kk * Y[L.g(l - 1)] - polfack * Y[L.g(l + 1)])
    poltrunc = L.lmax_pol / max(1, L.lmax_pol - 2)
    kk = kk_tab[L.lmax_pol]
    dY[L.g(L.lmax_pol)] = torch.where(
        clamp, torch.zeros_like(k),
        -kappa * Y[L.g(L.lmax_pol)] + k * poltrunc * Y[L.g(L.lmax_pol - 1)]
        - (L.lmax_pol + 3) * Y[L.g(L.lmax_pol)] * inv_tau)

    # ---- massless neutrinos ---------------------------------------------
    dY[L.scalar("dr")] = torch.where(rsa_r, torch.zeros_like(dr),
                                     -k * (4.0 / 3.0 * z + qr_e))
    kk1, kk12 = kk_tab[1], kk2_tab[1]
    dY[L.scalar("qr")] = torch.where(rsa_r, torch.zeros_like(qr), kk1 * dr_e - kk12 * pir)
    kk2_, kk22 = kk_tab[2], kk2_tab[2]
    dY[L.scalar("pir")] = torch.where(
        rsa_r, torch.zeros_like(pir),
        kk2_ * qr_e - kk22 * Y[L.fr(3)] + (8.0 / 15.0) * k * sigma)
    for l in range(3, L.lmax_r):
        kk, kk2 = kk_tab[l], kk2_tab[l]
        dY[L.fr(l)] = torch.where(rsa_r, torch.zeros_like(k),
                                  kk * Y[L.fr(l - 1)] - kk2 * Y[L.fr(l + 1)])
    kk = kk_tab[L.lmax_r]
    dY[L.fr(L.lmax_r)] = torch.where(
        rsa_r, torch.zeros_like(k),
        kk * Y[L.fr(L.lmax_r - 1)]
        - (L.lmax_r + 1) * Y[L.fr(L.lmax_r)] * inv_tau)

    # ---- massive neutrinos ----------------------------------------------
    vT = c["vnu"].permute(1, 2, 0)                          # (S, nq, T)
    kb = k.reshape(1, 1, T)
    zb = z.reshape(1, 1, T)
    dFnu = torch.zeros_like(Fnu)
    kk_l = [(k * l / (2.0 * l + 1.0)).reshape(1, 1, T) for l in range(L.lmax_nu + 2)]
    kk2_l = [(k * (l + 1.0) / (2.0 * l + 1.0)).reshape(1, 1, T)
             for l in range(L.lmax_nu + 2)]
    dFnu[..., 0, :] = -kb * (4.0 / 3.0 * zb + vT * Fnu[..., 1, :])
    dFnu[..., 1, :] = vT * (kk_l[1] * Fnu[..., 0, :] - kk2_l[1] * Fnu[..., 2, :])
    dFnu[..., 2, :] = (vT * (kk_l[2] * Fnu[..., 1, :] - kk2_l[2] * Fnu[..., 3, :])
                       + (8.0 / 15.0) * kb * sigma.reshape(1, 1, T))
    for l in range(3, L.lmax_nu):
        dFnu[..., l, :] = vT * (kk_l[l] * Fnu[..., l - 1, :]
                                - kk2_l[l] * Fnu[..., l + 1, :])
    l = L.lmax_nu
    dFnu[..., l, :] = (kb * vT * Fnu[..., l - 1, :]
                       - (l + 1) * Fnu[..., l, :] * inv_tau.reshape(1, 1, T))

    # Two masks that the numpy solver has and this port did not, which is why
    # 48 of 229 rows disagreed (numpy gave exact zeros where this gave ~1e4):
    #
    #  1. CAMB demotes the ladder to lmaxnu_high_ktau = 4 as soon as
    #     k*tau > max(20, lmaxnr-4) and keeps evolving those four moments, so
    #     l_hi becomes the top row and everything above it is pinned to zero.
    #  2. The hierarchy freezes entirely above k*tau_nu > max(15, k*taurend_nu)
    #     on the ACCUMULATED conformal time.  Without this the block has no sink
    #     and the cascade runs away.
    _zero = torch.zeros((), dtype=Y.dtype, device=Y.device)
    ktau_t = k * tau
    l_hi_m = min(int(s.lmax_nu_high_ktau), L.lmax_nu)
    hi_ktau = (ktau_t > max(20.0, float(L.lmax_nu - 4))) & (k > s.rsa_kmin_r)
    if bool(hi_ktau.any()):
        m = hi_ktau.reshape(1, 1, 1, T)
        dFnu[..., l_hi_m, :] = torch.where(
            m,
            kb * vT * Fnu[..., l_hi_m - 1, :]
            - (l_hi_m + 1) * Fnu[..., l_hi_m, :] * inv_tau.reshape(1, 1, T),
            dFnu[..., l_hi_m, :])
        for ll in range(l_hi_m + 1, L.lmax_nu + 1):
            dFnu[..., ll, :] = torch.where(m, _zero, dFnu[..., ll, :])
    if bool(rsa_r.any()):
        first = max(int(s.nu_freeze_from_l), 0)
        dFnu[..., first:, :] = torch.where(
            rsa_r.reshape(1, 1, 1, T), _zero, dFnu[..., first:, :])

    dY[L.offsets["Fnu"]] = dFnu.reshape(
        L.offsets["Fnu"].stop - L.offsets["Fnu"].start, T)

    # d/dtau -> d/dln a
    dY = dY * (1.0 / h).unsqueeze(0)
    return dY


def evolve_torch_adaptive(batch, z_out, a_end: float = 1.0, settings=None,
                          device=None, dtype=torch.float64, check_every: int = 0):
    """Batched RK4 on ``device`` driven by the numpy solver's step limiter.

    Every step advances the WHOLE batch by one ``hn = min`` over trajectories of
    the same angular-CFL / Thomson-stiffness / super-horizon / event-distance
    terms that ``perturbations.evolve`` uses.  That is why T=1 reproduces the
    numpy solver bit-for-bit (measured: 51494 steps, dc = 2.061496126141010e+05,
    rel = 0.00e+00), and why a mixed batch costs only 7% more steps than
    per-trajectory stepping (55231 vs 51494) instead of the 11x the raw spread of
    required steps suggested - an easy trajectory reaches a_end and is marked
    done long before the hard one finishes.

    The limiter is pure elementwise where/min/clamp over tensors already present
    in the coefficient dict, so it needs no host callback.  The only new
    host/device synchronisation is the single ``.item()`` that reads the step
    back, once per step.
    """
    from tenbol.physics.perturbations import initial_state, _event_times

    s = settings or batch.s
    L = batch.layout
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = batch.T

    Y = torch.as_tensor(np.asarray(initial_state(batch, s.a_start)),
                        dtype=dtype, device=device)

    z_out = np.atleast_1d(np.asarray(z_out, dtype=np.float64))
    a_out = np.sort(1.0 / (1.0 + z_out))
    rec = {key: np.full((T, a_out.size), np.nan) for key in
           ("dc", "db", "dg", "dr", "vb", "qg", "qr", "etak")}

    events = _event_times(batch, s.a_start, a_end)
    targets = np.sort(np.vstack([
        np.where(np.isfinite(events[n][:, 0]), events[n][:, 0], np.inf)
        for n in ("tc", "rsa_g", "rsa_r")]), axis=0).T
    next_ev = np.zeros(T, dtype=np.int64)
    pending = targets[:, 0].copy()

    a = np.full(T, s.a_start)
    done = np.zeros(T, dtype=bool)
    next_out = np.zeros(T, dtype=np.int64)
    steps = 0

    def _c(v):
        return torch.as_tensor(float(v), dtype=dtype, device=device)

    while not np.all(done):
        steps += 1
        if steps > s.max_steps:
            raise RuntimeError("device integration exceeded max_steps")

        ct0 = to_device(batch.coefficients(a), L, device, dtype)

        # ---- step limiter: mirror of perturbations.evolve ---------------
        kk = ct0["k"]
        kappa = ct0["kappa"]
        dtau = 1.0 / ct0["h"]
        ktau = kk * ct0["tau"]
        tcu = ct0["tight_coupling"] != 0
        rsa_g = (ktau > torch.maximum(_c(s.rsa_ktau), kk * ct0["taurend"])) & (kk > _c(s.rsa_kmin))
        rsa_r = (ktau > torch.maximum(_c(s.rsa_ktau_r), kk * ct0["taurend_nu"])) & (kk > _c(s.rsa_kmin_r))
        angular = ~(tcu | rsa_g)
        lim = torch.full_like(kk, s.dlna_max)
        lim = torch.where(angular, torch.minimum(lim, _c(s.dlna_acoustic) / torch.clamp(kk * dtau, min=1e-30)), lim)
        lim = torch.where(~rsa_r, torch.minimum(lim, _c(s.dlna_nu) / torch.clamp(kk * dtau, min=1e-30)), lim)
        lim = torch.where(ktau < 1.0, torch.minimum(lim, _c(s.dlna_superhorizon)), lim)
        lim = torch.where(~tcu, torch.minimum(lim, _c(s.stiff_factor) / torch.clamp(kappa * dtau, min=1e-30)), lim)

        target = np.minimum(np.full(T, a_end), np.where(pending > a, pending, a_end))
        dln_target = np.where(target > a, np.log(np.maximum(target, 1e-300) / a), 0.0)
        dln_target = np.where(done, np.inf, dln_target)

        hn = min(float(lim.min().item()), float(dln_target.min()), s.h_max)
        hn = max(hn, s.dlna_min)

        k1 = rhs_torch(Y, None, L, ct0, s)
        a_half = a * np.exp(0.5 * hn)
        ct1 = to_device(batch.coefficients(a_half), L, device, dtype)
        k2 = rhs_torch(Y + 0.5 * hn * k1, None, L, ct1, s)
        k3 = rhs_torch(Y + 0.5 * hn * k2, None, L, ct1, s)
        ct3 = to_device(batch.coefficients(a * np.exp(hn)), L, device, dtype)
        k4 = rhs_torch(Y + hn * k3, None, L, ct3, s)
        Y = Y + (hn / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if check_every and (steps % check_every == 0):
            # Diagnostic: a NaN that appears mid-run tells us which step and
            # scale factor it started at, which a final-only check cannot.
            bad = ~torch.isfinite(Y).all(dim=0)
            if bool(bad.any()):
                j = int(bad.nonzero()[0])
                raise RuntimeError(
                    f"non-finite state at step {steps}, a={float(a[j]):.3e}, "
                    f"k={float(batch.k[j]):.3e}, hn={hn:.3e}")

        # Pin a to a_end when done, otherwise it stops a few ulp short and the
        # output test a_out[j] <= a_now never fires - which left rec at its
        # initial NaN even though the state itself was perfectly finite.
        a = np.where(done, a_end, np.minimum(a * np.exp(hn), a_end))
        adv = (~done) & (a >= pending)
        if np.any(adv):
            next_ev = np.where(adv, next_ev + 1, next_ev)
            pending = np.where(adv, np.where(next_ev < 3,
                               targets[np.arange(T), np.minimum(next_ev, 2)], np.inf), pending)
        done = a >= a_end - 1e-14

        a_now = float(a[0])
        while next_out[0] < a_out.size and a_out[next_out[0]] <= a_now:
            j = int(next_out[0])
            f = Y.detach().cpu().numpy()
            for key in rec:
                rec[key][:, j] = f[L.scalar(key)]
            next_out[0] += 1

    # Belt and braces: if a_out has a point at or below the final a that the
    # in-loop test missed by a rounding step, fill it from the final state.
    while next_out[0] < a_out.size:
        j = int(next_out[0])
        f = Y.detach().cpu().numpy()
        for key in rec:
            rec[key][:, j] = f[L.scalar(key)]
        next_out[0] += 1

    rec["n_steps"] = steps
    return rec


def evolve_torch(batch, z_out, a_end: float = 1.0, settings=None,
                 device=None, dtype=torch.float64, dlna=None,
                 merge_tol: float = 1.0e-2):
    """Single-batch RK4 integration on ``device`` (no scaling, no events).

    This is a correctness/performance probe for the device path: it advances a
    fixed-step RK4 with the region masks recomputed each step, which is enough
    to compare against the numpy solver and to time the GPU.  Event snapping
    and adaptive step control stay in the numpy driver until the device path is
    trusted.
    """
    from tenbol.physics.perturbations import initial_state

    s = settings or batch.s
    L = batch.layout
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    a = np.full(batch.T, s.a_start)
    Y0 = initial_state(batch, s.a_start)
    Y = torch.as_tensor(np.asarray(Y0), dtype=dtype, device=device)

    z_out = np.atleast_1d(np.asarray(z_out, dtype=np.float64))
    a_out = np.sort(1.0 / (1.0 + z_out))
    rec = {key: np.full((batch.T, a_out.size), np.nan) for key in
           ("dc", "db", "dg", "dr", "vb", "qg", "qr", "etak")}

    # Segment the integration at the regime switches.  A uniform grid over the
    # whole range steps OVER those switches, which costs 2.2% accuracy even at
    # 16119 steps, while landing on them gives 0.154% at 8061 (both measured, see
    # evidence/gpu_segmented_scheme.json).  Between switches the batch advances
    # on a plain fixed grid, so the host still synchronises once per segment
    # rather than once per step - which is the whole point of a device backend.
    from tenbol.physics.perturbations import _event_times
    brk = [float(s.a_start), float(a_end)]
    for arr in _event_times(batch, s.a_start, a_end).values():
        for v in np.asarray(arr).ravel():
            if np.isfinite(v) and s.a_start < v < a_end:
                brk.append(float(v))
    # Merge near-duplicate switches.  On ONE trajectory there are about six and
    # nothing merges; on T=16384 the union is 28366 boundaries because no two
    # trajectories switch at exactly the same moment, and that costs more steps
    # than plain uniform stepping (32842 against 8060).  Merging in ln a at
    # 1e-2 drops it to 857 boundaries / 8667 steps (8.8 min) while the
    # single-trajectory accuracy stays at 2.35e-3 against 2.33e-3 unmerged.
    brk = sorted(set(brk))
    if merge_tol and merge_tol > 0:
        merged = [brk[0]]
        for v in brk[1:]:
            if np.log(v) - np.log(merged[-1]) > merge_tol:
                merged.append(v)
        brk = merged

    h = dlna if dlna is not None else s.dlna_max
    next_out = np.zeros(batch.T, dtype=np.int64)
    steps = 0
    for lo, hi in zip(brk[:-1], brk[1:]):
        ln_a = np.log(lo)
        ln_hi = np.log(hi)
        while ln_a < ln_hi - 1e-15:
            steps += 1
            hn = min(h, ln_hi - ln_a)
            c0 = batch.coefficients(np.exp(ln_a))
            ct0 = to_device(c0, L, device, dtype)
            k1 = rhs_torch(Y, None, L, ct0, s)

            a1 = np.exp(ln_a + 0.5 * hn)
            ct1 = to_device(batch.coefficients(a1), L, device, dtype)
            k2 = rhs_torch(Y + 0.5 * hn * k1, None, L, ct1, s)

            k3 = rhs_torch(Y + 0.5 * hn * k2, None, L, ct1, s)

            a3 = np.exp(ln_a + hn)
            ct3 = to_device(batch.coefficients(a3), L, device, dtype)
            k4 = rhs_torch(Y + hn * k3, None, L, ct3, s)

            Y = Y + (hn / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            ln_a += hn

            # record at requested outputs (nearest step; this probe is not the
            # production dense-output path)
            a_now = np.exp(ln_a)
            while next_out[0] < a_out.size and a_out[next_out[0]] <= a_now:
                j = int(next_out[0])
                f = Y.detach().cpu().numpy()
                for key in rec:
                    rec[key][:, j] = f[L.scalar(key)]
                next_out[0] += 1
    rec["n_steps"] = steps
    rec["segments"] = len(brk) - 1
    return rec
