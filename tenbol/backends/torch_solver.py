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

from tenbol.physics.perturbations import NU_W, _l_coeff

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
    grho_spT = c["grho_nu_species"].transpose(0, 1)
    T = Y.shape[1]
    n_s, n_q = L.n_species, L.n_q
    w = torch.as_tensor(NU_W, dtype=Y.dtype, device=Y.device)

    def sc(name):
        return Y[L.scalar(name)]

    dc, db, vb, dg, qg, pig = sc("dc"), sc("db"), sc("vb"), sc("dg"), sc("qg"), sc("pig")
    dr, qr, pir, etak = sc("dr"), sc("qr"), sc("pir"), sc("etak")
    Fnu = Y[L.offsets["Fnu"]].reshape(n_s, n_q, L.lmax_nu + 1, T)

    # ---- massive-neutrino projections ---------------------------------
    epsT = c["vnu"].reciprocal().permute(1, 2, 0)          # (S, nq, T)
    wq = w[None, :, None]
    den = (wq * epsT).sum(dim=1)                           # (S, T)
    clxnu = (wq * epsT * Fnu[..., 0, :]).sum(dim=1) / den
    qnu = (wq * Fnu[..., 1, :]).sum(dim=1) / (wq * epsT).sum(dim=1)
    dgrho_nu = (grho_spT * clxnu).sum(dim=0)
    dgq_nu = (grho_spT * qnu).sum(dim=0)

    dgrho_matter = grho_b * db + grho_c * dc + dgrho_nu
    dgq_matter = grho_b * vb + dgq_nu

    tcu = kappa > (k / s.tc_epsilon)
    ktau = k * tau
    rsa_g = (ktau > s.rsa_ktau) & (k > s.rsa_kmin)
    rsa_r = (ktau > s.rsa_ktau_nu) & (k > s.rsa_kmin_r)

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
    dgq = dgq_matter + grho_g * qg_e + grho_r * qr_e

    inv_tau = torch.minimum(tau.clamp_min(1e-300).reciprocal(), k)
    z = (0.5 * dgrho / k + etak) / h
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
    _ktau_nu = torch.maximum(
        torch.as_tensor(float(s.rsa_ktau_nu), dtype=Y.dtype, device=Y.device),
        k * c["taurend_nu"])
    nu_free = (ktau_t > _ktau_nu) & (k > s.rsa_kmin_r)
    if bool(nu_free.any()):
        dFnu = torch.where(nu_free.reshape(1, 1, 1, T), _zero, dFnu)

    dY[L.offsets["Fnu"]] = dFnu.reshape(
        L.offsets["Fnu"].stop - L.offsets["Fnu"].start, T)

    # d/dtau -> d/dln a
    dY = dY * (1.0 / h).unsqueeze(0)
    return dY


def evolve_torch(batch, z_out, a_end: float = 1.0, settings=None,
                 device=None, dtype=torch.float64):
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

    ln_a = np.log(s.a_start)
    ln_end = 0.0
    h = s.dlna_max
    next_out = np.zeros(batch.T, dtype=np.int64)
    steps = 0
    while ln_a < ln_end - 1e-14:
        steps += 1
        hn = min(h, ln_end - ln_a)
        c0 = batch.coefficients(np.exp(ln_a))
        ct0 = to_device(c0, L, device, dtype)
        k1 = rhs_torch(Y, None, L, ct0, s)

        a1 = np.exp(ln_a + 0.5 * hn)
        ct1 = to_device(batch.coefficients(a1), L, device, dtype)
        k2 = rhs_torch(Y + 0.5 * hn * k1, None, L, ct1, s)

        a2 = np.exp(ln_a + 0.5 * hn)
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
    return rec
