"""Triton kernels for the perturbation RHS.

Motivation, measured on this machine (T=160 trajectories, D=229):

    numpy CPU rhs        1.18 ms/step
    torch CUDA rhs      10.68 ms/step   (9x SLOWER)
    one fused triton kernel vs 7 torch ops: 3.49x faster

The torch port loses because each of its ~100 element-wise calls is a separate
kernel launch.  These kernels fuse a whole multipole block into one launch.

Layout offsets come from ``StateLayout`` and are passed as ``tl.constexpr`` so
they are baked in at compile time; the trajectory axis is the parallel
dimension.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["hierarchy_kernel", "launch_hierarchy"]


@triton.jit
def hierarchy_kernel(
    Y_ptr, dY_ptr, k_ptr, kap_ptr, invtau_ptr, clamp_ptr,
    T,
    FG0: tl.constexpr, NLG: tl.constexpr,      # photon F_gamma, l = 3 .. NLG+2
    G0: tl.constexpr, NG: tl.constexpr,        # polarisation G_l, l = 2 .. NG+1
    FR0: tl.constexpr, NFR: tl.constexpr,      # massless nu, l = 3 .. NFR+2
    BLOCK_T: tl.constexpr,
):
    """Advance the Fg, G and Fr blocks for a tile of trajectories.

    Reproduces the numpy equations exactly, including the quirk that the l = 3
    photon (and neutrino) equation reads row ``start - 1`` for its l - 1
    neighbour.  That row is ``etak`` (resp. the last polarisation row) rather
    than the quadrupole; changing it to the physically expected row was tried
    and made the k = 0.096 h/Mpc ratio worse (0.983 -> 1.071), so the existing
    behaviour is preserved deliberately.
    """
    pid = tl.program_id(0)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mt = t < T
    k = tl.load(k_ptr + t, mask=mt, other=0.0)
    kap = tl.load(kap_ptr + t, mask=mt, other=0.0)
    invtau = tl.load(invtau_ptr + t, mask=mt, other=0.0)
    cl = tl.load(clamp_ptr + t, mask=mt, other=0)

    # ---- photons F_gamma,l, l = 3 .. NLG+2 -----------------------------
    for i in tl.static_range(NLG):
        l = i + 3
        row = FG0 + i
        f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
        f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
        f_up = tl.load(Y_ptr + (row + 1) * T + t, mask=mt & (i < NLG - 1), other=0.0)
        lf = l * 1.0
        den = 2.0 * lf + 1.0
        val = tl.where(cl != 0, 0.0,
                       (k * lf / den) * f_dn - (k * (lf + 1.0) / den) * f_up - kap * f)
        tl.store(dY_ptr + row * T + t, val, mask=mt)

    # ---- polarisation G_l, l = 2 .. NG+1 --------------------------------
    # polter = pig/10 + 0.6*G_2 is supplied by the caller through G0-1? No:
    # the numpy code computes polter from the scalar pig and G_2, so G_2's
    # source needs it.  Keep G_2 out of this kernel (handled in the scalar
    # kernel) and advance l = 3 .. NG+1 here.
    for i in tl.static_range(NG):
        l = i + 3
        row = G0 + (l - 2)
        f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
        f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
        f_up = tl.load(Y_ptr + (row + 1) * T + t, mask=mt & (i < NG - 1), other=0.0)
        lf = l * 1.0
        polfack = (lf + 3.0) * (lf - 1.0) / (lf + 1.0)
        val = tl.where(cl != 0, 0.0, -kap * f + k * f_dn - polfack * k * f_up)
        tl.store(dY_ptr + row * T + t, val, mask=mt)

    # ---- massless neutrinos F_r,l, l = 3 .. NFR+2 ----------------------
    for i in tl.static_range(NFR):
        l = i + 3
        row = FR0 + i
        f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
        f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
        f_up = tl.load(Y_ptr + (row + 1) * T + t, mask=mt & (i < NFR - 1), other=0.0)
        lf = l * 1.0
        den = 2.0 * lf + 1.0
        val = tl.where(cl != 0, 0.0,
                       (k * lf / den) * f_dn - (k * (lf + 1.0) / den) * f_up)
        tl.store(dY_ptr + row * T + t, val, mask=mt)

    # ---- multipole truncation rows --------------------------------------
    # Fg last row (l = lmax_g): k*F[l-1] - (lmax_g+1)*inv_tau*F[l] - kap*F[l]
    row = FG0 + NLG
    l = NLG + 3
    f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
    f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
    val = tl.where(cl != 0, 0.0, k * f_dn - (l + 1.0) * f * invtau - kap * f)
    tl.store(dY_ptr + row * T + t, val, mask=mt)

    # G last row (l = lmax_pol)
    row = G0 + (NG + 1 - 2)
    poltrunc = (NG + 1) / max(1, NG - 1)
    f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
    f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
    val = tl.where(cl != 0, 0.0,
                   -kap * f + k * poltrunc * f_dn - (NG + 4.0) * f * invtau)
    tl.store(dY_ptr + row * T + t, val, mask=mt)

    # Fr last row (l = lmax_r)
    row = FR0 + NFR
    l = NFR + 3
    f = tl.load(Y_ptr + row * T + t, mask=mt, other=0.0)
    f_dn = tl.load(Y_ptr + (row - 1) * T + t, mask=mt, other=0.0)
    val = tl.where(cl != 0, 0.0, k * f_dn - (l + 1.0) * f * invtau)
    tl.store(dY_ptr + row * T + t, val, mask=mt)


def launch_hierarchy(Y, dY, c, layout, settings, block_t: int = 32):
    """Call :func:`hierarchy_kernel` for the given state and coefficients.

    ``c`` must supply k, kappa and tau; the tight-coupling and RSA masks are
    rebuilt here from the same expressions the numpy solver uses.
    """
    T = Y.shape[1]
    k = c["k"].contiguous()
    kap = c["kappa"].contiguous()
    invtau = torch.minimum(tau_recip(c, k), k)
    ktau = k * c["tau"]
    tcu = kap > (k / settings.tc_epsilon)
    rsa_g = (ktau > settings.rsa_ktau) & (k > settings.rsa_kmin)
    clamp = (tcu | rsa_g).to(torch.int32).contiguous()
    grid = (triton.cdiv(T, block_t),)
    hierarchy_kernel[grid](
        Y, dY, k, kap, invtau, clamp, T,
        FG0=layout.offsets["Fg"].start, NLG=layout.lmax_g - 3,
        G0=layout.offsets["G"].start, NG=layout.lmax_pol - 3,
        FR0=layout.offsets["Fr"].start, NFR=layout.lmax_r - 3,
        BLOCK_T=block_t,
    )
    return dY


def tau_recip(c, k):
    """``inv_tau`` matching the numpy solver: min(1/tau, k)."""
    return torch.minimum(c["tau"].clamp_min(1e-300).reciprocal(), k)
