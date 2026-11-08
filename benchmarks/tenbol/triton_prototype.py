import torch, triton, triton.language as tl, time
import numpy as np

# Prototype: advance a whole multipole block for all T trajectories in ONE kernel.
# Layout: F[l, t] with l = 0..n_l-1, contiguous in l.
@triton.jit
def hier_kernel(F_ptr, dF_ptr, k_ptr, kap_ptr, clamp_ptr, n_l, T,
                BLOCK_L: tl.constexpr, BLOCK_T: tl.constexpr):
    tl_t = tl.program_id(0)
    offs_l = tl.arange(0, BLOCK_L)
    mask_l = offs_l < n_l
    t = tl_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t < T
    m = mask_l[:, None] & mask_t[None, :]
    idx = offs_l[:, None] * T + t[None, :]
    f = tl.load(F_ptr + idx, mask=m, other=0.0)
    # neighbours (up/down one multipole)
    f_dn = tl.load(F_ptr + idx - T, mask=m & (offs_l[:, None] > 0), other=0.0)
    f_up = tl.load(F_ptr + idx + T, mask=m & (offs_l[:, None] < n_l - 1), other=0.0)
    k = tl.load(k_ptr + t, mask=mask_t, other=0.0)[None, :]
    kap = tl.load(kap_ptr + t, mask=mask_t, other=0.0)[None, :]
    cl = tl.load(clamp_ptr + t, mask=mask_t, other=0)[None, :]
    lf = offs_l[:, None].to(tl.float32)
    kk = k * lf / (2.0 * lf + 1.0)
    kk2 = k * (lf + 1.0) / (2.0 * lf + 1.0)
    val = kk * f_dn - kk2 * f_up - kap * f
    val = tl.where(cl != 0, 0.0, val)
    tl.store(dF_ptr + idx, val, mask=m)

torch.manual_seed(0)
T, n_l = 160, 24
F = torch.randn(n_l, T, device="cuda", dtype=torch.float32)
dF = torch.empty_like(F)
k = torch.rand(T, device="cuda", dtype=torch.float32) * 0.1
kap = torch.rand(T, device="cuda", dtype=torch.float32) * 50
clamp = (torch.rand(T, device="cuda") > 0.7).to(torch.int32)

BL = triton.next_power_of_2(n_l)
grid = (triton.cdiv(T, 32),)
hier_kernel[grid](F, dF, k, kap, clamp, n_l, T, BLOCK_L=BL, BLOCK_T=32)
torch.cuda.synchronize()

# reference in torch
lf = torch.arange(n_l, device="cuda", dtype=torch.float32)[:, None]
kk = k[None, :] * lf / (2 * lf + 1)
kk2 = k[None, :] * (lf + 1) / (2 * lf + 1)
f_dn = torch.zeros_like(F); f_dn[1:] = F[:-1]
f_up = torch.zeros_like(F); f_up[:-1] = F[1:]
ref = torch.where(clamp[None, :] != 0, torch.zeros_like(F), kk * f_dn - kk2 * f_up - kap[None, :] * F)
print("max abs diff vs torch:", float((dF - ref).abs().max()))

# timing: 200 iterations of the kernel vs the torch equivalent (batch of ops)
N = 200
for _ in range(3): hier_kernel[grid](F, dF, k, kap, clamp, n_l, T, BLOCK_L=BL, BLOCK_T=32)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N): hier_kernel[grid](F, dF, k, kap, clamp, n_l, T, BLOCK_L=BL, BLOCK_T=32)
torch.cuda.synchronize()
t_triton = (time.perf_counter() - t0) / N
t0 = time.perf_counter()
for _ in range(N):
    ref = torch.where(clamp[None, :] != 0, torch.zeros_like(F), kk * f_dn - kk2 * f_up - kap[None, :] * F)
torch.cuda.synchronize()
t_torch = (time.perf_counter() - t0) / N
print(f"triton 1 kernel : {t_triton*1e6:8.1f} us/step")
print(f"torch  ~7 ops   : {t_torch*1e6:8.1f} us/step   speedup {t_torch/t_triton:.2f}x")
