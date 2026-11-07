# GPU/tensor-native Boltzmann solvers: what to borrow

Survey date: 2026-11-07. Scope: projects whose *primary* design target is
GPU/tensor execution of the Einstein–Boltzmann system (not CPU codes with a GPU
port). For each, what it does, what is transferable to Tenbol, and what is not.

## 1. DISCO-EB / DISCO-DJ I (JAX + diffrax)

Hahn, List & Porqueres, JCAP 06 (2024) 063, arXiv:2311.03291;
<https://github.com/ohahn/DISCO-EB> (GPL-3.0).
The closest analogue to Tenbol: same equations, same target (matter power for
inference), tensor-native.

Design choices worth knowing:

* **Vectorises over k with `jax.vmap`**, one cosmology (or a parameter batch)
  at a time; the whole time loop is `jax.jit`-compiled, so per-step Python
  overhead is zero.
* **Uses a high-order implicit Runge–Kutta (diffrax) for the stiff system**
  instead of a hand-rolled explicit scheme. The Jacobian is obtained by
  autodiff, and they state the Jacobian cost grows as (number of equations)²:
  `8 + 2*l_gamma + l_nuUR + n_q*l_nuMassive`.
* **Reports that on GPU they use *no* approximation schemes**: they implemented
  tight coupling, the ultra-relativistic fluid approximation and the radiation
  streaming approximation, measured no O(1) gain, and therefore do not use
  them. The argument: branchy, criterion-driven variable switching adds
  algorithmic complexity that is hostile to GPU execution, while an implicit
  integrator handles the stiffness directly.
* **Neutrino momentum bins**: 3–5 Gauss–Laguerre-inspired nodes (CAMB's
  kernel), not a dense quadrature; `n_q` multiplies the state size and the
  Jacobian cost, so it is kept minimal.

Transferable to Tenbol:

1. **Kill the regime-switch machinery, not the stiffness.** Tenbol currently
   carries tight coupling + RSA + step-aligned events (decisions D5/D8), which
   is exactly the branchy structure DISCO-EB abandoned on GPU. The replacement
   is a stiff-stable integrator — for us the cheap version is a **2×2 analytic
   exponential (ETD) for the Thomson-coupled `(vb, qg)` block plus exact decay
   for the polarized moments**, which needs no Jacobian and no branches.
2. **Reduce the neutrino quadrature** from our 12 Laguerre nodes to 3–5 CAMB
   kernels. This cuts the state dimension (currently 229) by ~100 rows and all
   per-step work proportionally.
3. **Follow their equation-count budget** when choosing `lmax`: the dominant
   cost is quadratic in the state size for implicit methods and linear for
   explicit ones, so the optimum differs — measure, do not assume.

Not transferable / caveats: their accuracy validation is for CMB and P(k) at
percent level with `rtol=1e-3` defaults; our 0.5% nonlinear criterion needs a
tighter linear budget, and diffrax's implicit solvers are slower for very small
batches than a fused explicit kernel (see §3).

## 2. clax (JAX, CLASS reimplementation)

<https://github.com/smsharma/clax>. Same physics family, JAX end-to-end,
documented accuracy vs CLASS: sub-0.2% C_l for l = 20–1200, full ncdm
hierarchy (5 momentum bins × 18 multipoles), `jax_enable_x64`, 487 s on an H100
for the science preset.

Transferable:

* **Dense output, not step targeting.** clax and DISCO-EB both interpolate the
  solution to the requested output times instead of forcing the integrator to
  land on them. This is precisely the bug found in Tenbol on 2026-11-06:
  requesting `z = 1000` among the outputs changed `delta(z=0)` by three orders
  of magnitude because output times were used as step targets. Fix: integrate
  with the physics-based step only, and interpolate (cubic Hermite from the
  step's endpoint derivatives) to the requested scale factors.
* **Precision presets as first-class objects** (`fit_cl`, `planck_cl`, …) with
  measured accuracy per preset — the same pattern as our
  `NumericalProfile`, and evidence that profiles must be validated per regime.
* **`compute_pk_table` / interpolator API**: the nonlinear provider consumes a
  validated interpolator, not a raw array. Matches our
  `LinearSpectrumBundle` contract in `Tenbol_structure.md` §10.1.

Caveat: clax is HaloFit-based, so its nonlinear step is not a drop-in
comparison for our HMcode target.

## 3. "GPU-accelerating CMB cosmology with state-of-the-art ODE solvers"
(Berry et al., Cambridge; Cosmo-26 talk)

They report **new ODE solvers optimised for "many small ODEs at once", ~100×
faster than diffrax**, quoting the same structure Tenbol needs: one small ODE
per (k-mode, parameter) pair, batched. Two concrete lessons:

* The winning abstraction is a **batched stepper over independent trajectories
  with per-trajectory step sizes**, not a single global step and not a
  per-trajectory Python loop.
* Pure-JAX generic solvers leave a large constant factor on the table; a
  problem-specific fused kernel is worth writing once the algorithm is fixed.

Tenbol consequence: keep the plan of a per-trajectory adaptive/fixed stepper
(decision D7), but implement it as **buckets of trajectories sharing a step
size**, so a bucket's kernel is branch-free. Our current global-minimum step is
the thing to replace.

## 4. Bolt.jl (Julia)

<https://github.com/xzackli/Bolt.jl>. Differentiable Julia solver (diffrax-like
abstractions, GPU-capable). Useful for equation organisation and for its
gradient checks; it is not a GPU benchmark for our workload. Borrowed idea:
explicit declaration of which quantities are differentiated per stage, which
matches our planned `SensitivitySpec`.

## 5. XLB / Lettuce (GPU lattice-Boltzmann frameworks)

<https://github.com/Autodesk/XLB>, <https://github.com/lettucecfd/lettuce>.
Not the cosmological hierarchy, but they demonstrate the *execution* pattern
Tenbol wants for the hierarchy: a dense state array with **variable-major**
layout, one fused kernel per stage, and no host round-trip inside the step
loop. Our `Y[D, T]` layout (`tenbol/physics/state.py`) follows that pattern.

## What Tenbol adopts, in order

| # | Action | Source | Rationale |
| --- | --- | --- | --- |
| 1 | Dense output (Hermite) instead of output times as step targets | clax, DISCO-EB | removes the measured output-grid dependence |
| 2 | 2×2 analytic exponential for the Thomson `(vb, qg)` block; exact decay for polarized moments | DISCO-EB's "no approximations on GPU" | stiffness handled without branches or events |
| 3 | 3–5 neutrino momentum bins | DISCO-EB / CAMB kernel | state 229 → ~120 rows |
| 4 | Step-size buckets per trajectory group | Berry et al. | branch-free kernels, no global-minimum step |
| 5 | Keep `Y[D,T]` variable-major + fixed physics schedule | XLB/Lettuce pattern | already implemented |
| 6 | Profile presets with per-regime validation evidence | clax | already in the design |

Explicitly *not* adopted: JAX/diffrax as the production path. The immediate
goal is a correct explicit/ETD CPU reference; autodiff and Metal/CUDA codegen
come after the linear spectrum is validated (the same order DISCO-EB followed:
MPC-only scope first, gradients second).
