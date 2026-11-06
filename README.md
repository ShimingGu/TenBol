# Tenbol

A modular, batch-native Einstein–Boltzmann framework for MCMC matter power,
implemented in Python. This repository is a **work in progress**: the
background/thermal layers are implemented and validated against CAMB 2.0.5; the
perturbation solver is implemented but **not yet validated**, and everything
downstream of it (linear spectrum assembly, HMcode, batch pipeline, CUDA/MPS
backends, performance work) is not built. See "Status" below for the precise,
evidence-backed account.

## Design reference

The design documents live at the repository root:

* `Tenbol_idea.md` — feasibility audit and evidence boundaries;
* `Tenbol_structure.md` — module contracts, physics conventions, acceptance;
* `Tenbol_execution_prompt.md` — the implementation task and its rules;
* `Tenbol_previous_camb_attempt.md` — the earlier CAMB GPU port and its lessons.

## Install / run

```bash
cd /scratch/Tenbol
python3.13 -m pip install numpy scipy pytest camb   # camb only for validation
export PYTHONPATH=.
python3.13 -m pytest tests/tenbol -q
```

The environment used for the recorded results (see `Tenbol_progress.md`):
Python 3.13.17 at `/usr/local/bin/python3.13`, numpy 2.5.4, scipy 1.18.1,
camb 2.0.5, torch 2.11.0+cu128, 96 logical CPUs, 3× H100 NVL MIG slices.
No Apple hardware is available, so MPS can only be implemented and handed off.

## Package layout

```
tenbol/
  constants.py          units and CAMB-aligned constants
  errors.py             typed failures (invalid cosmology / numerical / execution)
  physics/background.py flat LCDM + thermal massive neutrinos, tau(a) table
  physics/bbn.py        BBN helium (fit to CAMB's PRIMAT predictor)
  physics/thermal.py    RECFAST-level recombination, reionization, opacity, c_s^2
  physics/perturbations.py  synchronous-gauge hierarchy, tight coupling, RSA, RK4
  adapters/camb_reference.py  CAMB validation adapter (never used in production)
profiles/tenbol/        workload manifest + evidence JSON
benchmarks/tenbol/      reference convergence, linear validation, CPU baseline
tests/tenbol/           unit and validation tests
```

## Usage (current, partial)

```python
import numpy as np
from tenbol.physics.background import Background, CosmologyBatch
from tenbol.physics.thermal import ThermalHistory, Reionization

batch = CosmologyBatch.from_columns(
    h=[0.67], omega_b=[0.0224], omega_c=[0.120],
    A_s=[2.1e-9], n_s=[0.965], m_nu=[[0.06]], N_eff=[3.046],
)
bg = Background(batch)                       # H(a), tau(a), densities
th = ThermalHistory(bg, Reionization(tau=0.0544))   # x_e(a), T_m, opacity, c_s^2
```

The perturbation layer (`tenbol.physics.perturbations.PerturbationBatch`,
`evolve`) is not part of the supported public API yet.

## Status

Verified against CAMB 2.0.5 (see `Tenbol_progress.md` for commands and numbers):

| Layer | State |
| --- | --- |
| Workload manifest, reference profile, convergence/sensitivity evidence | done |
| Background (H(a), conformal time, neutrino thermodynamics, degeneracies) | validated to 3e-4 in H, ~1e-6 in chi(z) |
| BBN helium | fit to CAMB's predictor, residual 1.6e-6 |
| Thermal history (RECFAST-level) | validated through the sound horizon: r\*(z\*) = 144.41 Mpc vs CAMB 144.41 (4e-5) |
| Perturbation solver | implemented (photons+polarization, massless/massive neutrinos, tight coupling, RSA, per-trajectory RK4); **low-k transfer shape correct, high-k incorrect, k ≳ 0.5 h/Mpc unstable** |
| Linear P(k), sigma(R), HMcode-2020 | not implemented |
| Batch API, CPU pipeline, CUDA, MPS, performance comparison | not implemented |

The acceptance criterion of the task (max 0.5% relative difference in final
nonlinear power including HMcode, per backend) is **not met and not claimed**.

## Benchmarks

```bash
PYTHONPATH=. python3.13 benchmarks/tenbol/reference_convergence.py --out profiles/tenbol/evidence
PYTHONPATH=. python3.13 benchmarks/tenbol/linear_validation.py --out profiles/tenbol/evidence
PYTHONPATH=. python3.13 benchmarks/tenbol/baseline_cpu.py --batch 32 --workers 8 16
```

`reference_convergence.py` establishes that the reference itself is converged
(AccuracyBoost 2 vs 3: 1e-4) and quantifies the sensitivity to thermal-history
and background settings. `linear_validation.py` records the candidate/reference
linear-power comparison (currently failing; the file is evidence of the
diagnosis). `baseline_cpu.py` measures the tuned parallel-CPU CAMB baseline;
it must be run with process isolation because CAMB plus `fork` deadlocks with
BLAS/OpenMP in this container.
