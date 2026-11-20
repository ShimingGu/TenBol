# TenBol

TenBol stands for **TENsor BOLtzmann code**. It's a Python package for predicting
how matter clumps as the universe evolves. It can run several flat ΛCDM models
at once and returns linear transfer functions for total matter, cold dark
matter, and baryons, plus nonlinear matter power. It supports massive neutrinos
and runs on CPUs with NumPy or NVIDIA GPUs with CUDA.

TenBol has achieved less than 0.62% relative error in nonlinear matter power
compared with CAMB in testing. For a batch of two cosmologies, TenBol was
**1.9× as fast as CAMB** for nonlinear matter power, taking 15.5 seconds on one
NVIDIA H100 slice versus 29.9 seconds on two CPU workers.

## Install

From the repository directory, run this with Python 3.10 or newer.

```bash
python -m pip install . "numpy>=2" "scipy>=1.12"
```

For an NVIDIA system that supports CUDA 13, install the CUDA extra.

```bash
python -m pip install ".[cuda]" "numpy>=2" "scipy>=1.12"
```

## Calculate matter power

```python
from tenbol.physics.background import CosmologyBatch
from tenbol.power import matter_power, matter_power_nl

cosmologies = CosmologyBatch.from_columns(
    h=[0.67, 0.70],
    omega_b=[0.0224, 0.0224],
    omega_c=[0.120, 0.125],
    A_s=[2.1e-9, 2.1e-9],
    n_s=[0.965, 0.970],
    m_nu=[[0.06], [0.10]],
)

k = [0.01, 0.1, 1.0]  # h/Mpc
z = [0.0, 1.0]
backend = "numpy"  # Set to "cuda" to use an NVIDIA GPU.

linear = matter_power(cosmologies, k, z, backend=backend)
nonlinear = matter_power_nl(cosmologies, k, z, backend=backend)

print(linear.delta_m.shape)  # (2, 3, 2)
print(nonlinear.shape)      # (2, 3, 2)
```

The result axes are cosmology, wavenumber, and redshift. The linear result
provides `delta_m`, `delta_cdm`, and `delta_b`. Nonlinear power is returned in
`(Mpc/h)^3`. Each row in `m_nu` lists the neutrino masses in eV for one
cosmology.

## Use TenBol with an MCMC sampler

The sampler proposes values for your model parameters. Call TenBol with the
cosmological parameters to get matter power `P(k, z)` on the grid your analysis
needs. Your own model takes that spectrum and any survey parameters and
predicts what you measured, whether that's galaxy clustering, shear, or
something else. Your likelihood compares those predictions with the data.

When several proposals are ready together, pass their cosmologies to TenBol
in one call. The spectra come back in proposal order. Keep that order as your
model predicts the measurements and you return one log probability per
proposal. In `emcee`, `vectorize=True` tells the sampler to call your log
probability function with a batch of proposals.
[emcee sampler guide](https://emcee.readthedocs.io/en/stable/user/sampler/)
