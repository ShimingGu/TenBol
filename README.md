# TenBol

TenBol stands for **TENsor BOLtzmann code**. It's a Python package for predicting
how matter clumps as the universe evolves. It can run several flat ΛCDM models
at once and returns linear transfer functions for total matter, cold dark
matter, and baryons, plus nonlinear matter power. It supports massive neutrinos
and runs on CPUs with NumPy or NVIDIA GPUs with CUDA.

TenBol has achieved less than 0.62% relative error in nonlinear matter power
compared with CAMB in testing.

## How we made it faster

We moved the part that follows the growth of density fluctuations to the GPU.
Each GPU thread follows one scale in one cosmology. Some scales need short
time steps. Others can take longer steps and finish sooner. The intermediate
results stay on the GPU until the calculation finishes.

The CPU works out the expansion and temperature history once per cosmology
and shares it across all scales. It applies the nonlinear correction at the end.
We switched the temperature calculations that work on one number at a time
to simpler math calls. We also moved more of the calculation points to scales
where the power spectrum changes quickly, keeping the total number of points
the same. On one H100 benchmark, median time for the same batch fell from
37.2 to 29.9 seconds.

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

The sampler proposes cosmologies and decides which proposals to accept.
TenBol predicts the matter power for those proposals. Your likelihood compares
each prediction with your data and gives the sampler one score per proposal.

Send the proposals that are ready together in one call. For example, tell
`emcee` to pass a group of walkers to your likelihood with `vectorize=True`.
Each row of the returned power array belongs to the corresponding proposal.
This lets the GPU work on several proposals at once. With a single chain,
each proposal depends on the result of the previous step.
[emcee sampler guide](https://emcee.readthedocs.io/en/stable/user/sampler/)

Install `emcee` with `python -m pip install emcee` to try the example below.
It uses the first spectrum from the example above as mock data and varies
`h` and the cold dark matter density, `omega_c`.

```python
import emcee
import numpy as np

data = nonlinear[0]
error = 0.05 * data


def log_probability(proposals):
    h, omega_c = np.asarray(proposals).T
    in_prior = (0.60 < h) & (h < 0.80) & (0.10 < omega_c) & (omega_c < 0.16)
    scores = np.full(len(h), -np.inf)
    if in_prior.any():
        batch = CosmologyBatch.from_columns(
            h=h[in_prior], omega_c=omega_c[in_prior],
            omega_b=0.0224, A_s=2.1e-9, n_s=0.965, m_nu=0.06,
        )
        prediction = matter_power_nl(batch, k, z, backend=backend)
        residual = (prediction - data) / error
        scores[in_prior] = -0.5 * np.sum(residual**2, axis=(1, 2))
    return scores


rng = np.random.default_rng(42)
start = rng.normal(loc=[0.67, 0.12], scale=[0.005, 0.002], size=(8, 2))
sampler = emcee.EnsembleSampler(8, 2, log_probability, vectorize=True)
sampler.run_mcmc(start, 1)
```

For a real fit, replace the mock data and error model with yours and set the
priors and number of steps for your analysis. The same pattern works with any
sampler that can score several proposals in one call.
