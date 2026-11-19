# TenBol

TenBol is a Python package for batched flat ΛCDM calculations. It returns
linear transfer functions for total matter, cold dark matter, and baryons, and
computes nonlinear matter power. It supports massive neutrinos and runs on
CPUs with NumPy or on supported NVIDIA GPUs with CUDA.

TenBol has achieved less than 0.62% relative error in nonlinear matter power
compared with CAMB in testing.

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
