"""Dense state tensor for the perturbation hierarchy.

One array `Y` of shape ``(D, T)`` holds every trajectory's full state:

* ``T`` is the flattened trajectory batch ``(sample, k)``;
* ``D`` is the state dimension, laid out **variable-major** so that row ``d``
  is contiguous across trajectories.  On a device this gives coalesced
  loads/stores for the per-mode scalars, and the right-hand side becomes a
  fixed sequence of row-slice operations with per-trajectory scalar
  coefficients -- precisely the form a generated CUDA/Metal kernel iterates.

Block order (all offsets computed by :class:`StateLayout`)::

    [ scalars ]            dc db vb dg qg pig dr qr pir etak
    [ photon moments ]     Fg(l) for l = 3 .. lmax_g
    [ polarization ]       G(l)  for l = 2 .. lmax_pol
    [ massless neutrino ]  Fr(l) for l = 3 .. lmax_r
    [ massive neutrino ]   Fnu(s, q, l) for each species s, momentum q, l = 0..lmax_nu

The layout object is the single source of truth for indices: the physics code
addresses fields through :meth:`StateLayout.slices` and never through literal
offsets, so a kernel generator can emit the same access pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCALARS = ("dc", "db", "vb", "dg", "qg", "pig", "dr", "qr", "pir", "etak")


@dataclass(frozen=True)
class StateLayout:
    """Row layout of the dense perturbation state."""

    lmax_g: int = 24
    lmax_pol: int = 20
    lmax_r: int = 24
    lmax_nu: int = 12
    n_species: int = 1
    n_q: int = 12

    # derived, filled in __post_init__
    offsets: dict = field(default_factory=dict, compare=False, repr=False)
    size: int = field(default=0, compare=False)

    def __post_init__(self):
        if self.lmax_g < 3 or self.lmax_pol < 2 or self.lmax_r < 3 or self.lmax_nu < 2:
            raise ValueError("hierarchy capacities too small")
        off = {}
        n = len(SCALARS)
        off["scalars"] = slice(0, n)
        for name, l0, l1 in (
            ("Fg", 3, self.lmax_g),
            ("G", 2, self.lmax_pol),
            ("Fr", 3, self.lmax_r),
        ):
            off[name] = slice(n, n + (l1 - l0 + 1))
            n += l1 - l0 + 1
        off["Fnu"] = slice(n, n + self.n_species * self.n_q * (self.lmax_nu + 1))
        n = off["Fnu"].stop
        object.__setattr__(self, "offsets", off)
        object.__setattr__(self, "size", n)

    # -- scalar fields -------------------------------------------------
    def scalar(self, name: str) -> int:
        return SCALARS.index(name)

    # -- hierarchy rows -------------------------------------------------
    def fg(self, l: int) -> int:
        """Row of the photon moment ``F_gamma,l`` (3 <= l <= lmax_g)."""
        return self.offsets["Fg"].start + (l - 3)

    def g(self, l: int) -> int:
        """Row of the polarization moment ``G_l`` (2 <= l <= lmax_pol)."""
        return self.offsets["G"].start + (l - 2)

    def fr(self, l: int) -> int:
        """Row of the massless-neutrino moment ``F_r,l`` (3 <= l <= lmax_r)."""
        return self.offsets["Fr"].start + (l - 3)

    def fnu(self, s: int, q: int, l: int) -> int:
        """Row of the massive-neutrino moment ``F_{s,q,l}``."""
        return (self.offsets["Fnu"].start
                + (s * self.n_q + q) * (self.lmax_nu + 1) + l)

    # -- bulk access ----------------------------------------------------
    def allocate(self, n_traj: int, dtype=np.float64) -> np.ndarray:
        return np.zeros((self.size, n_traj), dtype=dtype)

    def zeros_like(self, Y: np.ndarray) -> np.ndarray:
        return np.zeros_like(Y)

    def pack(self, fields: dict) -> np.ndarray:
        """Convert a dict of named blocks into the dense tensor."""
        scal = np.stack([np.asarray(fields[name]) for name in SCALARS], axis=0)
        n = scal.shape[1]
        Y = np.zeros((self.size, n), dtype=scal.dtype)
        Y[self.offsets["scalars"]] = scal
        Y[self.offsets["Fg"]] = fields["Fg"]
        Y[self.offsets["G"]] = fields["G"]
        Y[self.offsets["Fr"]] = fields["Fr"]
        Y[self.offsets["Fnu"]] = fields["Fnu"].reshape(
            self.n_species * self.n_q * (self.lmax_nu + 1), n)
        return Y

    def unpack(self, Y: np.ndarray) -> dict:
        """Inverse of :meth:`pack` (views, not copies)."""
        out = {name: Y[self.scalar(name)] for name in SCALARS}
        out["Fg"] = Y[self.offsets["Fg"]]
        out["G"] = Y[self.offsets["G"]]
        out["Fr"] = Y[self.offsets["Fr"]]
        out["Fnu"] = Y[self.offsets["Fnu"]].reshape(
            self.n_species, self.n_q, self.lmax_nu + 1, Y.shape[1])
        return out

    # -- description for kernel generation ------------------------------
    def describe(self) -> dict:
        """Machine-readable layout description (consumed by backend codegen)."""
        return {
            "size": self.size,
            "scalars": {name: self.scalar(name) for name in SCALARS},
            "blocks": {
                "Fg": {"start": self.offsets["Fg"].start, "l0": 3, "l1": self.lmax_g},
                "G": {"start": self.offsets["G"].start, "l0": 2, "l1": self.lmax_pol},
                "Fr": {"start": self.offsets["Fr"].start, "l0": 3, "l1": self.lmax_r},
                "Fnu": {"start": self.offsets["Fnu"].start, "lmax": self.lmax_nu,
                        "n_species": self.n_species, "n_q": self.n_q},
            },
        }
