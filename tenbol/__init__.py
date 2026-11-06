"""Tenbol: modular, batched Einstein-Boltzmann solver.

The public entry points are :func:`tenbol.api.compile_model`,
:func:`tenbol.api.evaluate` and :class:`tenbol.api.BatchExecutor`, mirroring
the design in ``Tenbol_structure.md``.  Physics definitions live in
``tenbol.physics``; numerics in ``tenbol.numerics``; device code generation in
``tenbol.backends``; observables and the nonlinear provider in their own
packages.  CAMB is only ever imported from ``tenbol.adapters`` for validation.
"""

from . import errors  # noqa: F401
from ._version import __version__  # noqa: F401

__all__ = ["__version__", "errors"]
