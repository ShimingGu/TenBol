"""Typed failures, following §13 of the design.

Invalid physical parameters must never be silently converted into a
numerically plausible spectrum, and numerical failures must be distinguishable
from prior exclusions.
"""


class TenbolError(Exception):
    """Base class for all Tenbol failures."""


class InvalidCosmology(TenbolError):
    """Parameter values violate the model's physical constraints."""


class UnsupportedConfiguration(TenbolError):
    """Physics/output/backend combination is not implemented or supported."""


class ProfileDomainError(TenbolError):
    """Request lies outside the validated numerical domain of the profile."""


class NumericalFailure(TenbolError):
    """Non-finite state, unconverged solve, or step/iteration limit."""


class ExecutionFailure(TenbolError):
    """Device memory, compilation or backend execution failure."""


class PartialBatchFailure(TenbolError):
    """One or more rows of a batch failed; per-row statuses are preserved."""

    def __init__(self, message, statuses, result=None):
        super().__init__(message)
        self.statuses = statuses
        self.result = result
