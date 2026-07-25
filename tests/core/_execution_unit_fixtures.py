from __future__ import annotations

from cayu.runtime.execution_units import ModelAttemptIdentity, new_model_step_identity


def model_attempt_identity() -> ModelAttemptIdentity:
    """Return an exact provider-attempt identity for low-level ledger tests."""

    return new_model_step_identity().new_attempt()
