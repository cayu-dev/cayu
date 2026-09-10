from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.work_attempt_semantics import (
    WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES,
    WorkAttemptRunSemantics,
    copy_work_attempt_run_semantics,
)


def test_work_attempt_semantics_round_trip_preserves_absolute_deadline() -> None:
    expiry = datetime.now(UTC) + timedelta(minutes=5)
    semantics = WorkAttemptRunSemantics(
        max_steps=7,
        limits=RunLimits(max_total_tokens=100, scope="session"),
        deadline_expires_at=expiry,
        request_metadata={"job": {"version": 1}},
    )
    restored = WorkAttemptRunSemantics.model_validate_json(semantics.model_dump_json())
    assert restored == semantics
    assert restored.deadline.expires_at == expiry
    assert restored.limits.scope == "session"


def test_work_attempt_semantics_detaches_source_and_copies() -> None:
    limits = RunLimits(max_total_tokens=100)
    metadata = {"job": {"version": 1}}
    semantics = WorkAttemptRunSemantics(max_steps=7, limits=limits, request_metadata=metadata)
    copied = copy_work_attempt_run_semantics(semantics)
    limits.max_total_tokens = 200
    metadata["job"]["version"] = 2
    semantics.limits.max_total_tokens = 300
    semantics.request_metadata["job"]["version"] = 3
    assert copied.limits.max_total_tokens == 100
    assert copied.request_metadata == {"job": {"version": 1}}


@pytest.mark.parametrize("value", [True, False, 0, 257, "7"])
def test_work_attempt_semantics_requires_bounded_strict_steps(value) -> None:
    with pytest.raises(ValidationError):
        WorkAttemptRunSemantics(max_steps=value)


def test_work_attempt_semantics_revalidates_mutated_nested_settings() -> None:
    semantics = WorkAttemptRunSemantics(max_steps=7)
    semantics.limits.max_total_tokens = True
    with pytest.raises(ValidationError):
        copy_work_attempt_run_semantics(semantics)


def test_work_attempt_semantics_bounds_portable_document() -> None:
    with pytest.raises(ValueError):
        WorkAttemptRunSemantics(
            max_steps=7,
            request_metadata={"oversized": "x" * WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES},
        )
