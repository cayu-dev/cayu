from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu.budgets import BudgetLimit, default_price_book
from cayu.budgets.base import MAX_REQUEST_BUDGET_LIMITS
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import MAX_STEPS
from cayu.runtime.work_attempt_semantics import (
    WORK_ATTEMPT_RUN_SEMANTICS_MAX_BUDGETS,
    WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES,
    WORK_ATTEMPT_RUN_SEMANTICS_MAX_ITEMS,
    WorkAttemptRunSemantics,
    copy_work_attempt_run_semantics,
)
from cayu.runtime.work_attempt_source import (
    WORK_ATTEMPT_SOURCE_MAX_BYTES,
    WORK_ATTEMPT_SOURCE_MAX_ITEMS,
    WorkAttemptSourceRequest,
)
from cayu.sessions.requests import RunRequest
from cayu.tasks.admission import WORK_ATTEMPT_ADMISSION_MAX_BYTES, WORK_ATTEMPT_ADMISSION_MAX_ITEMS
from cayu.tasks.contracts import WORK_COMPLETION_DECISION_MAX_BYTES


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


@pytest.mark.parametrize("value", [True, False, 0, MAX_STEPS + 1, "7"])
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


def _small_price_book() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )


@pytest.mark.parametrize(
    "budget_limits",
    [
        # Two limits priced with the bundled book: about 72 KB of settings.
        [
            BudgetLimit(max_estimated_cost=Decimal(cost), pricing=default_price_book())
            for cost in ("5", "50")
        ],
        # More limits than the old private 64-entry cap.
        [
            BudgetLimit(max_estimated_cost=Decimal(index + 1), pricing=_small_price_book())
            for index in range(65)
        ],
    ],
    ids=["bundled-price-books", "many-limits"],
)
def test_work_attempt_semantics_hold_what_governed_source_admission_accepts(
    budget_limits,
) -> None:
    request = RunRequest(agent_name="assistant", messages=[], budget_limits=budget_limits)
    source = WorkAttemptSourceRequest.capture(
        kind="initial",
        request=request.model_dump(mode="json", warnings=False),
        fields_set=tuple(sorted(request.model_fields_set)),
        source_request_sha256="0" * 64,
    )
    semantics = WorkAttemptRunSemantics(
        max_steps=request.max_steps,
        budget_limits=request.budget_limits,
        request_metadata=request.metadata,
    )
    assert source.request["budget_limits"]
    assert semantics.budget_limits == request.budget_limits


def test_work_attempt_semantics_bounds_derive_from_owner_limits() -> None:
    assert WORK_ATTEMPT_RUN_SEMANTICS_MAX_BUDGETS == MAX_REQUEST_BUDGET_LIMITS
    assert WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES > WORK_ATTEMPT_SOURCE_MAX_BYTES
    assert WORK_ATTEMPT_RUN_SEMANTICS_MAX_ITEMS > WORK_ATTEMPT_SOURCE_MAX_ITEMS
    # The durable admission holds a full source snapshot and full run settings.
    assert WORK_ATTEMPT_ADMISSION_MAX_BYTES >= (
        WORK_COMPLETION_DECISION_MAX_BYTES
        + WORK_ATTEMPT_SOURCE_MAX_BYTES
        + WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES
    )
    assert WORK_ATTEMPT_ADMISSION_MAX_ITEMS >= (
        WORK_ATTEMPT_SOURCE_MAX_ITEMS + WORK_ATTEMPT_RUN_SEMANTICS_MAX_ITEMS
    )
