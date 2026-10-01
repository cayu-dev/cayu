"""Provider rejections during automatic compaction keep their classification."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent, OpenAIAPIError
from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import IncompleteSessionRecoveryRequest, RunRequest, SessionStore
from cayu.storage import SQLiteSessionStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate, TaskStatus, TaskStore
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


class AssistantProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:compaction-rejection-assistant",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.completed({})


class RejectingCompactionProvider(ModelProvider):
    name = "openai"

    def __init__(self, error: OpenAIAPIError) -> None:
        self.error = error
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:compaction-rejection-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        raise self.error
        yield


def _rate_limit() -> OpenAIAPIError:
    return OpenAIAPIError(
        "Rate limit reached for requests",
        status_code=429,
        error_type="rate_limit_error",
        error_code="rate_limit_exceeded",
        retryable=True,
        retry_after_s=0.05,
    )


def _app(
    provider: RejectingCompactionProvider,
    *,
    max_attempts: int,
    budget_policy: BudgetPolicy | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> CayuApp:
    app = CayuApp(
        enable_logging=False,
        budget_policy=budget_policy,
        session_store=session_store,
        task_store=task_store,
        secret_redactor=secret_redactor,
    )
    app.register_provider(AssistantProvider(), default=True)
    app.register_provider(provider)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=max_attempts, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    return app


def _run(app: CayuApp, session_id: str, *, task_id: str | None = None) -> list[Event]:
    async def run() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    task_id=task_id,
                    messages=[
                        Message.text("user", "old"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current"),
                    ],
                )
            )
        ]

    return asyncio.run(run())


def _compaction_events(events: list[Event], event_type: EventType) -> list[Event]:
    return [
        event
        for event in events
        if event.type is event_type and event.payload.get("purpose") == "context_compaction"
    ]


def test_retryable_compaction_rejection_is_classified_and_backs_off() -> None:
    provider = RejectingCompactionProvider(_rate_limit())
    app = _app(provider, max_attempts=3)

    returned = _run(app, "sess_compaction_rate_limited")
    durable = asyncio.run(app.session_store.load_events("sess_compaction_rate_limited"))

    assert len(provider.requests) == 3
    for events in (returned, durable):
        started = _compaction_events(events, EventType.MODEL_STARTED)
        completed = _compaction_events(events, EventType.MODEL_COMPLETED)
        errors = _compaction_events(events, EventType.MODEL_ERROR)
        assert len(started) == len(completed) == len(errors) == 3
        # Each dispatched attempt keeps its unknown-usage accounting record.
        for record in completed:
            assert record.payload["compaction_outcome"] == "provider_error"
            assert record.payload["usage_unavailable_reason"]
            assert "status_code" not in record.payload
        for attempt, (start, error) in enumerate(zip(started, errors, strict=True), start=1):
            assert error.payload["model_attempt_id"] == start.payload["model_attempt_id"]
            assert {
                key: error.payload[key]
                for key in (
                    "attempt",
                    "max_attempts",
                    "status_code",
                    "provider_error_type",
                    "provider_error_code",
                    "retryable",
                    "retry_after_s",
                    "provider_retryable",
                    "reason",
                    "retry",
                    "retry_disposition",
                )
            } == {
                "attempt": attempt,
                "max_attempts": 3,
                "status_code": 429,
                "provider_error_type": "rate_limit_error",
                "provider_error_code": "rate_limit_exceeded",
                "retryable": True,
                "retry_after_s": 0.05,
                "provider_retryable": True,
                "reason": "http_status",
                "retry": attempt < 3,
                "retry_disposition": (
                    "retry_scheduled" if attempt < 3 else "configured_attempt_exhaustion"
                ),
            }
            assert "Rate limit reached" not in error.payload["error"]
        # The next dispatch waits for the provider's Retry-After directive.
        for error, next_start in zip(errors, started[1:], strict=False):
            assert (next_start.timestamp - error.timestamp).total_seconds() >= 0.05

        [failed] = [event for event in events if event.type is EventType.CONTEXT_COMPACTION_FAILED]
        assert {
            key: failed.payload[key]
            for key in (
                "phase",
                "reason",
                "provider_dispatch_disposition",
                "status_code",
                "provider_error_type",
                "provider_error_code",
                "provider_retryable",
                "retry_disposition",
                "chunk_count",
            )
        } == {
            "phase": "provider_dispatch",
            "reason": "provider_failed",
            "provider_dispatch_disposition": "dispatched",
            "status_code": 429,
            "provider_error_type": "rate_limit_error",
            "provider_error_code": "rate_limit_exceeded",
            "provider_retryable": True,
            "retry_disposition": "configured_attempt_exhaustion",
            "chunk_count": 3,
        }
        terminal = events[-1]
        assert terminal.type is EventType.SESSION_FAILED
        for key in (
            "status_code",
            "provider_error_type",
            "provider_error_code",
            "provider_retryable",
            "retry_disposition",
        ):
            assert terminal.payload["compaction_failure"][key] == failed.payload[key]
    terminal = returned[-1]
    assert terminal.type is EventType.SESSION_FAILED
    assert terminal.payload["compaction_failure"]["reason"] == "provider_failed"
    assert terminal.payload["compaction_failure"]["provider_dispatch_disposition"] == ("dispatched")


def test_permanent_compaction_rejection_is_classified_without_retry() -> None:
    provider = RejectingCompactionProvider(
        OpenAIAPIError(
            "Request too large",
            status_code=400,
            error_type="invalid_request_error",
            error_code="context_length_exceeded",
            retryable=False,
        )
    )
    app = _app(provider, max_attempts=5)

    events = _run(app, "sess_compaction_rejected")

    assert len(provider.requests) == 1
    [error] = _compaction_events(events, EventType.MODEL_ERROR)
    assert error.payload["status_code"] == 400
    assert error.payload["provider_error_code"] == "context_length_exceeded"
    assert error.payload["retry"] is False
    assert error.payload["retry_disposition"] == "explicit_nonretryable"
    [failed] = [event for event in events if event.type is EventType.CONTEXT_COMPACTION_FAILED]
    assert failed.payload["reason"] == "provider_failed"
    assert failed.payload["status_code"] == 400
    assert failed.payload["provider_error_type"] == "invalid_request_error"
    assert failed.payload["provider_retryable"] is False
    assert failed.payload["retry_disposition"] == "explicit_nonretryable"


@pytest.mark.parametrize("secret", [None, "context_compaction", "purpose"])
def test_compaction_rejection_error_does_not_contradict_budget_settlement(
    secret: str | None,
) -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPrice.fixed(
                provider_name="openai",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    budget_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=pricing,
                reservation=BudgetReservation(max_input_tokens=1_000, max_output_tokens=1_000),
            ),
        )
    )
    provider = RejectingCompactionProvider(_rate_limit())
    app = _app(
        provider,
        max_attempts=2,
        budget_policy=budget_policy,
        secret_redactor=SecretRedactor(secret),
    )

    events = _run(app, "sess_budgeted_compaction_rejection")
    summary = asyncio.run(app.session_store.inspect_summary("sess_budgeted_compaction_rejection"))

    assert len(_compaction_events(events, EventType.MODEL_ERROR)) == 1
    assert summary.budget is not None
    assert summary.budget.reservation_count == 1
    assert summary.budget.conservative_reconciliation_count == 1
    # The compaction model.error is diagnostic; the completion settles the attempt.
    assert summary.budget.cost_state == "unpriced"


def test_terminal_compaction_provider_fields_keep_lifecycle_bounds() -> None:
    provider = RejectingCompactionProvider(
        OpenAIAPIError(
            "provider rejected request",
            status_code=400,
            error_type="x" * 513,
            error_code="invalid_request",
            request_id="provider-request-id",
            retryable=False,
        )
    )
    app = _app(provider, max_attempts=2)
    returned = _run(app, "sess_bounded_rejection")
    durable = asyncio.run(app.session_store.load_events("sess_bounded_rejection"))
    for events in (returned, durable):
        [failed] = [event for event in events if event.type is EventType.CONTEXT_COMPACTION_FAILED]
        for payload in (failed.payload, events[-1].payload["compaction_failure"]):
            assert payload["status_code"] == 400
            assert payload["provider_error_code"] == "invalid_request"
            assert payload["provider_retryable"] is False
            assert payload["retry_disposition"] == "explicit_nonretryable"
            assert "provider_error_type" not in payload
            assert "request_id" not in payload


@pytest.mark.parametrize("field", ["error_type", "error_code"])
def test_linked_task_compaction_failure_redacts_provider_fields_before_persistence(
    field: str,
) -> None:
    secret = "synthetic-compaction-provider-secret-2065"
    provider = RejectingCompactionProvider(
        OpenAIAPIError(
            "provider rejected request",
            status_code=400,
            retryable=False,
            error_type=secret if field == "error_type" else None,
            error_code=secret if field == "error_code" else None,
        )
    )
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(task_id="compaction-task", type="review", assigned_agent_name="assistant")
        )
    )
    app = _app(
        provider,
        max_attempts=1,
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
    )
    returned = _run(app, "sess_task_rejection", task_id="compaction-task")
    durable = asyncio.run(app.session_store.load_events("sess_task_rejection"))
    task = asyncio.run(task_store.load_task("compaction-task"))
    checkpoint = asyncio.run(app.session_store.load_checkpoint("sess_task_rejection"))

    assert task is not None and task.status is TaskStatus.FAILED
    assert task.error is not None
    assert task.error["compaction_failure"][f"provider_{field}"] == REDACTED_SECRET
    for events in (returned, durable):
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["compaction_failure"][f"provider_{field}"] == REDACTED_SECRET
        assert secret not in json.dumps([event.model_dump(mode="json") for event in events])
    assert secret not in json.dumps(task.error)
    assert secret not in json.dumps(checkpoint)


@pytest.mark.parametrize("boundary", ["completion_precommit", "completion_ack", "promotion_ack"])
def test_compaction_rejection_publication_failure_keeps_diagnostic(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = RejectingCompactionProvider(_rate_limit())
    app = _app(provider, max_attempts=3)
    store = app.session_store
    method = (
        "promote_model_completion_stage"
        if boundary == "promotion_ack"
        else "complete_model_completion_stage"
    )
    original = getattr(store, method)
    failures = 0

    async def fail_once(*args, **kwargs):
        nonlocal failures
        if failures:
            return await original(*args, **kwargs)
        failures += 1
        if boundary != "completion_precommit":
            await original(*args, **kwargs)
        raise ConnectionError("injected compaction publication failure")

    monkeypatch.setattr(store, method, fail_once)
    returned = _run(app, "sess_rejection_publication_failure")
    durable = asyncio.run(store.load_events("sess_rejection_publication_failure"))

    assert failures == 1
    assert len(provider.requests) == 1
    assert returned[-1].type is EventType.SESSION_FAILED
    for events in (returned, durable):
        [completion] = _compaction_events(events, EventType.MODEL_COMPLETED)
        [error] = _compaction_events(events, EventType.MODEL_ERROR)
        assert error.payload["model_attempt_id"] == completion.payload["model_attempt_id"]
        assert error.payload["status_code"] == 429
        assert error.payload["provider_error_code"] == "rate_limit_exceeded"
    assert asyncio.run(store.load_active_model_completion_stage(returned[-1].session_id)) is None


def _run_rejection_until_process_exit(database: str, crash_point: str) -> None:
    from tests.core.test_runtime import ProcessLossAutomaticCompactionStore

    store = ProcessLossAutomaticCompactionStore(database, crash_point=crash_point)
    _run(
        _app(RejectingCompactionProvider(_rate_limit()), max_attempts=3, session_store=store),
        "sess_rejection_process_loss",
    )
    raise AssertionError("the child did not reach the compaction crash point")


@pytest.mark.parametrize("crash_point", ["completed", "promoted"])
def test_compaction_rejection_diagnostic_survives_process_loss(
    tmp_path: Path, crash_point: str
) -> None:
    from tests.core.test_runtime import _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE

    database = tmp_path / "rejection.sqlite"
    root = Path(__file__).resolve().parents[2]
    child_environment = os.environ.copy()
    child_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(root / "src"), child_environment.get("PYTHONPATH")) if path
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.core.test_compaction_provider_rejection import "
            "_run_rejection_until_process_exit as run; "
            f"run({str(database)!r}, {crash_point!r})",
        ],
        cwd=root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == _AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE, child.stderr

    async def recover() -> None:
        clock = [datetime.now(UTC)]
        store = SQLiteSessionStore(database, ownership_clock=lambda: clock[0])
        try:
            execution = await store.inspect_session_execution("sess_rejection_process_loss")
            assert execution.lease_expires_at is not None
            # The child has exited; expire its lease using the store's test clock.
            clock[0] = execution.lease_expires_at + timedelta(seconds=1)
            provider = RejectingCompactionProvider(_rate_limit())
            restarted = _app(provider, max_attempts=3, session_store=store)
            active = await store.load_active_model_completion_stage("sess_rejection_process_loss")
            if crash_point == "completed":
                assert active is not None
                assert active.stage.publication is not None
                [staged_error] = _compaction_events(
                    list(active.stage.publication.events), EventType.MODEL_ERROR
                )
                assert staged_error.payload["status_code"] == 429
            else:
                assert active is None
            with suppress(ModelCompletionManualRecoveryRequired):
                await restarted.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="sess_rejection_process_loss", inactive_for_seconds=0
                    )
                )
            durable = await store.load_events("sess_rejection_process_loss")
            [completion] = _compaction_events(durable, EventType.MODEL_COMPLETED)
            [error] = _compaction_events(durable, EventType.MODEL_ERROR)
            assert error.payload["status_code"] == 429
            assert error.payload["provider_error_code"] == "rate_limit_exceeded"
            assert error.payload["model_attempt_id"] == completion.payload["model_attempt_id"]
            assert provider.requests == []
        finally:
            await store.close()

    asyncio.run(recover())
