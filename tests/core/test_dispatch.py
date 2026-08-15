from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

import pytest
from tests.core._execution_profile_fixtures import (
    checkpoint_with_rebound_test_invocation_profile,
    runtime_interaction_started_event,
)

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    DispatchRequest,
    DispatchStatus,
    InMemorySessionStore,
    InMemoryTaskStore,
    RunRequest,
    SessionStatus,
    SessionStatusConflict,
    StructuredOutputSpec,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskStoreDispatcher,
)
from cayu.runtime._diagnostics import ExceptionDiagnostic, exception_diagnostic
from cayu.runtime.dispatch import copy_dispatch_request
from cayu.vaults import REDACTED_SECRET, SecretRedactor

_DISPATCH_TASK_TYPE = "cayu.dispatch"


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[list[ModelStreamEvent]]) -> None:
        self.event_batches = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch_index = len(self.requests)
        self.requests.append(request)
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class Harness(NamedTuple):
    app: CayuApp
    store: InMemorySessionStore
    tasks: TaskStore
    provider: FakeProvider
    dispatcher: TaskStoreDispatcher


class _SecretFreeDispatchRuntime:
    """Test runtime implementing the mandatory durable-dispatch boundary."""

    @staticmethod
    def redact_dispatch_request(request: DispatchRequest) -> DispatchRequest:
        return copy_dispatch_request(request)

    @staticmethod
    def redact_json(value: Any) -> Any:
        return SecretRedactor().redact_json(value)

    @staticmethod
    def redact_exception_diagnostic(
        error: BaseException,
        *,
        empty_message: str,
        nonportable_message: str,
    ) -> ExceptionDiagnostic:
        return exception_diagnostic(
            error,
            empty_message=empty_message,
            nonportable_message=nonportable_message,
        )


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(
    batches: list[list[ModelStreamEvent]],
    *,
    task_store: TaskStore | None = None,
    task_type: str = _DISPATCH_TASK_TYPE,
    recover_stalled_sessions_after_seconds: int | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> Harness:
    store = InMemorySessionStore()
    tasks = task_store if task_store is not None else InMemoryTaskStore()
    provider = FakeProvider(batches)
    dispatcher = TaskStoreDispatcher(
        tasks,
        task_type=task_type,
        recover_stalled_sessions_after_seconds=recover_stalled_sessions_after_seconds,
    )
    app = CayuApp(
        session_store=store,
        task_store=tasks,
        dispatcher=dispatcher,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return Harness(app, store, tasks, provider, dispatcher)


def _create_resumable_session(app: CayuApp, session_id: str) -> None:
    async def run() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "first request")],
            )
        ):
            pass

    asyncio.run(run())


def _dispatch_request(session_id: str, dispatch_id: str) -> DispatchRequest:
    return DispatchRequest(
        session_id=session_id,
        dispatch_id=dispatch_id,
        messages=[Message.text("user", "queued work")],
    )


def test_submit_enqueues_pending_task_without_running() -> None:
    # Only the initial run consumes a batch; the dispatch must NOT run on submit.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_submit")

    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_submit", "d_submit")))

    assert handle.status == DispatchStatus.SUBMITTED
    assert handle.backend == "task_store"
    assert handle.dispatch_id == "d_submit"
    # Not run yet: the provider only saw the initial request.
    assert len(h.provider.requests) == 1

    # The work was persisted as a claimable (session-unbound) PENDING dispatch task that
    # carries the serialized request; the target session_id rides in the payload.
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.type == _DISPATCH_TASK_TYPE
    assert task.status == TaskStatus.PENDING
    assert task.session_id is None
    assert task.input["request"]["session_id"] == "sess_submit"
    assert task.input["request"]["dispatch_id"] == "d_submit"


def test_submit_redacts_workload_secrets_before_durable_dispatch_write() -> None:
    secret = "dispatch-boundary-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_redaction")
    request = DispatchRequest(
        session_id="sess_dispatch_redaction",
        dispatch_id="d_redacted",
        messages=[Message.text("user", f"queued {secret}")],
        metadata={"note": f"metadata {secret}"},
    )

    handle = asyncio.run(h.app.dispatch(request))
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))

    assert task is not None
    serialized = str(task.input)
    assert secret not in serialized
    assert REDACTED_SECRET in serialized


def test_submit_rejects_protocol_minimal_runtime_before_persisting_secret_input() -> None:
    secret = "protocol-minimal-dispatch-input-secret"
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class ProtocolMinimalRuntime:
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield

    request = DispatchRequest(
        session_id="sess_protocol_minimal",
        dispatch_id="d_protocol_minimal",
        messages=[Message.text("user", secret)],
    )

    with pytest.raises(TypeError, match="redact_dispatch_request"):
        asyncio.run(dispatcher.submit(ProtocolMinimalRuntime(), request))

    assert asyncio.run(tasks.list_tasks(TaskQuery(type=_DISPATCH_TASK_TYPE))) == []


def test_worker_rejects_protocol_minimal_runtime_before_persisting_secret_failure() -> None:
    secret = "protocol-minimal-dispatch-failure-secret"
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class ProtocolMinimalRuntime:
        @staticmethod
        def redact_dispatch_request(request: DispatchRequest) -> DispatchRequest:
            return copy_dispatch_request(request)

        @staticmethod
        def redact_json(value: Any) -> Any:
            return value

        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield
            raise RuntimeError(secret)

    task = asyncio.run(
        tasks.create_task(
            TaskCreate(
                type=_DISPATCH_TASK_TYPE,
                input={
                    "request": _dispatch_request(
                        "sess_protocol_minimal_failure",
                        "d_protocol_minimal_failure",
                    ).model_dump(mode="json")
                },
            )
        )
    )

    with pytest.raises(TypeError, match="redact_exception_diagnostic"):
        asyncio.run(dispatcher.process_next(ProtocolMinimalRuntime(), worker_id="worker_a"))

    persisted = asyncio.run(tasks.load_task(task.id))
    assert persisted is not None
    assert persisted.status is TaskStatus.PENDING
    assert secret not in str(persisted)


def test_submit_rejects_secret_bearing_structured_output_without_changing_schema() -> None:
    secret = "dispatch-schema-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_schema")
    request = DispatchRequest(
        session_id="sess_dispatch_schema",
        dispatch_id="d_schema",
        messages=[Message.text("user", "queued work")],
        structured_output=StructuredOutputSpec(
            json_schema={
                "type": "object",
                "properties": {"answer": {"const": secret}},
                "required": ["answer"],
            }
        ),
    )

    with pytest.raises(ValueError, match="changing execution semantics"):
        asyncio.run(h.app.dispatch(request))

    assert len(h.provider.requests) == 1


def test_inline_dispatch_rejects_secret_bearing_structured_output_before_session_claim() -> None:
    secret = "inline-dispatch-schema-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    session_id = "sess_inline_dispatch_schema"
    _create_resumable_session(h.app, session_id)

    async def scenario() -> None:
        with pytest.raises(ValueError, match="workload secret"):
            async for _ in h.app.dispatch_inline(
                DispatchRequest(
                    session_id=session_id,
                    dispatch_id="d_inline_schema",
                    messages=[Message.text("user", "queued work")],
                    structured_output=StructuredOutputSpec(
                        json_schema={"type": "string", "const": secret},
                    ),
                )
            ):
                pass

    asyncio.run(scenario())
    session = asyncio.run(h.store.load(session_id))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert len(h.provider.requests) == 1


def test_process_next_redacts_runtime_failure_before_task_persistence(monkeypatch) -> None:
    secret = "dispatch-failure-canary"
    h = _build(
        [_batch("first answer")],
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_failure")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_failure", "d_failure")))

    async def fail_dispatch(request):
        del request
        if False:
            yield
        raise RuntimeError(f"dispatch failed with {secret}")

    monkeypatch.setattr(h.app, "dispatch_inline", fail_dispatch)
    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))

    assert result is not None
    assert result.status == DispatchStatus.FAILED
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert secret not in str(task.error)
    assert REDACTED_SECRET in str(task.error)


def test_stalled_recovery_log_redacts_workload_secret(
    monkeypatch,
    caplog,
) -> None:
    secret = "dispatch-recovery-log-canary"
    h = _build(
        [_batch("first answer")],
        recover_stalled_sessions_after_seconds=0,
        secret_redactor=SecretRedactor(secret),
    )
    _create_resumable_session(h.app, "sess_dispatch_recovery_log")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_dispatch_recovery_log", "d_recovery_log")))
    asyncio.run(
        h.store.transition_status(
            "sess_dispatch_recovery_log",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    async def fail_recovery(request):
        del request
        raise RuntimeError(f"recovery failed with {secret}")

    monkeypatch.setattr(h.app, "recover_incomplete_session", fail_recovery)
    with caplog.at_level("WARNING", logger="cayu.runtime.dispatch"):
        result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_recovery"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert secret not in caplog.text
    assert REDACTED_SECRET in caplog.text


def test_process_next_claims_runs_and_completes() -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    _create_resumable_session(h.app, "sess_run")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_run", "d_run")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.COMPLETED
    assert result.dispatch_id == "d_run"
    # The dispatched run actually executed (second provider request).
    assert len(h.provider.requests) == 2
    # The queue task is completed and the session ran to completion.
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    session = asyncio.run(h.store.load("sess_run"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_process_next_returns_none_when_queue_empty() -> None:
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_empty")
    assert asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a")) is None


def test_busy_session_requeues_dispatch_task() -> None:
    # A second dispatch for a session another worker is already running must be requeued,
    # not failed — the per-session serialization is preserved without losing the work.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_busy")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_busy", "d_busy")))
    # Simulate another worker already running the session.
    asyncio.run(
        h.store.transition_status(
            "sess_busy",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.SUBMITTED
    assert result.metadata.get("requeued") is True
    # The dispatched run never started, and the task is back to PENDING for a later retry.
    assert len(h.provider.requests) == 1
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.PENDING


def test_busy_session_conflict_leaves_fresh_session_alone() -> None:
    # A conflicting session with recent store activity looks live (another worker is
    # really running it), so the dispatcher must requeue without recovering it.
    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_fresh_conflict")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_fresh_conflict", "d_fresh")))
    asyncio.run(
        h.store.transition_status(
            "sess_fresh_conflict",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert "recovered_session" not in result.metadata
    session = asyncio.run(h.store.load("sess_fresh_conflict"))
    assert session is not None
    assert session.status == SessionStatus.RUNNING  # untouched


def test_busy_session_with_old_status_timestamp_but_recent_progress_is_not_recovered() -> None:
    h = _build(
        [_batch("first answer")],
        recover_stalled_sessions_after_seconds=60,
    )
    _create_resumable_session(h.app, "sess_recent_progress")
    asyncio.run(h.app.dispatch(_dispatch_request("sess_recent_progress", "d_recent_progress")))
    asyncio.run(
        h.store.transition_status(
            "sess_recent_progress",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
        )
    )

    async def age_status_then_record_progress() -> None:
        old = datetime.now(UTC) - timedelta(hours=1)
        async with h.store._lock:
            session = h.store._sessions["sess_recent_progress"]
            h.store._sessions[session.id] = session.model_copy(
                update={"updated_at": old, "last_activity_at": old}
            )
        await h.store.checkpoint("sess_recent_progress", {"step": 2})

    asyncio.run(age_status_then_record_progress())
    before = asyncio.run(h.store.load("sess_recent_progress"))
    assert before is not None
    assert before.updated_at < datetime.now(UTC) - timedelta(minutes=30)
    assert before.last_activity_at > datetime.now(UTC) - timedelta(seconds=5)

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.metadata.get("requeued") is True
    assert "recovered_session" not in result.metadata
    after = asyncio.run(h.store.load("sess_recent_progress"))
    assert after is not None
    assert after.status == SessionStatus.RUNNING


def test_conflict_after_worker_crash_recovers_stalled_session_and_reruns() -> None:
    # A worker crashed mid-run: its queue task was reclaimed, but the session row is
    # stranded RUNNING, so every re-claim conflicts. With the recovery horizon elapsed
    # (0 here), the dispatcher must recover the session and requeue, and the next
    # claim must run the dispatch to completion instead of conflict-spinning forever.
    h = _build(
        [_batch("first answer"), _batch("dispatch answer")],
        recover_stalled_sessions_after_seconds=0,
    )
    _create_resumable_session(h.app, "sess_crash")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_crash", "d_crash")))
    # Simulate the crash: the session is stuck RUNNING with no live run anywhere.
    interaction_id = "interaction_dispatch_worker_crash"
    asyncio.run(
        h.store.transition_status_and_checkpoint(
            "sess_crash",
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, checkpoint: (
                checkpoint_with_rebound_test_invocation_profile(
                    session,
                    checkpoint,
                    interaction_id=interaction_id,
                )
            ),
            interaction_started_event=runtime_interaction_started_event(
                h.app,
                session_id="sess_crash",
                interaction_id=interaction_id,
                agent_name="assistant",
            ),
            interaction_source_messages=[],
        )
    )

    first = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert first is not None
    assert first.status == DispatchStatus.SUBMITTED
    assert first.metadata.get("requeued") is True
    assert first.metadata.get("recovered_session") is True
    # The stranded session was finalized to a resumable status, not left RUNNING.
    session = asyncio.run(h.store.load("sess_crash"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    recovery_events = asyncio.run(h.store.load_events("sess_crash"))
    fenced = next(event for event in recovery_events if event.type == EventType.SESSION_RUN_FENCED)
    assert fenced.payload["previous_run_epoch"] == 3
    assert fenced.payload["run_epoch"] == 4

    second = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_b"))

    assert second is not None
    assert second.status == DispatchStatus.COMPLETED
    assert len(h.provider.requests) == 2
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    session = asyncio.run(h.store.load("sess_crash"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_recover_stalled_sessions_after_seconds_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="recover_stalled_sessions_after_seconds"):
        TaskStoreDispatcher(InMemoryTaskStore(), recover_stalled_sessions_after_seconds=-1)


def test_failed_run_marks_dispatch_task_failed() -> None:
    # A dispatch whose run raises (here: the session does not exist) fails the task.
    h = _build([_batch("first answer")])
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_missing", "d_missing")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.FAILED
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error is not None and "error" in task.error


def test_in_band_run_failure_marks_dispatch_task_failed() -> None:
    # The session exists, but the dispatched run fails mid-stream (the provider has no batch
    # for it, so the run emits a SESSION_FAILED event rather than raising). The queue task
    # must be recorded FAILED — not COMPLETED — so failure queries and retries can see it.
    h = _build([_batch("first answer")])  # no batch for the dispatched run
    _create_resumable_session(h.app, "sess_inband")
    handle = asyncio.run(h.app.dispatch(_dispatch_request("sess_inband", "d_inband")))

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is not None
    assert result.status == DispatchStatus.FAILED
    task = asyncio.run(h.tasks.load_task(handle.metadata["queue_task_id"]))
    assert task is not None
    assert task.status == TaskStatus.FAILED


def test_invalid_request_payload_fails_task_terminally() -> None:
    # A claimed task whose request payload no longer validates (e.g. an older serialization
    # after a schema change) must be failed terminally, not left to be reclaimed forever.
    h = _build([_batch("first answer")])
    task = asyncio.run(
        h.tasks.create_task(
            TaskCreate(type=_DISPATCH_TASK_TYPE, input={"request": {"bad": "data"}})
        )
    )

    result = asyncio.run(h.dispatcher.process_next(h.app, worker_id="worker_a"))

    assert result is None
    failed = asyncio.run(h.tasks.load_task(task.id))
    assert failed is not None
    assert failed.status == TaskStatus.FAILED


def test_submit_rejects_loop_policies() -> None:
    # loop_policies are process-local callables that cannot survive serialization; queuing a
    # dispatch that carries them must fail loudly rather than silently drop them.
    from cayu.runtime import LoopPolicy

    class _NoopPolicy(LoopPolicy):
        pass

    h = _build([_batch("first answer")])
    _create_resumable_session(h.app, "sess_lp")
    request = DispatchRequest(
        session_id="sess_lp",
        dispatch_id="d_lp",
        messages=[Message.text("user", "queued work")],
        loop_policies=(_NoopPolicy(),),
    )
    with pytest.raises(ValueError, match="loop_policies"):
        asyncio.run(h.app.dispatch(request))


def test_durable_request_redaction_rejects_loop_policies_for_custom_dispatchers() -> None:
    from cayu.runtime import LoopPolicy

    class _NoopPolicy(LoopPolicy):
        pass

    h = _build([_batch("first answer")])
    request = DispatchRequest(
        session_id="sess_custom_dispatch_policy",
        dispatch_id="d_custom_policy",
        messages=[Message.text("user", "queued work")],
        loop_policies=(_NoopPolicy(),),
    )

    with pytest.raises(ValueError, match="loop_policies"):
        h.app.redact_dispatch_request(request)


def test_reclaimed_dispatch_is_reprocessable() -> None:
    h = _build([_batch("first answer"), _batch("dispatch answer")])
    _create_resumable_session(h.app, "sess_reclaim")

    async def scenario() -> None:
        handle = await h.app.dispatch(_dispatch_request("sess_reclaim", "d_reclaim"))
        queue_task_id = handle.metadata["queue_task_id"]
        # A worker claims it with a short lease, then "dies" without completing.
        await h.tasks.claim_task(
            "dead_worker", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=1
        )
        assert await h.dispatcher.process_next(h.app, worker_id="live_worker") is None

        await asyncio.sleep(1.05)
        reclaimed = await h.tasks.reclaim_expired(query=TaskQuery(type=_DISPATCH_TASK_TYPE))
        assert [task.id for task in reclaimed] == [queue_task_id]

        result = await h.dispatcher.process_next(h.app, worker_id="live_worker")
        assert result is not None
        assert result.status == DispatchStatus.COMPLETED

    asyncio.run(scenario())


def test_run_worker_drains_queue_until_stopped() -> None:
    h = _build([_batch("a0"), _batch("b0"), _batch("a1"), _batch("b1")])
    _create_resumable_session(h.app, "sess_w_a")
    _create_resumable_session(h.app, "sess_w_b")

    async def scenario() -> None:
        h_a = await h.app.dispatch(_dispatch_request("sess_w_a", "d_w_a"))
        h_b = await h.app.dispatch(_dispatch_request("sess_w_b", "d_w_b"))
        stop = asyncio.Event()
        worker = asyncio.create_task(
            h.dispatcher.run_worker(h.app, worker_id="worker_a", stop=stop, poll_interval_s=0.01)
        )
        try:
            async with asyncio.timeout(5):
                while True:
                    t_a = await h.tasks.load_task(h_a.metadata["queue_task_id"])
                    t_b = await h.tasks.load_task(h_b.metadata["queue_task_id"])
                    if (
                        t_a is not None
                        and t_b is not None
                        and t_a.status == TaskStatus.COMPLETED
                        and t_b.status == TaskStatus.COMPLETED
                    ):
                        break
                    await asyncio.sleep(0.01)
        finally:
            stop.set()
            await worker

    asyncio.run(scenario())


def test_lease_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lease_seconds must be a positive integer"):
        TaskStoreDispatcher(InMemoryTaskStore(), lease_seconds=0)


def test_terminal_update_requires_owning_worker() -> None:
    # A worker that does not own the task's active lease cannot terminalize it.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        task = await h.tasks.create_task(TaskCreate(type=_DISPATCH_TASK_TYPE))
        await h.tasks.claim_task("worker_a", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=300)

        with pytest.raises(ValueError, match="does not own"):
            await h.tasks.complete_task(task.id, {"ok": True}, worker_id="worker_b")
        with pytest.raises(ValueError, match="does not own"):
            await h.tasks.fail_task(task.id, {"err": True}, worker_id="worker_b")

        done = await h.tasks.complete_task(task.id, {"ok": True}, worker_id="worker_a")
        assert done.status == TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_terminalize_does_not_clobber_a_reclaimed_task() -> None:
    # If a worker lost its lease and the task was reclaimed by another worker, the original
    # worker's terminal write must be rejected and leave the reclaimer's record untouched.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        task = await h.tasks.create_task(
            TaskCreate(type=_DISPATCH_TASK_TYPE, input={"request": {"x": 1}})
        )
        # The task is now owned by worker_b (stands in for a reclaim by another worker).
        await h.tasks.claim_task("worker_b", TaskQuery(type=_DISPATCH_TASK_TYPE), lease_seconds=300)
        request = _dispatch_request("sess_reclaimed", "d_reclaimed")

        handle = await h.dispatcher._terminalize(
            task.id, "worker_a", request, DispatchStatus.COMPLETED, {"status": "completed"}
        )

        assert handle.metadata.get("reclaimed") is True
        reloaded = await h.tasks.load_task(task.id)
        assert reloaded is not None
        assert reloaded.status == TaskStatus.CLAIMED  # not clobbered to COMPLETED
        assert reloaded.worker_id == "worker_b"  # still the reclaimer's

    asyncio.run(scenario())


def test_malformed_dispatch_claim_loss_returns_without_clobbering_new_owner() -> None:
    class ReclaimBeforeMalformedFailureStore(InMemoryTaskStore):
        async def fail_task(self, task_id, error, *, worker_id=None):
            if worker_id is not None:
                await super().release_task(task_id, worker_id)
                reclaimed = await super().claim_task("worker_b", lease_seconds=300)
                assert reclaimed is not None
                assert reclaimed.id == task_id
            return await super().fail_task(task_id, error, worker_id=worker_id)

    tasks = ReclaimBeforeMalformedFailureStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class UnusedRuntime(_SecretFreeDispatchRuntime):
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield

    async def scenario():
        task = await tasks.create_task(
            TaskCreate(
                type=_DISPATCH_TASK_TYPE,
                input={"request": "not-an-object"},
            )
        )
        result = await dispatcher.process_next(UnusedRuntime(), worker_id="worker_a")
        return result, await tasks.load_task(task.id)

    result, task = asyncio.run(scenario())

    assert result is None
    assert task is not None
    assert task.status is TaskStatus.CLAIMED
    assert task.worker_id == "worker_b"


def test_conflict_requeue_returns_reclaimed_handle_when_control_plane_wins() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    request = _dispatch_request("sess_conflict_claim_lost", "d_conflict_claim_lost")

    async def scenario():
        class SubmitRuntime(_SecretFreeDispatchRuntime):
            async def dispatch_inline(
                self,
                request: DispatchRequest,
            ) -> AsyncIterator[Event]:
                del request
                if False:
                    yield

        submitted = await dispatcher.submit(SubmitRuntime(), request)
        task_id = submitted.metadata["queue_task_id"]

        class TerminalizingConflictRuntime(_SecretFreeDispatchRuntime):
            async def dispatch_inline(
                self,
                request: DispatchRequest,
            ) -> AsyncIterator[Event]:
                del request
                await tasks.complete_task(task_id, {"winner": "control-plane"})
                if False:
                    yield
                raise SessionStatusConflict("session already running")

        result = await dispatcher.process_next(
            TerminalizingConflictRuntime(),
            worker_id="worker_a",
        )
        return result, await tasks.load_task(task_id)

    result, task = asyncio.run(scenario())

    assert result is not None
    assert result.status is DispatchStatus.SUBMITTED
    assert result.metadata["reclaimed"] is True
    assert result.metadata.get("requeued") is None
    assert task is not None
    assert task.status is TaskStatus.COMPLETED
    assert task.result == {"winner": "control-plane"}


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("provider failure\u0000with invalid text", "nul_character"),
        ("provider failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_dispatch_failure_with_nonportable_text_is_terminal_and_not_reclaimed(
    rejected_text: str,
    error_code: str,
) -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)

    class FailingRuntime(_SecretFreeDispatchRuntime):
        async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
            del request
            if False:
                yield
            raise RuntimeError(rejected_text)

    async def scenario():
        submitted = await dispatcher.submit(
            FailingRuntime(),
            _dispatch_request("sess_nonportable_failure", "d_nonportable_failure"),
        )
        result = await dispatcher.process_next(FailingRuntime(), worker_id="worker_a")
        task_id = submitted.metadata["queue_task_id"]
        task = await tasks.load_task(task_id)
        reclaimed = await tasks.reclaim_expired(query=TaskQuery(type=_DISPATCH_TASK_TYPE))
        second_claim = await tasks.claim_task(
            "worker_b",
            TaskQuery(type=_DISPATCH_TASK_TYPE),
            lease_seconds=300,
        )
        return result, task, reclaimed, second_claim

    result, task, reclaimed, second_claim = asyncio.run(scenario())

    assert result is not None
    assert result.status is DispatchStatus.FAILED
    assert result.metadata.get("reclaimed") is None
    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error == {
        "error": "Dispatch failed with a non-portable diagnostic.",
        "error_type": "RuntimeError",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
    }
    assert reclaimed == []
    assert second_claim is None


def test_concurrent_workers_claim_distinct_dispatch_tasks(postgres_dsn: str) -> None:
    # In-memory sessions + a real PostgresTaskStore queue: two concurrent workers must
    # claim distinct dispatch tasks through the actual FOR UPDATE SKIP LOCKED path. A
    # per-process-unique task type isolates this run from any leftover rows.
    from cayu.storage import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    task_type = f"cayu.dispatch.test.{os.getpid()}"

    async def scenario() -> None:
        tasks = PostgresTaskStore(
            postgres_dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
        )
        try:
            h = _build(
                [_batch("a0"), _batch("b0"), _batch("a1"), _batch("b1")],
                task_store=tasks,
                task_type=task_type,
            )
            for session_id in ("sess_pg_a", "sess_pg_b"):
                async for _ in h.app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first request")],
                    )
                ):
                    pass
                await h.app.dispatch(_dispatch_request(session_id, f"d_{session_id}"))

            results = await asyncio.gather(
                h.dispatcher.process_next(h.app, worker_id="worker_a"),
                h.dispatcher.process_next(h.app, worker_id="worker_b"),
            )
            claimed = [r for r in results if r is not None]
            assert len(claimed) == 2
            assert {r.session_id for r in claimed} == {"sess_pg_a", "sess_pg_b"}
            assert all(r.status == DispatchStatus.COMPLETED for r in claimed)
        finally:
            await tasks.close()

    asyncio.run(scenario())
