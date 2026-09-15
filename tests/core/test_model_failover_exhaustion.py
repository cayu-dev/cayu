"""Public exhaustion evidence distinguishes bounded exhaustion from suppression."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.runtime import _model_step_executor as model_executor
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions import EventQuery


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "failure_kind", ["attempt_limit", "candidate_chain", "partial", "permanent"]
)
def test_public_failover_exhaustion_evidence(monkeypatch, tmp_path, backend, failure_kind):

    class UnavailableProvider(_RecoveryProvider):
        async def stream(self, request):
            self.requests.append(request)
            if failure_kind == "partial":
                yield ModelStreamEvent.text_delta("accepted partial output")
            raise ModelProviderError(
                "service rejected request",
                provider=self.name,
                status_code=401 if failure_kind == "permanent" else 503,
                retryable=failure_kind != "permanent",
            )

    async def scenario():
        path = tmp_path / "exhaustion.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        try:
            primary, backup = UnavailableProvider("primary"), UnavailableProvider("backup")
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(primary, default=True)
            app.register_provider(backup)
            app.register_agent(AgentSpec(name="agent", model="small"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="exhaustion",
                        messages=[Message.text("user", "answer")],
                        retry_policy=RetryPolicy(
                            max_attempts=5 if failure_kind == "attempt_limit" else 1,
                            initial_delay_s=0,
                        ),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=(
                                3
                                if failure_kind == "attempt_limit"
                                else 20
                                if failure_kind == "candidate_chain"
                                else 1
                            ),
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_FAILED
            expected_primary = 3 if failure_kind == "attempt_limit" else 1
            expected_backup = int(failure_kind == "candidate_chain")
            assert len(primary.requests) == expected_primary
            assert len(backup.requests) == expected_backup
            assert events[-1].payload["error_type"] == "ModelProviderError"
            errors = [event for event in events if event.type is EventType.MODEL_ERROR]
            assert [event.payload["provider_name"] for event in errors] == (
                ["primary"] * expected_primary + ["backup"] * expected_backup
            )
            exhausted = [
                event for event in events if event.type is EventType.MODEL_FAILOVER_EXHAUSTED
            ]
            if failure_kind in {"partial", "permanent"}:
                assert not exhausted
                return
            assert len(exhausted) == 1
            payload = exhausted[0].payload
            assert payload["reason"] == failure_kind
            assert payload["provider"] == ("backup" if expected_backup else "primary")
            assert payload["attempts_used"] == expected_primary + expected_backup
            checkpoint = await store.load_checkpoint("exhaustion")
            assert checkpoint is not None
            progress = checkpoint["model_failover"]
            assert payload["stage_id"] == progress["stage_id"]
            assert payload["candidate_index"] == progress["candidate_index"]
            persisted = await store.load_events("exhaustion")
            stored_exhausted = [
                event for event in persisted if event.type is EventType.MODEL_FAILOVER_EXHAUSTED
            ]
            assert len(stored_exhausted) == 1
            records = await store.query_events(
                EventQuery(session_id="exhaustion", event_id=stored_exhausted[0].id, limit=2)
            )
            assert len(records) == 1
            # Public IDs deliberately alias private IDs by durable sequence.
            assert exhausted[0].id == f"cayu_event_{records[0].sequence}"
            assert payload == {
                **stored_exhausted[0].payload,
                "model_step_id": "[PRIVATE_EVENT_AUTHORITY]",
                "model_attempt_id": "[PRIVATE_EVENT_AUTHORITY]",
            }
            stored_errors = [event for event in persisted if event.type is EventType.MODEL_ERROR]
            for key in ("model_step_id", "model_attempt_id"):
                assert stored_exhausted[0].payload[key] == stored_errors[-1].payload[key]
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_events("exhaustion") == persisted
                assert await store.load_checkpoint("exhaustion") == checkpoint
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("local_attempts", [1, 2])
def test_public_incomplete_tool_call_never_executes_or_allows_fallback(
    tmp_path, backend, local_attempts
):
    invocations = []

    class EffectTool(Tool):
        spec = ToolSpec(
            name="effect",
            description="Record an invocation",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx, args):
            invocations.append(ctx.agent_name)
            return ToolResult(content="executed")

    class IncompleteProvider(_RecoveryProvider):
        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(id="unfinished", name="effect", arguments={})
            raise ModelProviderError(
                "service rejected request", provider=self.name, status_code=503, retryable=True
            )

    async def scenario():
        path = tmp_path / "incomplete-tool.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        try:
            primary, backup = IncompleteProvider("primary"), _RecoveryProvider("backup")
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(primary, default=True)
            app.register_provider(backup)
            app.register_agent(AgentSpec(name="agent", model="small"), tools=[EffectTool()])
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="incomplete-tool",
                        messages=[Message.text("user", "Use the effect tool.")],
                        retry_policy=RetryPolicy(max_attempts=local_attempts, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=4,
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_FAILED
            assert events[-1].payload["error_type"] == "ModelProviderError"
            assert len(primary.requests) == local_attempts
            assert not backup.requests and not invocations
            assert not any(
                event.type
                in {
                    EventType.MODEL_COMPLETED,
                    EventType.TOOL_CALL_STARTED,
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.MODEL_FAILOVER_EXHAUSTED,
                }
                for event in events
            )
            selections = [
                event for event in events if event.type is EventType.MODEL_FAILOVER_SELECTED
            ]
            assert len(selections) == 1 and selections[0].payload["provider"] == "primary"
            checkpoint = await store.load_checkpoint("incomplete-tool")
            assert checkpoint is not None
            assert checkpoint["model_failover"]["candidate_index"] == 0
            assert checkpoint["model_failover"]["attempts_used"] == local_attempts
            assert checkpoint.get("pending_tool_round") is None
            persisted = await store.load_events("incomplete-tool")
            assert not any(event.type is EventType.TOOL_CALL_STARTED for event in persisted)
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_events("incomplete-tool") == persisted
                assert await store.load_checkpoint("incomplete-tool") == checkpoint
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cleanup", ["success", "failure", "cancel"])
def test_public_fallback_waits_for_provider_cleanup(tmp_path, backend, cleanup, caplog, capsys):
    canary = "private-provider-close-credential"

    async def scenario():
        closing, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class ClosingProvider(_RecoveryProvider):
            @property
            def stream_deadlines(self):
                return ProviderStreamDeadlines(semantic_progress_timeout_s=10)

            async def stream(self, request):
                self.requests.append(request)
                failure = ModelProviderError(
                    "service unavailable", provider=self.name, status_code=503, retryable=True
                )
                try:
                    yield ModelStreamEvent.error(str(failure), cause=failure)
                finally:
                    closing.set()
                    try:
                        await release.wait()
                        if cleanup == "failure":
                            raise OSError(canary)
                    finally:
                        stopped.set()

        path = tmp_path / "cleanup-fallback.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        primary, backup = ClosingProvider("primary"), _RecoveryProvider("backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        events, cancellations = [], []

        async def consume():
            stream = app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="cleanup-fallback",
                    messages=[Message.text("user", "answer")],
                    retry_policy=RetryPolicy(max_attempts=1),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                    ),
                )
            )
            assert isinstance(stream, AsyncGenerator)
            try:
                async for event in stream:
                    events.append(event)
            except asyncio.CancelledError as failure:
                cancellations.append(failure)
                raise
            finally:
                await stream.aclose()

        caller = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(closing.wait(), 30)
            assert not caller.done() and not stopped.is_set()
            assert len(primary.requests) == 1 and not backup.requests
            active = await store.load_active_model_completion_stage("cleanup-fallback")
            assert active is not None and active.stage.intent["provider_name"] == "primary"
            checkpoint = await store.load_checkpoint("cleanup-fallback")
            assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 0
            if cleanup == "cancel":
                caller.cancel()
                assert caller.cancelling() == 1
            release.set()
            if cleanup == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(caller, 30)
                assert caller.cancelled() and caller.cancelling() == 1
                assert len(cancellations) == 1
            else:
                await asyncio.wait_for(caller, 30)
                assert events[-1].type is (
                    EventType.SESSION_COMPLETED
                    if cleanup == "success"
                    else EventType.SESSION_FAILED
                )
            assert stopped.is_set()
            assert len(primary.requests) == 1
            assert len(backup.requests) == int(cleanup == "success")
            checkpoint = await store.load_checkpoint("cleanup-fallback")
            assert checkpoint is not None
            assert checkpoint["model_failover"]["candidate_index"] == int(cleanup == "success")
            persisted = await store.load_events("cleanup-fallback")
            errors = [event for event in persisted if event.type is EventType.MODEL_ERROR]
            assert len(errors) == 1
            assert errors[0].payload["provider_name"] == "primary"
            assert errors[0].payload["status_code"] == 503
            assert not any(event.type is EventType.MODEL_FAILOVER_EXHAUSTED for event in persisted)
            assert sum(event.type is EventType.MODEL_FAILOVER_SELECTED for event in persisted) == (
                2 if cleanup == "success" else 1
            )
            assert canary not in str(events) + str(persisted) + str(checkpoint)
            captured = capsys.readouterr()
            assert canary not in caplog.text + captured.out + captured.err
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_checkpoint("cleanup-fallback") == checkpoint
                assert await store.load_events("cleanup-fallback") == persisted
        finally:
            release.set()
            if not caller.done():
                caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_cancellation_after_exhaustion_commit(monkeypatch, tmp_path, backend):

    async def scenario():
        path = tmp_path / "cancel-exhaustion.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        committed = asyncio.Event()
        append = store.append_event
        writes = 0

        async def append_event(session_id, event):
            nonlocal writes
            await append(session_id, event)
            if event.type is EventType.MODEL_FAILOVER_EXHAUSTED:
                writes += 1
                committed.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(store, "append_event", append_event)
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        events = []
        cancellations = []

        async def consume():
            stream = app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="cancel-exhaustion",
                    messages=[Message.text("user", "answer")],
                    retry_policy=RetryPolicy(max_attempts=1),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                        max_total_attempts=1,
                    ),
                )
            )
            assert isinstance(stream, AsyncGenerator)
            try:
                async for event in stream:
                    events.append(event)
            except asyncio.CancelledError as error:
                cancellations.append(error)
                raise
            finally:
                await stream.aclose()

        caller = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(committed.wait(), 30)
            caller.cancel()
            assert caller.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 1
            assert len(cancellations) == 1
            assert len(primary.requests) == writes == 1 and not backup.requests
            persisted = await store.load_events("cancel-exhaustion")
            assert sum(event.type is EventType.MODEL_FAILOVER_EXHAUSTED for event in persisted) == 1
            errors = [event for event in persisted if event.type is EventType.MODEL_ERROR]
            assert len(errors) == 1 and errors[0].payload["provider_name"] == "primary"
            checkpoint = await store.load_checkpoint("cancel-exhaustion")
            assert checkpoint is not None and checkpoint["model_failover"]["attempts_used"] == 1
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_events("cancel-exhaustion") == persisted
                assert await store.load_checkpoint("cancel-exhaustion") == checkpoint
        finally:
            if not caller.done():
                caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("publication", ["before_commit", "lost_ack", "unverified_commit"])
def test_public_exhaustion_publication_failure(monkeypatch, tmp_path, backend, publication):

    async def scenario():
        path = tmp_path / "publication.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        append_failure = OSError("exhaustion append acknowledgement failed")
        read_failure = LookupError("exhaustion exact readback failed")
        original_append = store.append_event
        original_query = store.query_events
        combined: list[BaseException] = []
        original_combine = model_executor._combine_authoritative_model_failure
        writes = 0
        verification_failed = False

        def observe_combine(authoritative, secondary, *, message):
            result = original_combine(authoritative, secondary, message=message)
            combined.append(result)
            return result

        async def append_event(session_id, event):
            nonlocal writes
            if event.type is not EventType.MODEL_FAILOVER_EXHAUSTED:
                return await original_append(session_id, event)
            writes += 1
            if publication != "before_commit":
                await original_append(session_id, event)
            raise append_failure

        async def query_events(query):
            nonlocal verification_failed
            if (
                publication == "unverified_commit"
                and not verification_failed
                and query.event_id is not None
                and query.event_id.startswith("evt_failover_exhausted_")
            ):
                verification_failed = True
                raise read_failure
            return await original_query(query)

        monkeypatch.setattr(store, "append_event", append_event)
        monkeypatch.setattr(store, "query_events", query_events)
        monkeypatch.setattr(model_executor, "_combine_authoritative_model_failure", observe_combine)
        try:
            primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(primary, default=True)
            app.register_provider(backup)
            app.register_agent(AgentSpec(name="agent", model="small"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="publication",
                        messages=[Message.text("user", "answer")],
                        retry_policy=RetryPolicy(max_attempts=1),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=1,
                        ),
                    )
                )
            ]
            assert len(primary.requests) == 1 and not backup.requests
            assert writes == 1
            assert events[-1].type is EventType.SESSION_FAILED
            evidence = events[-1].payload["failure_evidence"]
            assert evidence["exception_types"].count("ModelProviderError") == 1
            if publication == "lost_ack":
                assert not combined
                assert events[-1].payload["error_type"] == "ModelProviderError"
            else:
                assert len(combined) == 1
                group = combined[0]
                assert isinstance(group, ExceptionGroup)
                assert len(group.exceptions) == 2
                assert isinstance(group.exceptions[0], ModelProviderError)
                assert group.exceptions[0].status_code == 503
                assert group.exceptions[1] is append_failure
                assert evidence["exception_types"].count("OSError") == 1
                assert evidence["secondary_failures"] is True
            if publication == "unverified_commit":
                assert verification_failed
                assert append_failure.__cause__ is read_failure
                assert evidence["exception_types"].count("LookupError") == 1
            persisted = await store.load_events("publication")
            assert sum(event.type is EventType.MODEL_FAILOVER_EXHAUSTED for event in persisted) == (
                publication != "before_commit"
            )
            assert sum(event.type is EventType.MODEL_FAILOVER_EXHAUSTED for event in events) == (
                publication == "lost_ack"
            )
            checkpoint = await store.load_checkpoint("publication")
            assert checkpoint is not None
            assert checkpoint["model_failover"]["attempts_used"] == 1
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_events("publication") == persisted
                assert await store.load_checkpoint("publication") == checkpoint
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
