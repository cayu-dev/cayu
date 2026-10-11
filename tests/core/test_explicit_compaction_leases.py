"""Explicit compaction claim exclusion, renewal and cancellation."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core._explicit_compaction_fixtures import (
    CompletingProvider,
    RecordingCompactor,
    UsageCompactionProvider,
    _create_profiled_session,
)

import cayu.runtime._session_engine as session_engine_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ContextCompactor,
    ModelCompactor,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._compaction import explicit as _session_compaction
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import (
    CompactSessionRequest,
    ForkSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.exposure import ToolCapabilityCeiling


class BlockingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def provider_budget_identity(self, _session) -> None:
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.started.set()
        await self.release.wait()
        return CompactionResult(summary="summary", covered_message_count=len(request.messages))


class BlockingCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def stream(self, request: ModelRequest):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield ModelStreamEvent.text_delta("provider summary")
        yield ModelStreamEvent.completed(
            {"model": request.model, "usage": {"input_tokens": 8, "output_tokens": 2}}
        )


def _test_compaction_heartbeat_state(
    claim_expires_at: datetime,
) -> _session_compaction._SessionOperationClaimHeartbeatState:
    state = _session_compaction._SessionOperationClaimHeartbeatState()
    state.confirm_claim(
        claim_expires_at,
        claim_deadline_monotonic=(
            time.monotonic() + _session_compaction._SESSION_OPERATION_CLAIM_LEASE.total_seconds()
        ),
    )
    return state


def test_compact_session_fences_expired_recovery_owner_before_retry() -> None:
    async def run() -> None:
        now = datetime(2026, 7, 18, tzinfo=UTC)
        session_id = "sess_compact_expired_recovery_owner"

        class InterleavedTakeoverStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self) -> None:
                super().__init__()
                self.takeover_started = asyncio.Event()
                self.allow_takeover = asyncio.Event()

            async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
                self.takeover_started.set()
                await self.allow_takeover.wait()
                return await super().fence_run_and_transform_checkpoint(*args, **kwargs)

        store = InterleavedTakeoverStore()
        compactor = RecordingCompactor()
        app = CayuApp(session_store=store, enable_logging=False, clock=lambda: now)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(session_id, transcript)
        await store.update_status(session_id, SessionStatus.COMPLETED)
        stale_owner_ready = asyncio.Event()
        release_stale_owner = asyncio.Event()

        async def stale_recovery_owner() -> None:
            fenced = await store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.COMPLETED},
                inactive_for_seconds=0,
            )
            assert fenced is not None

            def add_expired_claim(_session, checkpoint):
                updated = {} if checkpoint is None else dict(checkpoint)
                updated["incomplete_session_recovery_claim"] = {
                    "version": 1,
                    "claim_id": "stale-recovery-owner",
                    "claimed_at": (now - timedelta(minutes=10)).isoformat(),
                    "claim_expires_at": (now - timedelta(minutes=5)).isoformat(),
                }
                return updated

            await store.transform_checkpoint(session_id, add_expired_claim)
            stale_owner_ready.set()
            await store.takeover_started.wait()
            await store.update_metadata(
                session_id,
                {"stale_owner_write_before_atomic_takeover": True},
            )
            store.allow_takeover.set()
            await release_stale_owner.wait()
            try:
                with pytest.raises(SessionRunFenced):
                    await store.update_metadata(session_id, {"stale_owner_write": True})
            finally:
                await store.release_run_fence(session_id)

        stale_owner_task = asyncio.create_task(stale_recovery_owner())
        await asyncio.wait_for(stale_owner_ready.wait(), timeout=10)
        owned = await store.load(session_id)
        assert owned is not None
        stale_epoch = owned.run_epoch

        with pytest.raises(
            ValueError,
            match="fenced an expired incomplete-session recovery owner",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="compact-after-expired-recovery",
                    expected_run_epoch=stale_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pytest.fail("Compaction emitted an event before fencing the stale owner.")

        after_takeover = await store.load(session_id)
        assert after_takeover is not None
        assert after_takeover.run_epoch > stale_epoch
        assert after_takeover.metadata["stale_owner_write_before_atomic_takeover"] is True
        assert after_takeover.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=())
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "incomplete_session_recovery_claim" not in checkpoint
        assert compactor.requests == []

        release_stale_owner.set()
        await asyncio.wait_for(stale_owner_task, timeout=5)

        compacted = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="compact-after-expired-recovery",
                    expected_run_epoch=after_takeover.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        assert compacted[-1].type == EventType.SESSION_CHECKPOINTED
        assert len(compactor.requests) == 1

    asyncio.run(run())


def test_compact_session_claim_blocks_concurrent_resume() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(CompletingProvider())
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_resume_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        compact_events = []

        async def compact() -> None:
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-race-1",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                compact_events.append(event)

        compact_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.resume(
                    ResumeRequest(
                        session_id=created.id,
                        messages=[Message.text("user", "race")],
                    )
                ):
                    pass
            with pytest.raises(RuntimeError, match="active durable operation"):
                async for _event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=created.id,
                        session_id="sess_compact_race_fork",
                    )
                ):
                    pass
        finally:
            compactor.release.set()
            await compact_task

        assert [event.type for event in compact_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


def test_compact_session_rejects_an_equivalent_unexpired_concurrent_claim() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        compactor = BlockingCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_same_key_race",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-same-key-race",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )
        first_events = []

        async def compact() -> None:
            async for event in app.compact_session(request):
                first_events.append(event)

        first_task = asyncio.create_task(compact())
        await compactor.started.wait()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                async for _event in app.compact_session(request):
                    pass
        finally:
            compactor.release.set()
            await first_task

        assert [event.type for event in first_events] == [
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.SESSION_CHECKPOINTED,
        ]

    asyncio.run(run())


@pytest.mark.parametrize("expiry_boundary", ["commit", "acknowledgement"])
def test_compact_session_expired_initial_claim_never_enters_compactor(
    expiry_boundary: str,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}

        class ExpiringInitialClaimStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            async def publish_session_operation_guarded_with_store_time(
                self,
                session_id: str,
                **kwargs,
            ):
                events = kwargs.get("events", [])
                if [event.type for event in events] != [EventType.CONTEXT_COMPACTION_STARTED]:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                commit_time_guard = kwargs["commit_time_guard"]
                if expiry_boundary == "commit":

                    def expire_then_guard(_commit_at: datetime) -> None:
                        now["value"] = accepted_at + timedelta(minutes=6)
                        commit_time_guard(now["value"])

                    kwargs["commit_time_guard"] = expire_then_guard
                result = await super().publish_session_operation_guarded_with_store_time(
                    session_id,
                    **kwargs,
                )
                if expiry_boundary == "acknowledgement":
                    now["value"] = accepted_at + timedelta(minutes=6)
                return result

        store = ExpiringInitialClaimStore(ownership_clock=lambda: now["value"])
        compactor = RecordingCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compact_initial_claim_{expiry_boundary}",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(
            _session_compaction.SessionCompactionAttemptSuperseded,
            match="(?:initial publication|operation claim).*expired",
        ):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key=f"compact-initial-claim-{expiry_boundary}",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert compactor.requests == []
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.CONTEXT_COMPACTION_COMPLETED not in {
            event.type for event in durable_events
        }
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_heartbeats_claim_during_blocked_provider_dispatch(
    monkeypatch,
) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore(ownership_clock=lambda: now["value"])
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_provider_claim_heartbeat",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-provider-claim-heartbeat",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        first_task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        now["value"] = accepted_at + timedelta(minutes=4)

        first_renewal_expiry = accepted_at + timedelta(minutes=9)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= first_renewal_expiry:
                    break
                await asyncio.sleep(0)

        now["value"] = accepted_at + timedelta(minutes=6)
        expected_expiry = accepted_at + timedelta(minutes=11)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= expected_expiry:
                    break
                await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="already running"):
            async for _event in app.compact_session(request):
                pass
        assert provider.calls == 1

        provider.release.set()
        events = await first_task
        assert provider.calls == 1
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1

    asyncio.run(run())


def test_compact_session_claim_heartbeat_loss_cancels_provider_and_fences_publication(
    monkeypatch,
) -> None:
    class ClaimStealingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = ClaimStealingStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_fenced",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="superseded"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-claim-heartbeat-fenced",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        assert store.stolen
        assert provider.calls == 1
        assert provider.cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        completions = [event for event in durable_events if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["compaction_outcome"] == "cancelled"
        assert completions[0].payload["usage_unavailable_reason"]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}

    asyncio.run(run())


def test_compact_session_claim_heartbeat_retries_transient_store_failure(monkeypatch) -> None:
    class TransientHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_calls = 0
            self.renewed = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_calls += 1
                if self.heartbeat_calls == 1:
                    raise ConnectionError("transient heartbeat write failure")
                result = await super().publish_session_operation_guarded_with_store_time(
                    session_id, **kwargs
                )
                self.renewed.set()
                return result
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_RETRY_SECONDS",
            0.01,
        )
        store = TransientHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_retry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-retry",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        await asyncio.wait_for(store.renewed.wait(), timeout=5)
        assert store.heartbeat_calls >= 2
        assert not task.done()

        provider.release.set()
        events = await task
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_claim_heartbeat_reconciles_lost_renewal_acknowledgement(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class LostRenewalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__(ownership_clock=lambda: now["value"])
            self.acknowledgement_lost = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if (
                kwargs.get("events") == []
                and now["value"] >= accepted_at + timedelta(minutes=4)
                and not self.acknowledgement_lost.is_set()
            ):
                self.acknowledgement_lost.set()
                raise ConnectionError("renewal acknowledgement lost after commit")
            return result

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = LostRenewalAcknowledgementStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        reconciliation_finished = asyncio.Event()
        reconcile_claim = (
            app._session_compaction._reconcile_compaction_operation_claim_before_deadline
        )

        async def observe_reconciliation(**kwargs):
            result = await reconcile_claim(**kwargs)
            if store.acknowledgement_lost.is_set():
                reconciliation_finished.set()
            return result

        monkeypatch.setattr(
            app._session_compaction,
            "_reconcile_compaction_operation_claim_before_deadline",
            observe_reconciliation,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_lost_ack",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-claim-heartbeat-lost-ack",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        now["value"] = accepted_at + timedelta(minutes=4)
        await asyncio.wait_for(store.acknowledgement_lost.wait(), timeout=5)
        await asyncio.wait_for(reconciliation_finished.wait(), timeout=5)
        now["value"] = accepted_at + timedelta(minutes=6)

        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert datetime.fromisoformat(record["claim_expires_at"]) == (
            accepted_at + timedelta(minutes=9)
        )
        assert now["value"] > accepted_at + _session_compaction._SESSION_OPERATION_CLAIM_LEASE
        assert not task.done()
        assert not provider.cancelled.is_set()

        provider.release.set()
        events = await task
        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert provider.calls == 1
        operation = await store.load_session_operation(created.id, request.idempotency_key)
        assert operation is not None
        assert operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_stops_when_renewal_acknowledgement_exceeds_lease_deadline(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
    now = {"value": accepted_at}

    class DelayedRenewalAcknowledgementStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__(ownership_clock=lambda: now["value"])
            self.renewal_committed = asyncio.Event()
            self.release_acknowledgement = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            result = await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )
            if kwargs.get("events") == [] and not self.delayed:
                self.delayed = True
                self.renewal_committed.set()
                await self.release_acknowledgement.wait()
            return result

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_LEASE",
            # Leave time for admission and provider startup on loaded workers;
            # the test expires the admitted claim, not the startup machinery.
            timedelta(seconds=5),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.2,
        )
        store = DelayedRenewalAcknowledgementStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_delayed_renewal_acknowledgement",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-delayed-renewal-acknowledgement",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=10)
            await asyncio.wait_for(store.renewal_committed.wait(), timeout=10)
            now["value"] = accepted_at + timedelta(seconds=10)
            monkeypatch.setattr(
                _session_compaction,
                "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
                60.0,
            )
            # Provider cancellation is the lease-deadline invariant. Returning
            # also includes bounded heartbeat/store cleanup, so do not make
            # that cleanup race a second, shorter caller-imposed deadline.
            await asyncio.wait_for(provider.cancelled.wait(), timeout=10)
            with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
                await asyncio.wait_for(task, timeout=10)
        finally:
            store.release_acknowledgement.set()
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)

        assert provider.calls == 1
        assert provider.cancelled.is_set()
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operation = checkpoint["session_operations"]["records"][
            "compact-delayed-renewal-acknowledgement"
        ]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= now["value"]

    asyncio.run(run())


def test_compaction_claim_heartbeat_cancellation_observes_concurrent_renewal_failure(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_compaction
        heartbeat_task: asyncio.Task[None] | None = None

        async def reconcile_claim(**_kwargs) -> datetime:
            return accepted_at + timedelta(minutes=5)

        async def renew_claim(**_kwargs) -> datetime:
            assert heartbeat_task is not None
            heartbeat_task.cancel("cancel heartbeat while renewal fails")
            raise RuntimeError("renewal failed concurrently with cancellation")

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        monkeypatch.setattr(engine, "_renew_compaction_operation_claim", renew_claim)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_renewal_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key="compaction-renewal-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel heartbeat while renewal fails",
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == ("renewal failed concurrently with cancellation")
        assert any(
            "claim renewal also failed during cancellation" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )

    asyncio.run(run())


def test_compaction_claim_heartbeat_cancellation_observes_concurrent_reconciliation_failure(
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_compaction
        heartbeat_task: asyncio.Task[None] | None = None

        async def reconcile_claim(**_kwargs) -> datetime:
            assert heartbeat_task is not None
            heartbeat_task.cancel("cancel heartbeat while reconciliation fails")
            raise _session_compaction.SessionCompactionAttemptSuperseded(
                "ownership changed during concurrent reconciliation"
            )

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_reconciliation_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key="compaction-reconciliation-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel heartbeat while reconciliation fails",
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        assert isinstance(
            exc_info.value.__cause__,
            _session_compaction.SessionCompactionAttemptSuperseded,
        )
        assert "ownership changed" in str(exc_info.value.__cause__)
        assert any(
            "claim reconciliation also failed during cancellation" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["reconciliation", "renewal"])
def test_compaction_claim_heartbeat_converts_concurrent_child_cancellation_to_failure(
    stage: str,
    monkeypatch,
) -> None:
    accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at,
        )
        engine = app._session_compaction
        heartbeat_task: asyncio.Task[None] | None = None

        async def cancel_child() -> datetime:
            assert heartbeat_task is not None
            child_task = asyncio.current_task()
            assert child_task is not None
            child_task.add_done_callback(
                lambda _completed: heartbeat_task.cancel(f"stop heartbeat during {stage}")
            )
            child_task.cancel(f"cancel {stage} child")
            await asyncio.sleep(0)
            raise AssertionError("cancelled child resumed unexpectedly")

        async def reconcile_claim(**_kwargs) -> datetime:
            if stage == "reconciliation":
                return await cancel_child()
            return accepted_at + timedelta(minutes=5)

        async def renew_claim(**_kwargs) -> datetime:
            if stage == "renewal":
                return await cancel_child()
            raise AssertionError("unexpected renewal")

        monkeypatch.setattr(
            engine,
            "_reconcile_compaction_operation_claim_expiry",
            reconcile_claim,
        )
        monkeypatch.setattr(engine, "_renew_compaction_operation_claim", renew_claim)
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_compaction_{stage}_child_cancellation_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        request = CompactSessionRequest(
            session_id=session.id,
            idempotency_key=f"compaction-{stage}-child-cancellation-race",
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )
        claim_expires_at = accepted_at + timedelta(minutes=5)
        task = asyncio.create_task(
            engine._heartbeat_compaction_operation_claim(
                session=session,
                request=request,
                operation_id="operation-id",
                attempt_id="attempt-id",
                claim_expires_at=claim_expires_at,
                stop=asyncio.Event(),
                state=_test_compaction_heartbeat_state(claim_expires_at),
            )
        )
        heartbeat_task = task

        with pytest.raises(
            asyncio.CancelledError, match=f"stop heartbeat during {stage}"
        ) as exc_info:
            await task

        assert task.cancelling() == 1
        assert task.cancelled()
        operational_failure = exc_info.value.__cause__
        assert isinstance(operational_failure, RuntimeError)
        assert (
            str(operational_failure)
            == f"Session compaction claim {stage} was cancelled without caller cancellation."
        )
        child_cancellation = operational_failure.__cause__
        assert isinstance(child_cancellation, asyncio.CancelledError)
        assert child_cancellation.args == (f"cancel {stage} child",)

    asyncio.run(run())


def test_claimed_compaction_surfaces_failure_carried_by_cancelled_heartbeat() -> None:
    async def run() -> None:
        heartbeat_started = asyncio.Event()

        async def heartbeat() -> None:
            heartbeat_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from RuntimeError("heartbeat ownership changed")

        heartbeat_task = asyncio.create_task(heartbeat())
        await heartbeat_started.wait()
        heartbeat_task.cancel("stop failed heartbeat")
        await asyncio.gather(heartbeat_task, return_exceptions=True)

        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True
            raise AssertionError("operation must not start after heartbeat failure")

        with pytest.raises(RuntimeError, match="heartbeat ownership changed"):
            await _session_compaction._run_while_session_operation_claimed(
                operation,
                heartbeat_task=heartbeat_task,
            )

        assert not operation_called
        assert heartbeat_task.cancelling() == 1
        assert heartbeat_task.cancelled()

    asyncio.run(run())


def test_claimed_compaction_observes_cancelled_heartbeat_cause_only_once() -> None:
    async def run() -> None:
        heartbeat_started = asyncio.Event()
        finish_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            heartbeat_started.set()
            await finish_heartbeat.wait()
            raise asyncio.CancelledError("heartbeat stopped") from RuntimeError(
                "heartbeat ownership changed during operation startup"
            )

        heartbeat_task = asyncio.create_task(heartbeat())
        await heartbeat_started.wait()
        finish_heartbeat.set()

        operation_called = False

        async def operation():
            nonlocal operation_called
            operation_called = True
            raise AssertionError("operation must not start after heartbeat failure")

        with pytest.raises(
            RuntimeError,
            match="heartbeat ownership changed during operation startup",
        ):
            await _session_compaction._run_while_session_operation_claimed(
                operation,
                heartbeat_task=heartbeat_task,
            )

        assert not operation_called
        assert heartbeat_task.cancelled()

    asyncio.run(run())


def test_compact_session_caller_cancellation_does_not_wait_for_uncertain_claim_commit(
    monkeypatch,
) -> None:
    class PostGuardStalledStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.guard_passed = asyncio.Event()
            self.release_renewal = asyncio.Event()
            self.renewal_finished = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.delayed and kwargs.get("events") == []:
                self.delayed = True
                kwargs["commit_guard"]()
                self.guard_passed.set()
                try:
                    await self.release_renewal.wait()
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.renewal_finished.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = PostGuardStalledStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_cancel_uncertain_claim_commit",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-cancel-uncertain-claim-commit",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        await asyncio.wait_for(store.guard_passed.wait(), timeout=5)
        task.cancel("cancel during uncertain claim commit")

        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during uncertain claim commit",
        ):
            await asyncio.wait_for(task, timeout=1)

        # Durable cancellation cleanup consumes the delivered request before
        # re-raising the same caller-visible cancellation signal.
        assert task.cancelling() == 0
        assert task.cancelled()
        assert provider.cancelled.is_set()
        assert not store.renewal_finished.is_set()

        store.release_renewal.set()
        await asyncio.wait_for(store.renewal_finished.wait(), timeout=5)

    asyncio.run(run())


def test_compact_session_claim_heartbeat_cannot_revive_an_expired_lease(monkeypatch) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = InMemorySessionStore(ownership_clock=lambda: now["value"])
        provider = BlockingCompactionProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_expired",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-expired",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        now["value"] = accepted_at + timedelta(minutes=6)
        with pytest.raises(RuntimeError, match="expired before (?:renewal|reconciliation)"):
            await asyncio.wait_for(task, timeout=5)

        assert provider.calls == 1
        assert provider.cancelled.is_set()
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is None or "context_compaction" not in checkpoint

    asyncio.run(run())


def test_compact_session_stalled_claim_renewal_is_bounded_by_lease_deadline(
    monkeypatch,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_LEASE",
            # A 100ms lease can expire during admission before the injected
            # stalled renewal is reached. Keep real expiry but permit startup.
            timedelta(seconds=5),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledHeartbeatStore()
        compactor = CancellableCompactor()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_stall",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-stall",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(compactor.started.wait(), timeout=10)
            await asyncio.wait_for(store.heartbeat_started.wait(), timeout=10)
            with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
                await asyncio.wait_for(task, timeout=10)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)

        assert store.heartbeat_cancelled.is_set()
        assert compactor.cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}

    asyncio.run(run())


def test_compact_session_stalled_claim_reconciliation_is_bounded_by_lease_deadline(
    monkeypatch,
) -> None:
    class StalledReconciliationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_reconciliation = False
            self.reconciliation_started = asyncio.Event()
            self.reconciliation_cancelled = asyncio.Event()

        async def transform_checkpoint_with_store_time(
            self, session_id: str, checkpoint_transform
        ) -> None:
            if self.block_reconciliation:
                self.reconciliation_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.reconciliation_cancelled.set()
                    raise
            await super().transform_checkpoint_with_store_time(
                session_id,
                checkpoint_transform,
            )

    class BlockingCompactorWithReconciliation(ContextCompactor):
        def __init__(self, store: StalledReconciliationStore) -> None:
            self.store = store
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.store.block_reconciliation = True
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(milliseconds=100),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledReconciliationStore()
        compactor = BlockingCompactorWithReconciliation(store)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_stalled_claim_reconciliation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-stalled-claim-reconciliation",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(compactor.started.wait(), timeout=10)
        await asyncio.wait_for(store.reconciliation_started.wait(), timeout=10)
        with pytest.raises(RuntimeError, match="reconciliation was not confirmed"):
            await asyncio.wait_for(task, timeout=1)

        assert store.reconciliation_cancelled.is_set()
        assert compactor.cancelled.is_set()
        assert EventType.SESSION_CHECKPOINTED not in {
            event.type for event in await store.load_events(created.id)
        }

    asyncio.run(run())


@pytest.mark.parametrize(
    "commit_started_before_stall",
    [False, True],
    ids=["before-commit-guard", "after-commit-guard"],
)
def test_sqlite_stalled_claim_renewal_cannot_keep_work_running_after_deadline(
    monkeypatch,
    tmp_path,
    commit_started_before_stall: bool,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledCommitGuardSQLiteStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(
            self,
            work_started: asyncio.Event,
            *,
            ownership_clock: Callable[[], datetime],
        ) -> None:
            super().__init__(
                tmp_path / "stalled-renewal.sqlite",
                ownership_clock=ownership_clock,
            )
            self.work_started = work_started
            self.guard_started = threading.Event()
            self.release_guard = threading.Event()
            self.failure_publication_started = asyncio.Event()
            self.failure_publication_finished = asyncio.Event()
            self.delayed_renewal = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.failure_publication_started.set()
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.failure_publication_finished.set()
            commit_guard = kwargs.get("commit_guard")
            if (
                self.work_started.is_set()
                and not self.delayed_renewal
                and kwargs.get("events") == []
                and commit_guard is not None
            ):
                self.delayed_renewal = True

                def delayed_commit_guard() -> None:
                    if commit_started_before_stall:
                        commit_guard()
                    self.guard_started.set()
                    if not self.release_guard.wait(timeout=15):
                        raise TimeoutError("test did not release the renewal commit guard")
                    if not commit_started_before_stall:
                        commit_guard()

                kwargs["commit_guard"] = delayed_commit_guard
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(seconds=5),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CancellableCompactor()
        store = StalledCommitGuardSQLiteStore(
            compactor.started,
            ownership_clock=lambda: now["value"],
        )
        try:
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                clock=lambda: now["value"],
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_stalled_renewal",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)

            async def collect() -> list[Event]:
                return [
                    event
                    async for event in app.compact_session(
                        CompactSessionRequest(
                            session_id=created.id,
                            idempotency_key="compact-sqlite-stalled-renewal",
                            expected_run_epoch=completed.run_epoch,
                            expected_transcript_cursor=len(transcript),
                        )
                    )
                ]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=15)
            await asyncio.wait_for(
                asyncio.to_thread(store.guard_started.wait),
                timeout=15,
            )
            now["value"] = accepted_at + timedelta(seconds=10)
            await asyncio.wait_for(compactor.cancelled.wait(), timeout=15)

            with pytest.raises(RuntimeError, match="not confirmed before its lease deadline"):
                await asyncio.wait_for(task, timeout=15)

            assert not store.failure_publication_started.is_set()
            assert not store.failure_publication_finished.is_set()
            store.release_guard.set()
            await asyncio.wait_for(store.failure_publication_started.wait(), timeout=15)
            await asyncio.wait_for(store.failure_publication_finished.wait(), timeout=15)

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            record = checkpoint["session_operations"]["records"]["compact-sqlite-stalled-renewal"]
            assert datetime.fromisoformat(record["claim_expires_at"]) == (
                accepted_at + timedelta(seconds=5)
            )
            assert "context_compaction" not in checkpoint
            assert EventType.SESSION_CHECKPOINTED not in {
                event.type for event in await store.load_events(created.id)
            }
        finally:
            store.release_guard.set()
            await store.close()

    asyncio.run(run())


def test_sqlite_caller_cancellation_does_not_wait_for_stalled_claim_write(
    monkeypatch,
    tmp_path,
) -> None:
    class CancellableCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class StalledClaimWriteSQLiteStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, work_started: asyncio.Event) -> None:
            super().__init__(tmp_path / "cancel-stalled-renewal.sqlite")
            self.work_started = work_started
            self.guard_started = threading.Event()
            self.release_guard = threading.Event()
            self.failure_publication_finished = asyncio.Event()
            self.delayed_renewal = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.failure_publication_finished.set()
            commit_guard = kwargs.get("commit_guard")
            if (
                self.work_started.is_set()
                and not self.delayed_renewal
                and kwargs.get("events") == []
                and commit_guard is not None
            ):
                self.delayed_renewal = True

                def delayed_commit_guard() -> None:
                    commit_guard()
                    self.guard_started.set()
                    if not self.release_guard.wait(timeout=5):
                        raise TimeoutError("test did not release the renewal commit guard")

                kwargs["commit_guard"] = delayed_commit_guard
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CancellableCompactor()
        store = StalledClaimWriteSQLiteStore(compactor.started)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor,
                    max_user_turns=1,
                ),
            )
            created = await _create_profiled_session(
                app,
                store,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_cancel_stalled_claim_write",
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            transcript = [
                Message.text("user", "old request"),
                Message.text("assistant", "old answer"),
                Message.text("user", "current request"),
            ]
            await store.append_transcript_messages(created.id, transcript)
            completed = await store.update_status(created.id, SessionStatus.COMPLETED)

            async def collect() -> list[Event]:
                return [
                    event
                    async for event in app.compact_session(
                        CompactSessionRequest(
                            session_id=created.id,
                            idempotency_key="compact-sqlite-cancel-stalled-claim-write",
                            expected_run_epoch=completed.run_epoch,
                            expected_transcript_cursor=len(transcript),
                        )
                    )
                ]

            task = asyncio.create_task(collect())
            await asyncio.wait_for(compactor.started.wait(), timeout=10)
            await asyncio.wait_for(
                asyncio.to_thread(store.guard_started.wait),
                timeout=5,
            )
            task.cancel("cancel while claim commit is stalled")

            with pytest.raises(
                asyncio.CancelledError,
                match="cancel while claim commit is stalled",
            ):
                await asyncio.wait_for(task, timeout=1)

            assert task.cancelled()
            assert compactor.cancelled.is_set()
            assert not store.failure_publication_finished.is_set()
            store.release_guard.set()
            await asyncio.wait_for(store.failure_publication_finished.wait(), timeout=5)
            assert EventType.CONTEXT_COMPACTION_FAILED in {
                event.type for event in await store.load_events(created.id)
            }
        finally:
            store.release_guard.set()
            await store.close()

    asyncio.run(run())


def test_compact_session_expired_claim_cannot_publish_terminal_checkpoint(monkeypatch) -> None:
    class StalledHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        worker_now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = StalledHeartbeatStore(ownership_clock=lambda: store_now["value"])
        compactor = BlockingCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: worker_now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_terminal_after_claim_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-terminal-after-claim-expiry",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(compactor.started.wait(), timeout=10)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=10)
        store_now["value"] = accepted_at + timedelta(minutes=6)
        compactor.release.set()
        with pytest.raises(RuntimeError, match="expired before terminal publication"):
            await asyncio.wait_for(task, timeout=5)

        assert store.heartbeat_cancelled.is_set()
        durable_events = [
            record.event
            for record in await store.query_events(EventQuery(session_id=created.id, limit=100))
        ]
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable_events}
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "context_compaction" not in checkpoint
        operation = checkpoint["session_operations"]["records"][
            "compact-terminal-after-claim-expiry"
        ]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= store_now["value"]

        recovered_events = await collect()
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED

    asyncio.run(run())


def test_compact_session_failure_publication_cannot_terminalize_after_claim_expiry() -> None:
    class FailOnceCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.calls = 0

        def provider_budget_identity(self, _session) -> None:
            return None

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("compaction failed before delayed publication")
            return CompactionResult(
                summary="recovered summary",
                covered_message_count=len(request.messages),
            )

    class DelayedFailureStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.failure_publication_started = asyncio.Event()
            self.release_failure_publication = asyncio.Event()
            self.delayed = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.delayed and any(
                event.type == EventType.CONTEXT_COMPACTION_FAILED
                for event in kwargs.get("events", [])
            ):
                self.delayed = True
                self.failure_publication_started.set()
                await self.release_failure_publication.wait()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        store = DelayedFailureStore(ownership_clock=lambda: now["value"])
        compactor = FailOnceCompactor()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_failure_publication_after_claim_expiry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-failure-after-claim-expiry",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        async def collect() -> list[Event]:
            return [event async for event in app.compact_session(request)]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(store.failure_publication_started.wait(), timeout=10)
        now["value"] = accepted_at + timedelta(minutes=6)
        store.release_failure_publication.set()
        with pytest.raises(RuntimeError, match="failed before delayed publication"):
            await asyncio.wait_for(task, timeout=5)

        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operation = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert operation["status"] == "running"
        assert datetime.fromisoformat(operation["claim_expires_at"]) <= now["value"]

        recovered_events = await collect()
        assert recovered_events[-1].type == EventType.SESSION_CHECKPOINTED
        assert compactor.calls == 2

    asyncio.run(run())


def test_compact_session_terminal_publication_wins_blocked_heartbeat_race(monkeypatch) -> None:
    class BlockingHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = BlockingHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_terminal_race",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-terminal-race",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=10)
        provider.release.set()
        events = await asyncio.wait_for(task, timeout=5)

        assert events[-1].type == EventType.SESSION_CHECKPOINTED
        assert store.heartbeat_cancelled.is_set()
        operation = await store.load_session_operation(
            created.id, "compact-claim-heartbeat-terminal-race"
        )
        assert operation is not None
        assert operation["status"] == "completed"

    asyncio.run(run())


def test_compact_session_claim_loss_waits_for_completed_dispatch_settlement(monkeypatch) -> None:
    class BlockingReconcileLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__()
            self.reconcile_started = asyncio.Event()
            self.release_reconcile = asyncio.Event()
            self.reconcile_cancelled = asyncio.Event()

        async def reconcile(self, **kwargs):
            self.reconcile_started.set()
            try:
                await self.release_reconcile.wait()
            except asyncio.CancelledError:
                self.reconcile_cancelled.set()
                raise
            return await super().reconcile(**kwargs)

    class ClaimLossDuringReconcileStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, ledger: BlockingReconcileLedger) -> None:
            super().__init__()
            self.ledger = ledger
            self.heartbeat_failed = asyncio.Event()
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                await self.ledger.reconcile_started.wait()
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
                try:
                    return await super().publish_session_operation_guarded_with_store_time(
                        session_id,
                        **kwargs,
                    )
                finally:
                    self.heartbeat_failed.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="compaction-provider",
                    model="summary-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        )
        ledger = BlockingReconcileLedger()
        store = ClaimLossDuringReconcileStore(ledger)
        provider = UsageCompactionProvider()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=pricing,
                        reservation=BudgetReservation(
                            max_input_tokens=100,
                            max_output_tokens=100,
                        ),
                    ),
                )
            ),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_loss_during_settlement",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-loss-during-settlement",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(ledger.reconcile_started.wait(), timeout=10)
        await asyncio.wait_for(store.heartbeat_failed.wait(), timeout=5)
        await asyncio.sleep(0)
        assert not task.done()
        assert not ledger.reconcile_cancelled.is_set()

        ledger.release_reconcile.set()
        with pytest.raises(RuntimeError, match="superseded"):
            await asyncio.wait_for(task, timeout=5)

        durable = await store.load_events(created.id)
        completions = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        reconciliations = [event for event in durable if event.type == EventType.BUDGET_RECONCILED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert len(reconciliations) == 1
        assert reconciliations[0].payload["actual_amount"] == "0.000012"
        record = next(iter(ledger._records.values()))
        assert record.status == "reconciled"
        assert record.actual_amount == Decimal("0.000012")
        assert provider.calls == 1

    asyncio.run(run())


def test_compact_session_claim_loss_retains_concurrent_result_telemetry(monkeypatch) -> None:
    class CompletionReportingProvider(ModelProvider):
        name = "reported-provider"

        def __init__(self) -> None:
            self.calls = 0
            self.result_ready = asyncio.Event()
            self.release_result = asyncio.Event()

        async def stream(self, request: ModelRequest):
            del request
            self.calls += 1
            self.result_ready.set()
            await self.release_result.wait()
            yield ModelStreamEvent.text_delta("completed before claim loss")
            yield ModelStreamEvent.completed(
                {
                    "model": "summary-model",
                    "usage": {"input_tokens": 8, "output_tokens": 2},
                }
            )

    class CompletionReportingCompactor(ContextCompactor):
        def __init__(self) -> None:
            self.provider = CompletionReportingProvider()
            self.calls = 0
            self.result_ready = self.provider.result_ready
            self.release_result = self.provider.release_result

        async def compact(self, request: CompactionRequest) -> CompactionResult:
            self.calls += 1
            return await ModelCompactor(
                provider=self.provider,
                model="summary-model",
                max_input_chars=100_000,
            ).compact(request)

    class ConcurrentClaimLossStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, compactor: CompletionReportingCompactor) -> None:
            super().__init__()
            self.compactor = compactor
            self.stolen = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stolen and kwargs.get("events") == []:
                await self.compactor.result_ready.wait()
                self.stolen = True

                def steal(_session, checkpoint):
                    assert checkpoint is not None
                    updated = dict(checkpoint)
                    operations = dict(updated["session_operations"])
                    records = dict(operations["records"])
                    key, record = next(iter(records.items()))
                    records[key] = {**record, "current_attempt_id": "replacement-attempt"}
                    operations["records"] = records
                    updated["session_operations"] = operations
                    return updated

                await self.transform_checkpoint(session_id, steal)
                self.compactor.release_result.set()
                await asyncio.sleep(0)
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        compactor = CompletionReportingCompactor()
        store = ConcurrentClaimLossStore(compactor)
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(enabled=False),
            enable_logging=False,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=compactor,
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_loss_result_telemetry",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        with pytest.raises(RuntimeError, match="superseded"):
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-claim-loss-result-telemetry",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            ):
                pass

        durable = await store.load_events(created.id)
        completions = [event for event in durable if event.type == EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["usage_metrics"]["total_tokens"] == 10
        assert EventType.SESSION_CHECKPOINTED not in {event.type for event in durable}
        assert compactor.calls == 1

    asyncio.run(run())


def test_compact_session_caller_cancellation_interrupts_blocked_claim_heartbeat(
    monkeypatch,
) -> None:
    class BlockingHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.heartbeat_cancelled = asyncio.Event()

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if kwargs.get("events") == []:
                self.heartbeat_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.heartbeat_cancelled.set()
                    raise
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    async def run() -> None:
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )
        store = BlockingHeartbeatStore()
        provider = BlockingCompactionProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model="summary-model"),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_heartbeat_cancel",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-claim-heartbeat-cancel",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        await asyncio.wait_for(store.heartbeat_started.wait(), timeout=10)
        task.cancel("cancel while claim heartbeat write is blocked")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(task, timeout=5)

        assert exc_info.value.args == ("cancel while claim heartbeat write is blocked",)
        assert task.cancelled()
        assert provider.cancelled.is_set()
        assert store.heartbeat_cancelled.is_set()

    asyncio.run(run())


def test_compact_session_renews_operation_claim_between_provider_dispatches(monkeypatch) -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            1.0,
        )

        class DelayedAttemptPublicationStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self) -> None:
                super().__init__(ownership_clock=lambda: now["value"])
                self.delayed = False

            async def publish_session_operation_guarded_with_store_time(
                self, session_id: str, **kwargs
            ):
                events = kwargs["events"]
                if not self.delayed and any(
                    event.type == EventType.MODEL_COMPLETED for event in events
                ):
                    self.delayed = True
                    # The operation transform was already created at minute four,
                    # but the store does not execute it until thirty seconds later.
                    now["value"] = accepted_at + timedelta(minutes=4, seconds=30)
                return await super().publish_session_operation_guarded_with_store_time(
                    session_id, **kwargs
                )

        class MultiDispatchProvider(ModelProvider):
            name = "compaction-provider"

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []
                self.second_dispatch_started = asyncio.Event()
                self.release_second_dispatch = asyncio.Event()

            async def stream(self, request: ModelRequest):
                self.requests.append(request)
                call = len(self.requests)
                if call == 1:
                    # The first durable completion should renew the claim from
                    # minute four through minute nine.
                    now["value"] = accepted_at + timedelta(minutes=4)
                elif call == 2:
                    self.second_dispatch_started.set()
                    await self.release_second_dispatch.wait()
                yield ModelStreamEvent.text_delta(f"summary {call}")
                yield ModelStreamEvent.completed(
                    {
                        "model": request.model,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )

        store = DelayedAttemptPublicationStore()
        provider = MultiDispatchProvider()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_claim_renewal",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        request = CompactSessionRequest(
            session_id=created.id,
            idempotency_key="compact-claim-renewal",
            expected_run_epoch=completed.run_epoch,
            expected_transcript_cursor=len(transcript),
        )

        first_events: list[Event] = []

        async def compact() -> None:
            async for event in app.compact_session(request):
                first_events.append(event)

        first_task = asyncio.create_task(compact())
        await asyncio.wait_for(provider.second_dispatch_started.wait(), timeout=10)
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        record = checkpoint["session_operations"]["records"][request.idempotency_key]
        assert datetime.fromisoformat(record["updated_at"]) == accepted_at + timedelta(
            minutes=4, seconds=30
        )
        assert datetime.fromisoformat(record["claim_expires_at"]) == (
            accepted_at + timedelta(minutes=9, seconds=30)
        )

        # The original claim has expired, but the first attempt's atomic event
        # publication renewed it. The renewed lease must fence an overlapping
        # same-key retry while the second dispatch is in flight.
        now["value"] = accepted_at + timedelta(minutes=6)
        heartbeat_expiry = accepted_at + timedelta(minutes=11)
        async with asyncio.timeout(5):
            while True:
                checkpoint = await store.load_checkpoint(created.id)
                assert checkpoint is not None
                record = checkpoint["session_operations"]["records"][request.idempotency_key]
                if datetime.fromisoformat(record["claim_expires_at"]) >= heartbeat_expiry:
                    break
                await asyncio.sleep(0)
        assert not first_task.done()
        with pytest.raises(RuntimeError, match="already running"):
            async for _event in app.compact_session(request):
                pass
        assert len(provider.requests) == 2

        provider.release_second_dispatch.set()
        await first_task
        assert first_events[-1].type == EventType.SESSION_CHECKPOINTED

    asyncio.run(run())


def test_compact_session_heartbeat_timeout_honors_concurrent_publication_renewal(
    monkeypatch,
) -> None:
    class StalledFirstHeartbeatStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self, *, ownership_clock: Callable[[], datetime]) -> None:
            super().__init__(ownership_clock=ownership_clock)
            self.heartbeat_started = asyncio.Event()
            self.release_heartbeat = asyncio.Event()
            self.heartbeat_retried = asyncio.Event()
            self.stalled = False

        async def publish_session_operation_guarded_with_store_time(
            self, session_id: str, **kwargs
        ):
            if not self.stalled and kwargs.get("events") == []:
                self.stalled = True
                self.heartbeat_started.set()
                await self.release_heartbeat.wait()
            elif kwargs.get("events") == []:
                self.heartbeat_retried.set()
            return await super().publish_session_operation_guarded_with_store_time(
                session_id, **kwargs
            )

    class PublishWhileHeartbeatStalledProvider(ModelProvider):
        name = "compaction-provider"

        def __init__(
            self,
            *,
            store: StalledFirstHeartbeatStore,
            advance_clock: Callable[[], None],
        ) -> None:
            self._store = store
            self._advance_clock = advance_clock
            self.calls = 0
            self.second_dispatch_started = asyncio.Event()
            self.release_second_dispatch = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.calls += 1
            if self.calls == 1:
                await self._store.heartbeat_started.wait()
                # Publish the first attempt late enough that its guarded claim
                # renewal extends beyond the original local lease deadline.
                self._advance_clock()
            else:
                self.second_dispatch_started.set()
                await self.release_second_dispatch.wait()
            yield ModelStreamEvent.text_delta(f"summary {self.calls}")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

    async def run() -> None:
        accepted_at = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        now = {"value": accepted_at}
        monotonic_now = {"value": 0.0}
        monkeypatch.setattr(
            session_engine_module,
            "time",
            SimpleNamespace(monotonic=lambda: monotonic_now["value"]),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_LEASE",
            timedelta(seconds=2),
        )
        monkeypatch.setattr(
            _session_compaction,
            "_SESSION_OPERATION_CLAIM_HEARTBEAT_INTERVAL_SECONDS",
            0.01,
        )

        def advance_publication_clock() -> None:
            now["value"] = accepted_at + timedelta(seconds=1)
            monotonic_now["value"] = 1.0

        store = StalledFirstHeartbeatStore(ownership_clock=lambda: now["value"])
        provider = PublishWhileHeartbeatStalledProvider(
            store=store,
            advance_clock=advance_publication_clock,
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: now["value"],
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=provider,
                    model="summary-model",
                    max_input_chars=1000,
                ),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_concurrent_publication_renewal",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "oversized " + "x" * 5000),
            Message.text("user", "current"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.compact_session(
                    CompactSessionRequest(
                        session_id=created.id,
                        idempotency_key="compact-concurrent-publication-renewal",
                        expected_run_epoch=completed.run_epoch,
                        expected_transcript_cursor=len(transcript),
                    )
                )
            ]

        task = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(provider.second_dispatch_started.wait(), timeout=10)
            # This is past the original two-second deadline, but remains
            # inside the two-second lease measured from event publication.
            monotonic_now["value"] = 2.1
            # Wait for the stalled renewal to time out and a new heartbeat to
            # start, with the runtime clock fixed inside the renewed lease.
            await asyncio.wait_for(store.heartbeat_retried.wait(), timeout=10)
            assert not task.done()
            assert provider.calls == 2

            checkpoint = await store.load_checkpoint(created.id)
            assert checkpoint is not None
            record = checkpoint["session_operations"]["records"][
                "compact-concurrent-publication-renewal"
            ]
            assert datetime.fromisoformat(record["claim_expires_at"]) > (
                accepted_at + timedelta(seconds=2)
            )

            # Let the second attempt publish under later authoritative store
            # time while the old heartbeat store call is still stalled. Its
            # own dispatch point establishes the next conservative deadline.
            now["value"] = accepted_at + timedelta(seconds=2)
            provider.release_second_dispatch.set()
            async with asyncio.timeout(5):
                while True:
                    checkpoint = await store.load_checkpoint(created.id)
                    assert checkpoint is not None
                    record = checkpoint["session_operations"]["records"][
                        "compact-concurrent-publication-renewal"
                    ]
                    if datetime.fromisoformat(record["claim_expires_at"]) >= (
                        accepted_at + timedelta(seconds=4)
                    ):
                        break
                    await asyncio.sleep(0)
            store.release_heartbeat.set()
            events = await asyncio.wait_for(task, timeout=5)
            assert events[-1].type == EventType.SESSION_CHECKPOINTED
        finally:
            store.release_heartbeat.set()
            provider.release_second_dispatch.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_expired_compaction_claim_does_not_block_resume() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_provider(CompletingProvider())
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_resume",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=first_app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-resume",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        store_now["value"] = accepted_at + timedelta(minutes=6)
        resumed_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        resumed_app.register_provider(CompletingProvider())
        resumed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
                compact_after_messages=100,
            ),
        )
        events = [
            event
            async for event in resumed_app.resume(
                ResumeRequest(
                    session_id=created.id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert EventType.SESSION_COMPLETED in [event.type for event in events]
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        operations = checkpoint["session_operations"]
        assert operations["active_operation_id"] is None
        assert operations["records"]["compact-expired-resume"]["status"] == "abandoned"

    asyncio.run(run())


def test_compaction_store_wait_ignores_application_clock_skew() -> None:
    async def run() -> None:
        store_now = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        application_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = InMemorySessionStore(ownership_clock=lambda: store_now)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: application_now,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_application_clock_skew",
                messages=[Message.text("user", "create only")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)

        events = [
            event
            async for event in app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-with-skewed-application-clock",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]
        operation = await store.load_session_operation(
            created.id,
            "compact-with-skewed-application-clock",
        )
        return events, operation

    events, operation = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_CHECKPOINTED
    assert operation is not None
    assert operation["status"] == "completed"


def test_expired_compaction_claim_does_not_block_fork_or_a_new_key() -> None:
    async def run() -> None:
        accepted_at = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
        store_now = {"value": accepted_at}
        store = InMemorySessionStore(ownership_clock=lambda: store_now["value"])
        first_app = CayuApp(session_store=store, enable_logging=False, clock=lambda: accepted_at)
        first_app.register_provider(CompletingProvider())
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        created = await _create_profiled_session(
            first_app,
            store,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expired_compact_fork",
                messages=[Message.text("user", "create only")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
                app=first_app,
            ),
        )
        transcript = [
            Message.text("user", "old request"),
            Message.text("assistant", "old answer"),
            Message.text("user", "current request"),
            Message.text("assistant", "current answer"),
        ]
        await store.append_transcript_messages(created.id, transcript)
        completed = await store.update_status(created.id, SessionStatus.COMPLETED)
        abandoned = first_app.compact_session(
            CompactSessionRequest(
                session_id=created.id,
                idempotency_key="compact-expired-fork",
                expected_run_epoch=completed.run_epoch,
                expected_transcript_cursor=len(transcript),
            )
        )
        await anext(abandoned)
        await abandoned.aclose()

        store_now["value"] = accepted_at + timedelta(minutes=6)
        recovered_app = CayuApp(
            session_store=store,
            enable_logging=False,
            clock=lambda: accepted_at + timedelta(minutes=6),
        )
        recovered_app.register_provider(CompletingProvider())
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=RecordingCompactor(),
                max_user_turns=1,
            ),
        )
        forked = [
            event
            async for event in recovered_app.fork_session(
                ForkSessionRequest(
                    source_session_id=created.id,
                    session_id="sess_after_expired_compact",
                )
            )
        ]
        compacted = [
            event
            async for event in recovered_app.compact_session(
                CompactSessionRequest(
                    session_id=created.id,
                    idempotency_key="compact-after-expired-key",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=len(transcript),
                )
            )
        ]

        assert forked[-1].type == EventType.SESSION_FORKED
        assert compacted[-1].type == EventType.SESSION_CHECKPOINTED
        checkpoint = await store.load_checkpoint(created.id)
        assert checkpoint is not None
        assert "session_operations" not in checkpoint
        abandoned_operation = await store.load_session_operation(
            created.id,
            "compact-expired-fork",
        )
        completed_operation = await store.load_session_operation(
            created.id,
            "compact-after-expired-key",
        )
        assert abandoned_operation is not None
        assert abandoned_operation["status"] == "abandoned"
        assert completed_operation is not None
        assert completed_operation["status"] == "completed"

    asyncio.run(run())
