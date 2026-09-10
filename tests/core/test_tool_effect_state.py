from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.core import Event, EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime._tool_effect_state import (
    ToolEffectConflict,
    ToolEffectIntent,
    ToolEffectObservation,
    ToolEffectStateOwner,
    ToolEffectTerminal,
    validate_tool_effect_uncertainty_event,
)
from cayu.runtime.sessions import RuntimePublicationCheckpointOperation, RuntimePublicationMutation
from cayu.runtime.tool_effects import ToolEffectReceipt, ToolEffectReconciliationResult
from cayu.storage.sqlite import SQLiteSessionStore


async def _intent(store, *, session_id="effect-session") -> ToolEffectIntent:
    session = await store.create(
        RunRequest(
            session_id=session_id, agent_name="agent", messages=[Message.text("user", "go")]
        ),
        identity=SessionIdentity(provider_name="scripted", model="test"),
    )
    return ToolEffectIntent(
        session_id=session.id,
        session_instance_id=session.instance_id,
        source_run_epoch=session.run_epoch,
        interaction_id="interaction",
        model_step_id="step",
        model_attempt_id="attempt",
        tool_round_id="round",
        tool_call_id="call",
        agent_name="agent",
        tool_name="deploy",
        idempotency_key="key",
        execution_profile_fingerprint="profile",
        schema_digest="a" * 64,
        arguments_digest="b" * 64,
    )


def _event(intent, *, event_id="terminal", failed=False):
    return Event(
        id=event_id,
        session_id=intent.session_id,
        agent_name=intent.agent_name,
        tool_name=intent.tool_name,
        type=EventType.TOOL_CALL_FAILED if failed else EventType.TOOL_CALL_COMPLETED,
        payload={
            **{
                field: getattr(intent, field)
                for field in (
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "tool_call_id",
                    "idempotency_key",
                )
            },
            "result": {"content": "done", "is_error": failed, "artifacts": [], "structured": None},
        },
    )


@pytest.mark.parametrize("mismatch", ["session", "epoch", "transition"])
def test_uncertainty_diagnostic_mismatch_does_not_publish(mismatch):
    from cayu.failure_evidence import FailureEvidence

    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
        before = await store.load_events(intent.session_id)
        evidence = FailureEvidence(
            classification="timeout",
            session_id="different" if mismatch == "session" else intent.session_id,
            run_epoch=intent.source_run_epoch + (1 if mismatch == "epoch" else 0),
        )
        event = _event(intent)
        with pytest.raises(ToolEffectConflict):
            await owner.transition(
                executing,
                state="completed" if mismatch == "transition" else "outcome_unknown",
                run_epoch=intent.source_run_epoch,
                failure_evidence=evidence,
                terminal=_terminal(event) if mismatch == "transition" else None,
                events=(event,) if mismatch == "transition" else (),
            )
        assert await owner.load(intent) == executing
        assert await store.load_events(intent.session_id) == before

    asyncio.run(scenario())


def _terminal(event, receipt=None):
    return ToolEffectTerminal(
        event_id=event.id,
        result_digest=sha256(
            canonical_durable_json_bytes(event.payload["result"], "result")
        ).hexdigest(),
        receipt=receipt,
        reconciliation_request_digest=None if receipt is None else "c" * 64,
    )


def _receipt(intent):
    return ToolEffectReceipt(
        receipt_id="receipt",
        receipt_schema="deployment",
        receipt_schema_version=1,
        tool_call_id=intent.tool_call_id,
        tool_name=intent.tool_name,
        idempotency_key=intent.idempotency_key,
        outcome="completed",
        message="done",
        source="reconciler",
        observed_at=datetime(2026, 9, 8, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("denied_by", "other_policy"),
        ("denied_by", None),
        ("decision", "allow"),
        ("decision", "unknown"),
        ("decision", {}),
        ("result", {"content": "Command denied by policy.", "is_error": False}),
    ],
)
def test_effect_rejects_unproven_blocked_terminal(field, value):
    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        executing = await owner.begin(intent, run_epoch=0)
        failed = _event(intent, failed=True)
        event = failed.model_copy(
            update={
                "type": EventType.TOOL_CALL_BLOCKED,
                "payload": {
                    **failed.payload,
                    "denied_by": "command_policy",
                    "decision": "deny",
                    field: value,
                },
            }
        )
        with pytest.raises(ToolEffectConflict):
            await owner.transition(
                executing, state="failed", run_epoch=0, terminal=_terminal(event), events=(event,)
            )
        assert await owner.load(intent) == executing
        assert await store.load_events(intent.session_id) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("executed", None),
        ("executed", True),
        ("executed", 0),
        ("outcome_unknown", True),
        ("outcome_unknown", 0),
    ],
)
def test_prepared_settlement_rejects_ambiguous_terminal(field, value):
    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        session = await store.load(intent.session_id)
        assert await owner.preserve_unresolved(
            session, tool_round_id=intent.tool_round_id, tool_call_ids=(intent.tool_call_id,)
        )
        assert await owner.load(intent) == prepared
        assert await store.load_events(intent.session_id) == []
        event = _event(intent, failed=True)
        event.payload["result"]["structured"] = {
            "recovery_reason": "tool_effect_not_dispatched",
            "executed": False,
            "outcome_unknown": False,
            field: value,
        }
        with pytest.raises(ToolEffectConflict, match="non-execution evidence"):
            await owner.transition(
                prepared, state="failed", run_epoch=0, terminal=_terminal(event), events=(event,)
            )
        assert await owner.load(intent) == prepared
        assert await store.load_events(intent.session_id) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("input_id", ["pause", "different-pause", None])
def test_selected_terminal_requires_exact_user_input_linkage(backend, input_id, tmp_path):
    async def scenario(store):
        original = await _intent(store)
        intent = ToolEffectIntent(**(original.model_dump() | {"pause_id": "pause"}))
        owner = ToolEffectStateOwner(store)
        executing = await owner.begin(intent, run_epoch=0)
        unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
        before_events = await store.load_events(intent.session_id)
        event = _event(intent)
        if input_id is not None:
            event.payload["input_id"] = input_id
        receipt = _receipt(intent)

        async def settle():
            return await owner.transition(
                unknown,
                state="reconciled_completed",
                terminal=_terminal(event, receipt),
                events=(event,),
                run_epoch=0,
            )

        if input_id == "pause":
            selected = await settle()
            assert selected.state == "reconciled_completed"
            assert selected.terminal.receipt == receipt
            assert (await store.load_events(intent.session_id))[-1].payload["input_id"] == "pause"
        else:
            with pytest.raises(ToolEffectConflict, match="terminal material conflicts"):
                await settle()
            assert await owner.load(intent) == unknown
            assert await store.load_events(intent.session_id) == before_events

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "pause-terminal.sqlite")
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_nonterminal_observation_is_atomic_exact_and_retains_partial_evidence(
    backend, lose_acknowledgement, tmp_path, monkeypatch
):
    async def scenario():
        path = str(tmp_path / "observation.db")
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=0)
            unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)

            def evidence(record, *, event_id, resources, digest="c" * 64):
                observation = ToolEffectObservation(
                    event_id=event_id,
                    request_digest=digest,
                    result=ToolEffectReconciliationResult(
                        outcome="not_found",
                        observation="partial",
                        resource_versions=resources,
                        retryable=True,
                    ),
                )
                event = Event(
                    id=event_id,
                    type=EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED,
                    session_id=intent.session_id,
                    interaction_id=intent.interaction_id,
                    agent_name=intent.agent_name,
                    tool_name=intent.tool_name,
                    payload={
                        "schema_version": 1,
                        "execution_profile_fingerprint": intent.execution_profile_fingerprint,
                        **{
                            name: getattr(intent, name)
                            for name in (
                                "model_step_id",
                                "model_attempt_id",
                                "tool_round_id",
                                "tool_call_id",
                                "idempotency_key",
                            )
                        },
                        "request_digest": digest,
                        "result": observation.result.model_dump(mode="json"),
                        "resource_versions": {**record.resource_versions, **resources},
                    },
                )
                return observation, event

            observation, event = evidence(unknown, event_id="partial", resources={"part-1": "v1"})
            with pytest.raises(ToolEffectConflict):
                await owner.transition(unknown, state="outcome_unknown", run_epoch=0)
            with pytest.raises(ToolEffectConflict):
                await owner.transition(
                    unknown, state="outcome_unknown", run_epoch=0, observation=observation
                )
            assert await owner.load(intent) == unknown
            uncertainty_events = await store.load_events(intent.session_id)
            assert [item.type for item in uncertainty_events] == [
                EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
            ]
            publish = store.publish_session_operation
            lose_once = lose_acknowledgement

            async def publish_then_lose_ack(*args, **kwargs):
                nonlocal lose_once
                result = await publish(*args, **kwargs)
                if lose_once:
                    lose_once = False
                    raise OSError("observation committed but acknowledgement lost")
                return result

            monkeypatch.setattr(store, "publish_session_operation", publish_then_lose_ack)
            observed = await owner.transition(
                unknown,
                state="outcome_unknown",
                run_epoch=0,
                observation=observation,
                events=(event,),
            )
            assert not lose_once
            assert (
                await owner.transition(
                    unknown,
                    state="outcome_unknown",
                    run_epoch=0,
                    observation=observation,
                    events=(event,),
                )
                == observed
            )
            assert len(await store.load_events(intent.session_id)) == 2
            assert observed.terminal is None
            assert observed.resource_versions == {"part-1": "v1"}
            with pytest.raises(ToolEffectConflict):
                await owner.begin(intent, run_epoch=0)

            if isinstance(store, SQLiteSessionStore):
                await store.close()
                store = SQLiteSessionStore(path)
            owner = ToolEffectStateOwner(store)
            assert await owner.load(intent) == observed
            empty, empty_event = evidence(observed, event_id="empty", resources={}, digest="d" * 64)
            later = await owner.transition(
                observed,
                state="outcome_unknown",
                run_epoch=0,
                observation=empty,
                events=(empty_event,),
            )
            assert later.resource_versions == {"part-1": "v1"}
            for resources in ({"part-1": "v2"}, {f"new-{i}": "v1" for i in range(32)}):
                conflicting, conflicting_event = evidence(
                    later, event_id="conflict", resources=resources
                )
                with pytest.raises((ToolEffectConflict, ValueError)):
                    await owner.transition(
                        later,
                        state="outcome_unknown",
                        run_epoch=0,
                        observation=conflicting,
                        events=(conflicting_event,),
                    )
                assert await owner.load(intent) == later
                assert len(await store.load_events(intent.session_id)) == 3
            terminal_event = _event(intent)
            settled = await owner.transition(
                later,
                state="completed",
                run_epoch=0,
                terminal=_terminal(terminal_event),
                events=(terminal_event,),
            )
            assert settled.resource_versions == {"part-1": "v1"}
            assert settled.observation == later.observation
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_effect_state_exact_commit_replay_conflict_and_reconstruction(backend, tmp_path):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(str(tmp_path / "effects.db"))
        )
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        assert await owner.prepare(intent, run_epoch=0) == prepared
        executing = await owner.transition(prepared, state="executing", run_epoch=0)
        with pytest.raises(ToolEffectConflict):
            await owner.transition(prepared, state="executing", run_epoch=0)
        unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
        event = _event(intent)
        receipt = _receipt(intent)
        mutation = RuntimePublicationMutation(
            operations=(
                RuntimePublicationCheckpointOperation(
                    key="effect_test_projection",
                    expected_value_digest=None,
                    action="set",
                    value={"selected": event.id},
                ),
            )
        )
        settled = await owner.transition(
            unknown,
            state="reconciled_completed",
            terminal=_terminal(event, receipt),
            run_epoch=0,
            mutation=mutation,
            events=(event,),
        )
        assert settled.revision == 3
        assert (
            await owner.transition(
                unknown,
                state="reconciled_completed",
                terminal=_terminal(event, receipt),
                run_epoch=0,
                mutation=mutation,
                events=(event,),
            )
            == settled
        )
        assert [item.type for item in await store.load_events(intent.session_id)] == [
            EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
            EventType.TOOL_CALL_COMPLETED,
        ]
        assert (await store.load_checkpoint(intent.session_id))["effect_test_projection"] == {
            "selected": event.id
        }
        conflicting = receipt.model_copy(update={"receipt_id": "another"})
        with pytest.raises(ToolEffectConflict):
            await owner.transition(
                unknown,
                state="reconciled_completed",
                terminal=_terminal(event, conflicting),
                run_epoch=0,
                mutation=mutation,
                events=(event,),
            )
        if backend == "sqlite":
            await store.close()
            store = SQLiteSessionStore(str(tmp_path / "effects.db"))
        reconstructed = await ToolEffectStateOwner(store).load(intent)
        assert reconstructed == settled
        historical = next(
            item
            for item in await store.load_events(intent.session_id)
            if item.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        )
        # Settlement advances the effect, not the identity of its historical
        # uncertainty evidence used by a retained workspace observation.
        validate_tool_effect_uncertainty_event(historical, reconstructed)
        for field in ("dispatch_id", "intent_digest", "tool_round_id", "tool_call_id"):
            conflicting_event = historical.model_copy(
                update={"payload": {**historical.payload, field: "conflicting"}}
            )
            with pytest.raises(ToolEffectConflict):
                validate_tool_effect_uncertainty_event(conflicting_event, reconstructed)
        assert await ToolEffectStateOwner(store).load(intent) == reconstructed
        if backend == "sqlite":
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_late_normal_completion_and_reconciliation_have_one_winner(backend, tmp_path):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(str(tmp_path / "race.db"))
        )
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        executing = await owner.transition(prepared, state="executing", run_epoch=0)
        unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
        normal = _event(intent, event_id="normal")
        recovered = _event(intent, event_id="recovered")
        outcomes = await asyncio.gather(
            owner.transition(
                unknown,
                state="completed",
                terminal=_terminal(normal),
                run_epoch=0,
                events=(normal,),
            ),
            owner.transition(
                unknown,
                state="reconciled_completed",
                terminal=_terminal(recovered, _receipt(intent)),
                run_epoch=0,
                events=(recovered,),
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(outcome, ToolEffectConflict) for outcome in outcomes) == 1
        events = await store.load_events(intent.session_id)
        assert len(events) == 2
        assert events[0].type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        assert (await owner.load(intent)).terminal.event_id == events[1].id
        if backend == "sqlite":
            await store.close()

    asyncio.run(scenario())


def test_record_cannot_select_missing_terminal_material():
    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        executing = await owner.transition(prepared, state="executing", run_epoch=0)
        with pytest.raises(ToolEffectConflict, match="no atomic terminal"):
            await owner.transition(
                executing, state="completed", terminal=_terminal(_event(intent)), run_epoch=0
            )
        assert await owner.load(intent) == executing
        assert await store.load_events(intent.session_id) == []

    asyncio.run(scenario())


def test_lost_acknowledgement_is_reconciled_by_exact_record():
    class CommitThenRaise(InMemorySessionStore):
        async def publish_session_operation(self, *args, **kwargs):
            await super().publish_session_operation(*args, **kwargs)
            raise OSError("ack lost")

    async def scenario():
        store = CommitThenRaise()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        executing = await owner.transition(prepared, state="executing", run_epoch=0)
        assert await owner.load(intent) == executing
        with pytest.raises(ToolEffectConflict):
            await owner.transition(prepared, state="executing", run_epoch=0)

        unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
        assert await owner.load(intent) == unknown
        assert [event.type for event in await store.load_events(intent.session_id)] == [
            EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        ]
        session = await store.load(intent.session_id)
        assert await owner.preserve_unresolved(
            session, tool_round_id=intent.tool_round_id, tool_call_ids=(intent.tool_call_id,)
        )
        assert len(await store.load_events(intent.session_id)) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("cancellation_count", [1, 2])
@pytest.mark.parametrize("state", ["executing", "outcome_unknown"])
def test_cancellation_waits_for_owned_commit_and_remains_cancellation(cancellation_count, state):
    class BarrierStore(InMemorySessionStore):
        def __init__(self):
            super().__init__()
            self.block = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def publish_session_operation(self, *args, **kwargs):
            if self.block:
                self.entered.set()
                await self.release.wait()
            return await super().publish_session_operation(*args, **kwargs)

    async def scenario():
        store = BarrierStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        if state == "outcome_unknown":
            prepared = await owner.transition(prepared, state="executing", run_epoch=0)
        store.block = True
        task = asyncio.create_task(owner.transition(prepared, state=state, run_epoch=0))
        await store.entered.wait()
        for _ in range(cancellation_count):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert task.cancelling() == cancellation_count
        assert (await owner.load(intent)).state == state
        assert [event.type for event in await store.load_events(intent.session_id)] == (
            [EventType.TOOL_EFFECT_OUTCOME_UNKNOWN] if state == "outcome_unknown" else []
        )

    asyncio.run(scenario())


def test_receipt_cannot_select_a_different_terminal_result():
    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        prepared = await owner.prepare(intent, run_epoch=0)
        executing = await owner.transition(prepared, state="executing", run_epoch=0)
        unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
        event = _event(intent)
        receipt = _receipt(intent).model_copy(update={"message": "a different result"})
        with pytest.raises(ToolEffectConflict, match="differs from its receipt"):
            await owner.transition(
                unknown,
                state="reconciled_completed",
                terminal=_terminal(event, receipt),
                run_epoch=0,
                events=(event,),
            )
        assert await owner.load(intent) == unknown
        assert [item.type for item in await store.load_events(intent.session_id)] == [
            EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        ]

    asyncio.run(scenario())


def test_publication_and_readback_failures_retain_ordered_originals():
    publication_error = OSError("write failed")
    readback_error = OSError("readback failed")

    class UnavailableStore(InMemorySessionStore):
        async def publish_session_operation(self, *args, **kwargs):
            raise publication_error

        async def load_session_operation(self, *args, **kwargs):
            raise readback_error

    async def scenario():
        store = UnavailableStore()
        intent = await _intent(store)
        with pytest.raises(ExceptionGroup) as caught:
            await ToolEffectStateOwner(store).prepare(intent, run_epoch=0)
        assert caught.value.exceptions == (publication_error, readback_error)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancellation_count", [1, 2])
def test_cancelled_publication_retains_write_and_readback_failures(cancellation_count):
    publication_error = OSError("publication failed after dispatch")
    readback_error = OSError("exact readback failed")

    class BarrierFailureStore(InMemorySessionStore):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def publish_session_operation(self, *args, **kwargs):
            self.entered.set()
            await self.release.wait()
            raise publication_error

        async def load_session_operation(self, *args, **kwargs):
            raise readback_error

    async def scenario():
        store = BarrierFailureStore()
        intent = await _intent(store)
        owner = asyncio.create_task(ToolEffectStateOwner(store).prepare(intent, run_epoch=0))
        try:
            await asyncio.wait_for(store.entered.wait(), timeout=5)
            for _ in range(cancellation_count):
                owner.cancel("stop receipt publication")
                await asyncio.sleep(0)
            assert not owner.done()
            # The shared shielded-wait owner temporarily consumes cancellation
            # requests while the write settles, then restores them on exit.
            assert owner.cancelling() == 0
            store.release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await owner
            assert owner.cancelled()
            assert owner.cancelling() == cancellation_count
            assert caught.value.args == ("stop receipt publication",)
            cause = caught.value.__cause__
            assert isinstance(cause, ExceptionGroup)
            assert cause.exceptions == (publication_error, readback_error)
            assert await store.load_events(intent.session_id) == []
        finally:
            store.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_child_only_cancellation_is_not_caller_cancellation():
    cancellation = asyncio.CancelledError()

    class ChildCancelledStore(InMemorySessionStore):
        async def publish_session_operation(self, *args, **kwargs):
            raise cancellation

    async def scenario():
        store = ChildCancelledStore()
        intent = await _intent(store)
        task = asyncio.create_task(ToolEffectStateOwner(store).prepare(intent, run_epoch=0))
        with pytest.raises(RuntimeError, match="without caller cancellation") as caught:
            await task
        assert caught.value.__cause__ is cancellation
        assert task.cancelling() == 0
        assert not task.cancelled()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_digest", "c" * 64),
        ("arguments_digest", "c" * 64),
        ("tool_name", "other"),
        ("idempotency_key", "other"),
        ("execution_profile_fingerprint", "other"),
        ("approval_id", "other"),
        ("reconciler_fingerprint", "other"),
        ("source_run_epoch", 1),
    ],
)
def test_stable_call_identity_does_not_hide_conflicting_intent(field, value):
    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        await owner.prepare(intent, run_epoch=0)
        changed = intent.model_copy(update={field: value})
        with pytest.raises(ToolEffectConflict):
            await owner.load(changed)
        with pytest.raises(ToolEffectConflict):
            await owner.prepare(changed, run_epoch=0)

    asyncio.run(scenario())
