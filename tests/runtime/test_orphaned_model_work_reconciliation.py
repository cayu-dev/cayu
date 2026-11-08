"""Model work a lost run left unfinished is settled without the run's environment.

A worker registers its per-run environment and dies inside model work other than
an ordinary assistant call: a context compaction, an auxiliary call a tool made, a background provider operation, or an answer the
provider gave that the worker recorded and didn't publish. An application without
the environment then ends the session, as the orphaned-session sweep and
``cayu session close`` do, under the run's fenced recovery claim:

- a compaction or auxiliary call that reached its provider has an unknown outcome,
  its reservations charged at their reserved amount;
- a background operation is cancelled through the provider's operation adapter,
  and its budget charged unless the cancellation settled it;
- a recorded answer is published, its budget settled from the usage it reports.

Each runs on SQLite and PostgreSQL; an in-memory store can't outlive the worker.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from decimal import Decimal

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import (
    _TOOL,
    _close,
    _config,
    _failure,
    _latest_interaction_event,
    _orphan,
    _slot,
    _stores,
)
from tests.runtime.test_orphaned_session_reconciliation import (
    database_url as _database_url,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
)
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
)
from cayu.server import create_server
from cayu.sessions import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions._model_completion_publication import ModelCompletionStageDisposition
from cayu.sessions.records import SessionStatusConflict
from cayu.tools.base import ToolSpec
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits

# Each test gets a database of its own: a SQLite file, or a PostgreSQL database.
database_url = _database_url
_MODULE = __name__
_SUMMARY_MODEL = "summary-model"
_OPERATION_ID = "resp_orphaned_scan"
_RETRY_OPERATION_ID = "resp_orphaned_scan_retry"
_USAGE = {"input_tokens": 3, "output_tokens": 2}
_LIMITS = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=30)


class _Recording(Tool):
    """The tool an answer asks for; the abandoned run never starts it."""

    spec = _TOOL

    async def run(self, ctx, args):
        raise AssertionError("The abandoned run started a tool.")


class _Summarize(Tool):
    """A tool that makes a nested model call through managed auxiliary inference."""

    spec = ToolSpec(
        name="summarize",
        description="Summarize with a nested model call.",
        input_schema={"type": "object"},
        auxiliary_inference=AuxiliaryInferencePolicy(limits=_LIMITS, purposes=("tool.summary",)),
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:orphan-summarize", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx, args):
        await ctx.inference.invoke(
            ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
            purpose="tool.summary",
            limits=_LIMITS,
        )
        raise AssertionError("The worker survived its nested model call.")


def _answer(*events: ModelStreamEvent) -> list[ModelStreamEvent]:
    return [*events, ModelStreamEvent.completed({"finish_reason": "stop", "usage": _USAGE})]


class _Provider(VersionedFakeProvider):
    """The scan's model. In the worker it dies where ``crash`` says."""

    def __init__(self, mode: str, *, crash: bool = False) -> None:
        if mode == "answered_tools":
            batches = [
                _answer(ModelStreamEvent.tool_call(id="call-1", name="record", arguments={}))
            ]
        elif mode == "auxiliary":
            batches = [
                _answer(ModelStreamEvent.tool_call(id="call-1", name="summarize", arguments={})),
                _answer(ModelStreamEvent.text_delta("nested summary")),
            ]
        elif mode.startswith("compaction"):
            batches = [
                _answer(ModelStreamEvent.text_delta("durable summary")),
                _answer(ModelStreamEvent.text_delta("done")),
            ]
        else:
            batches = [_answer(ModelStreamEvent.text_delta("done"))]
        super().__init__(batches)
        self.mode = mode
        self.crash = crash

    def prepare_auxiliary_request(self, request, *, max_output_tokens: int):
        # Managed auxiliary inference is opt-in per provider.
        return self._prepare_auxiliary_request(
            request,
            max_output_tokens=max_output_tokens,
            output_option_path=self.output_token_option_path(),
        )

    async def stream(self, request):
        if not self.crash:
            raise AssertionError("Abandoning called the model.")
        if (self.mode == "compaction" and request.model == _SUMMARY_MODEL) or (
            self.mode == "auxiliary" and self.requests
        ):
            # The worker died while the provider answered the call.
            os._exit(137)
        async for event in super().stream(request):
            yield event


class _Operations(ProviderOperationAdapter):
    """OpenAI-style background responses: the work runs at the provider."""

    def __init__(
        self,
        *,
        outcome: ProviderOperationStatus = ProviderOperationStatus.CANCELLED,
        supported: bool = True,
        operation_id: str = _OPERATION_ID,
        retrieved: ProviderOperationStatus | None = None,
    ) -> None:
        self.outcome = outcome
        self.supported = supported
        self.operation_id = operation_id
        self.retrieved = retrieved
        self.cancelled: list[str] = []

    async def start(self, request):
        async def events():
            # The operation is recorded; the worker dies while it runs.
            os._exit(137)
            yield  # pragma: no cover

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=self.operation_id,
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state):
        if self.retrieved is None:
            raise AssertionError("Abandoning collects no result.")
        return ProviderOperationSnapshot(state=state, status=self.retrieved)

    async def reconnect(self, state):
        raise AssertionError("Abandoning reconnects to nothing.")

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return (
            ProviderOperationCancellationSupport.SUPPORTED
            if self.supported
            else ProviderOperationCancellationSupport.UNSUPPORTED
        )

    async def cancel(self, state):
        self.cancelled.append(state.operation_id)
        return ProviderOperationSnapshot(state=state, status=self.outcome)


class _BackgroundProvider(VersionedFakeProvider):
    def __init__(self, operations: _Operations | None, *, background: bool = True) -> None:
        super().__init__([])
        self.operations = operations
        self.background = background

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return (
            ProviderOperationMode.BACKGROUND if self.background else super().provider_operation_mode
        )

    @property
    def provider_operations(self):
        return self.operations if self.background else None

    async def stream(self, request):
        raise AssertionError("A background provider's work runs through its operations.")
        yield  # pragma: no cover


def _budget() -> BudgetPolicy:
    # Each model call reserves its budget before it is dispatched.
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("10"),
                pricing=PriceBook(
                    prices=tuple(
                        ModelPrice.fixed(
                            provider_name="fake",
                            model=model,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("10"),
                        )
                        for model in ("fake-model", _SUMMARY_MODEL)
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=100),
            ),
        )
    )


def _application(
    store,
    mode: str,
    *,
    environment=None,
    crash: bool = False,
    operations: _Operations | None = None,
    background: bool = True,
):
    app = CayuApp(session_store=store, budget_policy=_budget(), enable_logging=False)
    if mode == "background":
        provider = _BackgroundProvider(operations or _Operations(), background=background)
    else:
        provider = _Provider(mode, crash=crash)
    app.register_provider(provider, default=True)
    context_policy = (
        CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model=_SUMMARY_MODEL),
            max_user_turns=1,
            compact_after_messages=1,
        )
        if mode.startswith("compaction")
        else None
    )
    app.register_agent(
        AgentSpec(name="scanner", model="fake-model"),
        tools=[_Recording(), _Summarize()],
        context_policy=context_policy,
    )
    if environment is not None:
        app.register_environment_factory(environment.environment.spec, environment)
    return app


def _run_worker() -> None:
    """The scan's worker: it registers the run's environment and dies in model work."""

    async def scenario() -> None:
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"], os.environ["CAYU_ORPHAN_TEST_STORE"]
        )
        store = stores.session_store
        mode = os.environ["CAYU_ORPHAN_TEST_CRASH"]
        if mode.startswith("answered") or mode == "compaction_answered":
            purpose = "context-compaction" if mode == "compaction_answered" else "assistant-turn"
            promote = type(store).promote_model_completion_stage

            async def die_before_publishing(self, session_id, **kwargs):
                active = await self.load_active_model_completion_stage(session_id)
                if active is not None and active.stage.purpose == purpose:
                    # The answer is recorded; the worker died before publishing it.
                    os._exit(137)
                return await promote(self, session_id, **kwargs)

            type(store).promote_model_completion_stage = die_before_publishing
        app = _application(store, mode, environment=_slot(), crash=True)
        messages = (
            [
                Message.text("user", "old"),
                Message.text("assistant", "old answer"),
                Message.text("user", "go"),
            ]
            if mode.startswith("compaction")
            else [Message.text("user", "go")]
        )
        async for _ in app.run(
            RunRequest(
                session_id="scan",
                agent_name="scanner",
                environment_name=_slot().environment.spec.name,
                messages=messages,
            )
        ):
            pass
        raise AssertionError("The worker did not die inside the session.")

    asyncio.run(scenario())


def _run_fallback_worker() -> None:
    """A worker that accepted a fallback retry of the lost operation and died retrying."""

    async def scenario() -> None:
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"], os.environ["CAYU_ORPHAN_TEST_STORE"]
        )
        store = stores.session_store
        if os.environ["CAYU_ORPHAN_TEST_CRASH"] == "before_retry":
            # The worker died after it took on the retry, before preparing it.
            async def die(*args, **kwargs):
                os._exit(137)

            type(store).prepare_model_completion_stage = die
        operations = _Operations(operation_id=_RETRY_OPERATION_ID)
        app = _application(store, "background", environment=_slot(), operations=operations)
        stage = (await store.load_active_model_completion_stage("scan")).stage
        session = await store.load("scan")
        async for _ in app.resolve_provider_operation(
            ProviderOperationResolutionRequest(
                session_id="scan",
                stage_id=stage.stage_id,
                expected_run_epoch=session.run_epoch,
                action=ProviderOperationResolutionAction.FALLBACK_RETRY,
                reason="The provider lost it; try again.",
            )
        ):
            pass
        raise AssertionError("The worker did not die retrying the operation.")

    asyncio.run(scenario())


def _worker_died_in(tmp_path, database_url: str, mode: str) -> None:
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        crash=mode,
        database_url=database_url,
        module=_MODULE,
    )


_PURPOSES = {
    "compaction": "context-compaction",
    "compaction_answered": "context-compaction",
    "auxiliary": "auxiliary-inference",
}


async def _unfinished(store, mode: str):
    """The model work the worker left: its stage, in flight or answered."""

    active = await store.load_active_model_completion_stage("scan")
    assert active is not None
    stage = active.stage
    assert stage.purpose == _PURPOSES.get(mode, "assistant-turn")
    assert stage.state == ("completed" if "answered" in mode else "in_flight")
    assert await store.load_model_completion_stage_dispatch("scan", stage.stage_id)
    assert stage.reservation_ids
    assert (await store.load("scan")).status is SessionStatus.RUNNING
    return stage


async def _budget_settled_once(stores, stage, *, conservative: bool) -> None:
    """Each reservation is settled once: at its reserved amount, or from its usage."""

    for reservation_id in stage.reservation_ids:
        reservation = await stores.budget_ledger.load_reservation(reservation_id)
        assert reservation.status == "reconciled"
        if conservative:
            assert reservation.actual_amount == reservation.reserved_amount
        else:
            assert reservation.actual_amount < reservation.reserved_amount
    reconciled = [
        event.payload.get("reservation_id")
        for event in await stores.session_store.load_events("scan")
        if event.type is EventType.BUDGET_RECONCILED
    ]
    for reservation_id in stage.reservation_ids:
        assert reconciled.count(reservation_id) == 1, reconciled


async def _require_ended(stores, stage, payload: dict, mode: str) -> None:
    """The session failed with its model work settled, and nothing is left holding it."""

    store = stores.session_store
    assert await store.load_active_model_completion_stage("scan") is None
    call = (payload.get("model_call_outcome_unknown") or payload.get("model_call_settled"))[0]
    assert (call["stage_id"], call["purpose"]) == (stage.stage_id, stage.purpose)
    if "answered" in mode:
        # The provider's answer is published, and its budget settled from its usage.
        assert "model_call_outcome_unknown" not in payload
        assert call["outcome"] == "published"
        assert await store.load_model_completion_stage_settlement("scan", stage.stage_id) is None
        await _budget_settled_once(stores, stage, conservative=False)
    elif stage.purpose == "auxiliary-inference":
        # Recorded as recovery records a lost auxiliary call: its outcome unknown.
        [settled] = [
            event
            for event in await store.load_events("scan")
            if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert settled.payload["auxiliary_outcome"] == "outcome_unknown"
        assert settled.payload["model_attempt_id"] == stage.intent["model_attempt_id"]
    else:
        settlement = await store.load_model_completion_stage_settlement("scan", stage.stage_id)
        assert (
            settlement.disposition
            is ModelCompletionStageDisposition.PROVIDER_EFFECT_OUTCOME_UNKNOWN
        )
    assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
    events = await store.load_events("scan")
    assert [event.type for event in events].count(EventType.INTERACTION_FAILED) == 1
    assert [event.type for event in events].count(EventType.SESSION_FAILED) == 1
    checkpoint = await store.load_checkpoint("scan") or {}
    assert "incomplete_session_recovery_claim" not in checkpoint
    assert "provider_operation_cancellation_claim" not in checkpoint
    assert (await store.inspect_session_execution("scan")).state == "terminal"


@pytest.mark.parametrize("mode", ["compaction", "auxiliary"])
def test_a_lost_compaction_or_auxiliary_call_is_recorded_as_unknown(
    tmp_path, database_url, mode
) -> None:
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, mode)
        stage = await _unfinished(store, mode)

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        assert "model call it was making has an unknown outcome" in result.message
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        [unknown] = payload["model_call_outcome_unknown"]
        assert unknown["model_attempt_id"] == stage.intent["model_attempt_id"]
        await _require_ended(stores, stage, payload, mode)
        # Nothing knows what the provider charged, so each reservation is charged in full.
        await _budget_settled_once(stores, stage, conservative=True)
        if mode == "auxiliary":
            # The tool that made the nested call was running too.
            assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT,)
            [tool_call] = payload["tool_effect_outcome_unknown"]
            assert tool_call["tool_name"] == "summarize"
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["answered", "answered_tools", "compaction_answered"])
def test_an_answer_the_lost_run_recorded_is_published(tmp_path, database_url, mode) -> None:
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        reader = _application(store, mode)
        stage = await _unfinished(store, mode)
        transcript = await store.load_transcript("scan")

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        assert result.actions == (IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,)
        assert "answer its model call had recorded was published" in result.message
        payload = await _failure(store, "scan")
        await _require_ended(stores, stage, payload, mode)
        # The run doesn't continue: nothing ran the tool the answer asked for.
        assert payload.get("tool_effect_outcome_unknown", []) == []
        published = await store.load_transcript("scan")
        if mode == "answered":
            assert len(published) == len(transcript) + 1
            assert published[-1].role.value == "assistant"
        else:
            # A compaction's summary is accounted for, not applied. An answer that asks
            # for tools publishes its message with the tool round, which never ran; the
            # abandonment record keeps that round.
            assert len(published) == len(transcript)
        events = await store.load_events("scan")
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in events)
        completed = [
            event
            for event in events
            if event.type is EventType.MODEL_COMPLETED
            and event.id in {published.id for published in stage.publication.events}
        ]
        assert len(completed) == 1
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("outcome", "supported", "cancellation"),
    [
        (ProviderOperationStatus.CANCELLED, True, "cancelled"),
        (ProviderOperationStatus.COMPLETED, True, "completed"),
        (ProviderOperationStatus.CANCELLED, False, "unsupported"),
    ],
)
def test_a_lost_background_operation_is_cancelled_through_its_provider(
    tmp_path, database_url, outcome, supported, cancellation
) -> None:
    _worker_died_in(tmp_path, database_url, "background")

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        operations = _Operations(outcome=outcome, supported=supported)
        reader = _application(store, "background", operations=operations)
        stage = await _unfinished(store, "background")

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        assert operations.cancelled == ([_OPERATION_ID] if supported else [])
        payload = await _failure(store, "scan")
        await _require_ended(stores, stage, payload, "background")
        if cancellation == "cancelled":
            [settled] = payload["model_call_settled"]
            assert settled["outcome"] == "cancelled"
            assert "model_call_outcome_unknown" not in payload
            assert "background provider operation of its model call was cancelled" in (
                result.message
            )
        else:
            # The provider finished it, or can't cancel it: its outcome is unknown, and
            # its reservations are charged in full.
            [settled] = payload["model_call_outcome_unknown"]
            await _budget_settled_once(stores, stage, conservative=True)
        assert settled["provider_operation"] == {
            "operation_id": _OPERATION_ID,
            "cancellation": cancellation,
        }
        cancel_events = [
            event.payload["cancellation_status"]
            for event in await store.load_events("scan")
            if event.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED
        ]
        assert cancel_events == [cancellation]
        for reservation_id in stage.reservation_ids:
            reservation = await stores.budget_ledger.load_reservation(reservation_id)
            assert reservation.status == "reconciled"
        # A later sweep or close doesn't call the provider again.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        closed = await reader.close_session("scan", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        assert operations.cancelled == ([_OPERATION_ID] if supported else [])
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


def test_close_ends_a_session_its_lost_worker_left_inside_a_background_operation(
    tmp_path, database_url
) -> None:
    _worker_died_in(tmp_path, database_url, "background")
    stores = _stores(database_url, tmp_path / "cayu.db")
    store = stores.session_store
    operations = _Operations()
    reader = _application(store, "background", operations=operations)
    with TestClient(create_server(reader, config=_config())) as client:
        stage = client.portal.call(_unfinished, store, "background")
        closed = _close(client, "scan")
        assert closed.status_code == 200, closed.text
        assert closed.json()["actions"] == ["closed"]
        payload = client.portal.call(_failure, store, "scan")
        assert payload["reason"] == "session_closed"
        client.portal.call(_require_ended, stores, stage, payload, "background")
        again = _close(client, "scan")
        assert again.status_code == 200, again.text
        assert operations.cancelled == [_OPERATION_ID]
        client.portal.call(stores.close)


def test_a_background_operation_without_its_provider_is_named_with_its_way_forward(
    tmp_path, database_url
) -> None:
    _worker_died_in(tmp_path, database_url, "background")

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        # The provider is registered without its background operations.
        reader = _application(store, "background", background=False)
        stage = await _unfinished(store, "background")
        with pytest.raises(SessionStatusConflict) as refused:
            await reader.close_session("scan", reason="Worker lost.")
        message = str(refused.value)
        assert f"background provider operation {_OPERATION_ID!r} at 'fake'" in message
        assert stage.stage_id in message
        assert (
            "register provider 'fake' with its background operations in this application "
            "and run `cayu session close scan --reason REASON` again"
        ) in message
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        checkpoint = await store.load_checkpoint("scan") or {}
        assert "abandoned_session" not in checkpoint
        assert "incomplete_session_recovery_claim" not in checkpoint
        for reservation_id in stage.reservation_ids:
            reservation = await stores.budget_ledger.load_reservation(reservation_id)
            assert reservation.status == "active"
        # Once it is registered, the close goes through.
        operations = _Operations()
        registered = _application(store, "background", operations=operations)
        closed = await registered.close_session("scan", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        assert operations.cancelled == [_OPERATION_ID]
        await reader.aclose()
        await registered.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["compaction", "answered", "background"])
def test_model_work_that_changed_before_the_fence_is_left_alone(
    tmp_path, database_url, monkeypatch, mode
) -> None:
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        operations = _Operations()
        reader = _application(store, mode, operations=operations)
        stage = await _unfinished(store, mode)
        recovery = reader._incomplete_recovery
        check = recovery._orphaned_model_call
        checks = 0

        async def changed_after_the_first_check(session, *, unavailable):
            nonlocal checks
            checks += 1
            current = await check(session, unavailable=unavailable)
            if checks > 1:
                return current
            # What the first check saw: the run hadn't started its compaction yet, its
            # provider hadn't answered, or its background operation had no identity.
            if mode == "compaction":
                return None
            if mode == "answered":
                return replace(
                    current, stage=current.stage.model_copy(update={"state": "in_flight"})
                )
            return replace(current, operation=None, provider=None)

        monkeypatch.setattr(recovery, "_orphaned_model_call", changed_after_the_first_check)
        with pytest.raises(SessionStatusConflict, match="moved on to other model work"):
            await reader.close_session("scan", reason="Worker lost.")
        assert checks == 2
        # Nothing changed: no abandonment, no claim, no cancellation, no budget settled.
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        checkpoint = await store.load_checkpoint("scan") or {}
        assert "abandoned_session" not in checkpoint
        assert "incomplete_session_recovery_claim" not in checkpoint
        assert (await store.load_active_model_completion_stage("scan")).stage == stage
        assert operations.cancelled == []
        for reservation_id in stage.reservation_ids:
            reservation = await stores.budget_ledger.load_reservation(reservation_id)
            assert reservation.status == "active"
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


_SETTLES = {
    "compaction": ("_model_completion_recovery", "settle_abandoned_dispatch_budget"),
    "auxiliary": ("_model_completion_recovery", "record_abandoned_auxiliary_attempt"),
    "answered_tools": ("_model_completion_recovery", "publish_abandoned_answer"),
    "background": ("_model_completion_recovery", "cancel_abandoned_provider_operation"),
}


@pytest.mark.parametrize("mode", sorted(_SETTLES))
def test_model_work_settles_once_across_a_lost_acknowledgement(
    tmp_path, database_url, monkeypatch, mode
) -> None:
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        operations = _Operations(outcome=ProviderOperationStatus.COMPLETED)
        reader = _application(store, mode, operations=operations)
        stage = await _unfinished(store, mode)
        transcript = await store.load_transcript("scan")
        owner_name, name = _SETTLES[mode]
        owner = getattr(reader._incomplete_recovery, owner_name)
        original = getattr(owner, name)
        calls = 0

        async def lost_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = await original(*args, **kwargs)
            if calls == 1:
                # The store committed; the acknowledgement was lost.
                raise RuntimeError("acknowledgement lost")
            return result

        monkeypatch.setattr(owner, name, lost_once)
        [first] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert first.actions == (IncompleteSessionRecoveryAction.FAILED,)
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        assert "incomplete_session_recovery_claim" not in (
            await store.load_checkpoint("scan") or {}
        )

        [retried] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert retried.status is SessionStatus.FAILED, retried.message
        payload = await _failure(store, "scan")
        assert payload["reason"] == "orphaned_session_reconciled"
        assert await store.load_active_model_completion_stage("scan") is None
        await _budget_settled_once(stores, stage, conservative=mode != "answered_tools")
        if mode == "answered_tools":
            # Published once; the retry found it published and nothing left to settle.
            assert len(await store.load_transcript("scan")) == len(transcript)
            published = {event.id for event in stage.publication.events}
            completions = [
                event
                for event in await store.load_events("scan")
                if event.type is EventType.MODEL_COMPLETED and event.id in published
            ]
            assert len(completions) == 1
        else:
            [unknown] = payload["model_call_outcome_unknown"]
            assert unknown["stage_id"] == stage.stage_id
            await _require_ended(stores, stage, payload, mode)
        # The provider was asked to cancel once; the retry replayed its answer.
        assert operations.cancelled == ([_OPERATION_ID] if mode == "background" else [])
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("crash", ["before_retry", "in_retry"])
def test_a_fallback_retry_the_lost_run_was_making_is_settled(tmp_path, database_url, crash) -> None:
    _worker_died_in(tmp_path, database_url, "background")

    async def unavailable() -> tuple:
        # Recovery where the environment was still registered found the operation gone
        # at the provider, and interrupted the session for a person to decide.
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        source = await _unfinished(store, "background")
        recovering = _application(
            store,
            "background",
            environment=_slot(),
            operations=_Operations(retrieved=ProviderOperationStatus.UNAVAILABLE),
        )
        await recovering.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="scan", inactive_for_seconds=0)
        )
        assert (await store.load("scan")).status is SessionStatus.INTERRUPTED
        await recovering.aclose()
        await stores.close()
        return source

    source = asyncio.run(unavailable())
    # A person chose a fallback retry; the worker running it died before preparing the
    # retry, or inside the retry's own background operation.
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        crash=crash,
        database_url=database_url,
        entry="_run_fallback_worker",
        module=_MODULE,
    )

    async def scenario() -> None:
        stores = _stores(database_url, tmp_path / "cayu.db")
        store = stores.session_store
        assert (await store.load("scan")).status is SessionStatus.RUNNING
        operations = _Operations()
        reader = _application(store, "background", operations=operations)
        if crash == "before_retry":
            assert await store.load_active_model_completion_stage("scan") is None
            assert "provider_operation_pending_resolution_disposition" in (
                await store.load_checkpoint("scan") or {}
            )
        else:
            retry = await _unfinished(store, "background")
            assert retry.stage_id != source.stage_id

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED, result.message
        payload = await _failure(store, "scan")
        if crash == "before_retry":
            # The accepted retry is retired; it is never dispatched.
            assert "disposition it had accepted was retired" in result.message
            assert payload["provider_operation_disposition"] == {
                "stage_id": source.stage_id,
                "resolution_id": payload["provider_operation_disposition"]["resolution_id"],
                "action": "fallback_retry",
            }
            assert operations.cancelled == []
            assert await _latest_interaction_event(store, "scan") is EventType.INTERACTION_FAILED
        else:
            # The retry had retired the disposition; its own operation is cancelled.
            assert "provider_operation_disposition" not in payload
            assert operations.cancelled == [_RETRY_OPERATION_ID]
            await _require_ended(stores, retry, payload, "background")
        # The source call is charged in full, once.
        await _budget_settled_once(stores, source, conservative=True)
        checkpoint = await store.load_checkpoint("scan") or {}
        assert "provider_operation_pending_resolution_disposition" not in checkpoint
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await stores.close()

    asyncio.run(scenario())
