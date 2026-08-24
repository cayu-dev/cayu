from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.core.task_invocation_fixtures import stored_session_invocation
from tests.core.test_completion_decision_application import (
    _contract,
    _persist_decision,
    _result,
)

from cayu import (
    CayuApp,
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverExecutionError,
    CompletionResultResolverRequest,
    CompletionResultResolverUnavailable,
    CompletionResultUnavailable,
    CompletionVerdict,
    Event,
    EventQuery,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    Task,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskStatus,
    WorkCompletionConflict,
)
from cayu.runtime.sessions import (
    SessionStore,
    fork_session_invocation,
    run_request_with_task_invocation,
)
from cayu.runtime.tasks import TaskStore


@dataclass(frozen=True)
class _PreparedResolution:
    app: CayuApp
    session_store: SessionStore
    request: CompletionResultResolutionRequest
    expected_result: dict[str, object]
    task: Task


class _CountingResolver(CompletionResultResolver):
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.calls = 0

    async def resolve(
        self,
        request: CompletionResultResolverRequest,
    ) -> dict[str, object]:
        del request
        self.calls += 1
        return self._result


class _TwoOwnerBarrierResolver(CompletionResultResolver):
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def resolve(
        self,
        request: CompletionResultResolverRequest,
    ) -> dict[str, object]:
        del request
        self.calls += 1
        if self.calls == 2:
            self.both_started.set()
        await self.release.wait()
        return self._result


class _FailResultEventOnceStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._fail_result_event_once = True

    def _append_events_unlocked(
        self,
        session: Session,
        events: list[Event],
    ) -> Session:
        if self._fail_result_event_once and any(
            event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events
        ):
            self._fail_result_event_once = False
            raise ConnectionError("result event acknowledgement lost")
        return super()._append_events_unlocked(session, events)


async def _prepare_resolution(
    store: TaskStore,
    *,
    ordinal: int,
    label: str,
    session_store: SessionStore | None = None,
) -> _PreparedResolution:
    session_store = session_store or InMemorySessionStore()
    session_id = f"session:resolver-conformance:{label}"
    contract = _contract(contract_id=f"resolver-conformance-contract-{label}")
    await store.publish_work_contract(contract)
    task = await store.create_task(
        TaskCreate(
            task_id=f"resolver-conformance-task-{label}",
            type="verified-work",
            session_id=session_id,
            work_contract=contract.reference(),
        )
    )
    await session_store.create(
        run_request_with_task_invocation(
            RunRequest(
                agent_name="resolver-conformance-agent",
                session_id=session_id,
                task_id=task.id,
                messages=[Message.text("user", "Resolve the accepted result.")],
            ),
            TaskInvocationSnapshot(
                id=task.id,
                session_id=task.session_id,
                invocation=task.invocation,
            ),
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
    )
    task = await store.start_task(
        task.id,
        session_id=session_id,
        session_invocation=await stored_session_invocation(session_store, session_id),
    )
    decision_id, _reference = await _persist_decision(
        store,
        task=task,
        ordinal=ordinal,
        verdict=CompletionVerdict.ACCEPTED,
    )
    request = CompletionResultResolutionRequest(
        task_id=task.id,
        decision_id=decision_id,
        idempotency_key=f"resolver-conformance-resolution-{label}",
    )
    return _PreparedResolution(
        app=CayuApp(
            session_store=session_store,
            task_store=store,
            enable_logging=False,
        ),
        session_store=session_store,
        request=request,
        expected_result=_result(str(ordinal)),
        task=task,
    )


async def _assert_one_result_event(prepared: _PreparedResolution) -> None:
    records = await prepared.session_store.query_events(
        EventQuery(
            session_id=prepared.task.session_id,
            event_types=(EventType.TASK_COMPLETION_RESULT_RESOLVED,),
            limit=10,
        )
    )
    assert len(records) == 1
    assert "contract_version" not in records[0].event.payload


async def assert_completion_result_resolver_store_conformance(
    store: TaskStore,
    *,
    store_kind: str,
) -> None:
    exact = await _prepare_resolution(
        store,
        ordinal=101,
        label=f"{store_kind}-exact",
    )
    exact_resolver = _CountingResolver(exact.expected_result)
    exact.app.register_completion_result_resolver(
        _contract().result_resolver,
        exact_resolver,
    )
    completed = await exact.app.resolve_completion_result(exact.request)
    replay_app = CayuApp(
        session_store=exact.session_store,
        task_store=store,
        enable_logging=False,
    )
    assert await replay_app.resolve_completion_result(exact.request) == completed
    assert exact_resolver.calls == 1
    await _assert_one_result_event(exact)

    missing = await _prepare_resolution(
        store,
        ordinal=102,
        label=f"{store_kind}-missing",
    )
    with pytest.raises(CompletionResultResolverUnavailable):
        await missing.app.resolve_completion_result(missing.request)
    missing_task = await store.load_task(missing.task.id)
    assert missing_task is not None
    assert missing_task.status is TaskStatus.RUNNING

    registration = await _prepare_resolution(
        store,
        ordinal=103,
        label=f"{store_kind}-registration",
    )
    original = _CountingResolver(registration.expected_result)
    replacement = _CountingResolver(registration.expected_result)
    registration.app.register_completion_result_resolver(
        _contract().result_resolver,
        original,
    )
    with pytest.raises(ValueError, match="already registered"):
        registration.app.register_completion_result_resolver(
            _contract().result_resolver,
            replacement,
        )
    assert (
        await registration.app.resolve_completion_result(registration.request)
    ).status is TaskStatus.COMPLETED
    assert original.calls == 1
    assert replacement.calls == 0

    mismatch = await _prepare_resolution(
        store,
        ordinal=104,
        label=f"{store_kind}-mismatch",
    )
    mismatch.app.register_completion_result_resolver(
        _contract().result_resolver,
        _CountingResolver(_result("wrong-digest")),
    )
    with pytest.raises(
        CompletionResultResolverExecutionError,
        match="invalid result content",
    ):
        await mismatch.app.resolve_completion_result(mismatch.request)
    mismatched_task = await store.load_task(mismatch.task.id)
    assert mismatched_task is not None
    assert mismatched_task.status is TaskStatus.RUNNING
    assert mismatched_task.result is None

    class BlockingResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    cancelled = await _prepare_resolution(
        store,
        ordinal=105,
        label=f"{store_kind}-cancelled",
    )
    blocking = BlockingResolver()
    cancelled.app.register_completion_result_resolver(
        _contract().result_resolver,
        blocking,
    )
    operation = asyncio.create_task(cancelled.app.resolve_completion_result(cancelled.request))
    await blocking.started.wait()
    operation.cancel("cancel resolver conformance")
    assert operation.cancelling() == 1
    with pytest.raises(asyncio.CancelledError, match="cancel resolver conformance"):
        await operation
    assert operation.cancelled()
    cancelled_task = await store.load_task(cancelled.task.id)
    assert cancelled_task is not None
    assert cancelled_task.status is TaskStatus.RUNNING
    retry = CayuApp(
        session_store=cancelled.session_store,
        task_store=store,
        enable_logging=False,
    )
    retry_resolver = _CountingResolver(cancelled.expected_result)
    retry.register_completion_result_resolver(
        _contract().result_resolver,
        retry_resolver,
    )
    assert (await retry.resolve_completion_result(cancelled.request)).status is TaskStatus.COMPLETED
    assert retry_resolver.calls == 1

    acknowledgement_store = _FailResultEventOnceStore()
    recovery = await _prepare_resolution(
        store,
        ordinal=106,
        label=f"{store_kind}-recovery",
        session_store=acknowledgement_store,
    )
    recovery_resolver = _CountingResolver(recovery.expected_result)
    recovery.app.register_completion_result_resolver(
        _contract().result_resolver,
        recovery_resolver,
    )
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await recovery.app.resolve_completion_result(recovery.request)
    committed = await store.load_task(recovery.task.id)
    assert committed is not None
    assert committed.status is TaskStatus.COMPLETED
    recovery_app = CayuApp(
        session_store=acknowledgement_store,
        task_store=store,
        enable_logging=False,
    )
    assert await recovery_app.resolve_completion_result(recovery.request) == committed
    assert recovery_resolver.calls == 1
    await _assert_one_result_event(recovery)

    concurrent = await _prepare_resolution(
        store,
        ordinal=107,
        label=f"{store_kind}-concurrent",
    )
    second_app = CayuApp(
        session_store=concurrent.session_store,
        task_store=store,
        enable_logging=False,
    )
    concurrent_resolver = _TwoOwnerBarrierResolver(concurrent.expected_result)
    concurrent.app.register_completion_result_resolver(
        _contract().result_resolver,
        concurrent_resolver,
    )
    second_app.register_completion_result_resolver(
        _contract().result_resolver,
        concurrent_resolver,
    )
    first_call = asyncio.create_task(concurrent.app.resolve_completion_result(concurrent.request))
    second_call = asyncio.create_task(second_app.resolve_completion_result(concurrent.request))
    await concurrent_resolver.both_started.wait()
    concurrent_resolver.release.set()
    first_result, second_result = await asyncio.gather(first_call, second_call)
    assert first_result == second_result
    assert concurrent_resolver.calls == 2
    await _assert_one_result_event(concurrent)


async def assert_completion_result_resolver_session_publication_conformance(
    session_store: SessionStore,
    *,
    store_kind: str,
) -> None:
    forged_digest = "a" * 64
    forged_publication_id = f"completion-result-publication:v1:{forged_digest}"
    forged_owner_id = f"completion-result-owner:v1:{'b' * 64}"
    forged_root = {
        "schema_version": 2,
        "reservations": {
            forged_publication_id: {
                "schema_version": 2,
                "publication_id": forged_publication_id,
                "authority_sha256": forged_digest,
                "owners": {
                    forged_owner_id: {
                        "schema_version": 2,
                        "owner_id": forged_owner_id,
                        "expires_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
                    }
                },
            }
        },
    }
    creation_session_id = f"session:resolver-conformance:{store_kind}-private-root-create"
    created = await session_store.create(
        RunRequest(
            agent_name="resolver-conformance-agent",
            session_id=creation_session_id,
            messages=[Message.text("user", "Do not accept caller-owned publication state.")],
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
        checkpoint_transform=lambda _session, _checkpoint: {
            "caller_state": {"version": 1},
            "completion_result_event_publications": forged_root,
        },
    )
    created_checkpoint = await session_store.load_checkpoint(created.id)
    assert created_checkpoint == {"caller_state": {"version": 1}}

    fork = Session(
        id=f"{creation_session_id}:fork",
        agent_name=created.agent_name,
        provider_name=created.provider_name,
        model=created.model,
        parent_session_id=created.id,
        causal_budget_id=created.causal_budget_id,
        invocation=fork_session_invocation(created),
        status=created.status,
    )
    created_fork = await session_store.create_fork(
        source_session_id=created.id,
        fork=fork,
        source_statuses={created.status},
        transcript_cursor=None,
        checkpoint_transform=lambda _session, checkpoint: {
            **({} if checkpoint is None else checkpoint),
            "fork_state": {"version": 1},
            "completion_result_event_publications": forged_root,
        },
        expected_source_run_epoch=created.run_epoch,
    )
    fork_checkpoint = await session_store.load_checkpoint(created_fork.id)
    assert fork_checkpoint == {
        "caller_state": {"version": 1},
        "fork_state": {"version": 1},
    }
    await session_store.delete_session(created_fork.id)
    await session_store.delete_session(created.id)

    admitted_session_id = f"session:resolver-conformance:{store_kind}-private-root-admitted-create"
    interaction_id = f"interaction:resolver-conformance:{store_kind}-private-root"
    admitted = await session_store.create(
        RunRequest(
            agent_name="resolver-conformance-agent",
            session_id=admitted_session_id,
            messages=[],
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
        interaction_started_event=Event(
            id=f"event:resolver-conformance:{store_kind}-private-root",
            type=EventType.INTERACTION_STARTED,
            session_id=admitted_session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[],
        checkpoint_transform=lambda _session, checkpoint: {
            **({} if checkpoint is None else checkpoint),
            "completion_result_event_publications": forged_root,
        },
    )
    admitted_checkpoint = await session_store.load_checkpoint(admitted.id)
    assert admitted_checkpoint is not None
    assert "initial_transcript_pending" in admitted_checkpoint
    assert "completion_result_event_publications" not in admitted_checkpoint
    await session_store.release_run_fence(admitted.id)
    await session_store.update_status(admitted.id, SessionStatus.INTERRUPTED)
    await session_store.append_event(
        admitted.id,
        Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id=admitted.id,
        ),
    )
    await session_store.delete_session(admitted.id)

    missing_task_store = InMemoryTaskStore()
    missing = await _prepare_resolution(
        missing_task_store,
        ordinal=301,
        label=f"{store_kind}-missing-session",
        session_store=session_store,
    )
    missing_resolver = _CountingResolver(missing.expected_result)
    missing.app.register_completion_result_resolver(
        _contract().result_resolver,
        missing_resolver,
    )
    original_session = await session_store.load(missing.task.session_id)
    assert original_session is not None
    assert missing.task.session_instance_id == original_session.instance_id
    await session_store.delete_session(missing.task.session_id)
    with pytest.raises(KeyError, match="Session not found"):
        await missing.app.resolve_completion_result(missing.request)
    assert missing_resolver.calls == 0
    unchanged = await missing_task_store.load_task(missing.task.id)
    assert unchanged is not None
    assert unchanged.status is TaskStatus.RUNNING
    assert unchanged.result is None

    bound_snapshot = await missing_task_store.load_invocation_snapshot(missing.task.id)
    assert bound_snapshot is not None
    with pytest.raises(ValueError, match="already bound to a session instance"):
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="resolver-conformance-agent",
                    session_id=missing.task.session_id,
                    task_id=missing.task.id,
                    messages=[Message.text("user", "Resolve the accepted result.")],
                ),
                bound_snapshot,
            ),
            identity=SessionIdentity(
                provider_name="resolver-conformance-provider",
                model="resolver-conformance-model",
            ),
        )
    replacement = await session_store.create(
        run_request_with_task_invocation(
            RunRequest(
                agent_name="resolver-conformance-agent",
                session_id=missing.task.session_id,
                task_id=missing.task.id,
                messages=[Message.text("user", "Resolve the accepted result.")],
            ),
            TaskInvocationSnapshot(
                id=missing.task.id,
                session_id=missing.task.session_id,
                invocation=missing.task.invocation,
            ),
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
    )
    assert replacement.instance_id != missing.task.session_instance_id
    assert replacement.invocation == original_session.invocation
    with pytest.raises(WorkCompletionConflict, match="source session instance changed"):
        await missing.app.resolve_completion_result(missing.request)
    assert missing_resolver.calls == 0
    unchanged = await missing_task_store.load_task(missing.task.id)
    assert unchanged is not None
    assert unchanged.status is TaskStatus.RUNNING
    assert unchanged.result is None
    await session_store.delete_session(missing.task.session_id)

    class BlockingResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.started.set()
            await self.release.wait()
            return _result("302")

    racing_task_store = InMemoryTaskStore()
    racing = await _prepare_resolution(
        racing_task_store,
        ordinal=302,
        label=f"{store_kind}-delete-race",
        session_store=session_store,
    )
    blocking = BlockingResolver()
    racing.app.register_completion_result_resolver(
        _contract().result_resolver,
        blocking,
    )
    operation = asyncio.create_task(racing.app.resolve_completion_result(racing.request))
    await blocking.started.wait()
    await session_store.checkpoint(
        racing.task.session_id,
        {"application_checkpoint": {"version": 1}},
    )
    replaced_checkpoint = await session_store.load_checkpoint(racing.task.session_id)
    assert replaced_checkpoint is not None
    assert replaced_checkpoint["application_checkpoint"] == {"version": 1}
    assert "completion_result_event_publications" in replaced_checkpoint
    await session_store.transform_checkpoint(
        racing.task.session_id,
        lambda _session, _checkpoint: {"transformed_checkpoint": {"version": 1}},
    )
    transformed_checkpoint = await session_store.load_checkpoint(racing.task.session_id)
    assert transformed_checkpoint is not None
    assert transformed_checkpoint["transformed_checkpoint"] == {"version": 1}
    assert "completion_result_event_publications" in transformed_checkpoint

    def mutate_callback_checkpoint_in_place(
        _session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assert checkpoint is not None
        checkpoint.pop("completion_result_event_publications", None)
        checkpoint["in_place_transform"] = {"version": 1}
        return checkpoint

    await session_store.transform_checkpoint(
        racing.task.session_id,
        mutate_callback_checkpoint_in_place,
    )
    in_place_checkpoint = await session_store.load_checkpoint(racing.task.session_id)
    assert in_place_checkpoint is not None
    assert in_place_checkpoint["in_place_transform"] == {"version": 1}
    assert "completion_result_event_publications" in in_place_checkpoint

    await session_store.publish_checkpoint_and_events(
        racing.task.session_id,
        checkpoint_transform=lambda _session, _checkpoint: {
            "public_publication_transform": {"version": 1},
            "completion_result_event_publications": {
                "schema_version": 1,
                "reservations": {},
            },
        },
        events=[],
    )
    public_publication_checkpoint = await session_store.load_checkpoint(racing.task.session_id)
    assert public_publication_checkpoint is not None
    assert public_publication_checkpoint["public_publication_transform"] == {"version": 1}
    assert "completion_result_event_publications" in public_publication_checkpoint
    assert public_publication_checkpoint["completion_result_event_publications"] != {
        "schema_version": 1,
        "reservations": {},
    }
    with pytest.raises(
        ValueError,
        match="completion-result event publication is incomplete",
    ):
        await session_store.delete_session(racing.task.session_id)
    blocking.release.set()
    assert (await operation).status is TaskStatus.COMPLETED
    await session_store.delete_session(racing.task.session_id)
    assert await session_store.load(racing.task.session_id) is None

    class SplitOutcomeResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.both_started = asyncio.Event()
            self.release_success = asyncio.Event()
            self.calls = 0

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.calls += 1
            call = self.calls
            if call == 1:
                self.first_started.set()
            else:
                self.both_started.set()
            await self.both_started.wait()
            if call == 1:
                raise CompletionResultUnavailable("first exact owner cannot resolve")
            await self.release_success.wait()
            return _result("303")

    shared_claim_task_store = InMemoryTaskStore()
    shared_claim = await _prepare_resolution(
        shared_claim_task_store,
        ordinal=303,
        label=f"{store_kind}-shared-publication-claim",
        session_store=session_store,
    )
    shared_claim_second_app = CayuApp(
        session_store=session_store,
        task_store=shared_claim_task_store,
        enable_logging=False,
    )
    split_resolver = SplitOutcomeResolver()
    shared_claim.app.register_completion_result_resolver(
        _contract().result_resolver,
        split_resolver,
    )
    shared_claim_second_app.register_completion_result_resolver(
        _contract().result_resolver,
        split_resolver,
    )
    failed_owner = asyncio.create_task(
        shared_claim.app.resolve_completion_result(shared_claim.request)
    )
    await split_resolver.first_started.wait()
    successful_owner = asyncio.create_task(
        shared_claim_second_app.resolve_completion_result(shared_claim.request)
    )
    await split_resolver.both_started.wait()
    with pytest.raises(CompletionResultUnavailable):
        await failed_owner
    with pytest.raises(
        ValueError,
        match="completion-result event publication is incomplete",
    ):
        await session_store.delete_session(shared_claim.task.session_id)
    checkpoint = await session_store.load_checkpoint(shared_claim.task.session_id)
    assert checkpoint is not None
    reservations = checkpoint["completion_result_event_publications"]["reservations"]
    assert len(reservations) == 1
    assert len(next(iter(reservations.values()))["owners"]) == 1

    split_resolver.release_success.set()
    assert (await successful_owner).status is TaskStatus.COMPLETED
    await session_store.delete_session(shared_claim.task.session_id)
    assert await session_store.load(shared_claim.task.session_id) is None

    expired_owner_session_id = (
        f"session:resolver-conformance:{store_kind}-expired-publication-owner"
    )
    await session_store.create(
        RunRequest(
            agent_name="resolver-conformance-agent",
            session_id=expired_owner_session_id,
            messages=[Message.text("user", "Exercise expired publication ownership.")],
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
    )
    expired_digest = "e" * 64
    expired_publication_id = f"completion-result-publication:v1:{expired_digest}"
    expired_owner_id = f"completion-result-owner:v1:{'f' * 64}"
    expired_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    await session_store._publish_completion_result_event_publication(
        expired_owner_session_id,
        checkpoint_transform=lambda _session, _checkpoint: {
            "completion_result_event_publications": {
                "schema_version": 2,
                "reservations": {
                    expired_publication_id: {
                        "schema_version": 2,
                        "publication_id": expired_publication_id,
                        "authority_sha256": expired_digest,
                        "owners": {
                            expired_owner_id: {
                                "schema_version": 2,
                                "owner_id": expired_owner_id,
                                "expires_at": expired_at,
                            }
                        },
                    }
                },
            }
        },
        events=[],
    )
    await session_store.delete_session(expired_owner_session_id)
    assert await session_store.load(expired_owner_session_id) is None

    corrupted_session_id = f"session:resolver-conformance:{store_kind}-corrupt-reservation"
    await session_store.create(
        RunRequest(
            agent_name="resolver-conformance-agent",
            session_id=corrupted_session_id,
            messages=[Message.text("user", "Exercise corrupted publication authority.")],
        ),
        identity=SessionIdentity(
            provider_name="resolver-conformance-provider",
            model="resolver-conformance-model",
        ),
    )
    for corrupted in (
        {"completion_result_event_publications": None},
        {
            "completion_result_event_publications": {
                "schema_version": True,
                "reservations": {},
            }
        },
    ):
        await session_store._publish_completion_result_event_publication(
            corrupted_session_id,
            checkpoint_transform=lambda _session, _checkpoint, value=corrupted: value,
            events=[],
        )
        with pytest.raises(ValueError, match="publication authority"):
            await session_store.delete_session(corrupted_session_id)
        assert await session_store.load(corrupted_session_id) is not None
    await session_store._publish_completion_result_event_publication(
        corrupted_session_id,
        checkpoint_transform=lambda _session, _checkpoint: {},
        events=[],
    )
    await session_store.delete_session(corrupted_session_id)


async def assert_completion_result_resolver_cross_instance_concurrency(
    first_store: TaskStore,
    second_store: TaskStore,
    *,
    store_kind: str,
) -> None:
    prepared = await _prepare_resolution(
        first_store,
        ordinal=201,
        label=f"{store_kind}-cross-instance",
    )
    second_app = CayuApp(
        session_store=prepared.session_store,
        task_store=second_store,
        enable_logging=False,
    )
    resolver = _TwoOwnerBarrierResolver(prepared.expected_result)
    prepared.app.register_completion_result_resolver(_contract().result_resolver, resolver)
    second_app.register_completion_result_resolver(_contract().result_resolver, resolver)
    first_call = asyncio.create_task(prepared.app.resolve_completion_result(prepared.request))
    second_call = asyncio.create_task(second_app.resolve_completion_result(prepared.request))
    await resolver.both_started.wait()
    resolver.release.set()
    first_result, second_result = await asyncio.gather(first_call, second_call)
    assert first_result == second_result
    assert resolver.calls == 2
    assert await first_store.load_task(prepared.task.id) == first_result
    assert await second_store.load_task(prepared.task.id) == second_result
    await _assert_one_result_event(prepared)
