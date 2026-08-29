from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import warnings
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from tests.core.completion_result_resolver_conformance import (
    assert_completion_result_resolver_cross_instance_concurrency,
    assert_completion_result_resolver_store_conformance,
)
from tests.core.task_invocation_fixtures import stored_session_invocation
from tests.core.test_completion_decision_application import (
    _assert_secret_absent_from_cayu_error,
    _contract,
    _persist_decision,
    _result,
    _running_task,
)

import cayu.runtime._completion_result_resolver_coordinator as resolver_coordinator_module
from cayu import (
    CayuApp,
    CompletionDecisionApplicationRequest,
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverExecutionError,
    CompletionResultResolverRef,
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
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskStatus,
    WorkCompletionConflict,
)
from cayu.runtime.sessions import (
    CheckpointTransform,
    SessionStore,
    run_request_with_task_invocation,
)
from cayu.runtime.tasks import CompletionDecisionApplicationReceipt, TaskStore
from cayu.vaults import SecretRedactor


class _Resolver(CompletionResultResolver):
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.requests: list[CompletionResultResolverRequest] = []

    async def resolve(self, request: CompletionResultResolverRequest) -> dict[str, object]:
        self.requests.append(request)
        return self.result


async def _seed_session(store: SessionStore) -> None:
    await store.create(
        RunRequest(
            agent_name="resolver-agent",
            session_id="session:application",
            messages=[Message.text("user", "Resolve the accepted result.")],
        ),
        identity=SessionIdentity(
            provider_name="resolver-provider",
            model="resolver-model",
        ),
    )


async def _prepared_app() -> tuple[
    CayuApp,
    InMemorySessionStore,
    TaskStore,
    str,
]:
    return await _prepared_app_with_stores()


async def _prepared_app_with_stores(
    *,
    session_store: InMemorySessionStore | None = None,
    task_store: TaskStore | None = None,
    secret_redactor: SecretRedactor | None = None,
) -> tuple[CayuApp, InMemorySessionStore, TaskStore, str]:
    session_store = session_store or InMemorySessionStore()
    task_store = task_store or InMemoryTaskStore()
    await _seed_session(session_store)
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    task = await _running_task(
        task_store,
        session_invocation=await stored_session_invocation(
            session_store,
            "session:application",
        ),
    )
    decision_id, _reference = await _persist_decision(
        task_store,
        task=task,
        ordinal=1,
        verdict=CompletionVerdict.ACCEPTED,
    )
    return app, session_store, task_store, decision_id


def _resolution_request(decision_id: str) -> CompletionResultResolutionRequest:
    return CompletionResultResolutionRequest(
        task_id="application-task",
        decision_id=decision_id,
        idempotency_key="resolve-result-1",
    )


def _group_leaves(error: BaseExceptionGroup) -> list[BaseException]:
    leaves: list[BaseException] = []
    pending = list(error.exceptions)
    while pending:
        current = pending.pop(0)
        if isinstance(current, BaseExceptionGroup):
            pending[0:0] = current.exceptions
        else:
            leaves.append(current)
    return leaves


def test_result_resolver_identity_is_required_and_fingerprinted() -> None:
    contract = _contract()
    changed = contract.model_copy(
        update={
            "result_resolver": CompletionResultResolverRef(
                resolver_id=contract.result_resolver.resolver_id,
                version="v2",
                configuration_fingerprint=sha256(b"resolver-v2").hexdigest(),
            )
        }
    )

    assert contract.result_resolver.resolver_id == "application-result"
    assert changed.result_resolver != contract.result_resolver
    with pytest.raises(ValueError, match="fingerprint conflicts"):
        type(contract).model_validate(changed.model_dump(mode="python", warnings=False))


def test_resolve_applies_result_and_replays_without_registered_resolver() -> None:
    async def scenario() -> None:
        app, session_store, task_store, decision_id = await _prepared_app()
        resolver_result = _result("1")
        resolver = _Resolver(resolver_result)
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        resolved = await app.resolve_completion_result(request)
        resolver_result["artifact_id"] = "mutated-after-return"

        assert resolved.status is TaskStatus.COMPLETED
        assert resolved.result == _result("1")
        assert len(resolver.requests) == 1
        assert resolver.requests[0].decision.decision_id == decision_id
        assert resolver.requests[0].result_reference == resolver.requests[0].proposal.result

        restarted = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        replayed = await restarted.resolve_completion_result(request)
        assert replayed == resolved

        records = await session_store.query_events(
            EventQuery(
                session_id="session:application",
                event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                limit=10,
            )
        )
        assert len(records) == 1
        event = records[0].event
        assert event.interaction_id is None
        assert event.timestamp == resolved.completed_at
        assert event.payload["result_digest"] == resolver.requests[0].result_reference.digest
        assert "contract_version" not in event.payload
        assert "result" not in event.payload

    asyncio.run(scenario())


def test_runtime_publication_expiry_is_not_rejected_by_short_secret_collision() -> None:
    async def scenario() -> None:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(str(datetime.now(UTC).year)),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            _Resolver(_result("1")),
        )

        completed = await app.resolve_completion_result(_resolution_request(decision_id))

        assert completed.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_task_created_before_its_session_resolves_through_task_backed_provenance() -> None:
    async def scenario() -> None:
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        contract = _contract()
        await task_store.publish_work_contract(contract)
        task = await task_store.create_task(
            TaskCreate(
                task_id="application-task",
                type="verified-work",
                session_id="session:application",
                work_contract=contract.reference(),
            )
        )
        await session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    agent_name="resolver-agent",
                    session_id="session:application",
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
                provider_name="resolver-provider",
                model="resolver-model",
            ),
        )
        task = await task_store.start_task(
            task.id,
            session_id="session:application",
            session_invocation=await stored_session_invocation(
                session_store,
                "session:application",
            ),
        )
        decision_id, _reference = await _persist_decision(
            task_store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(contract.result_resolver, resolver)

        completed = await app.resolve_completion_result(_resolution_request(decision_id))

        assert completed.status is TaskStatus.COMPLETED
        assert len(resolver.requests) == 1

    asyncio.run(scenario())


def test_missing_exact_resolver_fails_before_application() -> None:
    async def scenario() -> None:
        app, _session_store, task_store, decision_id = await _prepared_app()
        with pytest.raises(CompletionResultResolverUnavailable):
            await app.resolve_completion_result(_resolution_request(decision_id))
        task = await task_store.load_task("application-task")
        assert task is not None
        assert task.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_conflicting_task_session_fails_before_resolver_dispatch() -> None:
    class ConflictingTaskSessionStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.forge_task_session = False

        async def load_task(self, task_id: str) -> Task | None:
            task = await super().load_task(task_id)
            if task is None or not self.forge_task_session:
                return task
            return task.model_copy(update={"session_id": "session:conflicting"})

    async def scenario() -> None:
        task_store = ConflictingTaskSessionStore()
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            task_store=task_store,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        task_store.forge_task_session = True

        with pytest.raises(WorkCompletionConflict, match="Durable task authority conflicts"):
            await app.resolve_completion_result(_resolution_request(decision_id))

        assert resolver.requests == []
        task_store.forge_task_session = False
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_mutable_current_registration_cannot_replace_contract_resolver() -> None:
    async def scenario() -> None:
        app, _session_store, task_store, decision_id = await _prepared_app()
        wrong_reference = CompletionResultResolverRef(
            resolver_id=_contract().result_resolver.resolver_id,
            version="v2",
            configuration_fingerprint=sha256(b"application-result-v2").hexdigest(),
        )
        wrong_resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(wrong_reference, wrong_resolver)

        with pytest.raises(CompletionResultResolverUnavailable):
            await app.resolve_completion_result(_resolution_request(decision_id))

        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        assert wrong_resolver.requests == []

    asyncio.run(scenario())


def test_conflicting_duplicate_registration_is_rejected_without_replacement() -> None:
    async def scenario() -> None:
        app, _session_store, _task_store, decision_id = await _prepared_app()
        original = _Resolver(_result("1"))
        replacement = _Resolver(_result("changed"))
        app.register_completion_result_resolver(_contract().result_resolver, original)

        with pytest.raises(ValueError, match="already registered"):
            app.register_completion_result_resolver(
                _contract().result_resolver,
                replacement,
            )

        resolved = await app.resolve_completion_result(_resolution_request(decision_id))
        assert resolved.result == _result("1")
        assert len(original.requests) == 1
        assert replacement.requests == []

    asyncio.run(scenario())


def test_changed_result_content_fails_closed_before_application() -> None:
    async def scenario() -> None:
        app, _session_store, task_store, decision_id = await _prepared_app()
        resolver = _Resolver(_result("changed"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)

        with pytest.raises(
            CompletionResultResolverExecutionError,
            match="invalid result content",
        ):
            await app.resolve_completion_result(_resolution_request(decision_id))

        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        assert persisted.result is None

    asyncio.run(scenario())


def test_unavailable_result_uses_typed_safe_failure() -> None:
    class UnavailableResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise CompletionResultUnavailable("application detail")

    async def scenario() -> None:
        app, _session_store, task_store, decision_id = await _prepared_app()
        app.register_completion_result_resolver(
            _contract().result_resolver,
            UnavailableResolver(),
        )
        with pytest.raises(CompletionResultUnavailable, match="accepted completion result"):
            await app.resolve_completion_result(_resolution_request(decision_id))
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_resolver_failure_does_not_expose_workload_secret(
    caplog,
    capsys,
) -> None:
    secret = "completion-result-resolver-secret-canary"

    class FailingResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise RuntimeError(f"resolver exposed {secret}")

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            FailingResolver(),
        )
        with pytest.raises(CompletionResultResolverExecutionError) as captured:
            await app.resolve_completion_result(_resolution_request(decision_id))
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(warning.message) for warning in caught_warnings)


def test_resolver_failure_does_not_retain_secret_bearing_contract_content() -> None:
    secret = _contract().objective

    class FailingResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise RuntimeError("resolver failed")

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            FailingResolver(),
        )
        with pytest.raises(CompletionResultResolverExecutionError) as captured:
            await app.resolve_completion_result(_resolution_request(decision_id))
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_wrong_type_resolver_result_does_not_retain_extension_object() -> None:
    secret = "completion-result-object-secret-canary"

    class SecretValue:
        def __repr__(self) -> str:
            return secret

    class WrongTypeResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            return cast("dict[str, object]", SecretValue())

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            WrongTypeResolver(),
        )
        with pytest.raises(CompletionResultResolverExecutionError) as captured:
            await app.resolve_completion_result(_resolution_request(decision_id))
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_concurrent_exact_resolution_invokes_resolver_once() -> None:
    class BarrierResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return _result("1")

    async def scenario() -> None:
        app, _session_store, _task_store, decision_id = await _prepared_app()
        resolver = BarrierResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        first = asyncio.create_task(app.resolve_completion_result(request))
        await resolver.started.wait()
        second = asyncio.create_task(app.resolve_completion_result(request))
        await asyncio.sleep(0)
        assert resolver.calls == 1
        resolver.release.set()

        first_result, second_result = await asyncio.gather(first, second)
        assert first_result == second_result
        assert resolver.calls == 1

    asyncio.run(scenario())


def test_concurrent_application_instances_converge_on_one_receipt_and_event() -> None:
    class TwoOwnerBarrierResolver(CompletionResultResolver):
        def __init__(self) -> None:
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
            return _result("1")

    async def scenario() -> None:
        first, session_store, task_store, decision_id = await _prepared_app()
        second = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        resolver = TwoOwnerBarrierResolver()
        first.register_completion_result_resolver(_contract().result_resolver, resolver)
        second.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        first_call = asyncio.create_task(first.resolve_completion_result(request))
        second_call = asyncio.create_task(second.resolve_completion_result(request))
        await resolver.both_started.wait()
        resolver.release.set()
        first_result, second_result = await asyncio.gather(first_call, second_call)

        assert first_result == second_result
        assert resolver.calls == 2
        records = await session_store.query_events(
            EventQuery(
                session_id="session:application",
                event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                limit=10,
            )
        )
        assert len(records) == 1

    asyncio.run(scenario())


def test_completed_owner_does_not_retire_a_slower_concurrent_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowResolver(CompletionResultResolver):
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
            return _result("1")

    async def scenario() -> None:
        first, session_store, task_store, decision_id = await _prepared_app()
        second = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        slow = SlowResolver()
        fast = _Resolver(_result("1"))
        first.register_completion_result_resolver(_contract().result_resolver, slow)
        second.register_completion_result_resolver(_contract().result_resolver, fast)

        slow_operation = asyncio.create_task(
            first.resolve_completion_result(_resolution_request(decision_id))
        )
        await slow.started.wait()
        completed = await second.resolve_completion_result(_resolution_request(decision_id))
        assert completed.status is TaskStatus.COMPLETED
        await asyncio.sleep(0.05)

        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        reservations = checkpoint["completion_result_event_publications"]["reservations"]
        assert len(reservations) == 1
        assert len(next(iter(reservations.values()))["owners"]) == 1
        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")

        slow.release.set()
        assert await slow_operation == completed
        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" not in checkpoint

    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_LEASE_SECONDS",
        0.2,
    )
    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_HEARTBEAT_SECONDS",
        0.01,
    )
    asyncio.run(scenario())


def test_timed_out_resolver_remains_fenced_until_owned_task_settles() -> None:
    class SlowCancellationResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.settled = asyncio.Event()
            self.calls = 0

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.calls += 1
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            self.settled.set()
            return _result("1")

    async def scenario() -> None:
        app, _session_store, task_store, decision_id = await _prepared_app()
        resolver = SlowCancellationResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id).model_copy(
            update={"execution_timeout_seconds": 0.01}
        )

        with pytest.raises(CompletionResultResolverExecutionError, match="timeout"):
            await app.resolve_completion_result(request)
        with pytest.raises(CompletionResultResolverExecutionError, match="draining"):
            await app.resolve_completion_result(request)
        assert resolver.calls == 1

        resolver.release.set()
        await resolver.settled.wait()
        await asyncio.sleep(0)

        completed = await app.resolve_completion_result(request)
        assert completed.status is TaskStatus.COMPLETED
        assert resolver.calls == 2
        persisted = await task_store.load_task("application-task")
        assert persisted == completed

    asyncio.run(scenario())


def test_cancellation_opaque_resolver_remains_fenced_until_thread_settles() -> None:
    class ThreadedResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.settled = threading.Event()
            self.calls = 0

        def blocking_read(self) -> dict[str, object]:
            self.calls += 1
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("resolver test thread was not released")
            self.settled.set()
            return _result("1")

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            return await asyncio.to_thread(self.blocking_read)

    async def scenario() -> None:
        app, session_store, _task_store, decision_id = await _prepared_app()
        resolver = ThreadedResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        operation = asyncio.create_task(app.resolve_completion_result(request))
        try:
            assert await asyncio.to_thread(resolver.started.wait, 2)
            operation.cancel("stop cancellation-opaque resolver")
            with pytest.raises(
                asyncio.CancelledError,
                match="stop cancellation-opaque resolver",
            ):
                await operation
            for _ in range(70):
                with pytest.raises(CompletionResultResolverExecutionError, match="draining"):
                    await app.resolve_completion_result(request)
            assert resolver.calls == 1
            draining_checkpoint = await session_store.load_checkpoint("session:application")
            assert draining_checkpoint is not None
            reservations = draining_checkpoint["completion_result_event_publications"][
                "reservations"
            ]
            assert len(reservations) == 1
            assert len(next(iter(reservations.values()))["owners"]) == 1

            resolver.release.set()
            assert await asyncio.to_thread(resolver.settled.wait, 2)
            await asyncio.sleep(0)

            completed = await app.resolve_completion_result(request)
            assert completed.status is TaskStatus.COMPLETED
            assert resolver.calls == 2
            checkpoint = await session_store.load_checkpoint("session:application")
            assert checkpoint is not None
            assert "completion_result_event_publications" not in checkpoint
        finally:
            resolver.release.set()

    asyncio.run(scenario())


def test_missing_session_rejects_resolution_before_resolver_dispatch() -> None:
    async def scenario() -> None:
        app, session_store, task_store, decision_id = await _prepared_app()
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        await session_store.delete_session("session:application")

        with pytest.raises(KeyError, match="Session not found"):
            await app.resolve_completion_result(_resolution_request(decision_id))

        assert resolver.requests == []
        task = await task_store.load_task("application-task")
        assert task is not None
        assert task.status is TaskStatus.RUNNING
        assert task.result is None

    asyncio.run(scenario())


def test_session_delete_is_fenced_until_result_event_publication() -> None:
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
            return _result("1")

    async def scenario() -> None:
        app, session_store, _task_store, decision_id = await _prepared_app()
        resolver = BlockingResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await resolver.started.wait()

        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")
        assert await session_store.load("session:application") is not None

        resolver.release.set()
        assert (await operation).status is TaskStatus.COMPLETED
        await session_store.delete_session("session:application")
        assert await session_store.load("session:application") is None

    asyncio.run(scenario())


def test_foreground_resolver_renews_publication_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            return _result("1")

    async def scenario() -> None:
        app, session_store, _task_store, decision_id = await _prepared_app()
        resolver = BlockingResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await resolver.started.wait()
        await asyncio.sleep(0.12)

        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")

        resolver.release.set()
        assert (await operation).status is TaskStatus.COMPLETED

    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_LEASE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_HEARTBEAT_SECONDS",
        0.01,
    )
    asyncio.run(scenario())


def test_foreground_application_renews_publication_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingApplicationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.application_started = asyncio.Event()
            self.application_allowed = asyncio.Event()

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_started.set()
            await self.application_allowed.wait()
            return await super().apply_completion_decision(request)

    async def scenario() -> None:
        task_store = BlockingApplicationStore()
        app, session_store, _task_store, decision_id = await _prepared_app_with_stores(
            task_store=task_store,
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            _Resolver(_result("1")),
        )
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await task_store.application_started.wait()
        await asyncio.sleep(0.12)

        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")

        task_store.application_allowed.set()
        assert (await operation).status is TaskStatus.COMPLETED

    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_LEASE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_HEARTBEAT_SECONDS",
        0.01,
    )
    asyncio.run(scenario())


def test_foreground_event_publication_renews_publication_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingResultEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_completion_result_event_publication_reservations = True

        def __init__(self) -> None:
            super().__init__()
            self.publication_started = asyncio.Event()
            self.publication_allowed = asyncio.Event()

        async def _publish_completion_result_event_publication(
            self,
            session_id: str,
            *,
            checkpoint_transform: CheckpointTransform,
            events: list[Event],
        ) -> Session:
            if any(event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events):
                self.publication_started.set()
                await self.publication_allowed.wait()
            return await super()._publish_completion_result_event_publication(
                session_id,
                checkpoint_transform=checkpoint_transform,
                events=events,
            )

    async def scenario() -> None:
        session_store = BlockingResultEventStore()
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            _Resolver(_result("1")),
        )
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await session_store.publication_started.wait()
        await asyncio.sleep(0.12)

        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")

        session_store.publication_allowed.set()
        assert (await operation).status is TaskStatus.COMPLETED

    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_LEASE_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        resolver_coordinator_module,
        "_PUBLICATION_OWNER_HEARTBEAT_SECONDS",
        0.01,
    )
    asyncio.run(scenario())


def test_custom_store_reservation_overrides_fail_before_resolver_dispatch() -> None:
    class IncompleteCreateStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def create(
            self,
            request: RunRequest,
            *,
            identity: SessionIdentity,
            interaction_started_event: Event | None = None,
            interaction_source_messages: list[Message] | None = None,
            checkpoint_transform: CheckpointTransform | None = None,
        ) -> Session:
            return await super().create(
                request,
                identity=identity,
                interaction_started_event=interaction_started_event,
                interaction_source_messages=interaction_source_messages,
                checkpoint_transform=checkpoint_transform,
            )

    class IncompleteCheckpointStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
            await super().checkpoint(session_id, state)

    class IncompleteTransformStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def transform_checkpoint(
            self,
            session_id: str,
            checkpoint_transform: CheckpointTransform,
        ) -> None:
            await super().transform_checkpoint(session_id, checkpoint_transform)

    async def scenario() -> None:
        for session_store in (
            IncompleteCreateStore(),
            IncompleteCheckpointStore(),
            IncompleteTransformStore(),
        ):
            app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
                session_store=session_store,
            )
            resolver = _Resolver(_result("1"))
            app.register_completion_result_resolver(_contract().result_resolver, resolver)
            checkpoint_before = await session_store.load_checkpoint("session:application")

            with pytest.raises(
                NotImplementedError,
                match="owns publication reservation, checkpoint replacement, and deletion",
            ):
                await app.resolve_completion_result(_resolution_request(decision_id))

            assert resolver.requests == []
            persisted = await task_store.load_task("application-task")
            assert persisted is not None
            assert persisted.status is TaskStatus.RUNNING
            assert persisted.result is None
            assert await session_store.load_checkpoint("session:application") == checkpoint_before

    asyncio.run(scenario())


def test_cancellation_during_publication_reservation_release_remains_authoritative() -> None:
    class BlockingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_completion_result_event_publication_reservations = True

        def __init__(self) -> None:
            super().__init__()
            self.empty_publications = 0
            self.release_started = asyncio.Event()
            self.release_allowed = asyncio.Event()

        async def _publish_completion_result_event_publication(
            self,
            session_id: str,
            *,
            checkpoint_transform: CheckpointTransform,
            events: list[Event],
        ) -> Session:
            if not events:
                self.empty_publications += 1
                if self.empty_publications == 2:
                    self.release_started.set()
                    await self.release_allowed.wait()
            return await super()._publish_completion_result_event_publication(
                session_id,
                checkpoint_transform=checkpoint_transform,
                events=events,
            )

    class UnavailableResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise CompletionResultUnavailable("source result is unavailable")

    async def scenario() -> None:
        session_store = BlockingReleaseStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            UnavailableResolver(),
        )
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await session_store.release_started.wait()

        operation.cancel("cancel while publication ownership settles")
        assert operation.cancelling() == 1
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel while publication ownership settles",
        ) as captured:
            await operation
        assert operation.cancelled()
        assert operation.cancelling() == 1
        assert isinstance(captured.value.__cause__, CompletionResultUnavailable)

        with pytest.raises(CompletionResultResolverExecutionError, match="draining"):
            await app.resolve_completion_result(_resolution_request(decision_id))
        session_store.release_allowed.set()
        for _ in range(20):
            await asyncio.sleep(0)
            checkpoint = await session_store.load_checkpoint("session:application")
            if checkpoint is not None and "completion_result_event_publications" not in checkpoint:
                break

        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" not in checkpoint
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        assert persisted.result is None

    asyncio.run(scenario())


def test_primary_and_publication_release_failure_are_both_preserved() -> None:
    secret = "publication-release-ordinary-failure-secret-canary"

    class FailingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_completion_result_event_publication_reservations = True

        def __init__(self) -> None:
            super().__init__()
            self.empty_publications = 0

        async def _publish_completion_result_event_publication(
            self,
            session_id: str,
            *,
            checkpoint_transform: CheckpointTransform,
            events: list[Event],
        ) -> Session:
            if not events:
                self.empty_publications += 1
                if self.empty_publications == 2:
                    raise ConnectionError(secret)
            return await super()._publish_completion_result_event_publication(
                session_id,
                checkpoint_transform=checkpoint_transform,
                events=events,
            )

    class UnavailableResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise CompletionResultUnavailable(secret)

    async def scenario() -> BaseException:
        session_store = FailingReleaseStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            UnavailableResolver(),
        )

        with pytest.raises(CompletionResultUnavailable) as captured:
            await app.resolve_completion_result(_resolution_request(decision_id))
        assert isinstance(captured.value.__cause__, ConnectionError)
        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" in checkpoint
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_cancellation_preserves_primary_and_publication_release_failure() -> None:
    secret = "publication-release-combined-failure-secret-canary"

    class BlockingFailingReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_completion_result_event_publication_reservations = True

        def __init__(self) -> None:
            super().__init__()
            self.empty_publications = 0
            self.release_started = asyncio.Event()
            self.release_allowed = asyncio.Event()

        async def _publish_completion_result_event_publication(
            self,
            session_id: str,
            *,
            checkpoint_transform: CheckpointTransform,
            events: list[Event],
        ) -> Session:
            if not events:
                self.empty_publications += 1
                if self.empty_publications == 2:
                    self.release_started.set()
                    await self.release_allowed.wait()
                    raise ConnectionError(secret)
            return await super()._publish_completion_result_event_publication(
                session_id,
                checkpoint_transform=checkpoint_transform,
                events=events,
            )

    class UnavailableResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise CompletionResultUnavailable(secret)

    async def scenario() -> BaseException:
        session_store = BlockingFailingReleaseStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            UnavailableResolver(),
        )
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await session_store.release_started.wait()
        operation.cancel(secret)
        session_store.release_allowed.set()

        with pytest.raises(asyncio.CancelledError) as captured:
            await operation
        assert operation.cancelled()
        assert operation.cancelling() == 1
        cause = captured.value.__cause__
        assert isinstance(cause, CompletionResultUnavailable)
        session_store.release_allowed.set()
        for _ in range(20):
            await asyncio.sleep(0)
            try:
                await app.resolve_completion_result(_resolution_request(decision_id))
            except CompletionResultResolverExecutionError as error:
                if "draining" in str(error):
                    continue
                raise
            except ConnectionError:
                break
        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" in checkpoint
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_process_control_during_publication_reservation_release_is_authoritative() -> None:
    secret = "publication-release-process-control-secret-canary"

    class ProcessControlReleaseStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_completion_result_event_publication_reservations = True

        def __init__(self) -> None:
            super().__init__()
            self.empty_publications = 0
            self.release_started = asyncio.Event()
            self.release_allowed = asyncio.Event()

        async def _publish_completion_result_event_publication(
            self,
            session_id: str,
            *,
            checkpoint_transform: CheckpointTransform,
            events: list[Event],
        ) -> Session:
            if not events:
                self.empty_publications += 1
                if self.empty_publications == 2:
                    self.release_started.set()
                    await self.release_allowed.wait()
                    raise GeneratorExit(secret)
            return await super()._publish_completion_result_event_publication(
                session_id,
                checkpoint_transform=checkpoint_transform,
                events=events,
            )

    class UnavailableResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise CompletionResultUnavailable(secret)

    async def scenario() -> BaseException:
        session_store = ProcessControlReleaseStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            UnavailableResolver(),
        )

        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await session_store.release_started.wait()
        operation.cancel(secret)
        await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError) as captured:
            await operation
        assert operation.cancelled()
        assert operation.cancelling() == 1
        assert isinstance(captured.value.__cause__, CompletionResultUnavailable)
        session_store.release_allowed.set()
        for _ in range(20):
            await asyncio.sleep(0)
            try:
                await app.resolve_completion_result(_resolution_request(decision_id))
            except CompletionResultResolverExecutionError as error:
                if "draining" in str(error):
                    continue
                raise
            except GeneratorExit as late:
                captured_late = late
                break
        else:
            raise AssertionError("late process-control settlement was not observed")
        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" in checkpoint
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING
        return captured_late

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_known_application_conflict_releases_result_event_publication() -> None:
    async def scenario() -> None:
        app, session_store, _task_store, decision_id = await _prepared_app()
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)

        completed = await app.resolve_completion_result(_resolution_request(decision_id))
        assert completed.status is TaskStatus.COMPLETED

        conflicting_request = _resolution_request(decision_id).model_copy(
            update={"idempotency_key": "resolve-result-conflict"}
        )
        with pytest.raises(WorkCompletionConflict, match="already applied"):
            await app.resolve_completion_result(conflicting_request)

        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" not in checkpoint
        assert len(resolver.requests) == 2

        await session_store.delete_session("session:application")
        assert await session_store.load("session:application") is None

    asyncio.run(scenario())


def test_sqlite_resolution_replays_after_task_store_reopen_without_resolver(
    tmp_path,
) -> None:
    async def scenario() -> None:
        database = tmp_path / "completion-result-resolution.sqlite"
        first_store = SQLiteTaskStore(database)
        app, session_store, _task_store, decision_id = await _prepared_app_with_stores(
            task_store=first_store,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        completed = await app.resolve_completion_result(request)
        await first_store.close()

        reopened_store = SQLiteTaskStore(database)
        try:
            restarted = CayuApp(
                session_store=session_store,
                task_store=reopened_store,
                enable_logging=False,
            )
            replayed = await restarted.resolve_completion_result(request)
            assert replayed == completed
            assert len(resolver.requests) == 1
        finally:
            await reopened_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ("memory", "sqlite"))
def test_completion_result_resolver_store_conformance(store_kind: str, tmp_path) -> None:
    async def scenario() -> None:
        store: TaskStore
        if store_kind == "memory":
            store = InMemoryTaskStore()
        else:
            store = SQLiteTaskStore(tmp_path / "resolver-conformance.sqlite")
        try:
            await assert_completion_result_resolver_store_conformance(
                store,
                store_kind=store_kind,
            )
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


def test_sqlite_completion_result_resolution_is_atomic_across_store_instances(
    tmp_path,
) -> None:
    async def scenario() -> None:
        database = tmp_path / "resolver-cross-instance.sqlite"
        first = SQLiteTaskStore(database)
        second = SQLiteTaskStore(database)
        try:
            await assert_completion_result_resolver_cross_instance_concurrency(
                first,
                second,
                store_kind="sqlite",
            )
        finally:
            await first.close()
            await second.close()

    asyncio.run(scenario())


def test_sqlite_result_resolution_recovers_after_real_process_loss(tmp_path) -> None:
    session_database = tmp_path / "resolver-process-session.sqlite"
    task_database = tmp_path / "resolver-process-task.sqlite"
    resolver_marker = tmp_path / "resolver-called.txt"

    async def seed() -> str:
        session_store = SQLiteSessionStore(session_database)
        task_store = SQLiteTaskStore(task_database)
        try:
            await _seed_session(session_store)
            task = await _running_task(
                task_store,
                session_invocation=await stored_session_invocation(
                    session_store,
                    "session:application",
                ),
            )
            decision_id, _reference = await _persist_decision(
                task_store,
                task=task,
                ordinal=1,
                verdict=CompletionVerdict.ACCEPTED,
            )
            return decision_id
        finally:
            await session_store.close()
            await task_store.close()

    decision_id = asyncio.run(seed())
    repository = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name("_completion_result_resolver_process_worker.py")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(repository / "src"), str(repository))),
    }
    crashed = subprocess.run(
        [
            sys.executable,
            str(worker),
            str(session_database),
            str(task_database),
            str(resolver_marker),
            decision_id,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert crashed.returncode == 86, (crashed.stdout, crashed.stderr)
    assert resolver_marker.read_text(encoding="utf-8") == "resolved\n"

    async def recover() -> None:
        session_store = SQLiteSessionStore(session_database)
        task_store = SQLiteTaskStore(task_database)
        try:
            app = CayuApp(
                session_store=session_store,
                task_store=task_store,
                enable_logging=False,
            )
            request = _resolution_request(decision_id)
            completed = await app.resolve_completion_result(request)
            assert completed.status is TaskStatus.COMPLETED
            assert completed.result == _result("1")
            assert await app.resolve_completion_result(request) == completed
            records = await session_store.query_events(
                EventQuery(
                    session_id="session:application",
                    event_types=(EventType.TASK_COMPLETION_RESULT_RESOLVED,),
                    limit=10,
                )
            )
            assert len(records) == 1
        finally:
            await session_store.close()
            await task_store.close()

    asyncio.run(recover())


def test_event_publication_failure_replays_from_application_receipt() -> None:
    class FailResultEventOnceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_result_event_once = True

        def _append_events_unlocked(
            self,
            session: Session,
            events: list[Event],
        ) -> Session:
            if self.fail_result_event_once and any(
                event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events
            ):
                self.fail_result_event_once = False
                raise ConnectionError("result event publication failed")
            return super()._append_events_unlocked(session, events)

    async def scenario() -> None:
        session_store = FailResultEventOnceStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        with pytest.raises(ConnectionError, match="publication failed"):
            await app.resolve_completion_result(request)
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.COMPLETED

        restarted = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        replayed = await restarted.resolve_completion_result(request)
        assert replayed == persisted
        assert len(resolver.requests) == 1

        records = await session_store.query_events(
            EventQuery(
                session_id="session:application",
                event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                limit=10,
            )
        )
        assert len(records) == 1

    asyncio.run(scenario())


def test_receipt_replay_rejects_forged_session_incarnation_before_event_publication() -> None:
    class FailResultEventOnceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_result_event_once = True

        def _append_events_unlocked(
            self,
            session: Session,
            events: list[Event],
        ) -> Session:
            if self.fail_result_event_once and any(
                event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events
            ):
                self.fail_result_event_once = False
                raise ConnectionError("result event publication failed")
            return super()._append_events_unlocked(session, events)

    class ForgedReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_receipt = False

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None or not self.forge_receipt:
                return receipt
            forged_task = receipt.task.model_copy(update={"session_instance_id": str(uuid4())})
            return receipt.model_copy(update={"task": forged_task})

    async def scenario() -> None:
        session_store = FailResultEventOnceStore()
        task_store = ForgedReceiptStore()
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            task_store=task_store,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        with pytest.raises(ConnectionError, match="publication failed"):
            await app.resolve_completion_result(request)
        task_store.forge_receipt = True

        restarted = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        with pytest.raises(
            WorkCompletionConflict,
            match="durable immutable task authority",
        ):
            await restarted.resolve_completion_result(request)

        records = await session_store.query_events(
            EventQuery(
                session_id="session:application",
                event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                limit=10,
            )
        )
        assert records == []
        assert len(resolver.requests) == 1

    asyncio.run(scenario())


def test_receipt_replay_reacquires_publication_ownership_before_event_readback() -> None:
    class FailThenBlockResultEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_result_event_once = True
            self.block_result_readback = False
            self.target_event_id: str | None = None
            self.readback_started = asyncio.Event()
            self.readback_allowed = asyncio.Event()

        def _append_events_unlocked(
            self,
            session: Session,
            events: list[Event],
        ) -> Session:
            result_event = next(
                (
                    event
                    for event in events
                    if event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED
                ),
                None,
            )
            if self.fail_result_event_once and result_event is not None:
                self.fail_result_event_once = False
                self.target_event_id = result_event.id
                raise ConnectionError("result event publication failed")
            return super()._append_events_unlocked(session, events)

        async def query_events(self, query: EventQuery | None = None):
            if (
                self.block_result_readback
                and query is not None
                and query.event_id == self.target_event_id
            ):
                self.block_result_readback = False
                self.readback_started.set()
                await self.readback_allowed.wait()
            return await super().query_events(query)

    async def scenario() -> None:
        session_store = FailThenBlockResultEventStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)

        with pytest.raises(ConnectionError, match="publication failed"):
            await app.resolve_completion_result(request)

        expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()

        def expire_prior_owner(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            assert checkpoint is not None
            reservations = checkpoint["completion_result_event_publications"]["reservations"]
            for reservation in reservations.values():
                for owner in reservation["owners"].values():
                    owner["expires_at"] = expired_at
            return checkpoint

        await session_store._publish_completion_result_event_publication(
            "session:application",
            checkpoint_transform=expire_prior_owner,
            events=[],
        )

        restarted = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        session_store.block_result_readback = True
        replay = asyncio.create_task(restarted.resolve_completion_result(request))
        await session_store.readback_started.wait()
        with pytest.raises(
            ValueError,
            match="completion-result event publication is incomplete",
        ):
            await session_store.delete_session("session:application")
        session_store.readback_allowed.set()
        assert (await replay).status is TaskStatus.COMPLETED
        assert len(resolver.requests) == 1

    asyncio.run(scenario())


def test_caller_cancellation_during_publication_readback_preserves_failure() -> None:
    secret = "result-event-publication-readback-secret-canary"

    class FailAndBlockResultEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.fail_result_event_once = True
            self.block_readback_once = True
            self.target_event_id: str | None = None
            self.readback_started = asyncio.Event()
            self.readback_allowed = asyncio.Event()

        def _append_events_unlocked(
            self,
            session: Session,
            events: list[Event],
        ) -> Session:
            result_event = next(
                (
                    event
                    for event in events
                    if event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED
                ),
                None,
            )
            if self.fail_result_event_once and result_event is not None:
                self.fail_result_event_once = False
                self.target_event_id = result_event.id
                raise ConnectionError(secret)
            return super()._append_events_unlocked(session, events)

        async def query_events(self, query: EventQuery | None = None):
            if (
                self.block_readback_once
                and query is not None
                and query.event_id == self.target_event_id
            ):
                self.block_readback_once = False
                self.readback_started.set()
                await self.readback_allowed.wait()
            return await super().query_events(query)

    async def scenario() -> BaseException:
        session_store = FailAndBlockResultEventStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            secret_redactor=SecretRedactor(secret),
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        operation = asyncio.create_task(app.resolve_completion_result(request))
        try:
            await session_store.readback_started.wait()
            operation.cancel(secret)
            assert operation.cancelling() == 1
            with pytest.raises(asyncio.CancelledError) as captured:
                await operation
            assert operation.cancelled()
            assert operation.cancelling() == 1
            assert isinstance(captured.value.__cause__, ConnectionError)

            checkpoint = await session_store.load_checkpoint("session:application")
            assert checkpoint is not None
            assert "completion_result_event_publications" in checkpoint
            persisted = await task_store.load_task("application-task")
            assert persisted is not None
            assert persisted.status is TaskStatus.COMPLETED

            replayed = await app.resolve_completion_result(request)
            assert replayed == persisted
            assert len(resolver.requests) == 1
            records = await session_store.query_events(
                EventQuery(
                    session_id="session:application",
                    event_types={EventType.TASK_COMPLETION_RESULT_RESOLVED},
                    limit=10,
                )
            )
            assert len(records) == 1
            return captured.value
        finally:
            session_store.readback_allowed.set()

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_child_only_event_publication_cancellation_replays_from_receipt() -> None:
    secret = "result-event-child-cancellation-secret-canary"

    class CancelResultEventOnceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.cancel_result_event_once = True

        def _append_events_unlocked(
            self,
            session: Session,
            events: list[Event],
        ) -> Session:
            if self.cancel_result_event_once and any(
                event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events
            ):
                self.cancel_result_event_once = False
                raise asyncio.CancelledError(secret)
            return super()._append_events_unlocked(session, events)

    async def scenario() -> BaseException:
        session_store = CancelResultEventOnceStore()
        app, _session_store, task_store, decision_id = await _prepared_app_with_stores(
            session_store=session_store,
            secret_redactor=SecretRedactor(secret),
        )
        resolver = _Resolver(_result("1"))
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        operation = asyncio.create_task(app.resolve_completion_result(request))

        with pytest.raises(CompletionResultResolverExecutionError) as captured:
            await operation
        assert operation.cancelled() is False
        assert operation.cancelling() == 0
        persisted = await task_store.load_task("application-task")
        assert persisted is not None
        assert persisted.status is TaskStatus.COMPLETED

        replayed = await app.resolve_completion_result(request)
        assert replayed == persisted
        assert len(resolver.requests) == 1
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_child_only_resolver_cancellation_is_not_caller_cancellation() -> None:
    class ChildCancelledResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise asyncio.CancelledError("adapter cancelled itself")

    async def scenario() -> None:
        app, _session_store, _task_store, decision_id = await _prepared_app()
        app.register_completion_result_resolver(
            _contract().result_resolver,
            ChildCancelledResolver(),
        )
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        with pytest.raises(CompletionResultResolverExecutionError):
            await operation
        assert operation.cancelled() is False
        assert operation.cancelling() == 0

    asyncio.run(scenario())


def test_resolver_process_control_remains_authoritative_and_secret_safe() -> None:
    secret = "resolver-process-control-secret-canary"

    class ProcessControlResolver(CompletionResultResolver):
        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            raise BaseExceptionGroup(
                f"resolver group {secret}",
                [GeneratorExit(secret), RuntimeError(secret)],
            )

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        app.register_completion_result_resolver(
            _contract().result_resolver,
            ProcessControlResolver(),
        )
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.resolve_completion_result(_resolution_request(decision_id))
        return captured.value

    error = asyncio.run(scenario())
    assert any(isinstance(leaf, GeneratorExit) for leaf in error.exceptions)
    _assert_secret_absent_from_cayu_error(error, secret)


def test_late_resolver_process_control_remains_authoritative_after_cancellation() -> None:
    class LateProcessControlResolver(CompletionResultResolver):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def resolve(
            self,
            request: CompletionResultResolverRequest,
        ) -> dict[str, object]:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            raise GeneratorExit("late resolver process control")

    async def scenario() -> None:
        app, session_store, _task_store, decision_id = await _prepared_app()
        resolver = LateProcessControlResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        operation = asyncio.create_task(app.resolve_completion_result(request))
        await resolver.started.wait()
        operation.cancel("stop before resolver process control")
        with pytest.raises(asyncio.CancelledError):
            await operation

        resolver.release.set()
        for _ in range(30):
            await asyncio.sleep(0)
            try:
                await app.resolve_completion_result(request)
            except CompletionResultResolverExecutionError as error:
                if "draining" in str(error):
                    continue
                raise
            except GeneratorExit:
                break
        else:
            raise AssertionError("late resolver process-control signal was not observed")

        assert resolver.calls == 1
        checkpoint = await session_store.load_checkpoint("session:application")
        assert checkpoint is not None
        assert "completion_result_event_publications" not in checkpoint

    asyncio.run(scenario())


def test_caller_cancellation_remains_authoritative() -> None:
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
            return _result("1")

    async def scenario() -> None:
        app, _session_store, _task_store, decision_id = await _prepared_app()
        resolver = BlockingResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await resolver.started.wait()
        operation.cancel("stop result resolution")
        with pytest.raises(asyncio.CancelledError) as captured:
            await operation
        resolver.release.set()
        await asyncio.sleep(0)

        assert captured.value.args == ("stop result resolution",)
        assert operation.cancelled()
        assert operation.cancelling() == 1

    asyncio.run(scenario())


def test_caller_cancellation_diagnostic_is_secret_safe() -> None:
    secret = "completion-result-cancellation-secret-canary"

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
            return _result("1")

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        resolver = BlockingResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        operation = asyncio.create_task(
            app.resolve_completion_result(_resolution_request(decision_id))
        )
        await resolver.started.wait()
        operation.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as captured:
            await operation
        resolver.release.set()
        await asyncio.sleep(0)
        assert operation.cancelled()
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)


def test_lock_wait_cancellation_diagnostic_is_secret_safe() -> None:
    secret = "completion-result-lock-cancellation-secret-canary"

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
            return _result("1")

    async def scenario() -> BaseException:
        app, _session_store, _task_store, decision_id = await _prepared_app_with_stores(
            secret_redactor=SecretRedactor(secret),
        )
        resolver = BlockingResolver()
        app.register_completion_result_resolver(_contract().result_resolver, resolver)
        request = _resolution_request(decision_id)
        owner = asyncio.create_task(app.resolve_completion_result(request))
        await resolver.started.wait()
        waiter = asyncio.create_task(app.resolve_completion_result(request))
        await asyncio.sleep(0)
        waiter.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as captured:
            await waiter
        resolver.release.set()
        assert (await owner).status is TaskStatus.COMPLETED
        assert waiter.cancelled()
        return captured.value

    error = asyncio.run(scenario())
    _assert_secret_absent_from_cayu_error(error, secret)
