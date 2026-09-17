"""Public regressions for diagnostic detachment and exact-key convergence."""

from __future__ import annotations

import asyncio
import threading
import traceback
import warnings
from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_exports import (
    CONTEXT,
    SECRET,
    AcceptanceReader,
    Harness,
    harness,
    published,
)
from tests.core.test_session_exports import (
    backend as backend,
)

from cayu.collaboration._session_export_store import operation_key
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportCapacityExceeded,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.events import EventType
from cayu.sessions.base import InMemorySessionStore
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize(
    ("cleanup", "caller_cancels"),
    [
        ("success", 0),
        ("failure", 0),
        ("group", 0),
        ("dependency_cancel", 0),
        ("failure", 2),
        ("group", 2),
        ("dependency_cancel", 2),
    ],
)
@pytest.mark.parametrize("readback", ["unavailable", "conflict"])
def test_lookup_preserves_readback_failure_through_guard_cleanup(
    backend, cleanup, caller_cancels, readback, caplog, capsys
):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class FailedReadStore(base):
        session_export_version = 1
        fail_read = False

        async def load_session_operation(self, *args, **kwargs):
            if self.fail_read:
                if readback == "conflict":
                    raise SessionExportConflict()
                raise OSError("readback failed " + SECRET)
            return await super().load_session_operation(*args, **kwargs)

    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app(
                store_type=FailedReadStore, redactor=SecretRedactor(SECRET)
            )
            await case.create(store)
            request = await case.request(app)
            receipt = await app.export_session(request, context=CONTEXT)
            events = await store.load_events(case.session_id)
            acquire = policy.acquire
            entered = asyncio.Event()
            dependency = None
            guard_failure = None

            async def blocked_cleanup():
                entered.set()
                await asyncio.Event().wait()

            @asynccontextmanager
            async def guarded(context, **kwargs):
                nonlocal dependency, guard_failure
                async with acquire(context, **kwargs) as authority:
                    try:
                        yield authority
                    except BaseException as error:
                        guard_failure = error
                        try:
                            if cleanup == "failure":
                                raise ValueError("cleanup failed " + SECRET) from error
                            if cleanup == "group":
                                raise ExceptionGroup(
                                    "cleanup " + SECRET,
                                    [
                                        ValueError("cleanup first " + SECRET),
                                        RuntimeError("cleanup second " + SECRET),
                                    ],
                                ) from error
                            if cleanup == "dependency_cancel":
                                dependency = asyncio.create_task(blocked_cleanup())
                                await dependency
                            raise
                        finally:
                            for _ in range(caller_cancels):
                                assert caller.cancel(SECRET)

            policy.acquire = guarded
            store.fail_read = True
            caller = asyncio.create_task(app.lookup_session_export(request, context=CONTEXT))
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    if cleanup == "dependency_cancel":
                        await asyncio.wait_for(entered.wait(), 5)
                        assert dependency is not None
                        assert dependency.cancel(SECRET)
                    if cleanup == "success":
                        result = await asyncio.wait_for(caller, 5)
                        assert result.status == readback
                        failure = guard_failure
                    else:
                        with pytest.raises(
                            asyncio.CancelledError if caller_cancels else SessionExportUnavailable
                        ) as caught:
                            await asyncio.wait_for(caller, 5)
                        failure = caught.value
                assert failure is not None and guard_failure is not None
                assert caller.cancelled() is bool(caller_cancels)
                assert caller.cancelling() == caller_cancels
                if dependency is not None:
                    assert dependency.cancelled() and dependency.cancelling() == 1
                seen = set()
                nodes = []

                def inspect(error):
                    if error is None:
                        return
                    assert id(error) not in seen, "Each failure must occur exactly once"
                    seen.add(id(error))
                    nodes.append(error)
                    inspect(error.__cause__)
                    inspect(error.__context__)
                    if isinstance(error, BaseExceptionGroup):
                        for child in error.exceptions:
                            inspect(child)

                inspect(failure)
                primary_type = SessionExportConflict if readback == "conflict" else OSError
                assert sum(type(node) is primary_type for node in nodes) == 1
                if cleanup == "failure":
                    assert sum(type(node) is ValueError for node in nodes) == 1
                elif cleanup == "group":
                    group = next(node for node in nodes if isinstance(node, ExceptionGroup))
                    assert [type(child) for child in group.exceptions] == [ValueError, RuntimeError]
                elif cleanup == "dependency_cancel":
                    assert sum(type(node) is asyncio.CancelledError for node in nodes) == (
                        2 if caller_cancels else 1
                    )
                assert SECRET not in "".join(str(node) + repr(node) for node in nodes)
                assert SECRET not in "".join(traceback.format_exception(failure))
                assert SECRET not in str(captured_warnings)
                assert SECRET not in str(guard_failure)
                assert not app._session_export_coordinator.owners.pending
                assert projector.calls == 1
                assert await store.load_events(case.session_id) == events
                store.fail_read = False
                policy.acquire = acquire
                assert (
                    await app.lookup_session_export(request, context=CONTEXT)
                ).receipt == receipt
            finally:
                if dependency is not None and not dependency.done():
                    dependency.cancel()
                await asyncio.gather(caller, return_exceptions=True)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert SECRET not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize(
    "error_type",
    [
        SessionExportDenied,
        SessionExportConflict,
        SessionExportUnavailable,
        SessionExportCapacityExceeded,
    ],
)
@pytest.mark.parametrize("chain", ["cause", "context", "suppressed"])
def test_typed_policy_errors_detach_complete_diagnostics(
    backend, error_type, chain, caplog, capsys
):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app(redactor=SecretRedactor(SECRET))
            await case.create(store)
            request = await case.request(app)

            @asynccontextmanager
            async def denied(*args, **kwargs):
                try:
                    raise ExceptionGroup(
                        "private " + SECRET,
                        [OSError("first " + SECRET), ValueError("second " + SECRET)],
                    )
                except ExceptionGroup as dependency:
                    error = error_type()
                    error.args = (SECRET,)
                    error.add_note(SECRET)
                    if chain == "cause":
                        raise error from dependency
                    if chain == "suppressed":
                        raise error from None
                    raise error  # noqa: B904 -- exercise implicit dependency context
                yield  # pragma: no cover

            policy.acquire = denied
            with (
                warnings.catch_warnings(record=True) as captured_warnings,
                pytest.raises(error_type) as caught,
            ):
                await app.export_session(request, context=CONTEXT)
            failure = caught.value
            assert type(failure) is error_type
            seen = set()
            rendered = []

            def inspect(error):
                if error is None or id(error) in seen:
                    return
                seen.add(id(error))
                rendered.append(str(error) + repr(error) + repr(getattr(error, "__notes__", [])))
                inspect(error.__cause__)
                inspect(error.__context__)
                if isinstance(error, BaseExceptionGroup):
                    for child in error.exceptions:
                        inspect(child)

            inspect(failure)
            assert SECRET not in "".join(rendered)
            diagnostic = failure.__cause__ or failure.__context__
            assert isinstance(diagnostic, ExceptionGroup)
            assert [type(item) for item in diagnostic.exceptions] == [OSError, ValueError]
            assert "first" in str(diagnostic.exceptions[0])
            assert "second" in str(diagnostic.exceptions[1])
            assert SECRET not in "".join(traceback.format_exception(failure))
            assert SECRET not in str(captured_warnings)
            assert projector.calls == 0
            assert not await published(store, case.session_id)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert SECRET not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("grouped_cleanup", [False, True])
@pytest.mark.parametrize("cancel_count", [0, 1, 2])
def test_operation_and_guard_cleanup_failures_preserve_safe_graph(
    backend, grouped_cleanup, cancel_count, caplog, capsys
):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app(redactor=SecretRedactor(SECRET))
            await case.create(store)
            request = await case.request(app)
            acquire = policy.acquire

            def fail_projection(source):
                projector.calls += 1
                raise ValueError("projection failed " + SECRET)

            @asynccontextmanager
            async def failing_cleanup(context, **kwargs):
                async with acquire(context, **kwargs) as authority:
                    try:
                        yield authority
                    finally:
                        if "export" in kwargs["actions"]:
                            # Deliver real caller cancellation in the same turn
                            # that the owned operation settles its failure graph.
                            # No cancellation is sent to the retained owner.
                            for _ in range(cancel_count):
                                assert task.cancel(SECRET)
                            error = OSError("guard cleanup failed " + SECRET)
                            error.add_note(SECRET)
                            if grouped_cleanup:
                                raise ExceptionGroup(
                                    "cleanup " + SECRET,
                                    [error, RuntimeError("second cleanup failed " + SECRET)],
                                )
                            raise error

            projector.project = fail_projection
            policy.acquire = failing_cleanup
            cancellation_handled = False

            async def observe():
                nonlocal cancellation_handled
                try:
                    return await app.export_session(request, context=CONTEXT)
                except asyncio.CancelledError:
                    cancellation_handled = True
                    raise

            task = asyncio.create_task(observe())
            with (
                warnings.catch_warnings(record=True) as captured_warnings,
                pytest.raises(
                    asyncio.CancelledError if cancel_count else SessionExportUnavailable
                ) as caught,
            ):
                await task
            assert task.cancelled() is bool(cancel_count)
            assert task.cancelling() == cancel_count
            assert cancellation_handled is bool(cancel_count)
            assert not app._session_export_coordinator.owners.pending
            diagnostic = caught.value.__cause__
            assert diagnostic is not None
            primary = diagnostic.__context__
            assert type(primary) is ValueError
            assert "projection failed" in str(primary)
            if grouped_cleanup:
                assert isinstance(diagnostic, ExceptionGroup)
                assert [type(child) for child in diagnostic.exceptions] == [OSError, RuntimeError]
                assert "guard cleanup failed" in str(diagnostic.exceptions[0])
                assert "second cleanup failed" in str(diagnostic.exceptions[1])
            else:
                assert type(diagnostic) is OSError
                assert "guard cleanup failed" in str(diagnostic)

            seen = set()
            rendered = []

            def inspect(error):
                if error is None:
                    return
                assert id(error) not in seen, "Failure evidence must appear exactly once"
                seen.add(id(error))
                rendered.append(str(error) + repr(error) + repr(getattr(error, "__notes__", [])))
                inspect(error.__cause__)
                inspect(error.__context__)
                if isinstance(error, BaseExceptionGroup):
                    for child in error.exceptions:
                        inspect(child)

            inspect(caught.value)
            assert SECRET not in "".join(rendered)
            assert SECRET not in "".join(traceback.format_exception(caught.value))
            assert SECRET not in str(captured_warnings)
            assert projector.calls == 1
            assert not await published(store, case.session_id)
            assert (await app.lookup_session_export(request, context=CONTEXT)).status == "not_found"

    asyncio.run(run())
    captured = capsys.readouterr()
    assert SECRET not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("grouped_primary", [False, True])
def test_dependency_cleanup_cancellation_preserves_prior_failure(
    backend, grouped_primary, caplog, capsys
):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app(redactor=SecretRedactor(SECRET))
            await case.create(store)
            request = await case.request(app)
            acquire = policy.acquire
            entered = asyncio.Event()
            dependency = None

            def fail_projection(source):
                projector.calls += 1
                failure = ValueError("projection failed " + SECRET)
                if grouped_primary:
                    raise ExceptionGroup(
                        "projection " + SECRET,
                        [failure, OSError("source failed " + SECRET)],
                    )
                raise failure

            async def cleanup_dependency():
                entered.set()
                await asyncio.Event().wait()

            @asynccontextmanager
            async def cancelled_cleanup(context, **kwargs):
                nonlocal dependency
                async with acquire(context, **kwargs) as authority:
                    try:
                        yield authority
                    finally:
                        if "export" in kwargs["actions"]:
                            dependency = asyncio.create_task(cleanup_dependency())
                            await dependency

            projector.project = fail_projection
            policy.acquire = cancelled_cleanup
            caller = asyncio.create_task(app.export_session(request, context=CONTEXT))
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    await asyncio.wait_for(entered.wait(), 5)
                    assert dependency is not None
                    assert not caller.done() and caller.cancelling() == 0
                    assert dependency.cancel(SECRET)
                    with pytest.raises(SessionExportUnavailable) as caught:
                        await asyncio.wait_for(caller, 5)
                assert dependency.cancelled() and dependency.cancelling() == 1
                assert not caller.cancelled() and caller.cancelling() == 0
                assert not app._session_export_coordinator.owners.pending
                cancellation = caught.value.__cause__
                assert type(cancellation) is asyncio.CancelledError
                primary = cancellation.__context__
                if grouped_primary:
                    assert isinstance(primary, ExceptionGroup)
                    assert [type(child) for child in primary.exceptions] == [ValueError, OSError]
                    assert "projection failed" in str(primary.exceptions[0])
                    assert "source failed" in str(primary.exceptions[1])
                else:
                    assert type(primary) is ValueError
                    assert "projection failed" in str(primary)
                seen = set()
                rendered = []

                def inspect(error):
                    if error is None:
                        return
                    assert id(error) not in seen, "Failure evidence must appear exactly once"
                    seen.add(id(error))
                    rendered.append(
                        str(error) + repr(error) + repr(getattr(error, "__notes__", []))
                    )
                    inspect(error.__cause__)
                    inspect(error.__context__)
                    if isinstance(error, BaseExceptionGroup):
                        for child in error.exceptions:
                            inspect(child)

                inspect(caught.value)
                assert SECRET not in "".join(rendered)
                assert SECRET not in "".join(traceback.format_exception(caught.value))
                assert SECRET not in str(captured_warnings)
                assert projector.calls == 1
                assert not await published(store, case.session_id)
                assert (
                    await app.lookup_session_export(request, context=CONTEXT)
                ).status == "not_found"
            finally:
                if dependency is not None and not dependency.done():
                    dependency.cancel()
                await asyncio.gather(caller, return_exceptions=True)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert SECRET not in caplog.text + captured.out + captured.err


@pytest.mark.parametrize("bound", ["count", "pending", "bytes"])
def test_delayed_duplicate_replays_at_capacity(backend, bound):
    async def run():
        async with harness(backend) as case:
            limits = ExportLimits(
                max_exports=1 if bound == "count" else 4,
                max_pending=1 if bound == "pending" else 4,
                max_retained_bytes=65536 if bound == "bytes" else 4 * 65536,
            )
            winner, store, _, _ = case.app(limits=limits)
            delayed, _, _, projector = case.app(limits=limits)
            await case.create(store)
            request = await case.request(winner)
            projector.release = threading.Event()
            task = asyncio.create_task(delayed.export_session(request, context=CONTEXT))
            try:
                assert await asyncio.to_thread(projector.entered.wait, 5)
                receipt = await winner.export_session(request, context=CONTEXT)
                projector.release.set()
                assert await asyncio.wait_for(task, 10) == receipt
                assert len(await published(store, case.session_id)) == 1
            finally:
                projector.release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["release", "retire"])
@pytest.mark.parametrize("boundary", ["before_record", "after_record"])
def test_concurrent_exact_settlement_replays_terminal_winner(backend, mode, boundary):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class DelayedStore(base):
        session_export_version = 1
        blocked_key = None

        async def load_session_operation(self, session_id, key, **kwargs):
            selected = key == self.blocked_key
            if selected:
                self.blocked_key = None
                if boundary == "before_record":
                    entered.set()
                    await resume.wait()
            result = await super().load_session_operation(session_id, key, **kwargs)
            if selected and boundary == "after_record":
                entered.set()
                await resume.wait()
            return result

    async def run():
        nonlocal entered, resume
        entered, resume = asyncio.Event(), asyncio.Event()
        store = backend[1](DelayedStore)
        case = Harness(lambda store_type=None: store)
        task = None
        try:
            winner, _, _, _ = case.app(readers=(AcceptanceReader(),))
            delayed, _, _, _ = case.app(readers=(AcceptanceReader(),))
            await case.create(store)
            request = await case.request(winner)
            await winner.export_session(request, context=CONTEXT)
            command = SessionExportSettlementRequest(
                request=request,
                operation=request.ref.operation.model_copy(update={"caller_key": "settle"}),
                mode=mode,
            )
            store.blocked_key = operation_key(request.ref.operation)
            task = asyncio.create_task(delayed.settle_session_export(command, context=CONTEXT))
            await asyncio.wait_for(entered.wait(), 5)
            receipt = await winner.settle_session_export(command, context=CONTEXT)
            resume.set()
            assert await asyncio.wait_for(task, 10) == receipt
            events = await store.load_events(case.session_id)
            assert (
                sum(
                    event.type
                    in {EventType.SESSION_EXPORT_RELEASED, EventType.SESSION_EXPORT_RETIRED}
                    for event in events
                )
                == 1
            )
        finally:
            resume.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await case.close()

    entered = resume = None
    asyncio.run(run())


def test_cross_kind_key_reuse_is_conflict_not_unavailable(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            original = await case.request(app)
            other = await case.request(app, "other-export")
            await app.export_session(original, context=CONTEXT)
            await app.export_session(other, context=CONTEXT)
            command = SessionExportSettlementRequest(
                request=original, operation=other.ref.operation, mode="retire"
            )
            with pytest.raises(SessionExportConflict):
                await app.settle_session_export(command, context=CONTEXT)
            command = command.model_copy(
                update={
                    "operation": original.ref.operation.model_copy(update={"caller_key": "retire"})
                }
            )
            await app.settle_session_export(command, context=CONTEXT)
            reused = original.model_copy(
                update={"ref": original.ref.model_copy(update={"operation": command.operation})}
            )
            with pytest.raises(SessionExportConflict):
                await app.export_session(reused, context=CONTEXT)
            assert (await app.lookup_session_export(reused, context=CONTEXT)).status == "conflict"
            assert projector.calls == 2
            assert len(await published(store, case.session_id)) == 2

    asyncio.run(run())


@pytest.mark.parametrize("entrance", ["lookup", "settlement"])
def test_malformed_sibling_evidence_remains_unavailable(backend, entrance):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class MalformedStore(base):
        session_export_version = 1
        malformed_key = None

        async def load_session_operation(self, session_id, key, **kwargs):
            result = await super().load_session_operation(session_id, key, **kwargs)
            if key == self.malformed_key:
                field = "settlement" if entrance == "lookup" else "receipt"
                return {field: {"malformed": True}}
            return result

    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app(store_type=MalformedStore)
            await case.create(store)
            request = await case.request(app)
            await app.export_session(request, context=CONTEXT)
            if entrance == "lookup":
                store.malformed_key = operation_key(request.ref.operation)
                operation = app.lookup_session_export(request, context=CONTEXT)
            else:
                command = SessionExportSettlementRequest(
                    request=request,
                    operation=request.ref.operation.model_copy(update={"caller_key": "retire"}),
                    mode="retire",
                )
                store.malformed_key = operation_key(command.operation)
                operation = app.settle_session_export(command, context=CONTEXT)
            if entrance == "lookup":
                assert (await operation).status == "unavailable"
            else:
                with pytest.raises(SessionExportUnavailable):
                    await operation
            assert projector.calls == 1
            assert len(await published(store, case.session_id)) == 1

    asyncio.run(run())


def test_guard_wait_replays_winner_before_source_or_projector_access(backend):
    async def run():
        async with harness(backend) as case:
            winner, store, _, _ = case.app()
            delayed, _, policy, projector = case.app()
            await case.create(store)
            request = await case.request(winner)
            entered, resume = asyncio.Event(), asyncio.Event()
            acquire = policy.acquire

            @asynccontextmanager
            async def gated(context, **kwargs):
                if "export" in kwargs["actions"]:
                    entered.set()
                    await resume.wait()
                async with acquire(context, **kwargs) as authority:
                    yield authority

            def unavailable_projector(source):
                projector.calls += 1
                raise RuntimeError("Projection must not run for the committed receipt")

            policy.acquire = gated
            projector.project = unavailable_projector
            task = asyncio.create_task(delayed.export_session(request, context=CONTEXT))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                receipt = await winner.export_session(request, context=CONTEXT)
                if backend[0] == "sqlite":
                    # SQLite exposes public compaction: the retained export must
                    # remain replayable even though its original row is now gone.
                    assert await store.compact_transcript(case.session_id, keep_last=0) == 1
                    assert not (
                        await store.load_transcript_window(case.session_id, start_index=0, limit=1)
                    ).records
                before = await store.load_events(case.session_id)
                resume.set()
                assert await asyncio.wait_for(task, 10) == receipt
                assert projector.calls == 0
                assert await store.load_events(case.session_id) == before
                assert len(await published(store, case.session_id)) == 1
                assert await winner.read_session_export(request, context=CONTEXT) == {"count": 5}
            finally:
                resume.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["export", "release", "retire"])
@pytest.mark.parametrize("revoke", [False, True])
def test_concurrent_replay_holds_current_readback_permission(backend, mode, revoke):
    async def run():
        async with harness(backend) as case:
            winner, store, _, _ = case.app(readers=(AcceptanceReader(),))
            delayed, _, policy, projector = case.app(readers=(AcceptanceReader(),))
            await case.create(store)
            request = await case.request(winner)
            command = SessionExportSettlementRequest(
                request=request,
                operation=request.ref.operation.model_copy(update={"caller_key": "settle"}),
                mode="retire" if mode == "export" else mode,
            )
            if mode != "export":
                await winner.export_session(request, context=CONTEXT)
            entered, resume = asyncio.Event(), asyncio.Event()
            acquire = policy.acquire

            @asynccontextmanager
            async def gated(context, **kwargs):
                if mode in kwargs["actions"]:
                    entered.set()
                    await resume.wait()
                async with acquire(context, **kwargs) as authority:
                    yield authority

            policy.acquire = gated

            def invoke(app):
                return (
                    app.export_session(request, context=CONTEXT)
                    if mode == "export"
                    else app.settle_session_export(command, context=CONTEXT)
                )

            task = asyncio.create_task(invoke(delayed))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                receipt = await invoke(winner)
                before = await store.load_events(case.session_id)
                if revoke:
                    async with policy.lock:
                        policy.denied.add("readback")
                resume.set()
                if revoke:
                    with pytest.raises(SessionExportDenied):
                        await asyncio.wait_for(task, 10)
                    assert projector.calls == 0
                else:
                    assert await asyncio.wait_for(task, 10) == receipt
                assert await store.load_events(case.session_id) == before
                expected = (
                    ("readback", "source", "export") if mode == "export" else ("readback", mode)
                )
                assert any(call[2] == expected for call in policy.calls)
            finally:
                resume.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["export", "release", "retire"])
@pytest.mark.parametrize("phase", ["initial", "reconciliation"])
def test_delayed_receipt_read_cannot_outlive_authorization(backend, mode, phase):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class DelayedReadStore(base):
        session_export_version = 1
        delayed_key = None

        async def load_session_operation(self, session_id, key, **kwargs):
            result = await super().load_session_operation(session_id, key, **kwargs)
            if key == self.delayed_key:
                self.delayed_key = None
                entered.set()
                await resume.wait()
            return result

    async def run():
        nonlocal entered, resume
        entered, resume = asyncio.Event(), asyncio.Event()
        async with harness(backend) as case:
            app, store, policy, _ = case.app(
                readers=(AcceptanceReader(),), store_type=DelayedReadStore
            )
            # Share the native store, but retain an independent app/policy owner.
            original_factory = case.factory
            case.factory = lambda store_type=None: store
            winner, _, _, _ = case.app(readers=(AcceptanceReader(),))
            case.factory = original_factory
            await case.create(store)
            request = await case.request(app)
            command = SessionExportSettlementRequest(
                request=request,
                operation=request.ref.operation.model_copy(update={"caller_key": "settle"}),
                mode="retire" if mode == "export" else mode,
            )
            if mode != "export":
                await winner.export_session(request, context=CONTEXT)

            def invoke(client):
                return (
                    client.export_session(request, context=CONTEXT)
                    if mode == "export"
                    else client.settle_session_export(command, context=CONTEXT)
                )

            key = operation_key(request.ref.operation if mode == "export" else command.operation)
            if phase == "initial":
                await invoke(winner)
            acquire = policy.acquire
            expiry_ms = 0

            @asynccontextmanager
            async def expiring(context, **kwargs):
                nonlocal expiry_ms
                target = (
                    kwargs["actions"] == ("readback",)
                    if phase == "initial"
                    else mode in kwargs["actions"]
                )
                if target and phase == "reconciliation":
                    # Commit a competitor only after this caller's initial
                    # readback found no receipt, before its new-effect guard.
                    await invoke(winner)
                async with acquire(context, **kwargs) as authority:
                    if target:

                        def sample(_session, _checkpoint, now):
                            nonlocal expiry_ms
                            expiry_ms = int(now.timestamp() * 1000) + 1000
                            return None

                        await store.transform_checkpoint_with_store_time(case.session_id, sample)
                        authority = authority.model_copy(update={"expires_at_ms": expiry_ms})
                        store.delayed_key = key
                    yield authority

            policy.acquire = expiring
            task = asyncio.create_task(invoke(app))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                # Keep the read pending until the native store's own clock
                # proves expiry; no caller-clock substitution or fake timeout.
                expired = False

                async def await_expiry():
                    nonlocal expired
                    while not expired:

                        def sample(_session, _checkpoint, now):
                            nonlocal expired
                            expired = int(now.timestamp() * 1000) >= expiry_ms
                            return None

                        await store.transform_checkpoint_with_store_time(case.session_id, sample)
                        if not expired:
                            await asyncio.sleep(0.02)

                await asyncio.wait_for(await_expiry(), 5)
                before = await store.load_events(case.session_id)
                resume.set()
                with pytest.raises(SessionExportDenied):
                    await asyncio.wait_for(task, 5)
                assert await store.load_events(case.session_id) == before
                assert await invoke(winner) is not None
            finally:
                resume.set()
                await asyncio.gather(task, return_exceptions=True)

    entered = resume = None
    asyncio.run(run())
