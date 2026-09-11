from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import threading
import traceback
import warnings
import weakref
from pathlib import Path

import pytest
from tests.core.knowledge_publication_conformance import assert_concurrent_publication_conformance
from tests.sqlite_resources import SQLiteResourceLeak, SQLiteResourceScope, _failure_evidence

from cayu import InMemoryKnowledgeStore, KnowledgeAccessScope, SQLiteSessionStore

_STORAGE_CASES = (
    "tests/core/test_session_operation_fault_harness.py::"
    "test_session_operation_fault_harness_store_conformance[sqlite]",
    "tests/core/test_sqlite_knowledge_store.py::"
    "test_sqlite_knowledge_store_owned_publication_conformance",
    "tests/core/test_task_store.py::test_task_stores_filter_contract_queues[SQLiteTaskStore]",
    "tests/core/test_task_store.py::"
    "test_task_store_retry_conformance_for_acknowledgement_failures[SQLiteTaskStore]",
)


def test_scope_closes_real_resources_and_removes_its_root(tmp_path, request):
    async def scenario():
        async with SQLiteResourceScope(tmp_path, request.node.nodeid) as scope:
            root = scope.root
            scope.own(SQLiteSessionStore(scope.path("sessions.sqlite")))
            connection = scope.own(sqlite3.connect(scope.path("raw.sqlite")), kind="connection")
            cursor = scope.own(connection.cursor(), kind="cursor")
            cursor.execute("CREATE TABLE example (value INTEGER)")
            executor = scope.executor()
            assert await asyncio.wrap_future(executor.submit(lambda: 42)) == 42
        assert not root.exists()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    asyncio.run(scenario())


def test_scope_retains_database_until_dispatched_thread_settles(tmp_path, request):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid, timeout=0.02)
        await scope.__aenter__()
        release = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        path = scope.path()

        def write():
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE example (value INTEGER)")
                loop.call_soon_threadsafe(started.set)
                release.wait()
                connection.execute("INSERT INTO example VALUES (1)")
            connection.close()

        thread = scope.thread(write)
        thread.start()
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(SQLiteResourceLeak, match="kind=thread.*still-running"):
                await scope.aclose()
            assert path.exists()
        finally:
            release.set()
            scope.timeout = 2
            await scope.aclose()
        assert not thread.is_alive()
        assert not scope.root.exists()

    asyncio.run(scenario())


def test_scope_timeout_does_not_cancel_owned_work(tmp_path, request):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid, timeout=0.02)
        await scope.__aenter__()
        release = asyncio.Event()
        task = scope.task(release.wait())
        try:
            with pytest.raises(SQLiteResourceLeak, match="kind=task.*still-running"):
                await scope.aclose()
            assert not task.done()
            assert task.cancelling() == 0
        finally:
            release.set()
            scope.timeout = 2
            await scope.aclose()
        assert task.done()
        assert not task.cancelled()

    asyncio.run(scenario())


def test_scope_cleanup_failure_is_bounded_and_payload_free(tmp_path, capsys, caplog):
    canary = "PRIVATE-database-payload"

    class Registry:
        failed = False

        def close(self):
            if not self.failed:
                self.failed = True
                raise RuntimeError(canary)

    async def scenario():
        scope = SQLiteResourceScope(tmp_path, f"tests/test_example.py::test_registry[{canary}]")
        await scope.__aenter__()
        scope.own(Registry(), kind="registry")
        with pytest.raises(SQLiteResourceLeak) as caught:
            await scope.aclose()
        assert "kind=registry" in str(caught.value)
        assert "test_registry" in str(caught.value)
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)
        await scope.aclose()

    asyncio.run(scenario())
    assert canary not in caplog.text
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err


def test_scope_same_parent_has_distinct_roots(tmp_path, request):
    async def scenario():
        async with (
            SQLiteResourceScope(tmp_path, request.node.nodeid) as first,
            SQLiteResourceScope(tmp_path, request.node.nodeid) as second,
        ):
            assert first.path() != second.path()

    asyncio.run(scenario())


def test_cancellation_waits_for_closer_then_propagates_once(tmp_path, request):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class Store:
            async def close(self):
                started.set()
                await release.wait()

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)

        async def owner():
            async with scope:
                scope.own(Store())

        task = asyncio.create_task(owner())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        assert task.cancelling() == 1
        assert scope.root.exists()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert not scope.root.exists()

    asyncio.run(scenario())


def test_completed_failed_task_is_reported_without_unretrieved_exception(tmp_path, request):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()

        async def failed():
            raise RuntimeError("PRIVATE-task-payload")

        task = scope.task(failed())
        await asyncio.wait({task})
        with pytest.raises(ExceptionGroup) as caught:
            await scope.aclose()
        assert len(caught.value.exceptions) == 1
        assert "kind=task" in str(caught.value.exceptions[0])
        assert "PRIVATE" not in str(caught.value.exceptions[0])
        assert not scope.root.exists()

    asyncio.run(scenario())


def test_body_and_cleanup_failures_preserve_order(tmp_path, request):
    class Registry:
        fail = True

        def close(self):
            if self.fail:
                raise RuntimeError("PRIVATE-cleanup")

    async def scenario():
        registry = Registry()
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        primary = AssertionError("test assertion")
        try:
            with pytest.raises(ExceptionGroup) as caught:
                async with scope:
                    scope.own(registry, kind="registry")
                    raise primary
            assert caught.value.exceptions[0] is primary
            assert isinstance(caught.value.exceptions[1], SQLiteResourceLeak)
        finally:
            registry.fail = False
            await scope.aclose()

    asyncio.run(scenario())


@pytest.mark.process
@pytest.mark.parametrize("reverse", [False, True])
def test_storage_groups_repeat_in_one_process_without_scope_leftovers(tmp_path, reverse):
    cases = list(reversed(_STORAGE_CASES)) if reverse else list(_STORAGE_CASES)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pytest\n"
            "for _ in range(3):\n"
            "    result = pytest.main(sys.argv[1:])\n"
            "    if result: raise SystemExit(result)\n",
            "-q",
            "--basetemp",
            str(tmp_path / "pytest-roots"),
            *cases,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, "Repeated storage groups failed; run the documented sequence."
    assert result.stdout.count(f"{len(cases)} passed") == 3
    assert not list((tmp_path / "pytest-roots").rglob("sqlite-scope-*"))


def test_thread_failure_does_not_print_its_payload(tmp_path, request, capsys, caplog):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()

        def fail():
            raise RuntimeError("PRIVATE-thread-payload")

        thread = scope.thread(fail)
        thread.start()
        with pytest.raises(ExceptionGroup) as caught:
            await scope.aclose()
        assert "kind=thread" in str(caught.value.exceptions[0])
        assert not thread.is_alive()
        assert not scope.root.exists()

    asyncio.run(scenario())
    output = capsys.readouterr()
    assert "PRIVATE-thread-payload" not in output.out + output.err + caplog.text


def test_child_close_cancellation_is_not_owner_cancellation(tmp_path, request):
    class Store:
        failed = False

        async def close(self):
            if not self.failed:
                self.failed = True
                raise asyncio.CancelledError("PRIVATE-close")

    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()
        scope.own(Store())
        with pytest.raises(SQLiteResourceLeak, match="close-cancelled"):
            await scope.aclose()
        assert asyncio.current_task().cancelling() == 0
        await scope.aclose()

    asyncio.run(scenario())


def test_fixture_finalizer_detects_missing_context(tmp_path):
    scope = SQLiteResourceScope(tmp_path, "tests/test_example.py::test_missing_context")
    with pytest.raises(SQLiteResourceLeak, match="not-entered"):
        scope.assert_finished()
    assert not scope.root.exists()


def test_task_timeout_error_is_failure_not_unsettled_work(tmp_path, request):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()

        async def fail():
            raise TimeoutError("PRIVATE-provider-timeout")

        scope.task(fail())
        with pytest.raises(ExceptionGroup) as caught:
            await scope.aclose()
        assert "state=failed" in str(caught.value.exceptions[0])
        assert not scope.root.exists()

    asyncio.run(scenario())


def test_duplicate_registration_and_concurrent_close_have_one_owner(tmp_path, request):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        class Store:
            async def close(self):
                nonlocal calls
                calls += 1
                started.set()
                await release.wait()

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()
        store = scope.own(Store())
        scope.own(store)
        first = asyncio.create_task(scope.aclose())
        await asyncio.wait_for(started.wait(), 2)
        second = asyncio.create_task(scope.aclose())
        release.set()
        await asyncio.gather(first, second)
        assert calls == 1
        assert not scope.root.exists()

    asyncio.run(scenario())


def test_assertion_failure_closes_real_database_before_next_scope(tmp_path, request):
    async def scenario():
        first = SQLiteResourceScope(tmp_path, request.node.nodeid)
        with pytest.raises(AssertionError, match="test assertion"):
            async with first:
                connection = first.own(sqlite3.connect(first.path()), kind="connection")
                connection.execute("CREATE TABLE example (value INTEGER)")
                raise AssertionError("test assertion")
        assert not first.root.exists()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        async with SQLiteResourceScope(tmp_path, request.node.nodeid) as second:
            store = second.own(SQLiteSessionStore(second.path()))
            assert (await store.list_sessions()).sessions == []

    asyncio.run(scenario())


@pytest.mark.process
def test_real_fixture_reports_failure_and_isolates_the_following_test(tmp_path, monkeypatch):
    monkeypatch.setenv("CAYU_REQUIRE_CURRENT_TEST_DURATIONS", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--basetemp",
            str(tmp_path / "fixture-roots"),
            "tests/fixtures/sqlite_resource_cases.py",
        ],
        cwd=Path(__file__).resolve().parents[2],
        # These intentionally failing fixtures are not normal suite collection
        # entries and therefore do not participate in CI's duration census.
        env={**os.environ, "CAYU_REQUIRE_CURRENT_TEST_DURATIONS": "0"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "1 failed, 2 passed, 1 error" in result.stdout
    assert "test_missing_context" in result.stdout
    assert "state=not-entered" in result.stdout
    assert os.environ["CAYU_REQUIRE_CURRENT_TEST_DURATIONS"] == "1"
    assert not list((tmp_path / "fixture-roots").rglob("sqlite-scope-*"))


def test_cancelled_cleanup_failure_preserves_cancelled_task_state(tmp_path, request):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        class Store:
            fail = True

            async def close(self):
                started.set()
                await release.wait()
                if self.fail:
                    raise RuntimeError("PRIVATE-cleanup")

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()
        store = scope.own(Store())
        task = asyncio.create_task(scope.aclose())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        release.set()
        try:
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 1
            assert isinstance(caught.value.__cause__, SQLiteResourceLeak)
            assert "close-failed" in str(caught.value.__cause__)
        finally:
            store.fail = False
            await scope.aclose()

    asyncio.run(scenario())


def test_close_timeout_reuses_the_inflight_close(tmp_path, request):
    async def scenario():
        release = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        class Store:
            async def close(self):
                nonlocal calls
                calls += 1
                started.set()
                await release.wait()

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid, timeout=0.02)
        await scope.__aenter__()
        scope.own(Store())
        try:
            with pytest.raises(SQLiteResourceLeak, match="close-pending"):
                await scope.aclose()
            assert started.is_set() and scope.root.exists()
        finally:
            release.set()
            scope.timeout = 2
            await scope.aclose()
        assert calls == 1

    asyncio.run(scenario())


def test_executor_timeout_preserves_work_until_physical_completion(tmp_path, request):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid, timeout=0.02)
        await scope.__aenter__()
        executor = scope.executor()
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def work():
            loop.call_soon_threadsafe(started.set)
            release.wait()
            return 42

        future = executor.submit(work)
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(SQLiteResourceLeak, match="executor-work.*still-running"):
                await scope.aclose()
            assert not future.done() and scope.root.exists()
        finally:
            release.set()
            scope.timeout = 2
            await scope.aclose()
        assert future.result() == 42
        assert not scope.root.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["concurrent", "concurrent-chunk"])
def test_knowledge_conformance_settles_sibling_before_returning_failure(phase):
    async def scenario():
        started = asyncio.Event()
        stopped = asyncio.Event()

        class Store(InMemoryKnowledgeStore):
            async def publish_entry_revision(self, entry, chunks, *, operation_id):
                if operation_id not in {f"{phase}-a", f"{phase}-b"}:
                    return await super().publish_entry_revision(
                        entry, chunks, operation_id=operation_id
                    )
                if operation_id == f"{phase}-a":
                    await started.wait()
                    raise RuntimeError("controlled publication failure")
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        with pytest.raises(ExceptionGroup):
            await assert_concurrent_publication_conformance(
                Store(access_scope=KnowledgeAccessScope.privileged())
            )
        assert stopped.is_set()

    asyncio.run(scenario())


def test_work_and_cleanup_failures_keep_ordered_safe_diagnostics(tmp_path, request, caplog, capsys):
    async def scenario():
        class Registry:
            fail = True

            def close(self):
                if self.fail:
                    raise RuntimeError("PRIVATE-registry")

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()
        registry = scope.own(Registry(), kind="registry")

        async def fail():
            raise RuntimeError("PRIVATE-work")

        scope.task(fail())
        try:
            with pytest.raises(ExceptionGroup) as caught:
                await scope.aclose()
            assert len(caught.value.exceptions) == 2
            assert "kind=task" in str(caught.value.exceptions[0])
            assert "kind=registry" in str(caught.value.exceptions[1])
        finally:
            registry.fail = False
            with pytest.raises(ExceptionGroup):
                await scope.aclose()
        assert not scope.root.exists()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert "PRIVATE" not in output.out + output.err + caplog.text
    assert not recorded


def test_successful_teardown_releases_registered_object_references(tmp_path, request):
    class Store:
        def close(self):
            pass

    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        async with scope:
            store = scope.own(Store())
            reference = weakref.ref(store)
            del store
            assert reference() is not None
        assert reference() is None
        scope.assert_finished()

    asyncio.run(scenario())


@pytest.mark.parametrize("completed_first", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("startup_failure", [False, True])
def test_executor_worker_failure_cannot_cancel_teardown(
    tmp_path, request, completed_first, grouped, startup_failure
):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        started = asyncio.Event()
        waiting = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()

        def work():
            loop.call_soon_threadsafe(started.set)
            release.wait()
            signal = asyncio.CancelledError("PRIVATE-worker")
            if grouped:
                raise BaseExceptionGroup("PRIVATE-group", [signal, RuntimeError("PRIVATE-error")])
            raise signal

        original_wait = scope._wait_settled

        async def observed_wait(awaitable, deadline):
            waiting.set()
            return await original_wait(awaitable, deadline)

        scope._wait_settled = observed_wait
        startup_error = TimeoutError("controlled startup failure")
        with pytest.raises(ExceptionGroup) as caught:
            async with scope:
                connection = scope.own(sqlite3.connect(scope.path()), kind="connection")
                executor = scope.executor()
                closing = None
                try:
                    future = executor.submit(work)
                    await asyncio.wait_for(started.wait(), 2)
                    if startup_failure:
                        # Fail before aclose starts while the real worker is blocked.
                        raise startup_error
                    if completed_first:
                        release.set()
                        wrapped = asyncio.wrap_future(future)
                        await asyncio.wait({wrapped})
                        wrapped.exception()
                    closing = asyncio.create_task(scope.aclose())
                    if not completed_first:
                        await asyncio.wait_for(waiting.wait(), 2)
                        release.set()
                    await closing
                finally:
                    release.set()
                    if closing is not None:
                        await asyncio.gather(closing, return_exceptions=True)
        if startup_failure:
            assert caught.value.exceptions[0] is startup_error
            work_failure = caught.value.exceptions[1]
        else:
            assert closing is not None
            assert not closing.cancelled() and closing.cancelling() == 0
            work_failure = caught.value
        assert isinstance(work_failure, ExceptionGroup)
        assert len(work_failure.exceptions) == 1
        assert "executor-work" in str(work_failure.exceptions[0])
        assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
        assert not scope.root.exists()
        scope.assert_finished()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        assert future.done() and not future.cancelled()

    asyncio.run(scenario())


def test_context_exit_preserves_body_cleanup_and_cancellation(tmp_path, request):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        primary = AssertionError("body assertion")

        class Store:
            fail = True

            async def close(self):
                started.set()
                await release.wait()
                if self.fail:
                    raise RuntimeError("PRIVATE-closer")

        store = Store()
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)

        async def owner():
            async with scope:
                scope.own(store)
                raise primary

        task = asyncio.create_task(owner())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        release.set()
        try:
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 1
            evidence = caught.value.__cause__
            assert isinstance(evidence, BaseExceptionGroup)
            assert evidence.exceptions[0] is primary
            assert isinstance(evidence.exceptions[1], SQLiteResourceLeak)
            diagnostic = "".join(traceback.format_exception(caught.value))
            assert "body assertion" in diagnostic and "close-failed" in diagnostic
            assert "PRIVATE" not in diagnostic
        finally:
            store.fail = False
            await scope.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("completed_first", [False, True])
def test_grouped_task_failure_still_finishes_teardown(tmp_path, request, completed_first):
    async def scenario():
        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        await scope.__aenter__()
        release = asyncio.Event()
        waiting = asyncio.Event()

        async def fail():
            await release.wait()
            raise BaseExceptionGroup(
                "PRIVATE-group", [asyncio.CancelledError("PRIVATE-cancel"), ValueError("PRIVATE")]
            )

        task = scope.task(fail())
        original_wait = scope._wait_settled

        async def observed_wait(awaitable, deadline):
            waiting.set()
            return await original_wait(awaitable, deadline)

        scope._wait_settled = observed_wait
        if completed_first:
            release.set()
            await asyncio.wait({task})
        closing = asyncio.create_task(scope.aclose())
        try:
            if not completed_first:
                await asyncio.wait_for(waiting.wait(), 2)
                release.set()
            with pytest.raises(ExceptionGroup) as caught:
                await closing
            assert not closing.cancelled() and closing.cancelling() == 0
            assert "kind=task" in str(caught.value.exceptions[0])
            assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
            assert not scope.root.exists()
        finally:
            release.set()
            await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.process
def test_early_timeout_harvests_failed_sibling_at_process_finalization(tmp_path):
    script = """
import asyncio
import gc
import sys
from pathlib import Path
from tests.sqlite_resources import SQLiteResourceScope, SQLiteResourceLeak

async def main():
    scope = SQLiteResourceScope(Path(sys.argv[1]), 'safe-node', timeout=0.02)
    await scope.__aenter__()
    release = asyncio.Event()
    scope.task(release.wait())
    async def failed():
        raise RuntimeError('PRIVATE-sibling-payload')
    sibling = scope.task(failed())
    await asyncio.wait({sibling})
    try:
        await scope.aclose()
    except SQLiteResourceLeak as failure:
        assert 'still-running' in str(failure)
    else:
        raise AssertionError('expected timeout')
    # Settle the pending work without a second scope drain: reproduce abandonment
    # after the failed fixture, then force diagnostic emission during finalization.
    release.set()
    await asyncio.wait(set(scope._tasks))

asyncio.run(main())
gc.collect()
"""
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(["src", ".", os.environ.get("PYTHONPATH", "")]),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PRIVATE" not in result.stdout + result.stderr
    assert "never retrieved" not in result.stderr


def test_grouped_closer_failure_is_safe_and_retryable(tmp_path, request, capsys, caplog):
    async def scenario():
        class Store:
            calls = 0

            async def close(self):
                self.calls += 1
                if self.calls == 1:
                    raise BaseExceptionGroup(
                        "PRIVATE-close-group",
                        [asyncio.CancelledError("PRIVATE-cancel"), RuntimeError("PRIVATE-payload")],
                    )

        scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
        store = Store()
        try:
            with pytest.raises(SQLiteResourceLeak, match="close-failed") as caught:
                async with scope:
                    connection = scope.own(sqlite3.connect(scope.path()), kind="connection")
                    scope.own(store)
            assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
            assert scope.root.exists()
            # Reverse-order ownership keeps dependencies open until retry succeeds.
            connection.execute("SELECT 1")
        finally:
            await scope.aclose()
        assert store.calls == 2
        scope.assert_finished()
        assert not scope.root.exists()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert "PRIVATE" not in output.out + output.err + caplog.text
    assert not recorded


def test_nested_cancelled_scopes_preserve_every_cleanup_failure(tmp_path, request):
    async def scenario():
        scopes = [SQLiteResourceScope(tmp_path, request.node.nodeid) for _ in range(3)]
        entered = asyncio.Event()
        release = asyncio.Event()

        class Store:
            fail = True

            async def close(self):
                if self.fail:
                    raise RuntimeError("PRIVATE-close")

        stores = [Store() for _ in scopes]

        async def owner():
            async with scopes[0]:
                scopes[0].own(stores[0])
                async with scopes[1]:
                    scopes[1].own(stores[1])
                    async with scopes[2]:
                        scopes[2].own(stores[2])
                        entered.set()
                        await release.wait()

        task = asyncio.create_task(owner())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 1
            diagnostic = "".join(traceback.format_exception(caught.value))
            assert "PRIVATE" not in diagnostic
            paths = [scope.root.name for scope in reversed(scopes)]
            assert all(diagnostic.count(path) == 1 for path in paths)
            assert [diagnostic.index(path) for path in paths] == sorted(
                diagnostic.index(path) for path in paths
            )
            assert all(scope.root.exists() for scope in scopes)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            for scope, store in reversed(list(zip(scopes, stores, strict=True))):
                store.fail = False
                await scope.aclose()
        assert all(not scope.root.exists() for scope in scopes)

    asyncio.run(scenario())


def test_cleanup_evidence_preserves_groups_and_deduplicates_shared_failures():
    first = SQLiteResourceLeak("first")
    second = SQLiteResourceLeak("second")
    third = SQLiteResourceLeak("third")
    original = ExceptionGroup("inner", [first, second])
    overlapping = ExceptionGroup("outer", [second, third])
    combined = _failure_evidence(original, overlapping, first)
    assert isinstance(combined, ExceptionGroup)
    assert combined.exceptions[0] is original
    assert isinstance(combined.exceptions[1], ExceptionGroup)
    assert combined.exceptions[1].exceptions == (third,)
    assert _failure_evidence(original, original) is original
