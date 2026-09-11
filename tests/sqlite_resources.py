"""Explicit, same-loop ownership for SQLite test resources (never runtime API)."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
import shutil
import tempfile
import threading
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class _Closable(Protocol):
    def close(self) -> object: ...


C = TypeVar("C", bound=_Closable)


class SQLiteResourceLeak(AssertionError):
    """Content-free teardown failure, attributed to the pytest scope owner."""


class _CleanupDeadline(TimeoutError):
    pass


def _failure_evidence(*failures: BaseException) -> BaseException:
    """Keep ordered original evidence, without repeating shared exception objects."""
    seen: set[int] = set()

    def unique(failure: BaseException) -> BaseException | None:
        if id(failure) in seen:
            return None
        seen.add(id(failure))
        if isinstance(failure, BaseExceptionGroup):
            children = [child for item in failure.exceptions if (child := unique(item)) is not None]
            if not children:
                return None
            if len(children) != len(failure.exceptions) or any(
                left is not right for left, right in zip(children, failure.exceptions, strict=True)
            ):
                return failure.derive(children)
        return failure

    evidence = [item for failure in failures if (item := unique(failure)) is not None]
    return (
        evidence[0]
        if len(evidence) == 1
        else BaseExceptionGroup("Test body and SQLite teardown failed", evidence)
    )


@dataclass
class _Resource:
    kind: str
    close: Callable[[], object]
    identity: object
    pending: asyncio.Task[object] | None = None
    closed: bool = False


class SQLiteTestExecutor:
    """Only expose submission through the owning scope's work registry."""

    def __init__(self, scope: SQLiteResourceScope, max_workers: int) -> None:
        self._scope = scope
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        self._scope._check_open()
        future = self._executor.submit(fn)
        self._scope._futures.append(future)
        return future

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        failed = False

        def shutdown() -> None:
            nonlocal failed
            try:
                self._executor.shutdown(wait=True)
            except BaseException:
                failed = True
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(target=shutdown)
        thread.start()
        await done.wait()
        # The event is queued immediately before the thread exits.
        while thread.is_alive():
            await asyncio.sleep(0.001)
        if failed:
            raise RuntimeError("Test executor shutdown failed.")


class SQLiteResourceScope:
    """Drain owned work before reverse-order close and temporary-root removal.

    Explicit registration is intentional: no monkeypatching of sqlite, asyncio,
    global executors, or store classes. Enter and exit inside the test's loop.
    An over-budget operation remains owned; callers must establish quiescence
    and retry ``aclose`` before abandoning the loop or removing its files.
    """

    def __init__(self, parent: Path, nodeid: str, *, timeout: float = 5.0) -> None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Resource cleanup timeout must be finite and positive.")
        self.root = Path(tempfile.mkdtemp(prefix="sqlite-scope-", dir=parent))
        # Parameter representations are not needed to identify the owning test.
        self.nodeid = re.sub(r"[^a-zA-Z0-9_./:-]", "?", nodeid.split("[", 1)[0])[:240]
        if "[" in nodeid:
            self.nodeid += "[case-" + hashlib.sha256(nodeid.encode()).hexdigest()[:12] + "]"
        self.timeout = timeout
        self._resources: list[_Resource] = []
        self._tasks: list[asyncio.Task[Any]] = []
        self._futures: list[Future[Any]] = []
        self._threads: list[threading.Thread] = []
        self._thread_failures: list[bool] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._closing = False
        self._cleanup: asyncio.Task[None] | None = None

    async def __aenter__(self) -> SQLiteResourceScope:
        if self._loop is not None or self._closed:
            raise RuntimeError("Resource scope cannot be entered twice.")
        self._loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            await self.aclose()
        except BaseException as cleanup:
            if exc is not None and cleanup is not exc:
                if isinstance(exc, asyncio.CancelledError):
                    if exc.__cause__ is not None:
                        raise exc from _failure_evidence(exc.__cause__, cleanup)
                    raise exc from cleanup
                if isinstance(cleanup, asyncio.CancelledError):
                    prior = cleanup.__cause__
                    if prior is not None and prior is not exc:
                        raise cleanup from _failure_evidence(exc, prior)
                    raise cleanup from exc
                raise BaseExceptionGroup(
                    "Test body and SQLite teardown failed", [exc, cleanup]
                ) from None
            raise

    def _check_open(self) -> None:
        if self._closed or self._closing or self._loop is not asyncio.get_running_loop():
            raise RuntimeError("Resource registration requires its active owning loop.")

    def path(self, name: str = "database.sqlite") -> Path:
        self._check_open()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*", name):
            raise ValueError("Use a simple test-owned database filename.")
        return self.root / name

    def own(self, resource: C, *, kind: str = "store") -> C:
        """Register an object's close method; returns the unchanged real object."""
        self._check_open()
        if kind not in {"store", "connection", "cursor", "registry"}:
            raise ValueError("Unsupported resource kind.")
        if not any(item.identity is resource for item in self._resources):
            self._resources.append(_Resource(kind, resource.close, resource))
        return resource

    def task(self, coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        self._check_open()
        task = asyncio.create_task(coroutine)
        # Observe every outcome even if an earlier resource exceeds its deadline.
        # Retrieving the exception does not erase it from the task's result.
        task.add_done_callback(self._harvest)
        self._tasks.append(task)
        return task

    def thread(self, target: Callable[[], object]) -> threading.Thread:
        """Capture failures without invoking the payload-printing thread hook."""
        self._check_open()
        index = len(self._threads)
        self._thread_failures.append(False)

        def run() -> None:
            try:
                target()
            except BaseException:
                self._thread_failures[index] = True

        thread = threading.Thread(target=run)
        self._threads.append(thread)
        return thread

    def assert_finished(self) -> None:
        """Fixture finalizer: never silently abandon an unclosed scope."""
        if self._closed:
            return
        if self._loop is None:
            shutil.rmtree(self.root)
            self._closed = True
            raise self._failure("scope", 0, "not-entered")
        raise self._failure("scope", 0, "not-closed")

    def executor(self, *, max_workers: int = 1) -> SQLiteTestExecutor:
        self._check_open()
        executor = SQLiteTestExecutor(self, max_workers)
        self._resources.append(_Resource("executor", executor.close, executor))
        return executor

    def _failure(self, kind: str, index: int, state: str) -> SQLiteResourceLeak:
        return SQLiteResourceLeak(
            f"SQLite resources: node={self.nodeid} kind={kind} "
            f"resource={index} path=<test-root>/{self.root.name} state={state}"
        )

    @staticmethod
    def _harvest(done: asyncio.Future[Any]) -> None:
        if not done.cancelled():
            done.exception()

    async def _wait_settled(self, awaitable: Awaitable[T], deadline: float) -> asyncio.Future[T]:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        worker = asyncio.ensure_future(awaitable)
        # Harvest eventual failure even when the scope times out before readback.
        worker.add_done_callback(self._harvest)
        done, _ = await asyncio.wait({worker}, timeout=remaining)
        if not done:
            raise _CleanupDeadline
        return worker

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._loop is not asyncio.get_running_loop():
            raise RuntimeError("Resource cleanup requires its owning loop.")
        self._closing = True
        if self._cleanup is None or self._cleanup.done():
            self._cleanup = asyncio.create_task(self._drain_and_close())
        cleanup = self._cleanup
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as signal:
                if cleanup.done() and cleanup.cancelled():
                    break
                cancellation = cancellation or signal
            except Exception:
                break
        try:
            cleanup.result()
        except BaseException as failure:
            if cancellation is not None:
                raise cancellation from failure
            raise
        if cancellation is not None:
            raise cancellation

    async def _drain_and_close(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.timeout
        failures: list[Exception] = []

        def fail(kind: str, index: int, state: str) -> None:
            failure = self._failure(kind, index, state)
            if failures:
                raise ExceptionGroup(
                    "SQLite work and teardown failed", [*failures, failure]
                ) from None
            raise failure from None

        for index, task in enumerate(self._tasks):
            if not task.done():
                try:
                    await self._wait_settled(task, deadline)
                except _CleanupDeadline:
                    fail("task", index, "still-running")
            if task.done() and not task.cancelled() and task.exception() is not None:
                failures.append(self._failure("task", index, "failed"))
        for index, future in enumerate(self._futures):
            if not future.done():
                try:
                    await self._wait_settled(asyncio.wrap_future(future), deadline)
                except _CleanupDeadline:
                    fail("executor-work", index, "still-running")
            if future.done() and not future.cancelled() and future.exception() is not None:
                failures.append(self._failure("executor-work", index, "failed"))
        for index, thread in enumerate(self._threads):
            # Never use a blocking join on the owning event loop.
            while thread.is_alive():
                if asyncio.get_running_loop().time() >= deadline:
                    fail("thread", index, "still-running")
                await asyncio.sleep(0.001)
            if self._thread_failures[index]:
                failures.append(self._failure("thread", index, "failed"))
        for index, resource in reversed(list(enumerate(self._resources))):
            if resource.closed:
                continue
            if resource.pending is None:

                async def close_resource(item: _Resource = resource) -> object:
                    result = item.close()
                    return await result if inspect.isawaitable(result) else result

                resource.pending = asyncio.create_task(close_resource())
            try:
                settled = await self._wait_settled(resource.pending, deadline)
            except _CleanupDeadline:
                fail(resource.kind, index, "close-pending")
            if settled.cancelled():
                resource.pending = None
                fail(resource.kind, index, "close-cancelled")
            if settled.exception() is not None:
                resource.pending = None
                fail(resource.kind, index, "close-failed")
            resource.closed = True
        try:
            shutil.rmtree(self.root)
        except OSError:
            fail("root", 0, "remove-failed")
        self._resources.clear()
        self._tasks.clear()
        self._futures.clear()
        self._threads.clear()
        self._thread_failures.clear()
        self._closed = True
        if failures:
            raise ExceptionGroup("SQLite owned work failed", failures)
