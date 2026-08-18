"""Invocation-aware workspace and artifact capabilities handed to custom tools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn

from cayu._exception_groups import exception_tree_contains, rebuild_exception_group
from cayu._workspace_mutation import WorkspaceMutationSettlementError
from cayu.artifacts import (
    ArtifactReadResult,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.tools._operation_boundary import await_invocation_operation
from cayu.tools._redaction import (
    InvocationRedactorSnapshot,
    resolve_invocation_redactor_snapshot,
)
from cayu.workspaces import Workspace, WorkspaceReadResult


class InvocationResourceReadError(RuntimeError):
    """A custom-tool resource read could not be projected safely."""


def _bounded_workspace_mutation_failure(
    error: BaseException,
    *,
    operation: str,
) -> Exception:
    """Detach extension-owned control signals as ordinary operation failures."""

    message = f"{operation} failed with an unexpected control signal."
    if isinstance(error, Exception):
        return error
    if isinstance(error, BaseExceptionGroup):
        rebuilt = rebuild_exception_group(
            error,
            group_message=f"{operation} reported multiple failures.",
            leaf_mapper=lambda _leaf: RuntimeError(message),
            invalid_leaf_factory=lambda: RuntimeError(message),
        )
        if isinstance(rebuilt, Exception):
            return rebuilt
    return RuntimeError(message)


class InvocationWorkspaceMutationOwner:
    """Fence invocation-dispatched workspace mutations until proven settled."""

    def __init__(
        self,
        *,
        on_settlement_unproven: (Callable[[Callable[[], Awaitable[bool]]], None] | None) = None,
    ) -> None:
        if on_settlement_unproven is not None and not callable(on_settlement_unproven):
            raise TypeError("Workspace mutation settlement observer must be callable.")
        self.__active: set[asyncio.Task[Any]] = set()
        self.__sealed = False
        self.__settlement_unproven = False
        self.__on_settlement_unproven = on_settlement_unproven

    def fail_closed(
        self,
        settlement_probe: Callable[[], Awaitable[bool]],
    ) -> None:
        """Fence the window after a mutation loses positive settlement proof."""

        if not callable(settlement_probe):
            raise TypeError("Workspace mutation settlement probe must be callable.")
        self.__sealed = True
        self.__settlement_unproven = True
        if self.__on_settlement_unproven is not None:
            self.__on_settlement_unproven(settlement_probe)

    @contextmanager
    def track_current_task(self) -> Iterator[None]:
        """Register the current invocation task until its mutation settles."""

        if self.__sealed:
            raise RuntimeError("Workspace mutation is unavailable after invocation settlement.")
        current = asyncio.current_task()
        if current is None:  # pragma: no cover - coroutine execution invariant
            raise RuntimeError("Workspace mutation requires an asyncio task owner.")
        self.__active.add(current)
        try:
            yield
        finally:
            self.__active.discard(current)

    async def run(self, operation_factory: Callable[[], Awaitable[Any]]) -> Any:
        with self.track_current_task():
            outcome = await await_invocation_operation(
                operation_factory,
                request_child_cancellation=False,
                on_unsettled_supervisory_exit=self.fail_closed,
            )
        mutation_error = outcome.error
        if mutation_error is not None and exception_tree_contains(
            mutation_error,
            (GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            raise mutation_error
        if mutation_error is not None:
            mutation_error = _bounded_workspace_mutation_failure(
                mutation_error,
                operation="Invocation workspace mutation",
            )
        if outcome.cancellation is not None and mutation_error is not None:
            raise BaseExceptionGroup(
                "Invocation workspace mutation failed after caller cancellation.",
                [outcome.cancellation, mutation_error],
            ) from mutation_error
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if mutation_error is not None:
            raise mutation_error
        return outcome.result

    def seal_and_transfer_pending(self) -> None:
        """Transfer detached mutation tasks without delaying a supervisory exit."""

        self.__sealed = True
        current = asyncio.current_task()
        pending = tuple(task for task in self.__active if task is not current and not task.done())
        if pending:
            self.fail_closed(_RetainedWorkspaceMutationTasksProbe(pending))

    async def seal_and_wait(self) -> None:
        """Reject new mutations and wait for every dispatched mutation owner."""

        self.__sealed = True
        current = asyncio.current_task()
        owned = {task for task in self.__active if task is not current}
        pending = set(owned)
        cancellation: asyncio.CancelledError | None = None
        while pending:
            waiter = asyncio.gather(*pending, return_exceptions=True)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                pending = {task for task in pending if not task.done()}
                continue
            except BaseExceptionGroup as exc:
                if exception_tree_contains(
                    exc,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                ):
                    pending = {task for task in pending if not task.done()}
                    if pending:
                        self.fail_closed(_RetainedWorkspaceMutationTasksProbe(tuple(pending)))
                raise
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                pending = {task for task in pending if not task.done()}
                if pending:
                    self.fail_closed(_RetainedWorkspaceMutationTasksProbe(tuple(pending)))
                raise
            pending.clear()

        failures: list[BaseException] = []
        if self.__settlement_unproven:
            failures.append(
                WorkspaceMutationSettlementError(
                    "Workspace mutation settlement could not be proven."
                )
            )
        for task in owned:
            try:
                task.result()
            except BaseException as exc:
                failures.append(exc)
        process_failures = [
            failure
            for failure in failures
            if exception_tree_contains(
                failure,
                (GeneratorExit, KeyboardInterrupt, SystemExit),
            )
        ]
        if len(process_failures) == 1:
            raise process_failures[0]
        if process_failures:
            raise BaseExceptionGroup(
                "Workspace mutation settlement received process signals.",
                process_failures,
            )
        bounded_failures = [
            (
                failure
                if isinstance(failure, WorkspaceMutationSettlementError)
                else _bounded_workspace_mutation_failure(
                    failure,
                    operation="Detached invocation workspace mutation",
                )
            )
            for failure in failures
        ]
        del failures
        if cancellation is not None and bounded_failures:
            # The mutation adapter is extension-owned, so its failures may
            # retain private arguments or backend diagnostics. Keep caller
            # cancellation authoritative and attach only a fixed public-safe
            # indication that settlement also failed.
            del bounded_failures
            safe_failure = RuntimeError(
                "Workspace mutation settlement failed after caller cancellation."
            )
            raise cancellation from safe_failure
        if cancellation is not None:
            raise cancellation
        if len(bounded_failures) == 1:
            raise bounded_failures[0]
        if bounded_failures:
            raise ExceptionGroup(
                "Detached workspace mutations failed during settlement.",
                bounded_failures,
            )


class _RetainedWorkspaceMutationTasksProbe:
    """Join exact invocation tasks retained after a supervisory tool exit."""

    __slots__ = ("__tasks",)

    def __init__(self, tasks: tuple[asyncio.Task[Any], ...]) -> None:
        self.__tasks = tasks
        for task in tasks:
            task.add_done_callback(_observe_retained_workspace_mutation_task)

    async def __call__(self) -> bool:
        current_loop = asyncio.get_running_loop()
        if any(not task.done() and task.get_loop() is not current_loop for task in self.__tasks):
            return False
        pending = tuple(task for task in self.__tasks if not task.done())
        if pending:
            waiter = asyncio.gather(*pending, return_exceptions=True)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                raise
            except BaseException:
                if not all(task.done() for task in pending):
                    raise
        return all(task.done() for task in self.__tasks)


def _observe_retained_workspace_mutation_task(task: asyncio.Task[Any]) -> None:
    """Consume a detached mutation failure after quiescence is established."""

    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        return


_MAX_RESOURCE_CANCELLATION_REASON_BYTES = 2_048
_SAFE_RESOURCE_CANCELLATION_MESSAGE = "Resource read was cancelled."


@dataclass(frozen=True, slots=True)
class _ReadFailure:
    kind: str


@dataclass(frozen=True, slots=True)
class _ReadCancellation:
    args: tuple[object, ...]
    cause: Exception | None = None


_WorkspaceCapture = WorkspaceReadResult | _ReadFailure | _ReadCancellation
_ArtifactCapture = ArtifactReadResult | _ReadFailure | _ReadCancellation


def _bounded_workspace_read_limit(workspace: Workspace, max_bytes: int) -> int:
    value = workspace.bounded_read_limit(max_bytes)
    if type(value) is not int or value <= 0 or value > max_bytes:
        raise RuntimeError(
            f"{type(workspace).__name__}.bounded_read_limit() must return a positive integer "
            f"no greater than max_bytes={max_bytes}."
        )
    return value


class InvocationWorkspaceHandle(Workspace):
    """Workspace facade that redacts byte reads before their public bound."""

    __slots__ = (
        "__capture_observer",
        "__mutation_owner",
        "__redactor_snapshot_provider",
        "__workspace",
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        redactor_snapshot_provider: Callable[[], Any],
        capture_observer: Callable[[int], None],
        mutation_owner: InvocationWorkspaceMutationOwner | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("Invocation workspace delegate must implement Workspace.")
        if not callable(redactor_snapshot_provider):
            raise TypeError("Invocation workspace snapshot provider must be callable.")
        if not callable(capture_observer):
            raise TypeError("Invocation workspace capture observer must be callable.")
        if (
            mutation_owner is not None
            and type(mutation_owner) is not InvocationWorkspaceMutationOwner
        ):
            raise TypeError("Invocation workspace mutation owner is invalid.")
        self.__workspace = workspace
        self.__redactor_snapshot_provider = redactor_snapshot_provider
        self.__capture_observer = capture_observer
        self.__mutation_owner = mutation_owner
        self.id = workspace.id

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        snapshot = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
        for _ in range(2):
            captured = await self._capture_workspace_page(
                path=path,
                offset=offset,
                max_bytes=max_bytes,
                snapshot=snapshot,
            )
            if isinstance(captured, _ReadCancellation):
                del path, offset, max_bytes, snapshot, self
                _raise_clean_read_cancellation(captured)
            if isinstance(captured, _ReadFailure):
                del path, offset, max_bytes, snapshot, self
                _raise_workspace_read_failure(captured)
            current = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
            if current.revision == snapshot.revision and current.redactor.has_same_registry(
                snapshot.redactor
            ):
                self.__capture_observer(current.revision)
                return captured
            del captured
            snapshot = current
        del path, offset, max_bytes, snapshot, self
        raise InvocationResourceReadError(
            "Workspace output was omitted because its secret scope changed during the read."
        )

    async def _capture_workspace_page(
        self,
        *,
        path: str,
        offset: int,
        max_bytes: int | None,
        snapshot: InvocationRedactorSnapshot,
    ) -> _WorkspaceCapture:
        overlap = snapshot.redactor.pagination_overlap_utf8_bytes
        fetch_offset = max(0, offset - overlap)
        prefix_bytes = offset - fetch_offset
        fetch_max = None if max_bytes is None else prefix_bytes + max_bytes + overlap
        if fetch_max is not None:
            fetch_max = _bounded_workspace_read_limit(self.__workspace, fetch_max)
        captured = await _read_workspace_delegate(
            self.__workspace,
            path=path,
            offset=fetch_offset,
            max_bytes=fetch_max,
            redactor_snapshot_provider=self.__redactor_snapshot_provider,
        )
        if not isinstance(captured, WorkspaceReadResult):
            return captured
        if captured.offset != fetch_offset:
            return _ReadFailure("invalid_workspace_result")
        window_source_bytes = captured.source_bytes_read
        if window_source_bytes is None:  # pragma: no cover - dataclass invariant
            return _ReadFailure("invalid_workspace_result")
        window_end = captured.offset + window_source_bytes
        if window_end < offset:
            return _ReadFailure("redaction_window_unavailable")
        requested_end = (
            captured.total_bytes
            if max_bytes is None
            else min(captured.total_bytes, offset + max_bytes)
        )
        page_end = min(requested_end, window_end)
        source_bytes_read = page_end - offset
        if source_bytes_read == 0:
            projected = b""
            redaction_truncated = False
        else:
            required_window_end = min(captured.total_bytes, page_end + overlap)
            source_complete = window_end >= required_window_end
            projection_limit = (
                source_bytes_read if max_bytes is None else min(max_bytes, source_bytes_read)
            )
            projected, redaction_truncated = snapshot.redactor.redact_bytes_page(
                captured.content,
                window_offset=captured.offset,
                page_offset=offset,
                page_end=page_end,
                max_bytes=max(1, projection_limit),
                source_complete=source_complete,
            )
        raw_truncated = page_end < captured.total_bytes
        complete = offset == 0 and not raw_truncated
        return WorkspaceReadResult(
            content=projected,
            total_bytes=captured.total_bytes,
            truncated=raw_truncated,
            offset=offset,
            revision=captured.revision if complete else None,
            sha256=captured.sha256 if complete else None,
            source_bytes_read=source_bytes_read,
            redaction_truncated=redaction_truncated,
        )

    def bounded_read_limit(self, max_bytes: int) -> int:
        return self.__workspace.bounded_read_limit(max_bytes)

    async def write_bytes(self, path: str, content: bytes) -> None:
        await self._run_mutation(lambda: self.__workspace.write_bytes(path, content))

    async def delete(self, path: str) -> None:
        await self._run_mutation(lambda: self.__workspace.delete(path))

    async def create_bytes(self, path: str, content: bytes):
        return await self._run_mutation(lambda: self.__workspace.create_bytes(path, content))

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        return await self._run_mutation(
            lambda: self.__workspace.replace_bytes(
                path,
                content,
                expected_revision=expected_revision,
            )
        )

    async def delete_if_revision(self, path: str, *, expected_revision: str):
        return await self._run_mutation(
            lambda: self.__workspace.delete_if_revision(
                path,
                expected_revision=expected_revision,
            )
        )

    async def _run_mutation(self, operation_factory: Callable[[], Awaitable[Any]]) -> Any:
        if self.__mutation_owner is None:
            return await operation_factory()
        return await self.__mutation_owner.run(operation_factory)

    async def list(self, pattern: str = "**/*", *, limit: int | None = None):
        return await self.__workspace.list(pattern, limit=limit)

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return self.__workspace.resource_key

    def _authenticated_cwd_for_runner(self, runner: Any) -> str | None:
        """Return private workspace binding evidence to Cayu's runner facade."""

        from cayu.runners import LocalRunner
        from cayu.workspaces import LocalWorkspace, RunnerBoundWorkspace

        if isinstance(self.__workspace, RunnerBoundWorkspace):
            if not self.__workspace.is_bound_to_runner(runner):
                return None
            return self.__workspace.runner_cwd
        if isinstance(self.__workspace, LocalWorkspace) and isinstance(runner, LocalRunner):
            from pathlib import Path

            runner_root = Path(runner.resolve_cwd(None)).resolve()
            workspace_root = self.__workspace.root.resolve()
            try:
                workspace_root.relative_to(runner_root)
            except ValueError:
                return None
            return str(workspace_root)
        return None


class InvocationArtifactStoreHandle(ArtifactStore):
    """Artifact facade that redacts byte reads before their public bound."""

    __slots__ = (
        "__artifact_store",
        "__capture_observer",
        "__redactor_snapshot_provider",
    )

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        redactor_snapshot_provider: Callable[[], Any],
        capture_observer: Callable[[int], None],
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("Invocation artifact delegate must implement ArtifactStore.")
        if not callable(redactor_snapshot_provider):
            raise TypeError("Invocation artifact snapshot provider must be callable.")
        if not callable(capture_observer):
            raise TypeError("Invocation artifact capture observer must be callable.")
        self.__artifact_store = artifact_store
        self.__redactor_snapshot_provider = redactor_snapshot_provider
        self.__capture_observer = capture_observer
        self.id = artifact_store.id

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        snapshot = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
        for _ in range(2):
            captured = await self._capture_artifact(
                artifact_id=artifact_id,
                max_bytes=max_bytes,
                snapshot=snapshot,
            )
            if isinstance(captured, _ReadCancellation):
                del artifact_id, max_bytes, snapshot, self
                _raise_clean_read_cancellation(captured)
            if isinstance(captured, _ReadFailure):
                del artifact_id, max_bytes, snapshot, self
                _raise_artifact_read_failure(captured)
            current = resolve_invocation_redactor_snapshot(self.__redactor_snapshot_provider)
            if current.revision == snapshot.revision and current.redactor.has_same_registry(
                snapshot.redactor
            ):
                self.__capture_observer(current.revision)
                return captured
            del captured
            snapshot = current
        del artifact_id, max_bytes, snapshot, self
        raise InvocationResourceReadError(
            "Artifact output was omitted because its secret scope changed during the read."
        )

    async def _capture_artifact(
        self,
        *,
        artifact_id: str,
        max_bytes: int | None,
        snapshot: InvocationRedactorSnapshot,
    ) -> _ArtifactCapture:
        captured = await _read_artifact_delegate(
            self.__artifact_store,
            artifact_id=artifact_id,
            # ArtifactStore.max_bytes is a hard materialization boundary. A
            # truncated prefix is projected with source_complete=False so the
            # redactor withholds any suffix whose safety depends on unseen
            # bytes instead of over-reading for overlap.
            max_bytes=max_bytes,
            redactor_snapshot_provider=self.__redactor_snapshot_provider,
        )
        if not isinstance(captured, ArtifactReadResult):
            return captured
        source_available = captured.source_bytes_read
        if source_available is None:  # pragma: no cover - dataclass invariant
            return _ReadFailure("invalid_artifact_result")
        requested_source_bytes = (
            captured.total_bytes if max_bytes is None else min(captured.total_bytes, max_bytes)
        )
        source_bytes_read = min(source_available, requested_source_bytes)
        if source_bytes_read == 0:
            projected = b""
            redaction_truncated = False
        else:
            source_complete = not captured.truncated
            projection_limit = (
                source_bytes_read if max_bytes is None else min(max_bytes, source_bytes_read)
            )
            projected, redaction_truncated = snapshot.redactor.redact_bytes_page(
                captured.content,
                window_offset=0,
                page_offset=0,
                page_end=source_bytes_read,
                max_bytes=max(1, projection_limit),
                source_complete=source_complete,
            )
        return ArtifactReadResult(
            metadata=captured.metadata,
            content=projected,
            total_bytes=captured.total_bytes,
            truncated=source_bytes_read < captured.total_bytes,
            source_bytes_read=source_bytes_read,
            redaction_truncated=redaction_truncated,
        )

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any):
        return await self.__artifact_store.put_bytes(content, filename=filename, **kwargs)

    async def list(self, **kwargs: Any):
        return await self.__artifact_store.list(**kwargs)

    async def delete(self, artifact_id: str) -> None:
        await self.__artifact_store.delete(artifact_id)


async def _read_workspace_delegate(
    workspace: Workspace,
    *,
    path: str,
    offset: int,
    max_bytes: int | None,
    redactor_snapshot_provider: Callable[[], Any],
) -> _WorkspaceCapture:
    def operation_factory(
        workspace: Workspace = workspace,
        path: str = path,
        offset: int = offset,
        max_bytes: int | None = max_bytes,
    ) -> Awaitable[WorkspaceReadResult]:
        return workspace.read_bytes(path, offset=offset, max_bytes=max_bytes)

    operation = await_invocation_operation(operation_factory)
    del operation_factory, workspace, path
    try:
        outcome = await operation
    finally:
        del operation
    result = outcome.result
    error = outcome.error
    caller_cancellation = outcome.cancellation
    if caller_cancellation is not None:
        args = _safe_cancellation_args_with_latest_snapshot(
            caller_cancellation,
            redactor_snapshot_provider,
        )
        cause = (
            None
            if error is None or isinstance(error, asyncio.CancelledError)
            else _workspace_read_failure_exception(_workspace_failure_from_error(error))
        )
        del outcome, result, error, caller_cancellation, redactor_snapshot_provider
        return _ReadCancellation(args=args, cause=cause)
    if isinstance(error, asyncio.CancelledError):
        args = _safe_cancellation_args_with_latest_snapshot(
            error,
            redactor_snapshot_provider,
        )
        del outcome, result, error, caller_cancellation, redactor_snapshot_provider
        return _ReadCancellation(args)
    if error is not None:
        failure = _workspace_failure_from_error(error)
        del outcome, result, error, caller_cancellation, redactor_snapshot_provider
        return failure
    del outcome, error, caller_cancellation, redactor_snapshot_provider
    if type(result) is not WorkspaceReadResult:
        return _ReadFailure("invalid_workspace_result")
    try:
        copied = WorkspaceReadResult(
            content=result.content,
            total_bytes=result.total_bytes,
            truncated=result.truncated,
            offset=result.offset,
            revision=result.revision,
            sha256=result.sha256,
            source_bytes_read=result.source_bytes_read,
            redaction_truncated=result.redaction_truncated,
        )
        if max_bytes is not None and len(copied.content) > max_bytes:
            return _ReadFailure("invalid_workspace_result")
        return copied
    except BaseException:
        return _ReadFailure("invalid_workspace_result")


async def _read_artifact_delegate(
    artifact_store: ArtifactStore,
    *,
    artifact_id: str,
    max_bytes: int | None,
    redactor_snapshot_provider: Callable[[], Any],
) -> _ArtifactCapture:
    def operation_factory(
        artifact_store: ArtifactStore = artifact_store,
        artifact_id: str = artifact_id,
        max_bytes: int | None = max_bytes,
    ) -> Awaitable[ArtifactReadResult]:
        return artifact_store.read_bytes(artifact_id, max_bytes=max_bytes)

    operation = await_invocation_operation(operation_factory)
    del operation_factory, artifact_store
    try:
        outcome = await operation
    finally:
        del operation
    result = outcome.result
    error = outcome.error
    caller_cancellation = outcome.cancellation
    if caller_cancellation is not None:
        args = _safe_cancellation_args_with_latest_snapshot(
            caller_cancellation,
            redactor_snapshot_provider,
        )
        cause = (
            None
            if error is None or isinstance(error, asyncio.CancelledError)
            else _artifact_read_failure_exception(_artifact_failure_from_error(error))
        )
        del (
            artifact_id,
            outcome,
            result,
            error,
            caller_cancellation,
            redactor_snapshot_provider,
        )
        return _ReadCancellation(args=args, cause=cause)
    if isinstance(error, asyncio.CancelledError):
        args = _safe_cancellation_args_with_latest_snapshot(
            error,
            redactor_snapshot_provider,
        )
        del (
            artifact_id,
            outcome,
            result,
            error,
            caller_cancellation,
            redactor_snapshot_provider,
        )
        return _ReadCancellation(args)
    if error is not None:
        failure = _artifact_failure_from_error(error)
        del (
            artifact_id,
            outcome,
            result,
            error,
            caller_cancellation,
            redactor_snapshot_provider,
        )
        return failure
    del outcome, error, caller_cancellation, redactor_snapshot_provider
    if type(result) is not ArtifactReadResult:
        del artifact_id
        return _ReadFailure("invalid_artifact_result")
    try:
        copied = copy_artifact_read_result(
            result,
            expected_artifact_id=artifact_id,
            max_content_bytes=max_bytes,
        )
    except BaseException:
        del artifact_id, result
        return _ReadFailure("invalid_artifact_result")
    del artifact_id, result
    return copied


def _safe_cancellation_args(
    cancellation: asyncio.CancelledError,
    snapshot: InvocationRedactorSnapshot,
) -> tuple[object, ...]:
    try:
        args = BaseException.__dict__["args"].__get__(cancellation, BaseException)
    except BaseException:
        return (_SAFE_RESOURCE_CANCELLATION_MESSAGE,)
    if type(args) is not tuple or len(args) > 1:
        return (_SAFE_RESOURCE_CANCELLATION_MESSAGE,)
    if not args:
        return ()
    value = args[0]
    try:
        if type(value) is str:
            return (
                snapshot.redactor.redact_text_bounded(
                    value,
                    max_bytes=_MAX_RESOURCE_CANCELLATION_REASON_BYTES,
                ),
            )
        if value is None or type(value) in {bool, int, float}:
            return (value,)
    except BaseException:
        pass
    return (_SAFE_RESOURCE_CANCELLATION_MESSAGE,)


def _safe_cancellation_args_with_latest_snapshot(
    cancellation: asyncio.CancelledError,
    redactor_snapshot_provider: Callable[[], Any],
) -> tuple[object, ...]:
    """Project cancellation only after observing the post-await secret scope."""

    try:
        snapshot = resolve_invocation_redactor_snapshot(redactor_snapshot_provider)
    except BaseException:
        return (_SAFE_RESOURCE_CANCELLATION_MESSAGE,)
    return _safe_cancellation_args(cancellation, snapshot)


def _raise_clean_read_cancellation(cancellation: _ReadCancellation) -> NoReturn:
    args = cancellation.args
    cause = cancellation.cause
    del cancellation
    raise asyncio.CancelledError(*args) from cause


def _workspace_failure_from_error(error: BaseException) -> _ReadFailure:
    return _ReadFailure(
        "not_found" if isinstance(error, FileNotFoundError) else "workspace_unavailable"
    )


def _workspace_read_failure_exception(failure: _ReadFailure) -> Exception:
    if failure.kind == "not_found":
        return FileNotFoundError("Workspace path was not found.")
    return InvocationResourceReadError("Workspace read could not be completed safely.")


def _raise_workspace_read_failure(failure: _ReadFailure) -> NoReturn:
    error = _workspace_read_failure_exception(failure)
    del failure
    raise error from None


def _artifact_failure_from_error(error: BaseException) -> _ReadFailure:
    if isinstance(error, InvalidArtifactIdError):
        return _ReadFailure("invalid_artifact_id")
    if isinstance(error, FileNotFoundError):
        return _ReadFailure("not_found")
    return _ReadFailure("artifact_unavailable")


def _artifact_read_failure_exception(failure: _ReadFailure) -> Exception:
    if failure.kind == "invalid_artifact_id":
        return InvalidArtifactIdError("Artifact identifier is invalid.")
    if failure.kind == "not_found":
        return FileNotFoundError("Artifact was not found.")
    if failure.kind == "artifact_unavailable":
        return ArtifactStoreUnavailableError("Artifact store is unavailable.")
    return InvocationResourceReadError("Artifact read could not be completed safely.")


def _raise_artifact_read_failure(failure: _ReadFailure) -> NoReturn:
    error = _artifact_read_failure_exception(failure)
    del failure
    raise error from None


def invocation_workspace_handle(
    workspace: Workspace | None,
    *,
    redactor_snapshot_provider: Callable[[], Any],
    capture_observer: Callable[[int], None],
    mutation_owner: InvocationWorkspaceMutationOwner | None = None,
) -> InvocationWorkspaceHandle | None:
    if workspace is None:
        return None
    return InvocationWorkspaceHandle(
        workspace,
        redactor_snapshot_provider=redactor_snapshot_provider,
        capture_observer=capture_observer,
        mutation_owner=mutation_owner,
    )


def invocation_artifact_store_handle(
    artifact_store: ArtifactStore | None,
    *,
    redactor_snapshot_provider: Callable[[], Any],
    capture_observer: Callable[[int], None],
) -> InvocationArtifactStoreHandle | None:
    if artifact_store is None:
        return None
    return InvocationArtifactStoreHandle(
        artifact_store,
        redactor_snapshot_provider=redactor_snapshot_provider,
        capture_observer=capture_observer,
    )


def invocation_workspace_authenticated_cwd(workspace: Any, runner: Any) -> str | None:
    if type(workspace) is not InvocationWorkspaceHandle:
        return None
    return workspace._authenticated_cwd_for_runner(runner)


__all__ = [
    "InvocationArtifactStoreHandle",
    "InvocationResourceReadError",
    "InvocationWorkspaceHandle",
    "InvocationWorkspaceMutationOwner",
    "WorkspaceMutationSettlementError",
    "invocation_artifact_store_handle",
    "invocation_workspace_authenticated_cwd",
    "invocation_workspace_handle",
]
