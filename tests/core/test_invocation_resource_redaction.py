from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
from pydantic import SecretStr
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu import LocalArtifactStore, LocalWorkspace
from cayu.artifacts import ArtifactStoreUnavailableError
from cayu.core import AgentSpec, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.tools._resources import (
    InvocationResourceReadError,
    invocation_artifact_store_handle,
    invocation_workspace_handle,
)
from cayu.vaults import ResolvedSecret, SecretRedactor, SecretRef, StaticVault
from cayu.workspaces import WorkspaceReadResult


def test_invocation_workspace_read_redacts_before_source_page_bound(tmp_path) -> None:
    secret = "workspace-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    page = asyncio.run(handle.read_bytes("secret.txt", max_bytes=16))

    assert page.content == b""
    assert page.source_bytes_read == 16
    assert page.total_bytes == len(secret)
    assert page.truncated is True
    assert page.next_offset == 16
    assert secret[:16].encode() not in page.content


def test_invocation_artifact_read_preserves_unrelated_bounded_output(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"abcdef",
            filename="plain.txt",
            session_id="session",
        )
    )
    tracker = InvocationSecretTracker(SecretRedactor("unrelated-secret"))
    handle = invocation_artifact_store_handle(
        store,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    page = asyncio.run(handle.read_bytes(artifact.id, max_bytes=2))

    assert page.content == b"ab"
    assert page.source_bytes_read == 2
    assert page.total_bytes == 6
    assert page.truncated is True
    assert page.redaction_truncated is False


def test_invocation_artifact_read_never_exceeds_the_callers_hard_limit(tmp_path) -> None:
    class RecordingArtifactStore(LocalArtifactStore):
        observed_limits: list[int | None]

        def __init__(self) -> None:
            super().__init__(tmp_path / "artifacts", store_id="artifacts")
            self.observed_limits = []

        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            self.observed_limits.append(max_bytes)
            return await super().read_bytes(artifact_id, max_bytes=max_bytes)

    secret = "bounded-artifact-secret-canary-ABCDEFGHIJKLMNOP"
    store = RecordingArtifactStore()
    artifact = asyncio.run(
        store.put_bytes(
            secret.encode(),
            filename="secret.txt",
            session_id="session",
        )
    )
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_artifact_store_handle(
        store,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    page = asyncio.run(handle.read_bytes(artifact.id, max_bytes=16))

    assert store.observed_limits == [16]
    assert page.source_bytes_read == 16
    assert page.content == b""
    assert page.redaction_truncated is True
    assert secret[:16].encode() not in page.content


def test_late_secret_resolution_invalidates_custom_workspace_capture(tmp_path) -> None:
    secret = "late-workspace-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    page = asyncio.run(handle.read_bytes("secret.txt", max_bytes=16))
    tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))

    assert page.content == secret[:16].encode()
    assert tracker.seal_for_publication().unsafe_output is True


def test_workspace_read_retries_with_secret_registered_during_backend_await(tmp_path) -> None:
    secret = "concurrent-workspace-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class BlockingWorkspace(LocalWorkspace):
        started: asyncio.Event
        release: asyncio.Event
        calls = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    workspace = BlockingWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> WorkspaceReadResult:
        workspace.started = asyncio.Event()
        workspace.release = asyncio.Event()
        task = asyncio.create_task(handle.read_bytes("secret.txt", max_bytes=16))
        await workspace.started.wait()
        tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))
        workspace.release.set()
        return await task

    page = asyncio.run(scenario())

    assert workspace.calls == 2
    assert page.content == b""
    assert secret[:16].encode() not in page.content


def test_workspace_read_propagates_real_task_cancellation_without_raw_frames(tmp_path) -> None:
    secret = "cancelled-resource-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class BlockingWorkspace(LocalWorkspace):
        started: asyncio.Event

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    workspace = BlockingWorkspace(workspace_root, workspace_id="workspace")
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        workspace.started = asyncio.Event()
        task = asyncio.create_task(handle.read_bytes("secret.txt", max_bytes=16))
        await workspace.started.wait()
        task.cancel(secret)
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert secret not in repr(cancellation)


def test_workspace_cancellation_uses_secret_registered_during_backend_await(tmp_path) -> None:
    secret = "late-cancelled-workspace-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class BlockingWorkspace(LocalWorkspace):
        started: asyncio.Event

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    workspace = BlockingWorkspace(workspace_root, workspace_id="workspace")
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        workspace.started = asyncio.Event()
        task = asyncio.create_task(handle.read_bytes("secret.txt", max_bytes=16))
        await workspace.started.wait()
        tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))
        task.cancel(secret)
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert secret not in repr(cancellation)


def test_workspace_cleanup_failure_cannot_replace_real_task_cancellation(
    tmp_path,
) -> None:
    secret = "workspace-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class CleanupFailingWorkspace(LocalWorkspace):
        started: asyncio.Event

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError(secret)

    workspace = CleanupFailingWorkspace(workspace_root, workspace_id="workspace")
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        workspace.started = asyncio.Event()

        async def invoke() -> WorkspaceReadResult:
            return await handle.read_bytes("secret.txt", max_bytes=16)

        task = asyncio.create_task(invoke())
        await workspace.started.wait()
        task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert type(cancellation.__cause__) is InvocationResourceReadError
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


def test_workspace_preserves_pending_cancellation_suppressed_by_delegate(tmp_path) -> None:
    secret = "workspace-pending-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class CancellationSuppressingWorkspace(LocalWorkspace):
        calls = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            self.calls += 1
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    workspace = CancellationSuppressingWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("plain.txt", b"safe"))
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool, int]:
        async def invoke() -> tuple[asyncio.CancelledError, int]:
            task = asyncio.current_task()
            assert task is not None
            task.cancel(secret)
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await handle.read_bytes("plain.txt", max_bytes=16)
            cancelling = task.cancelling()
            await asyncio.sleep(0)
            task.uncancel()
            return exc_info.value, cancelling

        task = asyncio.create_task(invoke())
        cancellation, cancelling = await task
        return cancellation, cancelling, task.cancelled(), workspace.calls

    cancellation, cancelling, cancelled, calls = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is False
    assert calls == 0
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)


def test_workspace_does_not_reclassify_historical_task_cancellation(tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("plain.txt", b"safe"))
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[WorkspaceReadResult, int]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        historical_count = task.cancelling()
        try:
            return await handle.read_bytes("plain.txt", max_bytes=16), historical_count
        finally:
            task.uncancel()

    result, historical_count = asyncio.run(scenario())

    assert historical_count == 1
    assert result.content == b"safe"


def test_artifact_cancellation_uses_secret_registered_during_backend_await(tmp_path) -> None:
    secret = "late-cancelled-artifact-secret-canary-ABCDEFGHIJKLMNOP"

    class BlockingArtifactStore(LocalArtifactStore):
        started: asyncio.Event

        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            del artifact_id, max_bytes
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    store = BlockingArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    tracker = InvocationSecretTracker(SecretRedactor())
    handle = invocation_artifact_store_handle(
        store,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        store.started = asyncio.Event()
        task = asyncio.create_task(handle.read_bytes("art_1", max_bytes=16))
        await store.started.wait()
        tracker.record(ResolvedSecret(name="token", value=SecretStr(secret)))
        task.cancel(secret)
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert secret not in repr(cancellation)


def test_artifact_cleanup_failure_cannot_replace_real_task_cancellation(
    tmp_path,
) -> None:
    secret = "artifact-cleanup-secret-canary-ABCDEFGHIJKLMNOP"

    class CleanupFailingArtifactStore(LocalArtifactStore):
        started: asyncio.Event

        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            del artifact_id, max_bytes
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError(secret)

    store = CleanupFailingArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_artifact_store_handle(
        store,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        store.started = asyncio.Event()

        async def invoke():
            return await handle.read_bytes("art_1", max_bytes=16)

        task = asyncio.create_task(invoke())
        await store.started.wait()
        task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert type(cancellation.__cause__) is ArtifactStoreUnavailableError
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


def test_artifact_preserves_pending_cancellation_suppressed_by_delegate(tmp_path) -> None:
    secret = "artifact-pending-cancellation-secret-canary-ABCDEFGHIJKLMNOP"

    class CancellationSuppressingArtifactStore(LocalArtifactStore):
        calls = 0

        async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
            self.calls += 1
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
            return await super().read_bytes(artifact_id, max_bytes=max_bytes)

    store = CancellationSuppressingArtifactStore(
        tmp_path / "artifacts",
        store_id="artifacts",
    )
    artifact = asyncio.run(
        store.put_bytes(
            b"safe",
            filename="plain.txt",
            session_id="session",
        )
    )
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_artifact_store_handle(
        store,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, int, bool, int]:
        async def invoke() -> tuple[asyncio.CancelledError, int]:
            task = asyncio.current_task()
            assert task is not None
            task.cancel(secret)
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await handle.read_bytes(artifact.id, max_bytes=16)
            cancelling = task.cancelling()
            await asyncio.sleep(0)
            task.uncancel()
            return exc_info.value, cancelling

        task = asyncio.create_task(invoke())
        cancellation, cancelling = await task
        return cancellation, cancelling, task.cancelled(), store.calls

    cancellation, cancelling, cancelled, calls = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is False
    assert calls == 0
    assert cancellation.args == ("[REDACTED_SECRET]",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert secret not in repr(cancellation)


@pytest.mark.parametrize("resource_kind", ["workspace", "artifact"])
def test_resource_delegate_cannot_uncancel_the_callers_task(
    tmp_path,
    resource_kind: str,
) -> None:
    started: asyncio.Event

    if resource_kind == "workspace":
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()

        class UncancellingWorkspace(LocalWorkspace):
            async def read_bytes(
                self,
                path: str,
                *,
                offset: int = 0,
                max_bytes: int | None = None,
            ) -> WorkspaceReadResult:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is not None
                    task.uncancel()
                    return await super().read_bytes(
                        path,
                        offset=offset,
                        max_bytes=max_bytes,
                    )

        workspace = UncancellingWorkspace(workspace_root, workspace_id="workspace")
        asyncio.run(workspace.write_bytes("plain.txt", b"safe"))
        tracker = InvocationSecretTracker(SecretRedactor())
        handle = invocation_workspace_handle(
            workspace,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None

        async def read_resource():
            return await handle.read_bytes("plain.txt", max_bytes=16)

    else:

        class UncancellingArtifactStore(LocalArtifactStore):
            async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is not None
                    task.uncancel()
                    return await super().read_bytes(artifact_id, max_bytes=max_bytes)

        store = UncancellingArtifactStore(
            tmp_path / "artifacts",
            store_id="artifacts",
        )
        artifact = asyncio.run(
            store.put_bytes(
                b"safe",
                filename="plain.txt",
                session_id="session",
            )
        )
        tracker = InvocationSecretTracker(SecretRedactor())
        handle = invocation_artifact_store_handle(
            store,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None

        async def read_resource():
            return await handle.read_bytes(artifact.id, max_bytes=16)

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        nonlocal started
        started = asyncio.Event()
        task = asyncio.create_task(read_resource())
        await started.wait()
        task.cancel("authentic caller reason")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("authentic caller reason",)
    assert cancelling == 1
    assert cancelled is True


@pytest.mark.parametrize("resource_kind", ["workspace", "artifact"])
def test_resource_failure_cause_cannot_forge_the_caller_cancellation_reason(
    tmp_path,
    resource_kind: str,
) -> None:
    secret = "backend-controlled-secret-reason"
    started: asyncio.Event

    if resource_kind == "workspace":
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()

        class CausalFailureWorkspace(LocalWorkspace):
            async def read_bytes(
                self,
                path: str,
                *,
                offset: int = 0,
                max_bytes: int | None = None,
            ) -> WorkspaceReadResult:
                del path, offset, max_bytes
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("workspace cleanup failed") from asyncio.CancelledError(
                        secret
                    )

        workspace = CausalFailureWorkspace(workspace_root, workspace_id="workspace")
        tracker = InvocationSecretTracker(SecretRedactor(secret))
        handle = invocation_workspace_handle(
            workspace,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None
        expected_cause_type = InvocationResourceReadError

        async def read_resource():
            return await handle.read_bytes("plain.txt", max_bytes=16)

    else:

        class CausalFailureArtifactStore(LocalArtifactStore):
            async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
                del artifact_id, max_bytes
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("artifact cleanup failed") from asyncio.CancelledError(
                        secret
                    )

        store = CausalFailureArtifactStore(
            tmp_path / "artifacts",
            store_id="artifacts",
        )
        tracker = InvocationSecretTracker(SecretRedactor(secret))
        handle = invocation_artifact_store_handle(
            store,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None
        expected_cause_type = ArtifactStoreUnavailableError

        async def read_resource():
            return await handle.read_bytes("art_1", max_bytes=16)

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        nonlocal started
        started = asyncio.Event()
        task = asyncio.create_task(read_resource())
        await started.wait()
        task.cancel("authentic caller reason")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancellation.args == ("authentic caller reason",)
    assert type(cancellation.__cause__) is expected_cause_type
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)
    assert cancelling == 1
    assert cancelled is True


@pytest.mark.parametrize("resource_kind", ["workspace", "artifact"])
def test_resource_child_generator_exit_uses_established_failure_mapping(
    tmp_path,
    resource_kind: str,
) -> None:
    if resource_kind == "workspace":
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()

        class FatalWorkspace(LocalWorkspace):
            async def read_bytes(
                self,
                path: str,
                *,
                offset: int = 0,
                max_bytes: int | None = None,
            ) -> WorkspaceReadResult:
                del path, offset, max_bytes
                raise GeneratorExit("workspace terminated")

        workspace = FatalWorkspace(workspace_root, workspace_id="workspace")
        tracker = InvocationSecretTracker(SecretRedactor())
        handle = invocation_workspace_handle(
            workspace,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None
        operation = handle.read_bytes("fatal.txt", max_bytes=16)
        expected_error = InvocationResourceReadError
    else:

        class FatalArtifactStore(LocalArtifactStore):
            async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
                del artifact_id, max_bytes
                raise GeneratorExit("artifact store terminated")

        store = FatalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
        tracker = InvocationSecretTracker(SecretRedactor())
        handle = invocation_artifact_store_handle(
            store,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
        assert handle is not None
        operation = handle.read_bytes("art_fatal", max_bytes=16)
        expected_error = ArtifactStoreUnavailableError

    with pytest.raises(expected_error):
        asyncio.run(operation)


def test_workspace_read_omits_opaque_cancellation_reason_objects(tmp_path) -> None:
    secret = "opaque-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class SecretReason:
        def __str__(self) -> str:
            return secret

        def __repr__(self) -> str:
            return secret

    class BlockingWorkspace(LocalWorkspace):
        started: asyncio.Event

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    workspace = BlockingWorkspace(workspace_root, workspace_id="workspace")
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    handle = invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )
    assert handle is not None

    async def scenario() -> tuple[asyncio.CancelledError, bool]:
        workspace.started = asyncio.Event()
        task = asyncio.create_task(handle.read_bytes("secret.txt", max_bytes=16))
        await workspace.started.wait()
        task.cancel(SecretReason())
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelled()

    cancellation, cancelled = asyncio.run(scenario())

    assert cancelled is True
    assert cancellation.args == ("Resource read was cancelled.",)
    assert secret not in repr(cancellation)


def test_workspace_cleanup_failure_cancellation_stops_the_runtime_turn(tmp_path) -> None:
    secret = "runtime-workspace-cleanup-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "resource-cleanup-cancellation"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    class CleanupFailingWorkspace(LocalWorkspace):
        started: asyncio.Event

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            del path, offset, max_bytes
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError(secret)

    class ReadWorkspaceTool(Tool):
        spec = ToolSpec(
            name="read_workspace_during_cancellation",
            description="Read through the invocation workspace.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.workspace is not None
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="token"))
            await ctx.workspace.read_bytes("secret.txt", max_bytes=16)
            return ToolResult(content="unexpected")

    workspace = CleanupFailingWorkspace(workspace_root, workspace_id="workspace")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-read-workspace",
                    name="read_workspace_during_cancellation",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("unexpected"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=workspace,
            vault=StaticVault({"token": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ReadWorkspaceTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        workspace.started = asyncio.Event()
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "read")],
                ),
            )
        )
        await workspace.started.wait()
        task.cancel(secret)
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())
    events = asyncio.run(store.load_events(session_id))

    assert cancelling == 1
    assert cancelled is True
    assert len(provider.requests) == 1
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)
    assert secret not in repr(cancellation)
    assert secret not in repr(events)


def test_custom_workspace_read_is_safe_in_events_transcript_and_next_request(tmp_path) -> None:
    secret = "custom-workspace-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "resource-read-session"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="workspace")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))

    class ReadWorkspaceTool(Tool):
        spec = ToolSpec(
            name="read_workspace_directly",
            description="Exercise the documented custom workspace handle.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.workspace is not None
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="token"))
            page = await ctx.workspace.read_bytes("secret.txt", max_bytes=16)
            return ToolResult(
                content=page.content.decode(),
                structured={
                    "content": page.content.decode(),
                    "source_bytes_read": page.source_bytes_read,
                    "truncated": page.truncated,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-read-workspace",
                    name="read_workspace_directly",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=workspace,
            vault=StaticVault({"token": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ReadWorkspaceTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "read")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    completed = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    rendered = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "request": [
                message.model_dump(mode="json") for message in provider.requests[1].messages
            ],
        }
    )

    assert completed.payload["result"]["content"] == ""
    assert secret not in rendered
    assert secret[:16] not in rendered
