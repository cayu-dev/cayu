from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, cast

import pytest

import cayu.testing as testing
from cayu import (
    AgentSpec,
    CayuApp,
    LocalWorkspace,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.testing import (
    ToolEffectVerificationStatus,
    verify_tool_effect,
)


def _spec(name: str, effect: ToolEffect = ToolEffect.NONE) -> ToolSpec:
    return ToolSpec(
        name=name,
        effect=effect,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )


def _app(tool: Tool) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="worker", model="test-model"), tools=[tool])
    return app


class _CountingScandir:
    def __init__(self, entries: Any, counter: list[int]) -> None:
        self._entries = entries
        self._counter = counter

    def __enter__(self):
        self._entries.__enter__()
        return self

    def __exit__(self, *args):
        return self._entries.__exit__(*args)

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._entries)
        self._counter[0] += 1
        return entry


class _PureTool(Tool):
    spec = _spec("pure")

    def __init__(self) -> None:
        super().__init__()
        self.context: ToolContext | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.context = ctx
        assert ctx.workspace is not None
        existing = await ctx.workspace.read_bytes("input.txt")
        assert existing.content == b"unchanged"
        return ToolResult(content="done")


class _MutatingTool(Tool):
    spec = _spec("mutate")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.write_bytes("created.txt", b"created")
        await ctx.workspace.write_bytes("updated.txt", b"updated")
        await ctx.workspace.delete("deleted.txt")
        return ToolResult(content="changed")


class _ErrorTool(Tool):
    spec = _spec("error")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="failed", is_error=True)


class _MutateThenRaiseTool(Tool):
    spec = _spec("mutate_then_raise")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.write_bytes("created.txt", b"created")
        raise RuntimeError("test failure")


class _EffectfulTool(Tool):
    def __init__(self, effect: ToolEffect) -> None:
        super().__init__(_spec("effectful", effect))
        self.run_count = 0
        self.idempotency_key: str | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.run_count += 1
        self.idempotency_key = ctx.idempotency_key
        assert ctx.workspace is not None
        await ctx.workspace.write_bytes("effect.txt", str(self.run_count).encode())
        return ToolResult(content="observed")


class _InvalidResultTool(Tool):
    spec = _spec("invalid_result")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return cast("Any", "not a ToolResult")


class _OversizedWriteTool(Tool):
    spec = _spec("oversized_write")

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root: Path | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        self.workspace_root = workspace.root
        await workspace.write_bytes("large.bin", b"large")
        return ToolResult(content="written")


class _CancelledTool(Tool):
    spec = _spec("cancelled")

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root: Path | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        self.workspace_root = workspace.root
        raise asyncio.CancelledError


class _SlowTool(Tool):
    spec = _spec("slow")

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root: Path | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        self.workspace_root = workspace.root
        await asyncio.sleep(60)
        return ToolResult(content="late")


class _CancellationSwallowingTool(Tool):
    spec = _spec("cancellation_swallowing")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
        return ToolResult(content="late")


class _CallerCancellationErasingTool(Tool):
    spec = _spec("caller_cancellation_erasing")

    def __init__(self, started: asyncio.Event) -> None:
        super().__init__()
        self.started = started
        self.workspace_root: Path | None = None
        self.execution_task: asyncio.Task[ToolResult] | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        self.workspace_root = workspace.root
        task = asyncio.current_task()
        assert task is not None
        self.execution_task = task
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            task.uncancel()
            return ToolResult(content="cancellation erased")
        return ToolResult(content="unreachable")


class _BlockingTool(Tool):
    spec = _spec("blocking")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        time.sleep(0.03)
        return ToolResult(content="late")


class _LateMutatingTool(Tool):
    spec = _spec("late_mutation")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await ctx.workspace.write_bytes("late.txt", b"late")
        return ToolResult(content="late")


class _WhitespacePathTool(Tool):
    spec = _spec("whitespace_path")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.write_bytes(" report.txt ", b"created")
        return ToolResult(content="changed")


class _WhitespaceSeedMutationTool(Tool):
    spec = _spec("whitespace_seed_mutation")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.write_bytes(" updated.txt ", b"after")
        await ctx.workspace.delete(" deleted.txt ")
        return ToolResult(content="changed")


class _MetadataOnlyTool(Tool):
    spec = _spec("metadata_only")

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        stable = workspace.root / "stable.txt"
        (workspace.root / "empty").mkdir()
        os.symlink("stable.txt", workspace.root / "stable-link")
        stable.chmod(0o600)
        os.utime(stable, ns=(1_000_000_000, 1_000_000_000))
        return ToolResult(content="metadata changed")


class _EntryFloodTool(Tool):
    spec = _spec("entry_flood")

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root: Path | None = None

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        assert ctx.workspace is not None
        workspace = cast("LocalWorkspace", ctx.workspace)
        self.workspace_root = workspace.root
        for index in range(4):
            (workspace.root / f"empty-{index}").mkdir()
        return ToolResult(content="entries created")


def test_none_tool_reports_scoped_consistency_without_captured_content() -> None:
    tool = _PureTool()

    evidence = asyncio.run(
        verify_tool_effect(
            _app(tool),
            agent_name="worker",
            tool_name="pure",
            arguments={},
            workspace_files={"input.txt": b"unchanged"},
            unobserved_systems=("customer_api",),
            metadata={"tool_effect": "spoofed", "request_id": "request-1"},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.CONSISTENT
    assert evidence.declared_effect is ToolEffect.NONE
    assert evidence.observation_boundary == "isolated_workspace"
    assert evidence.observed_mutation is False
    assert evidence.execution_succeeded is True
    assert evidence.result_is_error is False
    assert evidence.exception_type is None
    assert evidence.timeout_seconds == 30.0
    assert evidence.workspace_max_entries == 2_000
    assert evidence.workspace_max_files == 1_000
    assert evidence.workspace_max_file_bytes == 16 * 1024 * 1024
    assert evidence.workspace_max_total_bytes == 64 * 1024 * 1024
    assert evidence.created_paths == ()
    assert "customer_api" in evidence.unobserved_systems
    assert "network_and_external_services" in evidence.unobserved_systems
    assert "unchanged" not in evidence.model_dump_json()

    assert tool.context is not None
    assert tool.context.agent_name == "worker"
    assert tool.context.environment_name == "isolated_workspace"
    assert tool.context.workspace_id == "isolated_workspace"
    assert tool.context.artifact_store is None
    assert tool.context.runner is None
    assert tool.context.vault is None
    assert tool.context.proxy is None
    assert tool.context.knowledge_store is None
    assert tool.context.metadata["tool_effect"] == ToolEffect.NONE.value
    assert tool.context.metadata["tool_call_id"] == "tool-effect-verification"
    assert tool.context.metadata["request_id"] == "request-1"


def test_none_tool_fails_scoped_verdict_on_create_update_and_delete() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_MutatingTool()),
            agent_name="worker",
            tool_name="mutate",
            arguments={},
            workspace_files={
                "updated.txt": b"before",
                "deleted.txt": b"delete me",
                "stable.txt": b"stable",
            },
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.MISMATCH
    assert evidence.observed_mutation is True
    assert evidence.execution_succeeded is True
    assert evidence.created_paths == ("created.txt",)
    assert evidence.updated_paths == ("updated.txt",)
    assert evidence.deleted_paths == ("deleted.txt",)


def test_workspace_paths_preserve_valid_leading_and_trailing_whitespace() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_WhitespacePathTool()),
            agent_name="worker",
            tool_name="whitespace_path",
            arguments={},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.MISMATCH
    assert evidence.execution_succeeded is True
    assert evidence.created_paths == (" report.txt ",)


def test_seeded_workspace_paths_preserve_valid_leading_and_trailing_whitespace() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_WhitespaceSeedMutationTool()),
            agent_name="worker",
            tool_name="whitespace_seed_mutation",
            arguments={},
            workspace_files={
                " updated.txt ": b"before",
                " deleted.txt ": b"delete me",
            },
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.MISMATCH
    assert evidence.execution_succeeded is True
    assert evidence.updated_paths == (" updated.txt ",)
    assert evidence.deleted_paths == (" deleted.txt ",)


@pytest.mark.skipif(os.name == "nt", reason="POSIX metadata and symlink semantics")
def test_empty_directories_symlinks_and_metadata_are_outside_the_observer() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_MetadataOnlyTool()),
            agent_name="worker",
            tool_name="metadata_only",
            arguments={},
            workspace_files={"stable.txt": b"stable"},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.CONSISTENT
    assert evidence.observed_mutation is False
    assert evidence.created_paths == ()
    assert any(
        "Empty directories, symlinks, non-regular entries" in item for item in evidence.limitations
    )


def test_none_mutation_remains_a_mismatch_when_the_tool_raises() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_MutateThenRaiseTool()),
            agent_name="worker",
            tool_name="mutate_then_raise",
            arguments={},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.MISMATCH
    assert evidence.execution_succeeded is False
    assert evidence.exception_type == "RuntimeError"
    assert evidence.created_paths == ("created.txt",)
    assert "test failure" not in evidence.model_dump_json()


def test_tool_error_without_mutation_reports_execution_failure() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_ErrorTool()),
            agent_name="worker",
            tool_name="error",
            arguments={},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.EXECUTION_FAILED
    assert evidence.execution_succeeded is False
    assert evidence.result_is_error is True
    assert evidence.exception_type is None
    assert evidence.observed_mutation is False


@pytest.mark.parametrize("effect", [ToolEffect.IDEMPOTENT, ToolEffect.EXTERNAL])
def test_effectful_tools_require_opt_in_and_are_never_presented_as_verified(
    effect: ToolEffect,
) -> None:
    tool = _EffectfulTool(effect)
    app = _app(tool)

    with pytest.raises(ValueError, match="allow_effectful_execution=True"):
        asyncio.run(
            verify_tool_effect(
                app,
                agent_name="worker",
                tool_name="effectful",
                arguments={},
            )
        )
    assert tool.run_count == 0

    evidence = asyncio.run(
        verify_tool_effect(
            app,
            agent_name="worker",
            tool_name="effectful",
            arguments={},
            idempotency_key="stable-operation",
            allow_effectful_execution=True,
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.OBSERVED
    assert evidence.declared_effect is effect
    assert evidence.observed_mutation is True
    assert evidence.created_paths == ("effect.txt",)
    assert tool.run_count == 1
    assert tool.idempotency_key == "stable-operation"


def test_verifier_uses_the_registered_effect_snapshot() -> None:
    tool = _PureTool()
    app = _app(tool)
    tool.spec = tool.spec.model_copy(update={"effect": ToolEffect.EXTERNAL})

    evidence = asyncio.run(
        verify_tool_effect(
            app,
            agent_name="worker",
            tool_name="pure",
            arguments={},
            workspace_files={"input.txt": b"unchanged"},
        )
    )

    assert evidence.declared_effect is ToolEffect.NONE
    assert evidence.status is ToolEffectVerificationStatus.CONSISTENT


def test_invalid_tool_result_is_reported_without_leaking_the_value() -> None:
    evidence = asyncio.run(
        verify_tool_effect(
            _app(_InvalidResultTool()),
            agent_name="worker",
            tool_name="invalid_result",
            arguments={},
        )
    )

    assert evidence.status is ToolEffectVerificationStatus.EXECUTION_FAILED
    assert evidence.exception_type == "InvalidToolResult"
    assert "not a ToolResult" not in evidence.model_dump_json()


def test_workspace_observation_bounds_fail_closed_and_cleanup() -> None:
    tool_before_execution = _PureTool()
    with pytest.raises(ValueError, match="max_file_bytes"):
        asyncio.run(
            verify_tool_effect(
                _app(tool_before_execution),
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files={"input.txt": b"unchanged"},
                max_file_bytes=1,
            )
        )
    assert tool_before_execution.context is None

    with pytest.raises(ValueError, match="max_files"):
        asyncio.run(
            verify_tool_effect(
                _app(_PureTool()),
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files={"input.txt": b"unchanged", "second.txt": b"second"},
                max_files=1,
            )
        )

    with pytest.raises(ValueError, match="max_total_bytes"):
        asyncio.run(
            verify_tool_effect(
                _app(_PureTool()),
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files={"input.txt": b"unchanged", "second.txt": b"second"},
                max_total_bytes=10,
            )
        )

    tool = _OversizedWriteTool()
    with pytest.raises(ValueError, match="max_file_bytes"):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="oversized_write",
                arguments={},
                max_file_bytes=1,
            )
        )
    assert tool.workspace_root is not None
    assert not tool.workspace_root.exists()

    entry_tool = _EntryFloodTool()
    with pytest.raises(ValueError, match="max_entries"):
        asyncio.run(
            verify_tool_effect(
                _app(entry_tool),
                agent_name="worker",
                tool_name="entry_flood",
                arguments={},
                max_entries=3,
            )
        )
    assert entry_tool.workspace_root is not None
    assert not entry_tool.workspace_root.exists()


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_entry_cap_stops_directory_and_symlink_heavy_traversal(
    entry_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if entry_kind == "symlink" and os.name == "nt":
        pytest.skip("symlink creation requires privileges on Windows")
    for index in range(10):
        path = tmp_path / f"entry-{index}"
        if entry_kind == "directory":
            path.mkdir()
        else:
            os.symlink("missing-target", path)

    original_scandir = os.scandir
    yielded_entries = [0]
    monkeypatch.setattr(
        testing.os,
        "scandir",
        lambda path: _CountingScandir(original_scandir(path), yielded_entries),
    )
    with pytest.raises(ValueError, match="max_entries"):
        testing._capture_workspace(
            LocalWorkspace(tmp_path),
            max_entries=3,
            max_files=100,
            max_file_bytes=1024,
            max_total_bytes=1024,
            clock=time.monotonic,
            deadline=float("inf"),
        )

    assert yielded_entries == [4]


def test_file_cap_stops_traversal_at_the_first_excess_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(10):
        (tmp_path / f"file-{index}.txt").write_bytes(b"content")

    original_scandir = os.scandir
    yielded_entries = [0]
    monkeypatch.setattr(
        testing.os,
        "scandir",
        lambda path: _CountingScandir(original_scandir(path), yielded_entries),
    )
    with pytest.raises(ValueError, match="max_files"):
        testing._capture_workspace(
            LocalWorkspace(tmp_path),
            max_entries=100,
            max_files=3,
            max_file_bytes=1024,
            max_total_bytes=1024,
            clock=time.monotonic,
            deadline=float("inf"),
        )

    assert yielded_entries == [4]


def test_overall_deadline_covers_workspace_seeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_seed = testing._seed_workspace
    tool = _PureTool()

    async def delayed_seed(*args: Any, **kwargs: Any) -> None:
        time.sleep(0.03)
        await original_seed(*args, **kwargs)

    monkeypatch.setattr(testing, "_seed_workspace", delayed_seed)
    with pytest.raises(TimeoutError):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files={"input.txt": b"unchanged"},
                timeout_seconds=0.01,
            )
        )

    assert tool.context is None


@pytest.mark.parametrize("delayed_capture", [1, 2], ids=["before", "after"])
def test_overall_deadline_is_forwarded_to_both_workspace_snapshots(
    delayed_capture: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_capture = testing._capture_workspace
    capture_count = 0
    observed_deadlines: list[float] = []
    tool = _PureTool()

    def delayed_snapshot(*args: Any, **kwargs: Any) -> dict[str, str]:
        nonlocal capture_count
        capture_count += 1
        deadline = kwargs["deadline"]
        observed_deadlines.append(deadline)
        if capture_count == delayed_capture:
            kwargs["clock"] = lambda: deadline
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(testing, "_capture_workspace", delayed_snapshot)
    with pytest.raises(TimeoutError):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files={"input.txt": b"unchanged"},
                timeout_seconds=10.0,
            )
        )

    assert capture_count == delayed_capture
    assert observed_deadlines == [observed_deadlines[0]] * delayed_capture
    assert (tool.context is not None) is (delayed_capture == 2)


def test_cancellation_propagates_and_cleans_the_workspace() -> None:
    tool = _CancelledTool()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="cancelled",
                arguments={},
            )
        )

    assert tool.workspace_root is not None
    assert not tool.workspace_root.exists()


def test_caller_cancellation_survives_child_tool_uncancel() -> None:
    async def exercise() -> _CallerCancellationErasingTool:
        started = asyncio.Event()
        tool = _CallerCancellationErasingTool(started)
        verification = asyncio.create_task(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="caller_cancellation_erasing",
                arguments={},
            )
        )
        await started.wait()
        verification.cancel()

        with pytest.raises(asyncio.CancelledError):
            await verification

        assert verification.cancelled()
        assert tool.execution_task is not None
        assert tool.execution_task is not verification
        return tool

    tool = asyncio.run(exercise())
    assert tool.workspace_root is not None
    assert not tool.workspace_root.exists()


def test_overall_deadline_raises_without_returning_incomplete_evidence() -> None:
    tool = _SlowTool()
    with pytest.raises(TimeoutError):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name="slow",
                arguments={},
                timeout_seconds=0.01,
            )
        )

    assert tool.workspace_root is not None
    assert not tool.workspace_root.exists()


@pytest.mark.parametrize(
    ("tool", "tool_name"),
    [
        (_CancellationSwallowingTool(), "cancellation_swallowing"),
        (_BlockingTool(), "blocking"),
        (_LateMutatingTool(), "late_mutation"),
    ],
)
def test_completion_after_deadline_never_returns_a_verdict(
    tool: Tool,
    tool_name: str,
) -> None:
    with pytest.raises(TimeoutError):
        asyncio.run(
            verify_tool_effect(
                _app(tool),
                agent_name="worker",
                tool_name=tool_name,
                arguments={},
                timeout_seconds=0.01,
            )
        )


def test_verification_rejects_unknown_tools_and_invalid_public_inputs() -> None:
    app = _app(_PureTool())

    with pytest.raises(KeyError, match="Tool not registered"):
        asyncio.run(
            verify_tool_effect(
                app,
                agent_name="worker",
                tool_name="missing",
                arguments={},
            )
        )
    with pytest.raises(TypeError, match="arguments must be a mapping"):
        asyncio.run(
            verify_tool_effect(
                app,
                agent_name="worker",
                tool_name="pure",
                arguments=cast("Any", []),
            )
        )
    with pytest.raises(TypeError, match="workspace_files values must be bytes"):
        asyncio.run(
            verify_tool_effect(
                app,
                agent_name="worker",
                tool_name="pure",
                arguments={},
                workspace_files=cast("Any", {"input.txt": "text"}),
            )
        )
    with pytest.raises(TypeError, match="unobserved_systems must be an iterable"):
        asyncio.run(
            verify_tool_effect(
                app,
                agent_name="worker",
                tool_name="pure",
                arguments={},
                unobserved_systems=cast("Any", "network"),
            )
        )
