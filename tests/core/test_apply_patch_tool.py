from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from cayu import (
    ApplyPatchTool,
    LocalArtifactStore,
    LocalWorkspace,
    ToolContext,
    ToolEffect,
    WorkspaceMoveAmbiguousError,
    WorkspaceMutationResult,
)
from cayu.core.tools import (
    DurableToolRecoveryAuthority,
    _bind_runtime_tool_invocation_authority,
)
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import (
    InvocationWorkspaceMutationOwner,
    invocation_workspace_handle,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _revision(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _context(tmp_path: Path, workspace: LocalWorkspace | None = None) -> ToolContext:
    selected = workspace or LocalWorkspace(tmp_path / "workspace", workspace_id="workspace")
    return ToolContext(
        session_id="session",
        workspace_id=selected.id,
        workspace=selected,
    )


def test_apply_patch_schema_and_effect_contract() -> None:
    tool = ApplyPatchTool()

    assert tool.spec.effect is ToolEffect.EXTERNAL
    assert tool.spec.workspace_mutation is True
    assert tool.spec.parallel_safe is False
    assert tool._publish_arguments is False
    assert tool.schema["additionalProperties"] is False
    operations = tool.schema["properties"]["operations"]
    assert operations["minItems"] == 1
    assert operations["maxItems"] == 100
    assert operations["items"]["additionalProperties"] is False
    assert operations["items"]["properties"]["edits"]["items"]["additionalProperties"] is False


def test_apply_patch_applies_create_two_updates_move_and_delete(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    originals = {
        "a.txt": b"alpha = 1\n",
        "b.txt": b"beta = 2\n",
        "legacy.txt": b"legacy\n",
        "obsolete.txt": b"obsolete\n",
    }
    for path, content in originals.items():
        (root / path).write_bytes(content)
    ctx = _context(tmp_path, LocalWorkspace(root, workspace_id="workspace"))

    result = asyncio.run(
        ApplyPatchTool().run(
            ctx,
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(originals["a.txt"]),
                        "edits": [{"old_text": "alpha = 1", "new_text": "alpha = 10"}],
                    },
                    {
                        "type": "update",
                        "path": "b.txt",
                        "expected_revision": _revision(originals["b.txt"]),
                        "edits": [{"old_text": "beta = 2", "new_text": "beta = 20"}],
                    },
                    {"type": "create", "path": "created.txt", "content": "created\n"},
                    {
                        "type": "move",
                        "from_path": "legacy.txt",
                        "to_path": "current.txt",
                        "expected_revision": _revision(originals["legacy.txt"]),
                    },
                    {
                        "type": "delete",
                        "path": "obsolete.txt",
                        "expected_revision": _revision(originals["obsolete.txt"]),
                    },
                ]
            },
        )
    )

    assert result.is_error is False
    assert result.structured["outcome"] == "applied"
    assert result.structured["cross_file_atomic"] is False
    assert result.structured["application_plan"] == [3, 2, 0, 1, 4]
    assert [item["status"] for item in result.structured["operations"]] == [
        "applied",
        "applied",
        "applied",
        "applied",
        "applied",
    ]
    assert result.structured["operations"][3]["move_fidelity"] == "link_unlink"
    assert (root / "a.txt").read_text() == "alpha = 10\n"
    assert (root / "b.txt").read_text() == "beta = 20\n"
    assert (root / "created.txt").read_text() == "created\n"
    assert (root / "current.txt").read_text() == "legacy\n"
    assert not (root / "legacy.txt").exists()
    assert not (root / "obsolete.txt").exists()
    assert "rename from legacy.txt" in result.structured["diff"]


@pytest.mark.parametrize(
    ("operations", "category"),
    [
        (
            [
                {"type": "create", "path": "same.txt", "content": "one"},
                {"type": "create", "path": "SAME.txt", "content": "two"},
            ],
            "move_chain_or_path_collision",
        ),
        (
            [{"type": "create", "path": ".git/config", "content": "unsafe"}],
            "protected_path",
        ),
        (
            [{"type": "create", "path": "C:/host.txt", "content": "unsafe"}],
            "absolute_path",
        ),
        (
            [
                {
                    "type": "move",
                    "from_path": "a",
                    "to_path": "b",
                    "expected_revision": "x",
                    "edits": [],
                }
            ],
            "invalid_operation_shape",
        ),
    ],
)
def test_apply_patch_graph_refusals_write_nothing(
    tmp_path: Path,
    operations: list[dict[str, object]],
    category: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    before = tuple(root.iterdir())

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, LocalWorkspace(root, workspace_id="workspace")),
            {"operations": operations},
        )
    )

    assert result.is_error is True
    assert result.structured["outcome"] == "precondition_failed"
    assert result.structured["category"] == category
    assert result.structured["mutated"] is False
    assert tuple(root.iterdir()) == before


def test_apply_patch_preflights_every_update_before_first_write(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    first = b"first = old\n"
    second = b"second = old\n"
    (root / "a.txt").write_bytes(first)
    (root / "b.txt").write_bytes(second)

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, LocalWorkspace(root, workspace_id="workspace")),
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(first),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    },
                    {
                        "type": "update",
                        "path": "b.txt",
                        "expected_revision": _revision(second),
                        "edits": [
                            {
                                "old_text": "missing",
                                "new_text": "new",
                                "expected_replacements": 1,
                            }
                        ],
                    },
                ]
            },
        )
    )

    assert result.structured["category"] == "replacement_count_mismatch"
    assert (root / "a.txt").read_bytes() == first
    assert (root / "b.txt").read_bytes() == second


class _SecondReplaceConflictsWorkspace(LocalWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, workspace_id="workspace")
        self.replace_calls = 0

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        self.replace_calls += 1
        if self.replace_calls == 2:
            await self.write_bytes(path, b"concurrent\n")
        return await super().replace_bytes(
            path,
            content,
            expected_revision=expected_revision,
        )


def test_apply_patch_reports_mid_plan_conflict_as_partial(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    first = b"first = old\n"
    second = b"second = old\n"
    (root / "a.txt").write_bytes(first)
    (root / "b.txt").write_bytes(second)
    workspace = _SecondReplaceConflictsWorkspace(root)

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, workspace),
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(first),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    },
                    {
                        "type": "update",
                        "path": "b.txt",
                        "expected_revision": _revision(second),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    },
                ]
            },
        )
    )

    assert result.structured["outcome"] == "partial"
    assert [item["status"] for item in result.structured["operations"]] == [
        "applied",
        "conflict",
    ]
    assert result.structured["requires_fresh_read"] is True
    assert (root / "a.txt").read_text() == "first = new\n"
    assert (root / "b.txt").read_text() == "concurrent\n"


class _ChangesBeforeFirstMutationWorkspace(LocalWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, workspace_id="workspace")
        self.scheduled = False

    async def read_bytes(self, path: str, **kwargs):
        result = await super().read_bytes(path, **kwargs)
        if path == "a.txt" and not self.scheduled:
            self.scheduled = True
            asyncio.get_running_loop().call_soon(
                (self.root / "a.txt").write_bytes,
                b"concurrent\n",
            )
        return result


def test_apply_patch_concurrent_change_before_first_mutation_is_precondition_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    original = b"value = old\n"
    (root / "a.txt").write_bytes(original)
    workspace = _ChangesBeforeFirstMutationWorkspace(root)

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, workspace),
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(original),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    }
                ]
            },
        )
    )

    assert result.structured["outcome"] == "precondition_failed"
    assert result.structured["operations"][0]["status"] == "conflict"
    assert (root / "a.txt").read_bytes() == b"concurrent\n"


class _CancelsAfterFirstReplaceWorkspace(LocalWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, workspace_id="workspace")
        self.replace_calls = 0

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        result = await super().replace_bytes(
            path,
            content,
            expected_revision=expected_revision,
        )
        self.replace_calls += 1
        if self.replace_calls == 1:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
        return result


def test_apply_patch_cancellation_between_operations_stops_remaining_plan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    first = b"first = old\n"
    second = b"second = old\n"
    (root / "a.txt").write_bytes(first)
    (root / "b.txt").write_bytes(second)
    workspace = _CancelsAfterFirstReplaceWorkspace(root)

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, workspace),
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(first),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    },
                    {
                        "type": "update",
                        "path": "b.txt",
                        "expected_revision": _revision(second),
                        "edits": [{"old_text": "old", "new_text": "new"}],
                    },
                ]
            },
        )
    )

    assert result.structured["outcome"] == "cancelled"
    assert [item["status"] for item in result.structured["operations"]] == [
        "applied",
        "not_started",
    ]
    assert (root / "a.txt").read_text() == "first = new\n"
    assert (root / "b.txt").read_bytes() == second


class _BlockedThreadReplaceWorkspace(LocalWorkspace):
    def __init__(self, root: Path) -> None:
        super().__init__(root, workspace_id="workspace")
        self.entered = threading.Event()
        self.release = threading.Event()

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        def mutate() -> WorkspaceMutationResult:
            before = (self.root / path).read_bytes()
            if _revision(before) != expected_revision:
                raise AssertionError("test mutation received a stale revision")
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test mutation was not released")
            (self.root / path).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            return WorkspaceMutationResult(
                operation="replace",
                before_revision=expected_revision,
                after_revision=f"sha256:{digest}",
                before_sha256=hashlib.sha256(before).hexdigest(),
                after_sha256=digest,
                before_bytes=len(before),
                after_bytes=len(content),
            )

        return await asyncio.to_thread(mutate)


def test_apply_patch_mutation_facade_waits_for_cancelled_dispatch_and_records_receipt(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        original = b"value = old\n"
        updated = b"value = new\n"
        (root / "a.txt").write_bytes(original)
        raw_workspace = _BlockedThreadReplaceWorkspace(root)
        receipts: list[tuple[str, str, object]] = []
        snapshot = InvocationRedactorSnapshot(0, SecretRedactor())
        facade = invocation_workspace_handle(
            raw_workspace,
            redactor_snapshot_provider=lambda: snapshot,
            capture_observer=lambda _revision: None,
            mutation_owner=InvocationWorkspaceMutationOwner(),
            direct_mutation_observer=lambda method, path, result: receipts.append(
                (method, path, result)
            ),
        )
        assert facade is not None
        ctx = ToolContext(
            session_id="session",
            workspace_id="workspace",
            workspace=facade,
        )
        ctx._bind_runtime_resource_authorities(workspace=raw_workspace, artifact_store=None)
        task = asyncio.create_task(
            ApplyPatchTool().run(
                ctx,
                {
                    "operations": [
                        {
                            "type": "update",
                            "path": "a.txt",
                            "expected_revision": _revision(original),
                            "edits": [{"old_text": "old", "new_text": "new"}],
                        }
                    ]
                },
            )
        )
        await asyncio.wait_for(asyncio.to_thread(raw_workspace.entered.wait), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        raw_workspace.release.set()
        result = await task

        assert result.structured["outcome"] == "cancelled"
        assert result.structured["operations"][0]["status"] == "applied"
        assert (root / "a.txt").read_bytes() == updated
        assert [(method, path) for method, path, _result in receipts] == [
            ("replace_bytes", "a.txt")
        ]

    asyncio.run(scenario())


class _AmbiguousMoveWorkspace(LocalWorkspace):
    async def move_if_revision(self, *args, **kwargs):
        result = await super().move_if_revision(*args, **kwargs)
        raise WorkspaceMoveAmbiguousError(result)


def test_apply_patch_reports_lost_move_acknowledgement_as_ambiguous(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    content = b"move me\n"
    (root / "source.txt").write_bytes(content)
    workspace = _AmbiguousMoveWorkspace(root, workspace_id="workspace")

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, workspace),
            {
                "operations": [
                    {
                        "type": "move",
                        "from_path": "source.txt",
                        "to_path": "destination.txt",
                        "expected_revision": _revision(content),
                    }
                ]
            },
        )
    )

    assert result.structured["outcome"] == "ambiguous"
    assert result.structured["operations"][0]["status"] == "unknown"
    assert result.structured["requires_fresh_read"] is True
    assert not (root / "source.txt").exists()
    assert (root / "destination.txt").read_bytes() == content


def test_apply_patch_redacts_diff_and_offloads_full_safe_manifest(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    secret = "patch-secret-canary-ABCDEFGHIJKLMNOP"
    original = (secret + "\nvalue = old\n" + ("x" * 400) + "\n").encode()
    (root / "a.txt").write_bytes(original)
    artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="session",
        workspace_id="workspace",
        workspace=LocalWorkspace(root, workspace_id="workspace"),
        artifact_store_id="artifacts",
        artifact_store=artifacts,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        ApplyPatchTool(
            max_diff_preview_bytes=128,
            max_file_diff_preview_bytes=128,
        ).run(
            ctx,
            {
                "operations": [
                    {
                        "type": "update",
                        "path": "a.txt",
                        "expected_revision": _revision(original),
                        "edits": [{"old_text": "value = old", "new_text": "value = new"}],
                    }
                ]
            },
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert result.structured["outcome"] == "applied"
    assert result.structured["diff_truncated"] is True
    assert result.structured["artifact_status"] == "stored"
    assert result.structured["artifact"]["artifact_id"].startswith("art_")
    assert secret not in rendered
    assert REDACTED_SECRET in rendered


class _RecordingArtifactStore(LocalArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root, store_id="artifacts")
        self.put_calls = 0

    async def put_bytes(self, *args, **kwargs):
        self.put_calls += 1
        return await super().put_bytes(*args, **kwargs)


def test_apply_patch_seals_dynamic_secret_scope_before_artifact_storage(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    long_path = "/".join(["segment-" + ("x" * 92)] * 6) + "/created.txt"
    arguments = {"operations": [{"type": "create", "path": long_path, "content": "created\n"}]}
    artifacts = _RecordingArtifactStore(tmp_path / "artifacts")
    workspace = LocalWorkspace(root, workspace_id="workspace")
    tracker = InvocationSecretTracker(SecretRedactor())
    tracker.begin_resolution()
    records: dict[str, dict[str, object]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        assert secondary == {}
        records[storage_key] = desired
        return desired

    ctx = ToolContext(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        artifact_store_id="artifacts",
        idempotency_key="tool-invocation",
        workspace=workspace,
        artifact_store=artifacts,
        invocation_secret_snapshot_provider=tracker.snapshot,
        invocation_secret_capture_observer=tracker.record_ambiguous_output_capture,
    )
    ctx._bind_runtime_resource_authorities(workspace=workspace, artifact_store=artifacts)
    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=3,
        model_step_id="mstep_00000000000000000000000000000000",
        model_attempt_id="mattempt_00000000000000000000000000000000",
        tool_round_id="tround_00000000000000000000000000000000",
        tool_call_id="call",
        tool_name="apply_patch",
        idempotency_key="tool-invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=tracker.seal_for_publication,
    )

    result = asyncio.run(ApplyPatchTool().run(ctx, arguments))

    assert result.is_error is True
    assert result.structured["error"] == "secret_redaction_scope_unstable"
    assert result.structured["workspace_outcome"] == "applied"
    assert artifacts.put_calls == 0
    assert (root / long_path).read_text() == "created\n"


class _AdvancingRedactorArtifactStore(_RecordingArtifactStore):
    def __init__(
        self,
        root: Path,
        *,
        redactor_state: list[SecretRedactor],
        secret: str,
    ) -> None:
        super().__init__(root)
        self.redactor_state = redactor_state
        self.secret = secret

    async def put_bytes(self, *args, **kwargs):
        artifact = await super().put_bytes(*args, **kwargs)
        self.redactor_state[0] = SecretRedactor(self.secret)
        return artifact


def test_apply_patch_removes_artifact_when_static_redactor_changes_during_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    secret = "late-patch-secret-ABCDEFGHIJKLMNOP"
    redactor_state = [SecretRedactor()]
    artifacts = _AdvancingRedactorArtifactStore(
        tmp_path / "artifacts",
        redactor_state=redactor_state,
        secret=secret,
    )
    long_path = "/".join(["segment-" + ("x" * 92)] * 6) + "/created.txt"
    ctx = ToolContext(
        session_id="session",
        workspace_id="workspace",
        workspace=LocalWorkspace(root, workspace_id="workspace"),
        artifact_store_id="artifacts",
        artifact_store=artifacts,
        invocation_secret_redactor=lambda: redactor_state[0],
    )

    result = asyncio.run(
        ApplyPatchTool().run(
            ctx,
            {"operations": [{"type": "create", "path": long_path, "content": secret}]},
        )
    )

    assert result.is_error is True
    assert result.structured["error"] == "secret_redaction_scope_unstable"
    assert result.structured["workspace_outcome"] == "applied"
    assert artifacts.put_calls == 1
    assert asyncio.run(artifacts.list()).artifacts == ()


def test_apply_patch_bounds_long_path_projection_and_offloads_manifest(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    long_path = "/".join(["segment-" + ("x" * 92)] * 6) + "/created.txt"
    artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="session",
        workspace_id="workspace",
        workspace=LocalWorkspace(root, workspace_id="workspace"),
        artifact_store_id="artifacts",
        artifact_store=artifacts,
    )

    result = asyncio.run(
        ApplyPatchTool().run(
            ctx,
            {"operations": [{"type": "create", "path": long_path, "content": "created\n"}]},
        )
    )

    operation = result.structured["operations"][0]
    assert result.structured["outcome"] == "applied"
    assert result.structured["manifest_truncated"] is True
    assert result.structured["manifest_truncation_reasons"] == ["path_bytes"]
    assert result.structured["artifact_status"] == "stored"
    assert operation["destination_path"].endswith("[path truncated]")
    assert len(operation["destination_path"].encode()) <= 512
    assert len(operation["destination_path_sha256"]) == 64
    assert len(json.dumps(result.model_dump(mode="json")["structured"]).encode()) <= 1024 * 1024
    assert (root / long_path).read_text() == "created\n"


def test_apply_patch_profile_identity_binds_limits_and_protected_paths() -> None:
    default = ApplyPatchTool()
    changed_limit = ApplyPatchTool(max_file_bytes=1024)
    changed_paths = ApplyPatchTool(protected_entry_names=(".git", ".private"))

    assert default.behavior_profile_id != changed_limit.behavior_profile_id
    assert default.behavior_profile_id != changed_paths.behavior_profile_id
    assert default._execution_profile_material()["cross_file_atomic"] is False


def test_apply_patch_refuses_hard_link_move_without_mutating_names(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    content = b"shared\n"
    (root / "source.txt").write_bytes(content)
    os.link(root / "source.txt", root / "alias.txt")

    result = asyncio.run(
        ApplyPatchTool().run(
            _context(tmp_path, LocalWorkspace(root, workspace_id="workspace")),
            {
                "operations": [
                    {
                        "type": "move",
                        "from_path": "source.txt",
                        "to_path": "destination.txt",
                        "expected_revision": _revision(content),
                    }
                ]
            },
        )
    )

    assert result.structured["outcome"] == "unsupported"
    assert result.structured["operations"][0]["status"] == "failed"
    assert (root / "source.txt").read_bytes() == content
    assert (root / "alias.txt").read_bytes() == content
    assert not (root / "destination.txt").exists()


def test_apply_patch_durable_recovery_reconstructs_without_replay_and_refuses_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    original = b"value = old\n" * 101
    updated = b"value = new\n" * 101
    (root / "a.txt").write_bytes(original)
    workspace = LocalWorkspace(root, workspace_id="workspace")
    arguments = {
        "operations": [
            {
                "type": "update",
                "path": "a.txt",
                "expected_revision": _revision(original),
                "edits": [
                    {
                        "old_text": "old",
                        "new_text": "new",
                        "expected_replacements": 101,
                    }
                ],
            }
        ]
    }
    records: dict[str, dict[str, object]] = {}

    async def load_operation(storage_key: str):
        return records.get(storage_key)

    async def compare_and_set_operation(storage_key, expected, desired, secondary):
        assert records.get(storage_key) == expected
        assert secondary == {}
        records[storage_key] = desired
        return desired

    ctx = ToolContext(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        workspace_id="workspace",
        idempotency_key="tool-invocation",
        workspace=workspace,
        metadata={"tool_call_id": "call"},
    )
    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task",
        parent_run_epoch=3,
        model_step_id="mstep_00000000000000000000000000000000",
        model_attempt_id="mattempt_00000000000000000000000000000000",
        tool_round_id="tround_00000000000000000000000000000000",
        tool_call_id="call",
        tool_name="apply_patch",
        idempotency_key="tool-invocation",
        effective_arguments=arguments,
        execution_profile_fingerprint="profile",
        environment_allocation_fingerprint="allocation",
        load_durable_operation=load_operation,
        compare_and_set_durable_operation=compare_and_set_operation,
        seal_durable_output=lambda record: record,
        secret_publication_sealer=lambda: None,
    )
    tool = ApplyPatchTool()
    applied = asyncio.run(tool.run(ctx, arguments))
    assert applied.structured["outcome"] == "applied"
    assert applied.structured["operations"][0]["replacement_count"] == 101
    assert (root / "a.txt").read_bytes() == updated

    journal = next(iter(records.values()))
    journal["state"] = "applying"
    journal["active_operation"] = 0
    journal["terminal_outcome"] = None
    journal["failure_category"] = None
    operation = journal["operations"][0]
    operation["status"] = "unknown"
    operation["after_revision_sha256"] = None
    operation["after_sha256"] = None
    operation["after_bytes"] = None

    recovery_authority = DurableToolRecoveryAuthority(
        agent_name="agent",
        environment_name="coding",
        workspace=workspace,
        artifact_reader=None,
        compare_and_set_operation=compare_and_set_operation,
    )

    recovery_arguments = {
        "parent_session_id": "session",
        "parent_run_epoch": 3,
        "execution_profile_fingerprint": "profile",
        "environment_name": "coding",
        "environment_allocation_fingerprint": "allocation",
        "model_step_id": "mstep_00000000000000000000000000000000",
        "model_attempt_id": "mattempt_00000000000000000000000000000000",
        "tool_round_id": "tround_00000000000000000000000000000000",
        "tool_call_id": "call",
        "idempotency_key": "tool-invocation",
        "arguments": arguments,
        "started": True,
        "load_operation": load_operation,
        "recovery_authority": recovery_authority,
    }
    recovered = asyncio.run(tool.reconcile_durable_tool_call(**recovery_arguments))
    assert recovered is not None
    assert recovered.structured["outcome"] == "applied"
    assert recovered.structured["recovered"] is True
    assert recovered.structured["operations"][0]["replacement_count"] == 101
    assert recovered.structured["operations"][0]["observed_source_state"] == "present"
    assert (
        recovered.structured["operations"][0]["observed_source_sha256"]
        == hashlib.sha256(updated).hexdigest()
    )
    assert next(iter(records.values()))["state"] == "terminal"
    assert "not replayed" in recovered.content
    assert (root / "a.txt").read_bytes() == updated

    drifted = asyncio.run(
        tool.reconcile_durable_tool_call(
            **{**recovery_arguments, "execution_profile_fingerprint": "changed-profile"}
        )
    )
    assert drifted is not None
    assert drifted.structured["outcome"] == "unsupported"
    assert drifted.structured["failure_category"] == "execution_profile_drift"
