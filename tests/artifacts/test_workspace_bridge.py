import asyncio

import pytest

from cayu.artifacts import (
    ArtifactScope,
    LocalArtifactStore,
    copy_artifact_to_workspace,
    copy_workspace_file_to_artifact,
)
from cayu.workspaces import LocalWorkspace


def _workspace(root, *, workspace_id: str | None = None) -> LocalWorkspace:
    root.mkdir()
    return LocalWorkspace(root, workspace_id=workspace_id)


def test_copy_artifact_to_workspace_is_explicit_one_way_copy(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    workspace = _workspace(tmp_path / "workspace", workspace_id="workspace")
    artifact = asyncio.run(
        store.put_bytes(
            b"invoice original",
            filename="invoice.txt",
            content_type="text/plain",
            session_id="sess_1",
            agent_name="assistant",
            environment_name="local-dev",
        )
    )

    result = asyncio.run(
        copy_artifact_to_workspace(
            store,
            workspace,
            artifact.id,
            "inputs/invoice.txt",
        )
    )
    asyncio.run(workspace.write_bytes("inputs/invoice.txt", b"workspace edit"))

    workspace_read = asyncio.run(workspace.read_bytes("inputs/invoice.txt"))
    artifact_read = asyncio.run(store.read_bytes(artifact.id))

    assert result.artifact == artifact
    assert result.workspace_path == "inputs/invoice.txt"
    assert result.bytes_written == len(b"invoice original")
    assert result.truncated is False
    assert workspace_read.content == b"workspace edit"
    assert artifact_read.content == b"invoice original"


def test_copy_artifact_to_workspace_rejects_partial_copy_by_default(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    workspace = _workspace(tmp_path / "workspace")
    artifact = asyncio.run(
        store.put_bytes(
            b"abcdef",
            filename="large.txt",
            session_id="sess_1",
        )
    )

    with pytest.raises(ValueError, match="partial workspace copy"):
        asyncio.run(
            copy_artifact_to_workspace(
                store,
                workspace,
                artifact.id,
                "inputs/large.txt",
                max_bytes=3,
            )
        )

    with pytest.raises(FileNotFoundError):
        asyncio.run(workspace.read_bytes("inputs/large.txt"))


def test_copy_artifact_to_workspace_can_allow_truncated_copy(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    workspace = _workspace(tmp_path / "workspace")
    artifact = asyncio.run(
        store.put_bytes(
            b"abcdef",
            filename="large.txt",
            session_id="sess_1",
        )
    )

    result = asyncio.run(
        copy_artifact_to_workspace(
            store,
            workspace,
            artifact.id,
            "inputs/large.txt",
            max_bytes=3,
            allow_truncated=True,
        )
    )
    workspace_read = asyncio.run(workspace.read_bytes("inputs/large.txt"))

    assert result.bytes_written == 3
    assert result.truncated is True
    assert workspace_read.content == b"abc"


def test_copy_workspace_file_to_artifact_stores_new_durable_artifact(tmp_path):
    workspace = _workspace(tmp_path / "workspace", workspace_id="workspace")
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    asyncio.run(workspace.write_bytes("results/summary.json", b'{"ok": true}'))

    result = asyncio.run(
        copy_workspace_file_to_artifact(
            workspace,
            store,
            "results/summary.json",
            scope=ArtifactScope.SESSION,
            session_id="sess_1",
            agent_name="assistant",
            environment_name="local-dev",
            metadata={"kind": "analysis_output"},
        )
    )
    artifact_read = asyncio.run(store.read_bytes(result.artifact.id))

    assert result.workspace_path == "results/summary.json"
    assert result.bytes_read == len(b'{"ok": true}')
    assert result.truncated is False
    assert result.artifact.filename == "summary.json"
    assert result.artifact.content_type == "application/json"
    assert result.artifact.metadata["kind"] == "analysis_output"
    assert result.artifact.metadata["source"] == "workspace"
    assert result.artifact.metadata["source_workspace_id"] == "workspace"
    assert result.artifact.metadata["source_workspace_path"] == "results/summary.json"
    assert result.artifact.metadata["source_workspace_total_bytes"] == len(b'{"ok": true}')
    assert result.artifact.metadata["source_workspace_truncated"] is False
    assert artifact_read.content == b'{"ok": true}'


def test_copy_workspace_file_to_artifact_rejects_partial_artifact_by_default(tmp_path):
    workspace = _workspace(tmp_path / "workspace")
    store = LocalArtifactStore(tmp_path / "artifacts")
    asyncio.run(workspace.write_bytes("results/large.txt", b"abcdef"))

    with pytest.raises(ValueError, match="partial artifact"):
        asyncio.run(
            copy_workspace_file_to_artifact(
                workspace,
                store,
                "results/large.txt",
                session_id="sess_1",
                max_bytes=3,
            )
        )

    artifacts = asyncio.run(store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    assert artifacts.artifacts == ()


def test_copy_workspace_file_to_artifact_can_allow_truncated_artifact(tmp_path):
    workspace = _workspace(tmp_path / "workspace")
    store = LocalArtifactStore(tmp_path / "artifacts")
    asyncio.run(workspace.write_bytes("results/large.txt", b"abcdef"))

    result = asyncio.run(
        copy_workspace_file_to_artifact(
            workspace,
            store,
            "results/large.txt",
            session_id="sess_1",
            max_bytes=3,
            allow_truncated=True,
        )
    )
    artifact_read = asyncio.run(store.read_bytes(result.artifact.id))

    assert result.bytes_read == 3
    assert result.truncated is True
    assert result.artifact.metadata["source_workspace_total_bytes"] == 6
    assert result.artifact.metadata["source_workspace_truncated"] is True
    assert artifact_read.content == b"abc"


def test_workspace_bridge_rejects_invalid_services(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    workspace = _workspace(tmp_path / "workspace")

    with pytest.raises(TypeError, match="artifact_store"):
        asyncio.run(
            copy_artifact_to_workspace(
                object(),  # type: ignore[arg-type]
                workspace,
                "art_1",
                "file.txt",
            )
        )

    with pytest.raises(TypeError, match="workspace"):
        asyncio.run(
            copy_workspace_file_to_artifact(
                object(),  # type: ignore[arg-type]
                store,
                "file.txt",
                session_id="sess_1",
            )
        )
