"""HTTP server inventory contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
import base64
import logging
import warnings
from typing import Any, cast

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, OneShotProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
)
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.environments.lifecycle import EnvironmentLifecyclePolicy
from cayu.server import create_server
from cayu.tools.commands import ExecCommandTool
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor
from cayu.workspaces.branches import (
    WorkspaceBranch,
    WorkspaceBranchCapabilities,
    WorkspaceBranchRequest,
)
from cayu.workspaces.local import LocalWorkspace


class CountingArtifactStore(ArtifactStore):
    id = "counting-artifacts"

    async def put_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        raise NotImplementedError

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        raise FileNotFoundError(artifact_id)

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        requested = min(limit or 0, 10_500)
        artifacts = tuple(
            ArtifactMetadata(
                id=f"artifact_{index:05d}",
                filename=f"artifact-{index:05d}.txt",
                content_type="text/plain",
                size_bytes=0,
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="local-review",
            )
            for index in range(requested)
        )
        return ArtifactListResult(
            artifacts=artifacts,
            total_count=20_000,
            truncated=True,
        )

    async def delete(self, artifact_id: str) -> None:
        return None


class InvalidArtifactDataStore(CountingArtifactStore):
    id = "invalid-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        metadata = ArtifactMetadata.model_construct(
            id=artifact_id,
            filename="invalid.txt",
            content_type="text/\ud800",
            size_bytes=0,
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            metadata={},
        )
        return ArtifactReadResult(metadata=metadata, content=b"", total_bytes=0)


class WrongArtifactDataStore(CountingArtifactStore):
    id = "wrong-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        content = b"different artifact content"
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id="different-artifact",
                filename="different.txt",
                content_type="text/plain",
                size_bytes=len(content),
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            ),
            content=content,
            total_bytes=len(content),
        )


class OverreadArtifactDataStore(CountingArtifactStore):
    id = "overread-artifact-data"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        content = b"store ignored the requested limit"
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id=artifact_id,
                filename="overread.txt",
                content_type="text/plain",
                size_bytes=len(content),
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            ),
            content=content,
            total_bytes=len(content),
        )


class UnavailableArtifactStore(CountingArtifactStore):
    id = "unavailable-artifacts"

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        raise PermissionError("Artifact backend is unavailable.")

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        raise PermissionError("Artifact backend is unavailable.")


class InvalidArtifactListStore(CountingArtifactStore):
    id = "invalid-artifact-list"

    async def list(
        self,
        *,
        scope: ArtifactScope | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        environment_name: str | None = None,
        limit: int | None = None,
    ) -> ArtifactListResult:
        return cast("ArtifactListResult", None)


def test_environment_capability_projection_preserves_controls_and_redacts_detail(
    tmp_path,
) -> None:
    canary = "extension-capability-secret"

    class DetailWorkspace(LocalWorkspace):
        def branch_capabilities(self) -> WorkspaceBranchCapabilities:
            return WorkspaceBranchCapabilities(detail_code=canary)

    app = CayuApp(
        secret_redactor=SecretRedactor([canary, "unsupported"]),
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="detail-workspace"),
            workspace=DetailWorkspace(tmp_path),
        ),
        default=True,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.get("/api/environments")

    assert response.status_code == 200
    capabilities = response.json()["environments"][0]["workspace_branch_capabilities"]
    assert capabilities["publication"] == "unsupported"
    assert capabilities["recovery"] == "unsupported"
    assert capabilities["retention"] == "unsupported"
    assert capabilities["detail_code"] == "[REDACTED_SECRET]"
    assert canary not in response.text


def test_environment_capability_projection_rejects_mutated_evidence_safely(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "mutated-server-capability-secret"

    class SecretBearingWrongType:
        def __repr__(self) -> str:
            return canary

        def __str__(self) -> str:
            return canary

    candidate = WorkspaceBranchCapabilities()
    object.__setattr__(candidate, "detail_code", SecretBearingWrongType())

    class MutatedCapabilityWorkspace(LocalWorkspace):
        def branch_capabilities(self) -> WorkspaceBranchCapabilities:
            return candidate

    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="mutated-capability"),
            workspace=MutatedCapabilityWorkspace(tmp_path),
        ),
        default=True,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        response = client.get("/api/environments")

    captured = capsys.readouterr()
    assert response.status_code == 200
    assert response.json()["environments"][0]["workspace_branch_capabilities"]["detail_code"] == (
        "workspace_branch_capability_evidence_invalid"
    )
    diagnostic_text = "\n".join(
        [
            response.text,
            repr(response),
            *(str(item.message) for item in caught),
            caplog.text,
            captured.out,
            captured.err,
        ]
    )
    assert canary not in diagnostic_text


def test_server_exposes_agent_environment_and_artifact_inventory(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root)

    async def attach_branch() -> WorkspaceBranch:
        from cayu.workspaces.revisions import (
            WorkspaceRevisionObservationLimits,
            observe_deterministic_workspace,
        )

        observation = await observe_deterministic_workspace(
            workspace,
            observer="server-lifecycle-test",
            limits=WorkspaceRevisionObservationLimits(),
        )
        created = await workspace.create_branch(WorkspaceBranchRequest(baseline=observation))
        assert created.branch is not None
        return created.branch

    attached_branch = asyncio.run(attach_branch())
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="fake-model",
            metadata={"team": "platform"},
            provider_options={"temperature": 0},
            system_prompt="Review runtime state.",
        ),
        tools=[UserInputTool(), ExecCommandTool()],
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="local-review",
                metadata={"tenant": "test"},
                lifecycle_policy=EnvironmentLifecyclePolicy(lifecycle_timeout_seconds=20.0),
            ),
            artifact_store=artifact_store,
            workspace=workspace,
            workspace_instructions="Use local workspace instructions.",
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment log\nstatus=ok\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"source": "test"},
        )
    )

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    agents = client.get("/api/agents")
    assert agents.status_code == 200
    agents_body = agents.json()
    assert agents_body["total_count"] == 1
    assert agents_body["agents"][0]["name"] == "reviewer"
    assert agents_body["agents"][0]["metadata"] == {"team": "platform"}
    assert agents_body["agents"][0]["has_system_prompt"] is True
    tools = {tool["name"]: tool for tool in agents_body["agents"][0]["tools"]}
    assert set(tools) == {"ask_user", "exec_command"}
    assert tools["ask_user"]["workspace_mutation"] is False
    assert tools["exec_command"]["workspace_mutation"] is True
    agent_detail = client.get("/api/agents/reviewer")
    assert agent_detail.status_code == 200
    detail_tools = {tool["name"]: tool for tool in agent_detail.json()["agents"][0]["tools"]}
    assert detail_tools["ask_user"]["workspace_mutation"] is False
    assert detail_tools["exec_command"]["workspace_mutation"] is True

    environments = client.get("/api/environments")
    assert environments.status_code == 200
    environments_body = environments.json()
    assert environments_body["total_count"] == 1
    assert environments_body["environments"][0]["name"] == "local-review"
    assert environments_body["environments"][0]["artifact_store_id"] == "test-artifacts"
    assert environments_body["environments"][0]["workspace_instructions"] == "inline"
    assert (
        environments_body["environments"][0]["lifecycle_policy"]["lifecycle_timeout_seconds"]
        == 20.0
    )
    assert environments_body["environments"][0]["workspace_branch_capabilities"] == {
        "detail_code": "process_local_workspace_branches",
        "isolation": True,
        "lifecycle_inspection": "attached",
        "net_changes": True,
        "publication": "cooperative_atomic",
        "recovery": "process_local",
        "retention": "process_local",
    }
    assert environments_body["environments"][0]["workspace_branch_lifecycle"] == {
        "attached_count": 1,
        "statuses": ["active"],
        "truncated": False,
    }
    assert attached_branch.lifecycle_status.value == "active"

    artifacts = client.get("/api/artifacts", params={"session_id": "sess_inventory"})
    assert artifacts.status_code == 200
    artifacts_body = artifacts.json()
    assert artifacts_body["total_count"] == 1
    assert artifacts_body["artifacts"][0]["id"] == artifact.id
    assert artifacts_body["artifacts"][0]["artifact_store_id"] == "test-artifacts"
    assert artifacts_body["artifacts"][0]["metadata"] == {"source": "test"}

    artifacts_by_agent = client.get("/api/artifacts", params={"agent_name": "reviewer"})
    assert artifacts_by_agent.status_code == 200
    artifacts_by_agent_body = artifacts_by_agent.json()
    assert artifacts_by_agent_body["total_count"] == 1
    assert artifacts_by_agent_body["artifacts"][0]["id"] == artifact.id

    artifacts_by_other_agent = client.get("/api/artifacts", params={"agent_name": "other"})
    assert artifacts_by_other_agent.status_code == 200
    artifacts_by_other_agent_body = artifacts_by_other_agent.json()
    assert artifacts_by_other_agent_body["total_count"] == 0
    assert artifacts_by_other_agent_body["artifacts"] == []

    read = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "test-artifacts", "max_bytes": 10},
    )
    assert read.status_code == 200
    read_body = read.json()
    assert read_body["artifact"]["id"] == artifact.id
    assert read_body["preview_base64"] == base64.b64encode(b"deployment").decode()
    assert read_body["text_preview"] == "deployment"
    assert read_body["total_bytes"] == len(b"deployment log\nstatus=ok\n")
    assert read_body["truncated"] is True

    json_with_charset = asyncio.run(
        artifact_store.put_bytes(
            b'{"status":"ok"}',
            filename="status.json",
            content_type="application/json; charset=utf-8",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    json_preview = client.get(
        f"/api/artifacts/{json_with_charset.id}",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert json_preview.status_code == 200
    assert json_preview.json()["text_preview"] == '{"status":"ok"}'

    malformed_without_store = client.get("/api/artifacts/not-a-local-artifact-id")
    assert malformed_without_store.status_code == 404
    assert malformed_without_store.json()["detail"] == "Artifact not found"

    malformed_with_store = client.get(
        "/api/artifacts/not-a-local-artifact-id",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert malformed_with_store.status_code == 404
    assert malformed_with_store.json()["detail"] == "Artifact not found"

    padded_id = client.get(
        f"/api/artifacts/%20{artifact.id}%20",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert padded_id.status_code == 404
    assert padded_id.json()["detail"] == "Artifact not found"


def test_artifact_content_endpoint_serves_bounded_downloads_safely(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"tenant": "test"}),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment log\nstatus=ok\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"source": "test"},
        )
    )

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    missing_store = client.get(f"/api/artifacts/{artifact.id}/content")
    assert missing_store.status_code == 422

    blank_store = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "   "},
    )
    assert blank_store.status_code == 422

    padded_store = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": " test-artifacts "},
    )
    assert padded_store.status_code == 422
    assert "must not start or end with whitespace" in padded_store.json()["detail"]

    content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert content.status_code == 200
    assert content.content == b"deployment log\nstatus=ok\n"
    assert content.headers["content-type"].startswith("text/plain")
    assert content.headers["x-cayu-artifact-id"] == artifact.id
    assert content.headers["x-cayu-artifact-store-id"] == "test-artifacts"
    assert content.headers["content-disposition"].startswith('attachment; filename="deploy.log"')
    assert content.headers["cache-control"] == "private, no-store"

    invalid_id = client.get(
        "/api/artifacts/not-a-local-artifact-id/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert invalid_id.status_code == 404
    assert invalid_id.json()["detail"] == "Artifact not found"

    padded_id = client.get(
        f"/api/artifacts/%20{artifact.id}%20/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert padded_id.status_code == 404
    assert padded_id.json()["detail"] == "Artifact not found"

    for malformed_id in (f"art_{'a' * 300}", "art_%00x", "art_%0Ax"):
        malformed_response = client.get(
            f"/api/artifacts/{malformed_id}/content",
            params={"artifact_store_id": "test-artifacts"},
        )
        assert malformed_response.status_code == 404
        assert malformed_response.json()["detail"] == "Artifact not found"

    oversized_content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts", "max_bytes": 10},
    )
    assert oversized_content.status_code == 413
    assert "exceeds the requested max_bytes" in oversized_content.json()["detail"]

    inline_content = client.get(
        f"/api/artifacts/{artifact.id}/content",
        params={"artifact_store_id": "test-artifacts", "disposition": "inline"},
    )
    assert inline_content.status_code == 200
    assert inline_content.headers["content-disposition"].startswith('inline; filename="deploy.log"')
    assert inline_content.headers["x-content-type-options"] == "nosniff"

    html_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"<script>alert('no inline')</script>",
            filename="unsafe.html",
            content_type="text/html",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    html_content = client.get(
        f"/api/artifacts/{html_artifact.id}/content",
        params={
            "artifact_store_id": "test-artifacts",
            "disposition": "inline",
        },
    )
    assert html_content.status_code == 200
    assert html_content.headers["content-disposition"].startswith(
        'attachment; filename="unsafe.html"'
    )
    assert html_content.headers["x-content-type-options"] == "nosniff"

    svg_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>",
            filename="unsafe.svg",
            content_type="image/svg+xml",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    svg_content = client.get(
        f"/api/artifacts/{svg_artifact.id}/content",
        params={
            "artifact_store_id": "test-artifacts",
            "disposition": "inline",
        },
    )
    assert svg_content.status_code == 200
    assert svg_content.headers["content-disposition"].startswith(
        'attachment; filename="unsafe.svg"'
    )
    assert svg_content.headers["x-content-type-options"] == "nosniff"

    with pytest.raises(ValueError, match="control characters"):
        asyncio.run(
            artifact_store.put_bytes(
                b"bad content type",
                filename="bad-content-type.txt",
                content_type="text/plain\r\nX-Bad: y",
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            )
        )
    with pytest.raises(ValueError, match="surrogate code points"):
        asyncio.run(
            artifact_store.put_bytes(
                b"bad filename",
                filename="bad\ud800.txt",
                content_type="text/plain",
                scope=ArtifactScope.SESSION,
                session_id="sess_inventory",
            )
        )
    unsafe_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"unsafe filename",
            filename="bad/path\r\nX: y.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    unsafe_content = client.get(
        f"/api/artifacts/{unsafe_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert unsafe_content.status_code == 200
    assert 'filename="bad_path__X: y.txt"' in unsafe_content.headers["content-disposition"]
    assert "bad%2Fpath" not in unsafe_content.headers["content-disposition"]
    assert "\r" not in unsafe_content.headers["content-disposition"]
    assert "\n" not in unsafe_content.headers["content-disposition"]
    assert "%0D" not in unsafe_content.headers["content-disposition"]
    assert "%0A" not in unsafe_content.headers["content-disposition"]

    bidi_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"unicode filename controls",
            filename="report\u202efdp\u2066\u2069.exe",
            content_type="application/octet-stream",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    bidi_filename_content = client.get(
        f"/api/artifacts/{bidi_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert bidi_filename_content.status_code == 200
    bidi_disposition = bidi_filename_content.headers["content-disposition"]
    assert 'filename="report_fdp__.exe"' in bidi_disposition
    assert "filename*=UTF-8''report_fdp__.exe" in bidi_disposition
    assert "%E2%80%AE" not in bidi_disposition
    assert "%E2%81%A6" not in bidi_disposition
    assert "%E2%81%A9" not in bidi_disposition

    app.register_environment(
        Environment(
            EnvironmentSpec(name="invalid-artifact-environment"),
            artifact_store=InvalidArtifactDataStore(),
        )
    )
    invalid_store_data = client.get(
        "/api/artifacts/invalid/content",
        params={"artifact_store_id": "invalid-artifact-data"},
    )
    assert invalid_store_data.status_code == 500
    assert invalid_store_data.json() == {"detail": "Artifact store returned invalid artifact data."}

    app.register_environment(
        Environment(
            EnvironmentSpec(name="wrong-artifact-environment"),
            artifact_store=WrongArtifactDataStore(),
        )
    )
    wrong_store_data = client.get(
        "/api/artifacts/requested/content",
        params={"artifact_store_id": "wrong-artifact-data"},
    )
    assert wrong_store_data.status_code == 500
    assert wrong_store_data.json() == {"detail": "Artifact store returned invalid artifact data."}

    app.register_environment(
        Environment(
            EnvironmentSpec(name="overread-artifact-environment"),
            artifact_store=OverreadArtifactDataStore(),
        )
    )
    overread_store_data = client.get(
        "/api/artifacts/requested/content",
        params={"artifact_store_id": "overread-artifact-data", "max_bytes": 1},
    )
    assert overread_store_data.status_code == 500
    assert overread_store_data.json() == {
        "detail": "Artifact store returned invalid artifact data."
    }

    app.register_environment(
        Environment(
            EnvironmentSpec(name="unavailable-artifact-environment"),
            artifact_store=UnavailableArtifactStore(),
        )
    )
    for unavailable_path in (
        "/api/artifacts/requested",
        "/api/artifacts/requested/content",
    ):
        unavailable_store = client.get(
            unavailable_path,
            params={"artifact_store_id": "unavailable-artifacts"},
        )
        assert unavailable_store.status_code == 503
        assert unavailable_store.json() == {"detail": "Artifact store is unavailable."}
        assert unavailable_store.headers["content-type"].startswith("application/json")

    unavailable_list = client.get(
        "/api/artifacts",
        params={"artifact_store_id": "unavailable-artifacts"},
    )
    assert unavailable_list.status_code == 503
    assert unavailable_list.json() == {"detail": "Artifact store is unavailable."}
    assert unavailable_list.headers["content-type"].startswith("application/json")

    app.register_environment(
        Environment(
            EnvironmentSpec(name="invalid-artifact-list-environment"),
            artifact_store=InvalidArtifactListStore(),
        )
    )
    invalid_list = client.get(
        "/api/artifacts",
        params={"artifact_store_id": "invalid-artifact-list"},
    )
    assert invalid_list.status_code == 500
    assert invalid_list.json() == {"detail": "Artifact store returned invalid artifact data."}
    assert invalid_list.headers["content-type"].startswith("application/json")

    long_filename_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"bounded header",
            filename=f"{'a' * 20_000}.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    bounded_header = client.get(
        f"/api/artifacts/{long_filename_artifact.id}/content",
        params={"artifact_store_id": "test-artifacts"},
    )
    assert bounded_header.status_code == 200
    disposition_header = bounded_header.headers["content-disposition"]
    assert len(disposition_header.encode("latin-1")) < 2048
    assert ".txt" in disposition_header

    symlink_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"artifact-content",
            filename="symlink.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
        )
    )
    outside_content = tmp_path / "outside-secret.txt"
    outside_content.write_bytes(b"host-secret-data")
    content_path = artifact_store.root / symlink_artifact.id / "content"
    content_path.unlink()
    try:
        content_path.symlink_to(outside_content)
    except OSError:
        pass
    else:
        symlink_content = client.get(
            f"/api/artifacts/{symlink_artifact.id}/content",
            params={"artifact_store_id": "test-artifacts"},
        )
        assert symlink_content.status_code == 500
        assert symlink_content.json() == {
            "detail": "Artifact store returned invalid artifact data."
        }
        assert symlink_content.content != outside_content.read_bytes()


def test_server_control_plane_inventory_redacts_configured_secrets(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="fake-model",
            metadata={"note": "agent secret-token"},
            provider_options={"header": "Bearer secret-token"},
        ),
        tools=[UserInputTool()],
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-review", metadata={"note": "env secret-token"}),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"deployment secret-token\n",
            filename="deploy.log",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_inventory",
            agent_name="reviewer",
            environment_name="local-review",
            metadata={"note": "artifact secret-token"},
        )
    )

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    agent = client.get("/api/agents").json()["agents"][0]
    environment = client.get("/api/environments").json()["environments"][0]
    artifact_list_item = client.get("/api/artifacts").json()["artifacts"][0]
    artifact_read_body = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "test-artifacts"},
    ).json()
    artifact_read = artifact_read_body["artifact"]

    assert agent["metadata"] == {"note": f"agent {REDACTED_SECRET}"}
    assert agent["provider_options"] == {"header": f"Bearer {REDACTED_SECRET}"}
    assert environment["metadata"] == {"note": f"env {REDACTED_SECRET}"}
    assert artifact_list_item["metadata"] == {"note": f"artifact {REDACTED_SECRET}"}
    assert artifact_read["metadata"] == {"note": f"artifact {REDACTED_SECRET}"}
    assert artifact_read_body["text_preview"] == f"deployment {REDACTED_SECRET}\n"
    assert (
        artifact_read_body["preview_base64"]
        == base64.b64encode(f"deployment {REDACTED_SECRET}\n".encode()).decode()
    )


def test_server_artifact_inventory_remains_usable_after_duplicate_store_rejection(
    tmp_path,
) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="duplicate-store")
    second_store = LocalArtifactStore(tmp_path / "second", store_id="duplicate-store")
    app.register_environment(
        Environment(EnvironmentSpec(name="first"), artifact_store=first_store),
        default=True,
    )
    with pytest.raises(ValueError, match="different registered store: duplicate-store"):
        app.register_environment(
            Environment(EnvironmentSpec(name="second"), artifact_store=second_store),
            default=True,
        )
    asyncio.run(
        first_store.put_bytes(
            b"first",
            filename="first.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="first",
        )
    )

    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    capabilities = client.get("/api/contract").json()["capabilities"]
    response = client.get("/api/artifacts")

    assert app.list_environments() == ("first",)
    assert app.get_environment().spec.name == "first"
    assert capabilities["surfaces"]["artifacts"]["read"]["enabled"] is True
    assert response.status_code == 200
    assert [artifact["filename"] for artifact in response.json()["artifacts"]] == ["first.txt"]


def test_server_artifact_inventory_paginates_across_registered_stores(tmp_path) -> None:
    app = CayuApp()
    first_store = LocalArtifactStore(tmp_path / "first", store_id="first-store")
    second_store = LocalArtifactStore(tmp_path / "second", store_id="second-store")
    app.register_environment(
        Environment(EnvironmentSpec(name="first"), artifact_store=first_store),
        default=True,
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="second"), artifact_store=second_store),
    )
    created = [
        (
            "first-store",
            asyncio.run(
                first_store.put_bytes(
                    b"one",
                    filename="one.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="first",
                )
            ),
        ),
        (
            "second-store",
            asyncio.run(
                second_store.put_bytes(
                    b"two",
                    filename="two.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="second",
                )
            ),
        ),
        (
            "first-store",
            asyncio.run(
                first_store.put_bytes(
                    b"three",
                    filename="three.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.ENVIRONMENT,
                    environment_name="first",
                )
            ),
        ),
    ]
    expected_ids = [
        artifact.id
        for _store_id, artifact in sorted(
            created,
            key=lambda item: (item[1].created_at.isoformat(), item[0], item[1].id),
            reverse=True,
        )
    ]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    first_page = client.get("/api/artifacts", params={"limit": 2})
    second_page = client.get("/api/artifacts", params={"limit": 2, "offset": 2})

    assert first_page.status_code == 200
    first_body = first_page.json()
    assert [artifact["id"] for artifact in first_body["artifacts"]] == expected_ids[:2]
    assert first_body["total_count"] == 3
    assert first_body["limit"] == 2
    assert first_body["offset"] == 0
    assert first_body["next_offset"] == 2
    assert first_body["truncated"] is True

    assert second_page.status_code == 200
    second_body = second_page.json()
    assert [artifact["id"] for artifact in second_body["artifacts"]] == expected_ids[2:]
    assert second_body["total_count"] == 3
    assert second_body["limit"] == 2
    assert second_body["offset"] == 2
    assert second_body["next_offset"] is None
    assert second_body["truncated"] is False


def test_server_artifact_inventory_includes_factory_registered_store(tmp_path) -> None:
    class Factory(EnvironmentFactory):
        def __init__(self, artifact_store: ArtifactStore, *, allow_create: bool = True) -> None:
            self.artifact_store = artifact_store
            self.allow_create = allow_create
            self.create_calls = 0

        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.create_calls += 1
            if not self.allow_create:
                raise AssertionError("artifact API must not materialize the environment factory")
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    artifact_store=self.artifact_store,
                )
            )

    artifact_root = tmp_path / "artifacts"
    first_store = LocalArtifactStore(artifact_root, store_id="factory-artifacts")
    first_factory = Factory(first_store)
    first_app = CayuApp()
    first_app.register_environment_factory(
        EnvironmentSpec(name="factory-env"),
        first_factory,
        artifact_store=first_store,
        default=True,
    )
    created = asyncio.run(
        first_app.get_environment_factory().create(
            EnvironmentFactoryRequest(
                session_id="sess_factory",
                agent_name="agent",
                environment_name="factory-env",
            )
        )
    )
    created_store = created.environment.artifact_store
    assert created_store is first_store
    artifact = asyncio.run(
        created_store.put_bytes(
            b"factory output",
            filename="result.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="sess_factory",
            environment_name="factory-env",
        )
    )

    restarted_store = LocalArtifactStore(artifact_root, store_id="factory-artifacts")
    restarted_factory = Factory(restarted_store, allow_create=False)
    restarted_app = CayuApp()
    restarted_app.register_environment_factory(
        EnvironmentSpec(name="factory-env"),
        restarted_factory,
        artifact_store=restarted_store,
        default=True,
    )
    client = TestClient(create_server(restarted_app, config=_LOCAL_SERVER_CONFIG))

    environments = client.get("/api/environments")
    artifacts = client.get("/api/artifacts", params={"session_id": "sess_factory"})
    read = client.get(
        f"/api/artifacts/{artifact.id}",
        params={"artifact_store_id": "factory-artifacts"},
    )

    assert environments.status_code == 200
    assert environments.json()["environments"][0]["artifact_store_id"] == "factory-artifacts"
    assert artifacts.status_code == 200
    assert [item["id"] for item in artifacts.json()["artifacts"]] == [artifact.id]
    assert read.status_code == 200
    assert read.json()["text_preview"] == "factory output"
    assert first_factory.create_calls == 1
    assert restarted_factory.create_calls == 0


def test_server_artifact_inventory_rejects_unbounded_offsets(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="test-artifacts")
    app = CayuApp()
    app.register_environment(
        Environment(EnvironmentSpec(name="local-review"), artifact_store=artifact_store),
        default=True,
    )

    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/artifacts",
        params={"offset": 10_001},
    )

    assert response.status_code == 422


def test_server_artifact_inventory_does_not_advertise_unusable_next_offset() -> None:
    app = CayuApp()
    app.register_environment(
        Environment(EnvironmentSpec(name="local-review"), artifact_store=CountingArtifactStore()),
        default=True,
    )

    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/artifacts",
        params={"offset": 10_000, "limit": 500},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["offset"] == 10_000
    assert body["limit"] == 500
    assert body["total_count"] == 20_000
    assert body["artifacts"]
    assert body["next_offset"] is None
    assert body["truncated"] is True
