from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    ArtifactScope,
    CayuApp,
    CorpusTarget,
    Environment,
    EnvironmentSpec,
    EvalScenarioDocumentV2,
    EvalScenarioDraftV2,
    EventType,
    ExecutionProfileBehaviorIdentity,
    LocalArtifactStore,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    RunRequest,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.server import AuthContext, DashboardConfig, EvalsConfig, ServerConfig, create_server
from cayu.storage.evals_sqlite import SQLiteEvalStore

_AUTH_HEADERS = {"Authorization": "Bearer valid"}


class _ApprovalProvider(ModelProvider):
    name = "server-scenario-approval-provider"

    def __init__(self) -> None:
        self.request_count = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:server-scenario-approval-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.request_count += 1
        if self.request_count == 1:
            yield ModelStreamEvent.tool_call(
                id="call-server-approval",
                name="review_action",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("approved scenario result")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ReviewTool(Tool):
    spec = ToolSpec(
        name="review_action",
        description="Perform one reviewed action.",
        input_schema={"type": "object", "properties": {}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:server-scenario-review-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        self.run_count = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.run_count += 1
        return ToolResult(content="reviewed")


def _authenticate(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != "Bearer valid":
        raise HTTPException(status_code=401, detail="unauthorized")
    return AuthContext(subject="scenario-operator")


def _target(
    tmp_path,
    provider: ScriptedModelProvider | None = None,
) -> tuple[CorpusTarget, LocalArtifactStore, ScriptedModelProvider]:
    provider = ScriptedModelProvider([]) if provider is None else provider
    artifact_store = LocalArtifactStore(
        tmp_path / "artifacts",
        store_id="scenario-server-artifacts",
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scenario-model"))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="files"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    target = CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(
            agent_name="assistant",
            messages=[],
            environment_name="files",
            max_steps=8,
        ),
        application_release_id="release-current",
    )
    return target, artifact_store, provider


def _server(target: CorpusTarget, store: SQLiteEvalStore):
    return create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                poll_interval_seconds=0.02,
                lease_seconds=5,
                shutdown_grace_seconds=2,
            ),
        ),
    )


def _scenario(
    requirement: ScenarioArtifactRequirementV2 | None = None,
) -> EvalScenarioDocumentV2:
    content = [ScenarioTextPartV2(text="Review this retained request.")]
    if requirement is not None:
        content.append(ScenarioFilePartV2(artifact_requirement_id=requirement.id))
    return EvalScenarioDocumentV2.create(
        id="retained-request",
        target_key="assistant.default",
        name="Retained request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create((ScenarioUserMessageV2.create(content),)),
            ),
        ),
        artifact_requirements=(() if requirement is None else (requirement,)),
    )


def test_scenario_editor_preview_save_catalog_and_download_are_target_scoped(
    tmp_path,
) -> None:
    target, _, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    scenario = _scenario()
    draft = EvalScenarioDraftV2.from_scenario(scenario)
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/scenarios/preview",
                    json={"draft": draft.model_dump(mode="json")},
                ).status_code
                == 401
            )
            preview = client.post(
                "/api/evals/scenarios/preview",
                headers=_AUTH_HEADERS,
                json={
                    "draft": draft.model_dump(mode="json"),
                    "settings": {"trials": 1, "max_concurrency": 1},
                },
            )
            assert preview.status_code == 200
            assert preview.json()["scenario"] == scenario.model_dump(mode="json")
            assert preview.json()["preflight"]["ready"] is True

            stale = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": "sha256:" + "0" * 64,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert stale.status_code == 409
            assert stale.json() == {"detail": "Eval scenario changed after the reviewed revision."}

            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert saved.status_code == 201
            assert saved.json()["entry"]["revision"] == scenario.revision
            assert saved.json()["preflight"]["ready"] is True

            catalog = client.get(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
            )
            assert catalog.status_code == 200
            assert [item["revision"] for item in catalog.json()["items"]] == [scenario.revision]
            loaded = client.get(
                f"/api/evals/scenarios/{scenario.revision}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == scenario.model_dump(mode="json")
            downloaded = client.get(
                f"/api/evals/scenarios/{scenario.revision}/download",
                headers=_AUTH_HEADERS,
            )
            assert downloaded.status_code == 200
            assert downloaded.content.endswith(b"\n")
            assert EvalScenarioDocumentV2.model_validate_json(downloaded.content) == scenario
        assert provider.requests == []
    finally:
        asyncio.run(store.close())


def test_saved_scenario_launches_without_python_eval_configuration_and_exports_result(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("current scenario result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    store = SQLiteEvalStore(tmp_path / "scenario-launch.db")
    scenario = _scenario()
    try:
        with TestClient(_server(target, store)) as client:
            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert saved.status_code == 201
            binding_revision = saved.json()["preflight"]["binding"]["revision"]

            stale = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "stale-scenario-launch"},
                json={
                    "expected_binding_revision": "sha256:" + "0" * 64,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert stale.status_code == 409

            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-launch"},
                json={
                    "expected_binding_revision": binding_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]
            replayed = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-launch"},
                json={
                    "expected_binding_revision": binding_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert replayed.status_code == 202
            assert replayed.json()["spec"]["run_id"] == run_id
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                run = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert run.status_code == 200
                if run.json()["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Scenario run did not terminalize.")
            assert run.json()["status"] == "completed"
            assert run.json()["scenario_progress"]["trials"][0]["phase"] == "completed"

            result = client.get(f"/api/evals/runs/{run_id}/result", headers=_AUTH_HEADERS)
            assert result.status_code == 200
            assert result.json()["result"]["run"]["status"] == "passed"
            assert (
                result.json()["result"]["run"]["cases"][0]["trials"][0]["output"]["text"]
                == "current scenario result"
            )
            report = client.get(
                f"/api/evals/runs/{run_id}/report.html",
                headers=_AUTH_HEADERS,
            )
            assert report.status_code == 200
            assert b"current scenario result" in report.content
    finally:
        asyncio.run(store.close())


def test_scenario_cancellation_terminalizes_while_awaiting_approval(tmp_path) -> None:
    provider = _ApprovalProvider()
    tool = _ReviewTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scenario-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    target = CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[], max_steps=8),
        application_release_id="release-current",
    )
    scenario = EvalScenarioDocumentV2.create(
        id="cancel-awaiting-approval",
        target_key=target.key,
        name="Cancel awaiting approval",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (ScenarioTextPartV2(text="Review this action."),)
                        ),
                    )
                ),
            ),
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="review-approval",
                tool_name="review_action",
                occurrence=1,
            ),
        ),
    )
    store = SQLiteEvalStore(tmp_path / "scenario-cancel.db")
    try:
        with TestClient(_server(target, store)) as client:
            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert saved.status_code == 201
            binding_revision = saved.json()["preflight"]["binding"]["revision"]
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "cancel-scenario-approval"},
                json={
                    "expected_binding_revision": binding_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert current.status_code == 200
                progress = current.json().get("scenario_progress")
                if progress is not None and progress["trials"][0]["phase"] == "awaiting_approval":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Scenario did not pause for approval.")

            cancellation = client.post(
                f"/api/evals/runs/{run_id}/cancel",
                headers=_AUTH_HEADERS,
            )
            assert cancellation.status_code == 202
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                terminal = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert terminal.status_code == 200
                if terminal.json()["status"] == "cancelled":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Cancelled scenario did not terminalize.")

            assert provider.request_count == 1
            assert tool.run_count == 0
            assert (
                client.get(
                    f"/api/evals/runs/{run_id}/result",
                    headers=_AUTH_HEADERS,
                ).status_code
                == 409
            )
    finally:
        asyncio.run(store.close())


def test_scenario_approval_route_is_fresh_fenced_and_actor_attributed(tmp_path) -> None:
    provider = _ApprovalProvider()
    tool = _ReviewTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scenario-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    target = CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[], max_steps=8),
        application_release_id="release-current",
    )
    scenario = EvalScenarioDocumentV2.create(
        id="approve-fresh-action",
        target_key=target.key,
        name="Approve fresh action",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (
                        ScenarioUserMessageV2.create(
                            (ScenarioTextPartV2(text="Review this action."),)
                        ),
                    )
                ),
            ),
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="review-approval",
                tool_name="review_action",
                occurrence=1,
            ),
        ),
    )
    store = SQLiteEvalStore(tmp_path / "scenario-approval.db")
    try:
        with TestClient(_server(target, store)) as client:
            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert saved.status_code == 201
            binding_revision = saved.json()["preflight"]["binding"]["revision"]
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "approve-scenario-action"},
                json={
                    "expected_binding_revision": binding_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                current = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert current.status_code == 200
                progress = current.json().get("scenario_progress")
                if progress is not None and progress["trials"][0]["phase"] == "awaiting_approval":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Scenario did not pause for approval.")

            approval_body = {
                "expected_progress_revision": progress["revision"],
                "trial_number": 1,
                "event_id": "review-approval",
                "decision": "approve",
            }
            unauthorized = client.post(
                f"/api/evals/runs/{run_id}/scenario-approval",
                json=approval_body,
            )
            assert unauthorized.status_code == 401
            stale = client.post(
                f"/api/evals/runs/{run_id}/scenario-approval",
                headers=_AUTH_HEADERS,
                json={
                    **approval_body,
                    "expected_progress_revision": "sha256:" + "0" * 64,
                },
            )
            assert stale.status_code == 409
            approved = client.post(
                f"/api/evals/runs/{run_id}/scenario-approval",
                headers=_AUTH_HEADERS,
                json=approval_body,
            )
            assert approved.status_code == 200
            recorded_approval = approved.json()["scenario_progress"]["trials"][0]["approval"]
            assert recorded_approval["decision"] == "approve"
            assert recorded_approval["reason"] is None
            assert recorded_approval["actor_id"] == "scenario-operator"
            assert type(recorded_approval["submitted_at"]) is str

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                terminal = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert terminal.status_code == 200
                if terminal.json()["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Approved scenario did not terminalize.")

            assert terminal.json()["status"] == "completed"
            assert tool.run_count == 1
            assert provider.request_count == 2
            session_id = terminal.json()["scenario_progress"]["trials"][0]["session_id"]

        events = asyncio.run(target.app.session_store.load_events(session_id))
        resumed = next(event for event in events if event.type is EventType.SESSION_RESUMED)
        assert resumed.payload["resolved_by"] == {
            "subject": "scenario-operator",
            "source": "http_auth",
            "tenant": None,
        }
    finally:
        asyncio.run(store.close())


def test_scenario_artifact_preparation_returns_a_ready_unsaved_revision(tmp_path) -> None:
    target, artifact_store, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")

    async def seed():
        content = b"retained production attachment"
        artifact = await artifact_store.put_bytes(
            content,
            filename="request.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="production-session",
            environment_name="files",
        )
        return content, artifact

    content, artifact = asyncio.run(seed())
    requirement = ScenarioArtifactRequirementV2(
        id="request-file",
        source="artifact_reference",
        reference=artifact.id,
        content_sha256=hashlib.sha256(content).hexdigest(),
        filename=artifact.filename,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
    )
    scenario = _scenario(requirement)
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/scenarios/preview",
                headers=_AUTH_HEADERS,
                json={"draft": EvalScenarioDraftV2.from_scenario(scenario).model_dump(mode="json")},
            )
            assert preview.status_code == 200
            assert [item["code"] for item in preview.json()["preflight"]["diagnostics"]] == [
                "artifact_binding_required"
            ]

            prepared = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert prepared.status_code == 200
            body = prepared.json()
            assert body["preflight"]["ready"] is True
            updated = EvalScenarioDocumentV2.model_validate(body["materialization"]["scenario"])
            assert updated.revision != scenario.revision
            fixture_id = body["materialization"]["artifact_id"]
            assert updated.artifact_requirements[0].reference == fixture_id
            fixture = asyncio.run(artifact_store.read_bytes(fixture_id))
            assert fixture.content == content
            assert fixture.metadata.scope is ArtifactScope.ENVIRONMENT

            repeated = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert repeated.status_code == 200
            assert repeated.json()["materialization"] == body["materialization"]

            repeated_updated = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": updated.revision,
                    "scenario": updated.model_dump(mode="json"),
                },
            )
            assert repeated_updated.status_code == 200
            assert repeated_updated.json()["materialization"] == body["materialization"]
        assert provider.requests == []
    finally:
        asyncio.run(store.close())
