from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.base import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.execution import CorpusTarget
from cayu.evals.scenario import (
    EvalScenarioDocumentV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
)
from cayu.evals.scenario_authoring import EvalScenarioDraftV2
from cayu.evals.store import EvalScenarioTrialPhase, EvalStoreResultTooLarge
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvalsConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
)
from cayu.server import routes as routes_module
from cayu.server.evals_registry import EvalTargetRegistration
from cayu.sessions.requests import RunRequest
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

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


def _approval_scenario(target_key: str) -> EvalScenarioDocumentV2:
    return EvalScenarioDocumentV2.create(
        id="approve-fresh-action",
        target_key=target_key,
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
    monkeypatch,
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
            execution_profile_revision = saved.json()["execution_profile_revision"]

            stale = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "stale-scenario-launch"},
                json={
                    "expected_binding_revision": "sha256:" + "0" * 64,
                    "expected_execution_profile_revision": execution_profile_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert stale.status_code == 409

            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-launch"},
                json={
                    "expected_binding_revision": binding_revision,
                    "expected_execution_profile_revision": execution_profile_revision,
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
                    "expected_execution_profile_revision": execution_profile_revision,
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

            def unavailable_profile(*, model: str) -> None:
                del model
                raise RuntimeError("provider temporarily unavailable")

            monkeypatch.setattr(provider, "preflight_model_target", unavailable_profile)
            replayed_while_unavailable = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-launch"},
                json={
                    "expected_binding_revision": binding_revision,
                    "expected_execution_profile_revision": execution_profile_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert replayed_while_unavailable.status_code == 202
            assert replayed_while_unavailable.json()["spec"]["run_id"] == run_id
    finally:
        asyncio.run(store.close())


def test_scenario_selected_environment_is_profiled_and_used_for_execution(tmp_path) -> None:
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("alternate environment result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    target.app.register_environment(
        Environment(
            EnvironmentSpec(name="alternate"),
            artifact_store=LocalArtifactStore(
                tmp_path / "alternate-artifacts",
                store_id="alternate-scenario-artifacts",
            ),
        )
    )
    target = target.model_copy(
        update={"request_base": target.request_base.model_copy(update={"environment_name": None})}
    )
    store = SQLiteEvalStore(tmp_path / "scenario-environment.db")
    scenario = _scenario()
    settings = {"environment_name": "alternate", "timeout_seconds": 30}
    try:
        with TestClient(_server(target, store)) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS)
            assert catalog.status_code == 200
            base_profile_revision = catalog.json()["items"][0]["execution_profile"]["revision"]
            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                    "settings": settings,
                },
            )
            assert saved.status_code == 201
            saved_body = saved.json()
            binding = saved_body["preflight"]["binding"]
            assert binding["environment_name"] == "alternate"
            assert saved_body["execution_profile_revision"] != base_profile_revision

            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "alternate-environment"},
                json={
                    "expected_binding_revision": binding["revision"],
                    "expected_execution_profile_revision": saved_body["execution_profile_revision"],
                    "settings": settings,
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                terminal = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert terminal.status_code == 200
                if terminal.json()["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Environment-selected scenario did not terminalize.")
            terminal_body = terminal.json()
            assert terminal_body["status"] == "completed"
            session_id = terminal_body["scenario_progress"]["trials"][0]["session_id"]
            session = asyncio.run(target.app.session_store.load(session_id))
            assert session is not None
            assert session.environment_name == "alternate"
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
            execution_profile_revision = saved.json()["execution_profile_revision"]
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "cancel-scenario-approval"},
                json={
                    "expected_binding_revision": binding_revision,
                    "expected_execution_profile_revision": execution_profile_revision,
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
    scenario = _approval_scenario(target.key)
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
            execution_profile_revision = saved.json()["execution_profile_revision"]
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "approve-scenario-action"},
                json={
                    "expected_binding_revision": binding_revision,
                    "expected_execution_profile_revision": execution_profile_revision,
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]

            deadline = time.monotonic() + 20
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

            deadline = time.monotonic() + 20
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


def test_scenario_rejects_profile_drift_before_approved_tool_dispatch(tmp_path) -> None:
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
    scenario = _approval_scenario(target.key)
    store = SQLiteEvalStore(tmp_path / "scenario-profile-drift.db")
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
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "profile-drift-scenario"},
                json={
                    "expected_binding_revision": saved.json()["preflight"]["binding"]["revision"],
                    "expected_execution_profile_revision": saved.json()[
                        "execution_profile_revision"
                    ],
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                current = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                progress = current.json().get("scenario_progress")
                if progress is not None and progress["trials"][0]["phase"] == "awaiting_approval":
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Scenario did not pause for approval.")

            app.register_agent(AgentSpec(name="late-agent", model="scenario-model"))
            approved = client.post(
                f"/api/evals/runs/{run_id}/scenario-approval",
                headers=_AUTH_HEADERS,
                json={
                    "expected_progress_revision": progress["revision"],
                    "trial_number": 1,
                    "event_id": "review-approval",
                    "decision": "approve",
                },
            )
            assert approved.status_code == 200

            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                terminal_response = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert terminal_response.status_code == 200
                terminal = terminal_response.json()
                if terminal["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Profile-drift scenario did not terminalize.")
            trial = terminal["scenario_progress"]["trials"][0]
            assert terminal["status"] == "failed"
            assert terminal["failure_code"] == "target_unavailable"
            assert trial["phase"] == "error"
            assert trial["failure_code"] == "execution_profile_changed"
            assert provider.request_count == 1
            assert tool.run_count == 0
    finally:
        asyncio.run(store.close())


@pytest.mark.parametrize(
    ("drift_running_update", "expected_provider_requests"),
    ((1, 0), (2, 1)),
    ids=("initial-dispatch", "approval-continuation"),
)
def test_scenario_rechecks_profile_after_running_progress_write(
    tmp_path,
    monkeypatch,
    drift_running_update,
    expected_provider_requests,
) -> None:
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
    scenario = _approval_scenario(target.key)
    store = SQLiteEvalStore(tmp_path / "scenario-progress-write-drift.db")
    original_update = store.update_scenario_trial
    running_updates = 0

    async def drift_during_second_running_update(claim, trial):
        nonlocal running_updates
        updated = await original_update(claim, trial)
        if trial.phase is EvalScenarioTrialPhase.RUNNING:
            running_updates += 1
            if running_updates == drift_running_update:
                app.register_agent(AgentSpec(name="late-agent", model="scenario-model"))
        return updated

    monkeypatch.setattr(store, "update_scenario_trial", drift_during_second_running_update)
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
            launched = client.post(
                f"/api/evals/scenarios/{scenario.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "progress-write-profile-drift"},
                json={
                    "expected_binding_revision": saved.json()["preflight"]["binding"]["revision"],
                    "expected_execution_profile_revision": saved.json()[
                        "execution_profile_revision"
                    ],
                    "settings": {"timeout_seconds": 30},
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["spec"]["run_id"]

            if drift_running_update == 2:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    current = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                    progress = current.json().get("scenario_progress")
                    if (
                        progress is not None
                        and progress["trials"][0]["phase"] == "awaiting_approval"
                    ):
                        break
                    time.sleep(0.01)
                else:
                    raise AssertionError("Scenario did not pause for approval.")

                approved = client.post(
                    f"/api/evals/runs/{run_id}/scenario-approval",
                    headers=_AUTH_HEADERS,
                    json={
                        "expected_progress_revision": progress["revision"],
                        "trial_number": 1,
                        "event_id": "review-approval",
                        "decision": "approve",
                    },
                )
                assert approved.status_code == 200

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                terminal_response = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert terminal_response.status_code == 200
                terminal = terminal_response.json()
                if terminal["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Profile-drift scenario did not terminalize.")

            trial = terminal["scenario_progress"]["trials"][0]
            assert running_updates == drift_running_update
            assert terminal["status"] == "failed"
            assert terminal["failure_code"] == "target_unavailable"
            assert trial["phase"] == "error"
            assert trial["failure_code"] == "execution_profile_changed"
            assert provider.request_count == expected_provider_requests
            assert tool.run_count == 0
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


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_scenario_authoring_preserves_auth_and_private_http_boundary(sqlite_resources, prefix):
    async def exercise():
        async with sqlite_resources as resources:
            target, _, provider = _target(resources.path("target"))
            store = resources.own(SQLiteEvalStore(resources.path()))
            calls = []

            def authenticate(request: Request):
                calls.append(request.url.path)
                return _authenticate(request)

            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    authenticate,
                    api=ServerApiConfig(path=prefix),
                    dashboard=DashboardConfig(enabled=False),
                    evals=EvalsConfig(target=target, store=store),
                ),
            )
            root = f"{prefix}/evals/scenarios"
            paths = [
                f"{root}/preview",
                root,
                f"{root}/artifacts/{{requirement_id}}/materialize",
                root,
                f"{root}/{{scenario_revision}}",
                f"{root}/{{scenario_revision}}/download",
            ]
            routes = [r for r in server.routes if isinstance(r, APIRoute)]
            authoring = [r for r in routes if r.path in paths]
            assert [r.path for r in authoring] == paths
            dependency = (
                next(r for r in routes if r.path == f"{prefix}/sessions").dependencies[0].dependency
            )
            for route in authoring:
                assert route.dependencies[0].dependency is dependency
                assert isinstance(route, routes_module._BoundedEvalsRoute)
                assert type(route).preparse_auth is authenticate
            scenario = _scenario()
            save_body = {
                "scenario": scenario.model_dump(mode="json"),
                "expected_scenario_revision": scenario.revision,
            }
            requests = [
                (
                    "POST",
                    f"{root}/preview",
                    {"draft": EvalScenarioDraftV2.from_scenario(scenario).model_dump(mode="json")},
                    200,
                ),
                ("POST", root, save_body, 201),
                ("POST", f"{root}/artifacts/missing/materialize", save_body, 404),
                ("GET", root, None, 200),
                ("GET", f"{root}/{scenario.revision}", None, 200),
                ("GET", f"{root}/{scenario.revision}/download", None, 200),
            ]
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for method, path, payload, status in requests:
                    calls.clear()
                    denied = await client.request(
                        method,
                        path,
                        content=b"private invalid input",
                        headers={"Content-Length": str(authoring[0].max_request_bytes + 1)},
                    )
                    assert denied.status_code == 401
                    assert denied.json() == {"detail": "unauthorized"}
                    assert denied.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    calls.clear()
                    response = await client.request(
                        method, path, headers=_AUTH_HEADERS, json=payload
                    )
                    assert response.status_code == status
                    assert response.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                for _, path, _, _ in requests[:3]:
                    for content, headers, status, detail in (
                        (
                            b"{}",
                            {"Content-Length": str(authoring[0].max_request_bytes + 1)},
                            413,
                            "Evals request exceeds the server byte limit.",
                        ),
                        (
                            b'{"private":"secret","private":"other"}',
                            {},
                            422,
                            "Invalid Evals request.",
                        ),
                        (b"NaN", {}, 422, "Invalid Evals request."),
                    ):
                        calls.clear()
                        response = await client.post(
                            path,
                            content=content,
                            headers={
                                **_AUTH_HEADERS,
                                "Content-Type": "application/json",
                                **headers,
                            },
                        )
                        assert response.status_code == status
                        assert response.json() == {"detail": detail}
                        assert response.headers["cache-control"] == "private, no-store"
                        assert calls == [path]

                def deny_access():
                    raise HTTPException(status_code=403, detail="scenario access revoked")

                server.dependency_overrides[dependency] = deny_access
                for method, path, payload, _ in requests:
                    response = await client.request(
                        method, path, headers=_AUTH_HEADERS, json=payload
                    )
                    assert response.status_code == 403
                    assert response.json() == {"detail": "scenario access revoked"}
                    assert response.headers["cache-control"] == "private, no-store"
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["invalid", "oversized", "unsupported", "hidden", "missing"])
def test_scenario_revision_reads_preserve_private_errors_for_authoring_and_launch(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            target, _, provider = _target(resources.path("target"))
            store = resources.own(SQLiteEvalStore(resources.path()))
            reads = []
            template = _scenario()
            hidden = EvalScenarioDocumentV2.create(
                id=template.id,
                name=template.name,
                target_key="assistant.unpublished",
                events=template.events,
            )

            async def load(revision):
                reads.append(revision)
                if failure == "invalid":
                    raise ValueError("private storage input")
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(1024)
                return hidden if failure == "hidden" else None

            monkeypatch.setattr(store, "load_scenario", load)
            if failure == "unsupported":
                monkeypatch.setattr(store, "scenarios", False)
            server = _server(target, store)
            revision = "sha256:" + "0" * 64
            root = f"/api/evals/scenarios/{revision}"
            expected = {
                "invalid": (422, "Invalid Evals query."),
                "oversized": (413, "Eval scenario exceeds the server byte limit."),
                "unsupported": (409, "Durable scenario persistence is not available."),
                "hidden": (404, "Eval scenario not found."),
                "missing": (404, "Eval scenario not found."),
            }[failure]
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for method, path, payload in (
                    ("GET", root, None),
                    ("GET", f"{root}/download", None),
                    (
                        "POST",
                        f"{root}/runs",
                        {
                            "expected_binding_revision": revision,
                            "expected_execution_profile_revision": revision,
                        },
                    ),
                ):
                    response = await client.request(
                        method,
                        path,
                        json=payload,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-private-read"},
                    )
                    assert response.status_code == expected[0]
                    assert response.json() == {"detail": expected[1]}
                    assert response.headers["cache-control"] == "private, no-store"
            assert reads == ([] if failure == "unsupported" else [revision] * 3)
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["invalid", "unavailable"])
def test_scenario_preflight_preserves_private_errors_before_save_or_launch(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            target, _, provider = _target(resources.path("target"))
            store = resources.own(SQLiteEvalStore(resources.path()))
            server = _server(target, store)
            scenario = _scenario()
            await store.save_scenario(scenario, redact_json=target.app.redact_json)
            calls = []

            def rejected_target(self):
                calls.append(self.target.key)
                error = ValueError if failure == "invalid" else RuntimeError
                raise error("private target details")

            async def unexpected_save(*args, **kwargs):
                pytest.fail("Failed preflight must not save a scenario or admit a run.")

            monkeypatch.setattr(EvalTargetRegistration, "execution_target", rejected_target)
            monkeypatch.setattr(store, "save_scenario", unexpected_save)
            monkeypatch.setattr(store, "admit_run", unexpected_save)
            expected = (
                (400, "Eval scenario or its launch selections are invalid.")
                if failure == "invalid"
                else (409, "Attached eval target is unavailable for scenario preflight.")
            )
            root = "/api/evals/scenarios"
            revision = "sha256:" + "0" * 64
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for path, payload in (
                    (
                        f"{root}/preview",
                        {
                            "draft": EvalScenarioDraftV2.from_scenario(scenario).model_dump(
                                mode="json"
                            )
                        },
                    ),
                    (
                        root,
                        {
                            "scenario": scenario.model_dump(mode="json"),
                            "expected_scenario_revision": scenario.revision,
                        },
                    ),
                    (
                        f"{root}/{scenario.revision}/runs",
                        {
                            "expected_binding_revision": revision,
                            "expected_execution_profile_revision": revision,
                        },
                    ),
                ):
                    response = await client.post(
                        path,
                        json=payload,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-preflight-failure"},
                    )
                    assert response.status_code == expected[0]
                    assert response.json() == {"detail": expected[1]}
                    assert response.headers["cache-control"] == "private, no-store"
            assert calls == [target.key] * 3
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_scenario_materialization_rebinds_artifacts_and_cancels_before_publication(
    sqlite_resources, monkeypatch, cancel
):
    async def exercise():
        async with sqlite_resources as resources:
            target, artifacts, provider = _target(resources.path("target"))
            store = resources.own(SQLiteEvalStore(resources.path()))
            content = b"retained request attachment"
            sources = [
                await artifacts.put_bytes(
                    content,
                    filename="request.txt",
                    content_type="text/plain",
                    scope=ArtifactScope.SESSION,
                    session_id=f"source-{i}",
                    environment_name="files",
                )
                for i in range(2)
            ]
            requirement = ScenarioArtifactRequirementV2(
                id="request-file",
                source="artifact_reference",
                reference=sources[0].id,
                content_sha256=hashlib.sha256(content).hexdigest(),
                filename="request.txt",
                content_type="text/plain",
                size_bytes=len(content),
            )
            scenario = _scenario(requirement)
            entered = asyncio.Event()
            release = asyncio.Event()
            events = []
            original_read = artifacts.read_bytes
            original_put = artifacts.put_bytes

            async def read(artifact_id, **kwargs):
                events.append(("read", artifact_id))
                if artifact_id == sources[1].id:
                    entered.set()
                    await release.wait()
                return await original_read(artifact_id, **kwargs)

            async def put(data, **kwargs):
                events.append(("put", kwargs["artifact_id"]))
                return await original_put(data, **kwargs)

            monkeypatch.setattr(artifacts, "read_bytes", read)
            monkeypatch.setattr(artifacts, "put_bytes", put)
            server = _server(target, store)
            path = f"/api/evals/scenarios/artifacts/{requirement.id}/materialize"
            body = {
                "scenario": scenario.model_dump(mode="json"),
                "expected_scenario_revision": scenario.revision,
                "settings": {"artifact_references": {requirement.id: sources[1].id}},
            }
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                stale = await client.post(
                    path,
                    headers=_AUTH_HEADERS,
                    json={**body, "expected_scenario_revision": "sha256:" + "0" * 64},
                )
                assert stale.status_code == 409
                assert stale.json() == {
                    "detail": "Eval scenario changed after the reviewed revision."
                }
                assert events == []
                request = resources.task(client.post(path, headers=_AUTH_HEADERS, json=body))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    assert events == [("read", sources[1].id)]
                    assert not request.done()
                    if cancel:
                        request.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await request
                        assert events == [("read", sources[1].id)]
                    else:
                        release.set()
                        response = await asyncio.wait_for(request, 10)
                        assert response.status_code == 200
                        result = response.json()
                        fixture_id = result["materialization"]["artifact_id"]
                        updated = EvalScenarioDocumentV2.model_validate(
                            result["materialization"]["scenario"]
                        )
                        assert updated.revision != scenario.revision
                        assert updated.artifact_requirements[0].reference == fixture_id
                        assert result["preflight"]["ready"] is True
                        assert result["execution_profile_revision"] is not None
                        assert events == [
                            ("read", sources[1].id),
                            ("put", fixture_id),
                            ("read", fixture_id),
                        ]
                        fixture = await original_read(fixture_id)
                        assert fixture.content == content
                        assert fixture.metadata.scope is ArtifactScope.ENVIRONMENT
                        assert await store.load_scenario(updated.revision) is None
                    assert await store.load_scenario(scenario.revision) is None
                finally:
                    release.set()
                    await asyncio.gather(request, return_exceptions=True)
            assert provider.requests == []

    asyncio.run(exercise())
