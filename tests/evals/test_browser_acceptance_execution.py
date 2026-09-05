from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import http.client
import json
import multiprocessing
import os
import signal
import time
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import scripts.run_browser_acceptance as command

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    BudgetLimit,
    BudgetReservation,
    CayuApp,
    EnvironmentSpec,
    LocalArtifactStore,
    Message,
    ModelPrice,
    PriceBook,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    VirtualEgressEnvironmentFactory,
    WebBridge,
)
from cayu.egress import (
    EgressAuthorityCutoverStrategy,
    EgressBinding,
    HttpxUpstream,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
)
from cayu.environments import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
)
from cayu.evals import (
    AssertionEvidenceView,
    BrowserAcceptanceCaseCategory,
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceFaultScenario,
    BrowserAcceptanceFixtureV1,
    BrowserAcceptanceLimitsV1,
    BrowserAcceptanceManifestV1,
    BrowserAcceptanceMode,
    BrowserAcceptancePlanV1,
    BrowserAcceptanceSemanticOracle,
    BrowserAcceptanceState,
    BrowserAcceptanceTrialReceiptV1,
    EvalCase,
    EvalPlan,
    EvalSuite,
    EvaluationEvidencePolicySpec,
    SessionCompleted,
    deterministic_browser_acceptance_manifest,
    inspect_browser_acceptance_runtime_identity,
    project_assertion_evidence_view,
    project_browser_acceptance_diagnostic,
    project_browser_acceptance_trial,
    run_browser_acceptance,
)
from cayu.evals import browser_acceptance as acceptance_module
from cayu.evals.corpus import _content_revision
from cayu.evals.internal import browser_acceptance as internal_acceptance
from cayu.evals.internal.browser_acceptance import build as build_internal_browser_acceptance
from cayu.providers import ModelStreamEvent
from cayu.runners import (
    PINNED_BROWSER_SESSION_WORKLOAD,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkloadAuthority,
)
from cayu.runtime._event_projection import public_event_id, public_event_sequence


class _ProtocolBrowserRunner(Runner):
    def __init__(self, upstream_origin: str) -> None:
        self._upstream_origin = upstream_origin
        self._revision = 0
        self._pages: dict[str, dict[str, Any]] = {}
        self._active_page_id: str | None = None
        self._total_operations = 0
        self._total_observations = 0
        self._total_page_creations = 0

    def _page_set(self) -> dict[str, Any]:
        return {
            "session_id": next(iter(self._pages.values()))["session_id"],
            "active_page_id": self._active_page_id,
            "pages": [
                {
                    "page_id": page["page_id"],
                    "lifecycle": page["lifecycle"],
                    "creation_epoch": page["creation_epoch"],
                    "control_epoch": page["control_epoch"],
                    "opener_page_id": page["opener_page_id"],
                    "creating_operation_id_sha256": page["creating_operation_id_sha256"],
                    "revision": page["revision"],
                    "url": page["url"],
                    "title": page["title"],
                    "load_state": "loaded",
                    "access_state": "available",
                    "last_observation_revision": page["last_observation_revision"],
                    "last_operation_id_sha256": page["last_operation_id_sha256"],
                    "terminal_reason": page["terminal_reason"],
                    "operation_count": page["operation_count"],
                    "observation_count": page["observation_count"],
                    "ref_count": page["ref_count"],
                    "request_count": 0,
                    "artifact_count": page["artifact_count"],
                }
                for page in sorted(self._pages.values(), key=lambda item: item["creation_epoch"])
            ],
            "total_page_creations": self._total_page_creations,
            "total_operations": self._total_operations,
            "total_observations": self._total_observations,
            "total_refs": sum(page["ref_count"] for page in self._pages.values()),
            "total_requests": 0,
            "total_artifacts": sum(page["artifact_count"] for page in self._pages.values()),
            "cleanup_operation_count": sum(
                page["lifecycle"] == "closed" for page in self._pages.values()
            ),
        }

    def _observe(self, page: dict[str, Any]) -> dict[str, Any]:
        self._revision += 1
        self._total_observations += 1
        page["observation_count"] += 1
        page["ref_count"] += 22
        page["revision"] = f"br_acceptance_revision_{self._revision}"
        page["last_observation_revision"] = page["revision"]
        names = (
            "Account",
            "Apply",
            "Bottom action",
            "Continue",
            "Covered action",
            "Detach me",
            "Download oversized file",
            "Download report",
            "Frame value",
            "Hidden action",
            "Name",
            "New",
            "Old",
            "Open blank popup",
            "Open cross-origin popup",
            "Open navigating popup",
            "Open popup",
            "Open popup burst",
            "Open redirecting popup",
            "Region",
            "Save",
            "Unavailable",
        )
        refs = [
            {
                "ref": "ref_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
                "role": "textbox" if name in {"Account", "Frame value", "Name"} else "button",
                "name": name,
            }
            for name in names
        ]
        return {
            "session_id": page["session_id"],
            "page_id": page["page_id"],
            "revision": page["revision"],
            "creation_epoch": page["creation_epoch"],
            "control_epoch": page["control_epoch"],
            "url": page["url"],
            "title": page["title"],
            "snapshot": "\n".join(
                f'- button "{item["name"]}" [ref={item["ref"]}]' for item in refs
            ),
            "refs": refs,
            "load_state": "loaded",
            "access_state": "available",
            "idle_timeout_seconds": 900,
            "truncation_reasons": [],
            "backend_identity": {
                "backend": "playwright",
                "backend_version": "1.62.0",
                "browser": "chromium",
                "browser_version": "acceptance-fixture",
                "worker_protocol": "cayu.browser-session.v3",
                "worker_version": "7",
            },
        }

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
        request = json.loads(kwargs["stdin"])
        operation = request["operation"]
        delta: dict[str, Any] = {
            "created_page_ids": [],
            "admitted_page_ids": [],
            "closed_page_ids": [],
            "crashed_page_ids": [],
            "refused": [],
        }
        failure: str | None = None
        observation: dict[str, Any] | None = None
        if operation == "navigate":
            upstream = urlsplit(self._upstream_origin)
            target = urlsplit(request["url"])
            assert upstream.hostname is not None
            assert upstream.port is not None
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=2)
            try:
                connection.request("GET", target.path or "/")
                response = connection.getresponse()
                response.read()
                assert response.status < 500
            finally:
                connection.close()
            page = {
                "session_id": request["session_id"],
                "page_id": request["page_id"],
                "lifecycle": "active",
                "creation_epoch": 1,
                "control_epoch": 1,
                "opener_page_id": None,
                "creating_operation_id_sha256": None,
                "revision": None,
                "url": request["url"],
                "title": "Acceptance fixture",
                "last_observation_revision": None,
                "last_operation_id_sha256": hashlib.sha256(
                    request["operation_id"].encode("utf-8")
                ).hexdigest(),
                "terminal_reason": None,
                "operation_count": 1,
                "observation_count": 0,
                "ref_count": 0,
                "artifact_count": 0,
            }
            self._pages[page["page_id"]] = page
            self._active_page_id = page["page_id"]
            self._total_page_creations = 1
            self._total_operations += 1
            observation = self._observe(page)
            delta["created_page_ids"] = [page["page_id"]]
            delta["admitted_page_ids"] = [page["page_id"]]
        elif operation == "list_pages":
            self._total_operations += 1
        elif operation == "close":
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v3",
                        "worker_version": "7",
                        "playwright_version": "1.62.0",
                        "kind": "success",
                        "allocation_disposition": "retired",
                        "closed": True,
                    }
                )
            )
        else:
            page = self._pages[request["page_id"]]
            if operation == "switch_page":
                if self._active_page_id is not None and self._active_page_id != page["page_id"]:
                    current = self._pages[self._active_page_id]
                    current["lifecycle"] = "background"
                    current["control_epoch"] += 1
                    current["revision"] = f"br_acceptance_revision_{self._revision + 1}"
                page["lifecycle"] = "active"
                page["control_epoch"] += 1
                self._active_page_id = page["page_id"]
                page["operation_count"] += 1
                page["last_operation_id_sha256"] = hashlib.sha256(
                    request["operation_id"].encode("utf-8")
                ).hexdigest()
                self._total_operations += 1
                observation = self._observe(page)
            elif operation == "close_page":
                page["lifecycle"] = "closed"
                page["control_epoch"] += 1
                page["revision"] = None
                page["terminal_reason"] = "closed_by_model"
                delta["closed_page_ids"] = [page["page_id"]]
                if self._active_page_id == page["page_id"]:
                    self._active_page_id = next(
                        (
                            candidate["page_id"]
                            for candidate in sorted(
                                self._pages.values(),
                                key=lambda item: item["creation_epoch"],
                            )
                            if candidate["lifecycle"] == "background"
                        ),
                        None,
                    )
                    if self._active_page_id is not None:
                        self._pages[self._active_page_id]["lifecycle"] = "active"
            else:
                page["control_epoch"] += 1
                page["operation_count"] += 1
                page["last_operation_id_sha256"] = hashlib.sha256(
                    request["operation_id"].encode("utf-8")
                ).hexdigest()
                self._total_operations += 1
                if operation == "click" and urlsplit(page["url"]).path.startswith("/popup"):
                    path = urlsplit(page["url"]).path
                    if path == "/popup-burst":
                        failure = "resource_exhausted"
                    elif path == "/popup-redirect":
                        failure = "policy_denied"
                    else:
                        self._total_page_creations += 1
                        popup_id = f"bp_acceptance_popup_{self._total_page_creations}"
                        popup_url = (
                            "https://static.browser.test/popup"
                            if path == "/popup-cross-origin"
                            else "https://docs.browser.test/popup-child"
                        )
                        popup = {
                            **page,
                            "page_id": popup_id,
                            "lifecycle": "background",
                            "creation_epoch": self._total_page_creations,
                            "control_epoch": 1,
                            "opener_page_id": page["page_id"],
                            "creating_operation_id_sha256": hashlib.sha256(
                                request["operation_id"].encode("utf-8")
                            ).hexdigest(),
                            "revision": f"br_acceptance_popup_{self._total_page_creations}",
                            "url": popup_url,
                            "title": "Acceptance popup",
                            "last_observation_revision": None,
                            "last_operation_id_sha256": None,
                            "terminal_reason": None,
                            "operation_count": 0,
                            "observation_count": 0,
                            "ref_count": 0,
                            "artifact_count": 0,
                        }
                        self._pages[popup_id] = popup
                        delta["created_page_ids"] = [popup_id]
                        delta["admitted_page_ids"] = [popup_id]
                observation = self._observe(page)
        artifacts = []
        if operation == "screenshot":
            artifacts.append(
                {
                    "kind": "screenshot",
                    "filename": "acceptance.png",
                    "content_type": "image/png",
                    "content_base64": base64.b64encode(b"acceptance-screenshot").decode("ascii"),
                }
            )
            self._pages[request["page_id"]]["artifact_count"] += 1
        page_set = self._page_set()
        payload: dict[str, Any] = {
            "protocol_version": "cayu.browser-session.v3",
            "worker_version": "7",
            "playwright_version": "1.62.0",
            "kind": "error" if failure is not None else "success",
            "allocation_disposition": "live",
            "page_set": page_set,
            "page_delta": delta,
            "artifacts": artifacts,
        }
        if failure is not None:
            payload["error"] = failure
        elif observation is not None:
            payload["observation"] = observation
        return ExecResult(stdout=json.dumps(payload))

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return ExecutionAdmissionCandidate(
            candidate="browser-acceptance-runner",
            evidence=ExecutionCapabilityEvidence(
                subject="browser-acceptance-runner",
                claims=tuple(
                    ExecutionCapabilityClaim.available(capability)
                    for capability in (
                        "deny_by_default_network",
                        "brokered_egress",
                        "confirmed_cancellation",
                        "confirmed_cleanup",
                    )
                ),
            ),
        )

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        if name == PINNED_BROWSER_SESSION_WORKLOAD.name:
            return PINNED_BROWSER_SESSION_WORKLOAD
        return None

    def output_secret_values_present(self) -> bool:
        return False


class _ProtocolEgressAdapter(SandboxEgressAdapter):
    runner_kind = "docker"
    process_external_allocation = False
    egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

    def __init__(self, upstream_origin: str) -> None:
        self._upstream_origin = upstream_origin

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        del session_id, grants, broker
        return EgressBinding(
            env={"HTTPS_PROXY": "http://browser-acceptance-proxy.invalid:8080"},
            ca_cert_pem=b"",
            runner_kind=self.runner_kind,
            guest_ca_path="/tmp/cayu-browser-acceptance-ca.pem",
        )

    async def create_runner(self, request):  # type: ignore[no-untyped-def]
        del request
        return _ProtocolBrowserRunner(self._upstream_origin)

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        if not isinstance(runner, _ProtocolBrowserRunner):
            raise TypeError("Protocol browser fixture received another runner.")
        return "a" * 64

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        del runner
        return ExecutionCapabilityEvidence(
            subject=self.runner_kind,
            claims=tuple(
                ExecutionCapabilityClaim.available(capability)
                for capability in (
                    "deny_by_default_network",
                    "brokered_egress",
                    "confirmed_cancellation",
                    "confirmed_cleanup",
                )
            ),
        )

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        del outcome
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


class _UntrustedScriptedProviderSubclass(ScriptedModelProvider):
    pass


def _run_protocol_process_scenario(
    upstream_origin: str,
    root_value: str,
    upstream_routes: dict[str, str],
    hosts: tuple[str, ...],
    case_document: dict[str, Any],
    seccomp_value: str,
    scenario_value: str,
    session_id: str,
) -> None:
    internal_acceptance.DockerEgressAdapter = (  # ty: ignore[invalid-assignment]
        lambda **kwargs: _ProtocolEgressAdapter(upstream_origin)
    )
    internal_acceptance._process_scenario_worker(
        root_value,
        upstream_routes,
        hosts,
        case_document,
        seccomp_value,
        scenario_value,
        session_id,
    )


def _block_process_scenario_until_killed(
    root_value: str,
    upstream_routes: dict[str, str],
    hosts: tuple[str, ...],
    case_document: dict[str, Any],
    seccomp_value: str,
    scenario_value: str,
    session_id: str,
) -> None:
    del upstream_routes, hosts, case_document, seccomp_value, scenario_value, session_id
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    root = Path(root_value)
    root.joinpath("blocked-child.pid").write_text(str(os.getpid()), encoding="ascii")
    while True:
        time.sleep(1)


def _persist_trial_receipt_until_staging_is_durable(
    receipt_document: dict[str, Any],
    receipt_directory: str,
) -> None:
    receipt = acceptance_module.BrowserAcceptanceTrialReceiptV1.model_validate(receipt_document)

    def exit_before_publication(_source: object, _destination: object) -> None:
        os._exit(87)

    with patch.object(os, "link", exit_before_publication):
        acceptance_module._persist_trial_receipt(receipt, receipt_directory)


def _plan(
    tmp_path: Path,
    fixture: BrowserAcceptanceFixtureV1,
    *,
    factory_hosts: tuple[str, ...] = ("docs.browser.test",),
    navigation_url: str = "https://docs.browser.test/basic",
    provider_type: type[ScriptedModelProvider] = ScriptedModelProvider,
    repetitions: int = 1,
) -> BrowserAcceptancePlanV1:
    executable = BrowserAcceptanceCaseV1.build(
        case_id="navigation",
        category=BrowserAcceptanceCaseCategory.SUCCESS,
        expected_state=BrowserAcceptanceState.PASSED,
        semantic_oracle=BrowserAcceptanceSemanticOracle.OBSERVATION,
        semantic_success_required=True,
        required=True,
        fixture_route="/basic",
        operations=("navigate",),
        oracle_parameters={"required_operations": ["navigate"]},
    )
    unsupported = BrowserAcceptanceCaseV1.build(
        case_id="reload",
        category=BrowserAcceptanceCaseCategory.CAPABILITY,
        expected_state=BrowserAcceptanceState.UNSUPPORTED,
        semantic_oracle=BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED,
        semantic_success_required=False,
        required=True,
        operations=("reload",),
        oracle_parameters={"operation": "reload"},
    )
    cases = (executable, unsupported)
    manifest = BrowserAcceptanceManifestV1.build(
        corpus_revision=_content_revision(
            {"cases": [case.revision for case in cases]},
            "browser acceptance execution test corpus",
        ),
        suite_id="browser-acceptance-public-flow",
        mode=BrowserAcceptanceMode.DETERMINISTIC,
        enabled=True,
        trial_count=1,
        allowed_origins=("https://docs.browser.test",),
        limits=BrowserAcceptanceLimitsV1(
            max_destinations=1,
            max_browser_operations=4,
            max_model_steps=2,
            max_wall_time_ms=30_000,
            max_artifact_bytes=4 << 20,
            max_concurrency=1,
        ),
        cases=cases,
    )
    script = [
        [
            ModelStreamEvent.tool_call(
                id="browser-call",
                name="browser_session",
                arguments={
                    "operation": "navigate",
                    "url": navigation_url,
                    "operation_id": "navigation",
                },
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                }
            ),
        ],
        [
            ModelStreamEvent.text_delta("The task is complete."),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                }
            ),
        ],
    ]
    provider = provider_type(script * repetitions)
    artifact_store = LocalArtifactStore(
        tmp_path / "artifacts",
        store_id="browser-acceptance-artifacts",
    )
    factory = VirtualEgressEnvironmentFactory(
        policies={
            "browser-acceptance": BrowserEgressPolicy(
                name="browser-acceptance",
                allowed_hosts=factory_hosts,
                allowed_path_prefixes=("/",),
            )
        },
        approved_destinations=tuple(
            ApprovedEgressDestination(
                destination=host,
                policy_name="browser-acceptance",
            )
            for host in factory_hosts
        ),
        adapter=_ProtocolEgressAdapter(fixture.upstream_origin),
        upstream=HttpxUpstream(routes=fixture.upstream_routes),
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=artifact_store,
    )
    bridge = WebBridge.sandboxed_browser(
        environment=factory,
        browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        interactive=True,
        interactive_options={"max_artifact_bytes": 1 << 20, "max_operations": 4},
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="browser"),
        factory,
        default=True,
    )
    bridge.register_agent(
        app,
        AgentSpec(name="browser-agent", model="scripted-browser-v1"),
    )
    suite = EvalSuite(
        id=manifest.suite_id,
        cases=[
            EvalCase(
                id=executable.case_id,
                request=RunRequest(
                    agent_name="browser-agent",
                    messages=[Message.text("user", "Open the deterministic fixture.")],
                    max_steps=2,
                    limits=RunLimits(max_tool_calls=4, max_elapsed_seconds=30),
                ),
                assertions=[SessionCompleted()],
                metadata={"browser_acceptance_case_revision": executable.revision},
            )
        ],
    )
    return BrowserAcceptancePlanV1(
        manifest=manifest,
        eval_plan=EvalPlan(app=app, suite=suite),
        bridge=bridge,
    )


async def _project_scenario_execution(
    plan: BrowserAcceptancePlanV1,
    case: BrowserAcceptanceCaseV1,
    result: acceptance_module.BrowserAcceptanceScenarioExecutionV1,
    fixture: BrowserAcceptanceFixtureV1,
) -> BrowserAcceptanceTrialReceiptV1:
    assert result.trial.trajectory is not None
    evidence = project_assertion_evidence_view(
        result.app,
        result.trial.trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.create(
            include_tool_arguments=True,
            include_tool_results=True,
        ),
    )
    route_count = (
        fixture.request_counts().get(case.fixture_route, 0)
        if case.fixture_route is not None and case.fixture_route.startswith("/")
        else None
    )
    return project_browser_acceptance_trial(
        case=case,
        run_identity_revision="sha256:" + "9" * 64,
        trial=result.trial,
        evidence=evidence,
        fixture_route_observed=None if route_count is None else route_count > 0,
        fixture_route_request_count=route_count,
        public_operations=acceptance_module._browser_public_operations(
            acceptance_module._registered_browser_acceptance_tool(plan)
        ),
        fault=result.fault,
    )


def test_browser_acceptance_runs_through_public_app_webbridge_and_runner(tmp_path: Path) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(tmp_path, fixture),
                deterministic_fixture=fixture,
            )
        )

    assert report.aggregate.overall_status.value == "passed"
    navigation, unsupported = report.rows
    assert navigation.case_id == "navigation"
    assert navigation.semantic_state.value == "passed"
    assert navigation.diagnostic.operations[0].operation == "navigate"
    assert navigation.diagnostic.operations[0].allocation_disposition.value == "live"
    assert navigation.diagnostic.fixture_route_observed is True
    assert navigation.diagnostic.fixture_route_request_count == 1
    assert navigation.usage.input_tokens == 5
    assert navigation.usage.output_tokens == 3
    assert navigation.usage.total_tokens == 8
    assert report.runtime_identity.chromium_identity == "acceptance-fixture"
    assert report.runtime_identity.provider_name == "scripted"
    assert report.runtime_identity.model == "scripted-browser-v1"
    assert report.runtime_identity.execution_profile_fingerprint != "7" * 64
    assert unsupported.case_id == "reload"
    assert unsupported.observed_state.value == "unsupported"
    assert unsupported.diagnostic.state.value == "not_requested"


def test_browser_acceptance_uses_portable_result_evidence_after_externalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_exec = _ProtocolBrowserRunner.exec

    async def exec_with_large_snapshot(
        self: _ProtocolBrowserRunner,
        command: ExecCommand,
        **kwargs: Any,
    ) -> ExecResult:
        result = await original_exec(self, command, **kwargs)
        document = json.loads(result.stdout)
        document["observation"]["snapshot"] = "x" * 5_000
        return ExecResult(stdout=json.dumps(document))

    monkeypatch.setattr(_ProtocolBrowserRunner, "exec", exec_with_large_snapshot)
    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(tmp_path, fixture),
                deterministic_fixture=fixture,
            )
        )

    navigation = report.rows[0]
    assert navigation.semantic_state.value == "passed"
    assert navigation.completion_state.value == "complete"
    assert navigation.diagnostic.truncated_categories == ()
    assert navigation.diagnostic.operations[0].snapshot_bytes == 5_000


def test_cayu_owned_deterministic_target_binds_every_executable_manifest_case() -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = asyncio.run(build_internal_browser_acceptance(fixture))

    assert plan.eval_plan.suite is not None
    expected = tuple(
        case.case_id
        for case in plan.manifest.cases
        if case.expected_state is not BrowserAcceptanceState.UNSUPPORTED
    )
    assert tuple(case.id for case in plan.eval_plan.suite.cases) == expected
    assert plan.scenario_executor is not None
    assert plan.scenario_executor_revision is not None
    assert all(
        case.fault_scenario is not None
        for case in plan.manifest.cases
        if case.category
        in {
            BrowserAcceptanceCaseCategory.CRASH,
            BrowserAcceptanceCaseCategory.CANCELLATION,
        }
    )
    assert all(
        case.metadata["browser_acceptance_case_revision"]
        == next(
            manifest_case.revision
            for manifest_case in plan.manifest.cases
            if manifest_case.case_id == case.id
        )
        for case in plan.eval_plan.suite.cases
    )


@pytest.mark.parametrize(
    "action",
    ["click", "download", "fill", "press", "screenshot", "select", "wait"],
)
def test_stale_reference_cases_reuse_pre_action_revision_and_reference(action: str) -> None:
    case = next(
        item
        for item in deterministic_browser_acceptance_manifest().cases
        if item.case_id == f"revision-stale-ref-after-{action}"
    )
    before = {
        "session_id": "browser-session",
        "page_id": "browser-page",
        "revision": "before-action",
        "control_epoch": 1,
        "refs": [
            {"name": "Save", "ref": "save-before"},
            {"name": "Name", "ref": "name-before"},
            {"name": "Region", "ref": "region-before"},
            {"name": "Download report", "ref": "download-before"},
        ],
    }
    after = {
        **before,
        "revision": "after-action",
        "control_epoch": 2,
        "refs": [
            {"name": "Save", "ref": "save-after"},
            {"name": "Download report", "ref": "download-after"},
        ],
    }

    stale_arguments = internal_acceptance._operation_arguments(
        case_id=case.case_id,
        operation=case.operations[2],
        operation_index=2,
        fixture_route=case.fixture_route,
        results=(before, after),
    )

    assert stale_arguments["expected_revision"] == "before-action"
    assert stale_arguments["ref"] in {"save-before", "download-before"}


@pytest.mark.parametrize(
    ("case_id", "expected_dispatches"),
    [
        ("cancellation-during-intent-publication", 1),
        ("cancellation-after-dispatched-marker", 1),
        ("cancellation-during-guest-effect", 1),
        ("cancellation-during-artifact-publication", 2),
        ("cancellation-after-final-receipt", 1),
    ],
)
def test_cayu_owned_fault_executor_delivers_real_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    expected_dispatches: int,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        plan = await build_internal_browser_acceptance(fixture)
        case = next(item for item in plan.manifest.cases if item.case_id == case_id)
        executor = plan.scenario_executor
        assert executor is not None

        result = await executor(case, 1, 1, 30)

        assert result.fault.scenario is case.fault_scenario
        assert result.fault.boundary_observed is True
        assert result.fault.cancellation_delivered is True
        assert result.fault.browser_dispatches == expected_dispatches
        assert result.trial.trajectory is not None
        if case_id in {
            "cancellation-during-intent-publication",
            "cancellation-after-dispatched-marker",
        }:
            projected = await _project_scenario_execution(plan, case, result, fixture)
            assert projected.semantic_state.value == "passed"

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize(
    ("case_id", "expected_dispatches"),
    [
        ("crash-before-dispatch", 1),
        ("crash-during-execution", 2),
        ("crash-after-effect", 1),
        ("crash-during-cleanup", 2),
        ("page-allocation-loss", 1),
    ],
)
def test_cayu_owned_fault_executor_crashes_browser_without_crashing_cayu(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    expected_dispatches: int,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )

        async def simulate_browser_signal(ctx: Any, session_id: str, signal_number: int) -> None:
            del ctx, session_id, signal_number

        monkeypatch.setattr(
            internal_acceptance,
            "_signal_browser_daemon",
            simulate_browser_signal,
        )
        plan = await build_internal_browser_acceptance(fixture)
        case = next(item for item in plan.manifest.cases if item.case_id == case_id)
        executor = plan.scenario_executor
        assert executor is not None

        result = await executor(case, 1, 1, 30)

        assert result.fault.scenario is case.fault_scenario
        assert result.fault.boundary_observed is True
        assert result.fault.process_loss_observed is False
        assert result.fault.recovered_in_fresh_app is False
        assert result.fault.browser_dispatches == expected_dispatches
        projected = await _project_scenario_execution(plan, case, result, fixture)
        assert projected.semantic_state.value == "passed", projected.model_dump(mode="json")

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


def test_browser_acceptance_rejects_incomplete_browser_operation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        plan = await build_internal_browser_acceptance(fixture)
        source_case = next(
            item
            for item in plan.manifest.cases
            if item.case_id == "cancellation-after-final-receipt"
        )
        executor = plan.scenario_executor
        assert executor is not None
        result = await executor(source_case, 1, 1, 30)
        assert result.trial.trajectory is not None
        evidence = project_assertion_evidence_view(
            result.app,
            result.trial.trajectory,
            evidence_policy=EvaluationEvidencePolicySpec.create(
                include_tool_arguments=True,
                include_tool_results=True,
            ),
        )
        case = BrowserAcceptanceCaseV1.build(
            case_id="operation-evidence-contract",
            category=BrowserAcceptanceCaseCategory.SUCCESS,
            expected_state=BrowserAcceptanceState.PASSED,
            semantic_oracle=BrowserAcceptanceSemanticOracle.OBSERVATION,
            semantic_success_required=True,
            required=True,
            operations=("navigate",),
            screenshot_checkpoints=(),
            oracle_parameters={"required_operations": ["navigate"]},
        )

        for mutation in ("missing_execution", "malformed_execution", "missing_allocation"):
            document = evidence.model_dump(mode="json", exclude={"revision"})
            structured = document["tool_calls"][0]["result"]["value"]["structured"]
            if mutation == "missing_execution":
                structured.pop("execution")
            elif mutation == "malformed_execution":
                structured["execution"]["terminal"] = "future_terminal_state"
            else:
                structured.pop("allocation_disposition")
            revision_document = copy.deepcopy(document)
            if type(document.get("total_tokens")) is str:
                document["total_tokens"] = int(document["total_tokens"])
            document["revision"] = _content_revision(revision_document, "assertion evidence")
            mutated = AssertionEvidenceView.model_validate(document)
            receipt = project_browser_acceptance_trial(
                case=case,
                run_identity_revision="sha256:" + "9" * 64,
                trial=result.trial,
                evidence=mutated,
                public_operations=acceptance_module._browser_public_operations(
                    acceptance_module._registered_browser_acceptance_tool(plan)
                ),
            )

            assert receipt.semantic_state.value == "failed"
            assert receipt.completion_state.value == "incomplete"
            diagnostic = project_browser_acceptance_diagnostic(mutated)
            assert diagnostic.truncated_categories

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize(
    ("case_id", "expected_dispatches"),
    [
        ("recovery-process-loss-intent", 0),
        ("recovery-process-loss-dispatched", 0),
        ("recovery-process-loss-guest-terminal", 1),
        ("recovery-process-loss-artifact-publication", 2),
        ("recovery-process-loss-acknowledgement", 1),
    ],
)
def test_cayu_owned_fault_executor_recovers_process_loss_in_fresh_app(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    expected_dispatches: int,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        plan = await build_internal_browser_acceptance(fixture)
        case = next(item for item in plan.manifest.cases if item.case_id == case_id)
        executor = plan.scenario_executor
        assert isinstance(executor, internal_acceptance._DeterministicScenarioExecutor)
        executor._process_worker = partial(
            _run_protocol_process_scenario,
            fixture.upstream_origin,
        )

        result = await executor(case, 1, 1, 30)

        assert result.fault.process_loss_observed is True
        assert result.fault.recovered_in_fresh_app is True
        assert result.fault.browser_dispatches == expected_dispatches
        assert result.trial.trajectory is not None

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


def test_interrupted_evidence_requires_positive_runtime_event_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        plan = await build_internal_browser_acceptance(fixture)
        case = next(
            item
            for item in plan.manifest.cases
            if item.case_id == "cancellation-after-dispatched-marker"
        )
        executor = plan.scenario_executor
        assert isinstance(executor, internal_acceptance._DeterministicScenarioExecutor)
        result = await executor(case, 1, 1, 30)
        assert result.trial.session_id is not None
        journal = (
            executor._root / f"{case.case_id}-1-1" / internal_acceptance._OBSERVED_EVENTS_FILENAME
        )
        records = tuple(
            json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
        )
        assert records
        omitted_event_id = records[-1]["event_id"]
        retained = tuple(record for record in records if record["event_id"] != omitted_event_id)
        journal.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                for record in retained
            ),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="conflicts with durable recovery state"):
            await internal_acceptance._load_observed_events(
                result.app,
                journal,
                session_id=result.trial.session_id,
            )

        journal.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        original_sequence = public_event_sequence(records[-1]["event_id"])
        assert original_sequence is not None
        records[-1]["event_id"] = public_event_id(original_sequence + 1)
        journal.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="conflicts with durable recovery state"):
            await internal_acceptance._load_observed_events(
                result.app,
                journal,
                session_id=result.trial.session_id,
            )

        duplicate_records = tuple(
            json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
        )
        duplicate_records[-1]["event_id"] = duplicate_records[0]["event_id"]
        duplicate_records[-1]["event_type"] = duplicate_records[0]["event_type"]
        journal.write_text(
            "".join(
                json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                for record in duplicate_records
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="conflicts with durable recovery state"):
            await internal_acceptance._load_observed_events(
                result.app,
                journal,
                session_id=result.trial.session_id,
            )

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM-resistant process fixture is POSIX-only")
def test_cayu_owned_fault_executor_quiesces_child_before_redelivering_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        monkeypatch.setattr(
            internal_acceptance,
            "_PROCESS_TERMINATE_GRACE_SECONDS",
            0.05,
        )
        monkeypatch.setattr(
            internal_acceptance,
            "_PROCESS_KILL_GRACE_SECONDS",
            1.0,
        )
        plan = await build_internal_browser_acceptance(fixture)
        case = next(
            item for item in plan.manifest.cases if item.case_id == "recovery-process-loss-intent"
        )
        executor = plan.scenario_executor
        assert isinstance(executor, internal_acceptance._DeterministicScenarioExecutor)
        executor._process_worker = _block_process_scenario_until_killed
        task = asyncio.create_task(executor(case, 1, 1, 30))
        pid_path = executor._root / f"{case.case_id}-1-1" / "blocked-child.pid"
        for _ in range(500):
            if pid_path.is_file():
                break
            await asyncio.sleep(0.01)
        assert pid_path.is_file()
        pid = int(pid_path.read_text(encoding="ascii"))

        task.cancel()
        assert task.cancelling() == 1
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 2
        assert task.cancelled()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


def test_browser_acceptance_rejects_unbounded_public_run_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture)
        assert plan.eval_plan.suite is not None
        source_case = plan.eval_plan.suite.cases[0]
        unbounded_case = EvalCase(
            id=source_case.id,
            request=RunRequest(
                agent_name="browser-agent",
                messages=[Message.text("user", "Open the deterministic fixture.")],
                max_steps=2,
            ),
            assertions=source_case.assertions,
            metadata=source_case.metadata,
        )
        unbounded_plan = BrowserAcceptancePlanV1(
            manifest=plan.manifest,
            eval_plan=EvalPlan(
                app=plan.eval_plan.app,
                suite=EvalSuite(id=plan.manifest.suite_id, cases=[unbounded_case]),
            ),
            bridge=plan.bridge,
        )

        with pytest.raises(ValueError, match="browser-operation ceiling"):
            asyncio.run(
                run_browser_acceptance(
                    unbounded_plan,
                    deterministic_fixture=fixture,
                )
            )


def test_browser_acceptance_rejects_unregistered_plan_bridge_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        assert source.eval_plan.app is not None
        factory = source.eval_plan.app.get_environment_factory("browser")
        substitute = WebBridge.sandboxed_browser(
            environment=factory,
            browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            interactive=True,
            interactive_options={"max_artifact_bytes": 1 << 19, "max_operations": 4},
        )
        plan = BrowserAcceptancePlanV1(
            manifest=source.manifest,
            eval_plan=source.eval_plan,
            bridge=substitute,
        )
        provider = source.eval_plan.app.get_provider("scripted")

        with pytest.raises(ValueError, match="not the registered browser_session tool"):
            asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))

        assert isinstance(provider, ScriptedModelProvider)
        assert provider.requests == []


def test_browser_acceptance_rejects_split_aggregate_artifact_budget_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        manifest = BrowserAcceptanceManifestV1.build(
            corpus_revision=source.manifest.corpus_revision,
            suite_id=source.manifest.suite_id,
            mode=source.manifest.mode,
            enabled=source.manifest.enabled,
            trial_count=source.manifest.trial_count,
            allowed_origins=source.manifest.allowed_origins,
            limits=source.manifest.limits.model_copy(update={"max_artifact_bytes": (4 << 20) - 1}),
            cases=source.manifest.cases,
        )
        plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            eval_plan=source.eval_plan,
            bridge=source.bridge,
        )
        assert source.eval_plan.app is not None
        provider = source.eval_plan.app.get_provider("scripted")

        with pytest.raises(ValueError, match="aggregate artifact ceiling"):
            asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))

        assert isinstance(provider, ScriptedModelProvider)
        assert provider.requests == []


def test_browser_acceptance_rejects_broader_egress_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(
            tmp_path,
            fixture,
            factory_hosts=("docs.browser.test", "unexpected.browser.test"),
        )

        with pytest.raises(ValueError, match="manifest allowlist"):
            asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))


def test_live_browser_acceptance_binds_exact_pricing_identity(tmp_path: Path) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        limits = source.manifest.limits.model_copy(
            update={
                "max_input_tokens": 1_000,
                "max_output_tokens": 500,
                "max_estimated_cost": "1.00 USD",
            }
        )
        manifest = BrowserAcceptanceManifestV1.build(
            corpus_revision=source.manifest.corpus_revision,
            suite_id=source.manifest.suite_id,
            mode=BrowserAcceptanceMode.LIVE_PUBLIC,
            enabled=True,
            trial_count=3,
            allowed_origins=source.manifest.allowed_origins,
            limits=limits,
            cases=source.manifest.cases,
        )
        with pytest.raises(ValueError, match="exact pricing evidence"):
            BrowserAcceptancePlanV1(
                manifest=manifest,
                eval_plan=source.eval_plan,
                bridge=source.bridge,
            )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="scripted-browser-v1",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    currency="USD",
                ),
            )
        )
        plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            eval_plan=source.eval_plan,
            bridge=source.bridge,
            pricing=pricing,
            cost_currencies=("USD",),
        )
        identity = asyncio.run(inspect_browser_acceptance_runtime_identity(plan))

    assert identity.pricing_profile_fingerprint is not None
    assert identity.cost_currencies == ("USD",)


def test_live_browser_acceptance_rejects_mismatched_budget_pricing_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        assert source.eval_plan.app is not None
        assert source.eval_plan.suite is not None
        source_case = source.eval_plan.suite.cases[0]
        limits = source.manifest.limits.model_copy(
            update={
                "max_input_tokens": 1_000,
                "max_output_tokens": 500,
                "max_estimated_cost": "1.00 USD",
            }
        )
        manifest = BrowserAcceptanceManifestV1.build(
            corpus_revision=source.manifest.corpus_revision,
            suite_id=source.manifest.suite_id,
            mode=BrowserAcceptanceMode.LIVE_PUBLIC,
            enabled=True,
            trial_count=1,
            allowed_origins=source.manifest.allowed_origins,
            limits=limits,
            cases=source.manifest.cases,
        )
        report_pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="scripted-browser-v1",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    currency="USD",
                ),
            )
        )
        enforcement_pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="scripted-browser-v1",
                    input_per_million=Decimal("0.01"),
                    output_per_million=Decimal("0.02"),
                    currency="USD",
                ),
            )
        )
        bounded_case = EvalCase(
            id=source_case.id,
            request=source_case.request.model_copy(
                update={
                    "limits": RunLimits(
                        max_tool_calls=4,
                        max_elapsed_seconds=30,
                        max_input_tokens=1_000,
                        max_output_tokens=500,
                    ),
                    "budget_limits": (
                        BudgetLimit(
                            scope="app",
                            max_estimated_cost=Decimal("1"),
                            pricing=report_pricing,
                        ),
                        BudgetLimit(
                            scope="session",
                            max_estimated_cost=Decimal("1"),
                            pricing=enforcement_pricing,
                        ),
                    ),
                }
            ),
            assertions=source_case.assertions,
            metadata=source_case.metadata,
        )
        plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            eval_plan=EvalPlan(
                app=source.eval_plan.app,
                suite=EvalSuite(id=manifest.suite_id, cases=[bounded_case]),
            ),
            bridge=source.bridge,
            pricing=report_pricing,
            cost_currencies=("USD",),
        )
        provider = source.eval_plan.app.get_provider("scripted")

        with pytest.raises(ValueError, match="report pricing authority"):
            asyncio.run(run_browser_acceptance(plan))

        assert isinstance(provider, ScriptedModelProvider)
        assert provider.requests == []


def test_live_browser_acceptance_rejects_split_app_budget_authorities_before_dispatch(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        assert source.eval_plan.app is not None
        assert source.eval_plan.suite is not None
        source_case = source.eval_plan.suite.cases[0]
        manifest_cases = tuple(
            BrowserAcceptanceCaseV1.build(
                **{
                    **source.manifest.cases[0].model_dump(
                        mode="python",
                        exclude={"revision", "case_id"},
                    ),
                    "case_id": case_id,
                }
            )
            for case_id in ("navigation-a", "navigation-b")
        )
        limits = source.manifest.limits.model_copy(
            update={
                "max_model_steps": 4,
                "max_input_tokens": 1_000,
                "max_output_tokens": 500,
                "max_estimated_cost": "1.00 USD",
            }
        )
        manifest = BrowserAcceptanceManifestV1.build(
            corpus_revision=_content_revision(
                {"cases": [case.revision for case in manifest_cases]},
                "browser acceptance split-budget test corpus",
            ),
            suite_id=source.manifest.suite_id,
            mode=BrowserAcceptanceMode.LIVE_PUBLIC,
            enabled=True,
            trial_count=1,
            allowed_origins=source.manifest.allowed_origins,
            limits=limits,
            cases=manifest_cases,
        )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="scripted",
                    model="scripted-browser-v1",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    currency="USD",
                ),
            )
        )
        reservation = BudgetReservation(max_input_tokens=1_000, max_output_tokens=500)
        suite_cases = tuple(
            EvalCase(
                id=case.case_id,
                request=source_case.request.model_copy(
                    update={
                        "limits": RunLimits(
                            max_tool_calls=4,
                            max_elapsed_seconds=30,
                            max_input_tokens=1_000,
                            max_output_tokens=500,
                        ),
                        "budget_limits": (
                            BudgetLimit(
                                scope="app",
                                max_estimated_cost=Decimal("0.60"),
                                pricing=pricing,
                                reservation=reservation,
                            ),
                        ),
                    }
                ),
                assertions=source_case.assertions,
                metadata={"browser_acceptance_case_revision": case.revision},
            )
            for case in manifest_cases
        )
        plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            eval_plan=EvalPlan(
                app=source.eval_plan.app,
                suite=EvalSuite(id=manifest.suite_id, cases=suite_cases),
            ),
            bridge=source.bridge,
            pricing=pricing,
            cost_currencies=("USD",),
        )
        provider = source.eval_plan.app.get_provider("scripted")

        with pytest.raises(ValueError, match="exact reserving app-wide cost ceiling"):
            asyncio.run(run_browser_acceptance(plan))

        distinct_authority_cases = tuple(
            EvalCase(
                id=eval_case.id,
                request=eval_case.request.model_copy(
                    update={
                        "budget_limits": (
                            BudgetLimit(
                                scope="app",
                                max_estimated_cost=Decimal("1.00"),
                                pricing=pricing,
                                reservation=BudgetReservation(
                                    max_input_tokens=1_000,
                                    max_output_tokens=500 - index,
                                ),
                            ),
                        )
                    }
                ),
                assertions=eval_case.assertions,
                metadata=eval_case.metadata,
            )
            for index, eval_case in enumerate(suite_cases)
        )
        distinct_authority_plan = BrowserAcceptancePlanV1(
            manifest=manifest,
            eval_plan=EvalPlan(
                app=source.eval_plan.app,
                suite=EvalSuite(id=manifest.suite_id, cases=distinct_authority_cases),
            ),
            bridge=source.bridge,
            pricing=pricing,
            cost_currencies=("USD",),
        )
        with pytest.raises(ValueError, match="share one exact app-budget authority"):
            asyncio.run(run_browser_acceptance(distinct_authority_plan))

        assert isinstance(provider, ScriptedModelProvider)
        assert provider.requests == []


def test_deterministic_browser_acceptance_rejects_identical_untrusted_provider_subclass(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(
            tmp_path,
            fixture,
            provider_type=_UntrustedScriptedProviderSubclass,
        )
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")

        with pytest.raises(ValueError, match="exact scripted provider"):
            asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))

        assert isinstance(provider, ScriptedModelProvider)
        assert provider.requests == []


def test_browser_acceptance_wrong_fixture_route_cannot_pass_semantic_oracle(
    tmp_path: Path,
) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(
                    tmp_path,
                    fixture,
                    navigation_url="https://docs.browser.test/forms",
                ),
                deterministic_fixture=fixture,
            )
        )

    assert report.rows[0].observed_state.value == "passed"
    assert report.rows[0].semantic_state.value == "failed"
    assert report.rows[0].conformance.value == "failed"
    assert report.aggregate.overall_status.value == "failed"


def test_browser_acceptance_retains_incomplete_row_when_diagnostic_projection_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_projection(*args: Any, **kwargs: Any):
        del args, kwargs
        raise RuntimeError("projection detail must not enter the report")

    monkeypatch.setattr(
        acceptance_module,
        "project_assertion_evidence_view",
        fail_projection,
    )
    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(tmp_path, fixture),
                deterministic_fixture=fixture,
            )
        )

    navigation, unsupported = report.rows
    assert navigation.observed_state.value == "passed"
    assert navigation.completion_state.value == "incomplete"
    assert navigation.diagnostic.error_code == "diagnostic_projection_failed"
    assert navigation.usage.model_steps == 2
    assert navigation.usage.input_tokens == 5
    assert navigation.usage.output_tokens == 3
    assert navigation.usage.browser_operations is None
    assert report.aggregate.total_model_steps == 2
    assert report.aggregate.total_browser_operations is None
    assert report.aggregate.overall_status.value == "incomplete"
    assert unsupported.observed_state.value == "unsupported"
    assert "projection detail" not in report.model_dump_json()


def test_browser_acceptance_retains_trial_when_execution_cannot_initialize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_execution(*args: Any, **kwargs: Any):
        del args, kwargs
        raise RuntimeError("BROWSER_EXECUTION_SECRET_CANARY")

    monkeypatch.setattr(acceptance_module, "run_eval_suite", fail_execution)
    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(tmp_path, fixture),
                deterministic_fixture=fixture,
                receipt_directory=tmp_path / "receipts",
            )
        )

    failed, unsupported = report.rows
    assert failed.case_id == "navigation"
    assert failed.observed_state is BrowserAcceptanceState.UNAVAILABLE
    assert failed.infrastructure_state.value == "unavailable"
    assert failed.completion_state.value == "incomplete"
    assert failed.diagnostic.error_code == "trial_execution_unavailable"
    assert failed.usage.model_steps is None
    assert unsupported.observed_state is BrowserAcceptanceState.UNSUPPORTED
    assert report.aggregate.overall_status.value == "incomplete"
    assert "BROWSER_EXECUTION_SECRET_CANARY" not in report.model_dump_json()


def test_browser_acceptance_persists_receipts_and_retries_selected_trial(
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture, repetitions=2)
        initial = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")
        assert isinstance(provider, ScriptedModelProvider)
        dispatched_requests = len(provider.requests)
        resumed = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        assert resumed == initial
        assert len(provider.requests) == dispatched_requests
        retried = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
                previous_report=initial,
                retry_trials=(("navigation", 1),),
            )
        )

    receipts = tuple(receipt_directory.glob("*.trial.json"))
    assert len(receipts) == 3
    assert retried.rows[0].attempt_number == 2
    assert retried.rows[1].attempt_number == 1
    assert retried.prior_rows == (initial.rows[0],)
    assert retried.source_report_revision == initial.revision


def test_browser_acceptance_retry_replays_committed_attempt_after_acknowledgement_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    original_persist = acceptance_module._persist_trial_receipt
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture, repetitions=2)
        initial = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")
        assert isinstance(provider, ScriptedModelProvider)

        def commit_then_interrupt(receipt, directory):  # type: ignore[no-untyped-def]
            original_persist(receipt, directory)
            if receipt.case_id == "navigation" and receipt.attempt_number == 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(
            acceptance_module,
            "_persist_trial_receipt",
            commit_then_interrupt,
        )
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                run_browser_acceptance(
                    plan,
                    deterministic_fixture=fixture,
                    receipt_directory=receipt_directory,
                    previous_report=initial,
                    retry_trials=(("navigation", 1),),
                )
            )
        requests_after_commit = len(provider.requests)
        monkeypatch.setattr(
            acceptance_module,
            "_persist_trial_receipt",
            original_persist,
        )

        replayed = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
                previous_report=initial,
                retry_trials=(("navigation", 1),),
            )
        )

    assert len(provider.requests) == requests_after_commit
    assert replayed.rows[0].attempt_number == 2
    assert replayed.prior_rows == (initial.rows[0],)


def test_browser_acceptance_prepared_attempt_is_not_redispatched_after_receipt_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    original_persist = acceptance_module._persist_trial_receipt
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture)
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")
        assert isinstance(provider, ScriptedModelProvider)

        def lose_before_receipt(receipt, directory):  # type: ignore[no-untyped-def]
            if receipt.case_id == "navigation":
                raise KeyboardInterrupt
            original_persist(receipt, directory)

        monkeypatch.setattr(
            acceptance_module,
            "_persist_trial_receipt",
            lose_before_receipt,
        )
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                run_browser_acceptance(
                    plan,
                    deterministic_fixture=fixture,
                    receipt_directory=receipt_directory,
                )
            )
        requests_after_interruption = len(provider.requests)
        assert requests_after_interruption == 2
        assert tuple(receipt_directory.glob("*.intent.json"))
        monkeypatch.setattr(
            acceptance_module,
            "_persist_trial_receipt",
            original_persist,
        )

        replayed = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )

    assert len(provider.requests) == requests_after_interruption
    assert replayed.rows[0].completion_state.value == "incomplete"
    assert replayed.rows[0].diagnostic.error_code == "trial_execution_interrupted"
    assert not tuple(receipt_directory.glob("*.intent.json"))


def test_browser_acceptance_serializes_live_owner_and_journal_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        receipt_directory = tmp_path / "receipts"
        plan = _plan(tmp_path, fixture)
        entered = asyncio.Event()
        release = asyncio.Event()
        dispatches = 0
        original_run = acceptance_module.run_eval_suite

        async def blocking_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal dispatches
            dispatches += 1
            entered.set()
            await release.wait()
            return await original_run(*args, **kwargs)

        monkeypatch.setattr(acceptance_module, "run_eval_suite", blocking_run)
        first = asyncio.create_task(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        cancelled_waiter = asyncio.create_task(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        await asyncio.sleep(0.1)
        assert dispatches == 1
        assert not second.done()
        cancelled_waiter.cancel()
        assert cancelled_waiter.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        assert cancelled_waiter.cancelled()
        release.set()
        first_report, second_report = await asyncio.gather(first, second)
        assert first_report == second_report
        assert dispatches == 1

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock regression")
def test_browser_acceptance_journal_wait_is_bounded_by_campaign_deadline(
    tmp_path: Path,
) -> None:
    import fcntl

    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    descriptor = os.open(
        receipt_directory / ".browser-acceptance.lock",
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with BrowserAcceptanceFixtureV1() as fixture:
            source = _plan(tmp_path, fixture)
            manifest = BrowserAcceptanceManifestV1.build(
                corpus_revision=source.manifest.corpus_revision,
                suite_id=source.manifest.suite_id,
                mode=source.manifest.mode,
                enabled=source.manifest.enabled,
                trial_count=source.manifest.trial_count,
                allowed_origins=source.manifest.allowed_origins,
                limits=source.manifest.limits.model_copy(update={"max_wall_time_ms": 50}),
                cases=source.manifest.cases,
            )
            plan = BrowserAcceptancePlanV1(
                manifest=manifest,
                eval_plan=source.eval_plan,
                bridge=source.bridge,
            )
            assert source.eval_plan.app is not None
            provider = source.eval_plan.app.get_provider("scripted")

            with pytest.raises(TimeoutError, match="journal ownership"):
                asyncio.run(
                    run_browser_acceptance(
                        plan,
                        deterministic_fixture=fixture,
                        receipt_directory=receipt_directory,
                    )
                )

            assert isinstance(provider, ScriptedModelProvider)
            assert provider.requests == []
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@pytest.mark.parametrize("kind", ["trial", "intent"])
def test_browser_acceptance_discards_partial_journal_staging(
    kind: str,
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    staging = receipt_directory / f".{('a' * 64)}.{kind}.json.partial.staging"
    staging.write_bytes(b'{"record_type":')

    with BrowserAcceptanceFixtureV1() as fixture:
        report = asyncio.run(
            run_browser_acceptance(
                _plan(tmp_path, fixture),
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )

    assert report.aggregate.overall_status.value == "passed"
    assert not staging.exists()


def test_browser_acceptance_recovers_fsynced_staging_after_process_loss_without_redispatch(
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture, repetitions=2)
        initial = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")
        assert isinstance(provider, ScriptedModelProvider)
        prior = initial.rows[0]
        future = type(prior).build(
            **{
                field_name: getattr(prior, field_name)
                for field_name in type(prior).model_fields
                if field_name not in {"revision", "row_id", "attempt_number", "conformance"}
            },
            attempt_number=2,
        )
        process = multiprocessing.get_context("spawn").Process(
            target=_persist_trial_receipt_until_staging_is_durable,
            args=(future.model_dump(mode="json"), str(receipt_directory)),
        )
        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert process.exitcode == 87
        process.close()
        assert not tuple(receipt_directory.glob(f"{future.row_id[7:]}*.trial.json"))
        assert tuple(receipt_directory.glob(".*.trial.json.*.staging"))
        requests_after_process_loss = len(provider.requests)

        replayed = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
                previous_report=initial,
                retry_trials=(("navigation", 1),),
            )
        )

    assert len(provider.requests) == requests_after_process_loss
    assert replayed.rows[0] == future
    assert not tuple(receipt_directory.glob(".*.trial.json.*.staging"))


def test_browser_acceptance_retry_rejects_future_journal_attempt_before_dispatch(
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "receipts"
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture)
        initial = asyncio.run(
            run_browser_acceptance(
                plan,
                deterministic_fixture=fixture,
                receipt_directory=receipt_directory,
            )
        )
        assert plan.eval_plan.app is not None
        provider = plan.eval_plan.app.get_provider("scripted")
        assert isinstance(provider, ScriptedModelProvider)
        requests_before_retry = len(provider.requests)
        prior = initial.rows[0]
        future = type(prior).build(
            **{
                field_name: getattr(prior, field_name)
                for field_name in type(prior).model_fields
                if field_name not in {"revision", "row_id", "attempt_number", "conformance"}
            },
            attempt_number=3,
        )
        acceptance_module._persist_trial_receipt(future, receipt_directory)

        with pytest.raises(ValueError, match="future attempt"):
            asyncio.run(
                run_browser_acceptance(
                    plan,
                    deterministic_fixture=fixture,
                    receipt_directory=receipt_directory,
                    previous_report=initial,
                    retry_trials=(("navigation", 1),),
                )
            )

    assert len(provider.requests) == requests_before_retry


def test_fault_case_requires_owned_scenario_executor(tmp_path: Path) -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        source = _plan(tmp_path, fixture)
        fault_case = BrowserAcceptanceCaseV1.build(
            case_id="navigation",
            category=BrowserAcceptanceCaseCategory.CRASH,
            expected_state=BrowserAcceptanceState.AMBIGUOUS,
            semantic_oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
            semantic_success_required=False,
            fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_DISPATCHED,
            required=True,
            fixture_route="/basic",
            operations=("navigate",),
            oracle_parameters={"error": "outcome_ambiguous"},
        )
        manifest = BrowserAcceptanceManifestV1.build(
            corpus_revision=_content_revision(
                {"cases": [fault_case.revision]},
                "browser acceptance fault execution test corpus",
            ),
            suite_id=source.manifest.suite_id,
            mode=BrowserAcceptanceMode.DETERMINISTIC,
            enabled=True,
            trial_count=1,
            allowed_origins=source.manifest.allowed_origins,
            limits=source.manifest.limits,
            cases=(fault_case,),
        )

        with pytest.raises(ValueError, match="scenario executor"):
            BrowserAcceptancePlanV1(
                manifest=manifest,
                eval_plan=source.eval_plan,
                bridge=source.bridge,
            )


def test_browser_acceptance_command_writes_both_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded: dict[str, BrowserAcceptancePlanV1] = {}

    async def load_plan(
        target: str,
        *,
        deterministic_fixture: BrowserAcceptanceFixtureV1 | None = None,
    ) -> BrowserAcceptancePlanV1:
        assert target == command._DETERMINISTIC_TARGET
        assert deterministic_fixture is not None
        if "plan" in loaded:
            return loaded["plan"]
        plan = _plan(tmp_path, deterministic_fixture)
        loaded["plan"] = plan
        return plan

    monkeypatch.setattr(command, "_load_plan", load_plan)
    monkeypatch.setattr(
        command,
        "deterministic_browser_acceptance_manifest",
        lambda: loaded["plan"].manifest,
    )
    output = tmp_path / "reports"

    status = asyncio.run(
        command._run(
            argparse.Namespace(
                target=None,
                mode="deterministic",
                output_directory=output,
                resume_report=None,
                retry=[],
            )
        )
    )

    assert status == 0
    summary = capsys.readouterr().out
    assert '"overall_status":"passed"' in summary
    assert len(tuple(output.glob("*.json"))) == 1
    assert len(tuple(output.glob("*.html"))) == 1
    plan = loaded["plan"]
    assert plan.eval_plan.app is not None
    provider = plan.eval_plan.app.get_provider("scripted")
    assert isinstance(provider, ScriptedModelProvider)
    requests_after_first_run = len(provider.requests)

    html_path = next(output.glob("*.html"))
    html_path.unlink()
    replay_status = asyncio.run(
        command._run(
            argparse.Namespace(
                target=None,
                mode="deterministic",
                output_directory=output,
                resume_report=None,
                retry=[],
            )
        )
    )

    assert replay_status == 0
    assert len(provider.requests) == requests_after_first_run
    assert html_path.exists()


def test_authenticated_browser_acceptance_stops_before_loading_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def unexpected_load(target: str) -> BrowserAcceptancePlanV1:
        raise AssertionError(f"loaded disabled target {target}")

    monkeypatch.setattr(command, "_load_plan", unexpected_load)

    with pytest.raises(RuntimeError, match="disabled"):
        asyncio.run(
            command._run(
                argparse.Namespace(
                    target="application.acceptance:authenticated",
                    mode="live_authenticated",
                    output_directory=tmp_path,
                )
            )
        )


def test_browser_acceptance_command_error_does_not_render_target_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "BROWSER_COMMAND_SECRET_CANARY"
    monkeypatch.setattr(
        command,
        "_arguments",
        lambda: argparse.Namespace(
            target="application.acceptance:build",
            mode="deterministic",
            output_directory=tmp_path,
        ),
    )

    async def fail(args: argparse.Namespace) -> int:
        del args
        raise RuntimeError(secret)

    monkeypatch.setattr(command, "_run", fail)

    assert command.main() == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert captured.err == "browser acceptance unavailable (builtins.RuntimeError)\n"
