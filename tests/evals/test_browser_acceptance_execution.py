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
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import scripts.run_browser_acceptance as command

from cayu import (
    AESGCMBrowserProfileKeyAuthority,
    AgentSpec,
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileDestinationPolicy,
    BrowserProfileScope,
    BudgetLimit,
    BudgetReservation,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    LocalArtifactStore,
    Message,
    ModelPrice,
    PriceBook,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    SQLiteBrowserProfileStore,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolResult,
    ToolSpec,
    VirtualEgressEnvironmentFactory,
    WebBridge,
)
from cayu._validation import freeze_json_value
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
    ExecutionExecutableEvidence,
    ExecutionToolRequirementEvidence,
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
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runners import (
    PINNED_BROWSER_SESSION_WORKLOAD,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkloadAuthority,
)
from cayu.runtime._event_projection import public_event_id, public_event_sequence


class _ProtocolBrowserRunner(Runner):
    def __init__(self, upstream_origin: str, evidence_path: str | None = None) -> None:
        self._upstream_origin = upstream_origin
        self._revision = 0
        self._pages: dict[str, dict[str, Any]] = {}
        self._active_page_id: str | None = None
        self._total_operations = 0
        self._total_observations = 0
        self._total_page_creations = 0
        self._evidence_path = evidence_path
        self._current_url = "https://docs.browser.test/basic"
        self._profile_state: dict[str, Any] = {"cookies": [], "origins": []}
        self._profile_active = False
        self.operations: list[str] = []

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
                    "url": None if self._profile_active else page["url"],
                    "title": None if self._profile_active else page["title"],
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
        session_component = hashlib.sha256(page["session_id"].encode("utf-8")).hexdigest()[:16]
        page["revision"] = f"br_acceptance_{session_component}_{self._revision}"
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
        if urlsplit(page["url"]).path == "/upload":
            names += ("Upload file",)
        page["ref_count"] += len(names)
        refs: list[dict[str, Any]] = [
            {
                "ref": "ref_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
                "role": "textbox" if name in {"Account", "Frame value", "Name"} else "button",
                "name": name,
            }
            for name in names
        ]
        for item in refs:
            if item["name"] == "Upload file":
                item.update(element_type="file_input", allows_multiple_files=False)
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
                "worker_protocol": "cayu.browser-session.v4",
                "worker_version": "13",
            },
        }

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
        request = json.loads(kwargs["stdin"])
        operation = request["operation"]
        self.operations.append(operation)
        if operation == "profile_restore":
            profile = request["browser_profile"]
            self._profile_state = copy.deepcopy(profile["restore_state"])
            self._profile_active = True
            self._record_evidence(
                operation,
                cookie_count=len(self._profile_state["cookies"]),
                origin_count=len(self._profile_state["origins"]),
                session_id=request["session_id"],
            )
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "13",
                        "playwright_version": "1.62.0",
                        "kind": "profile_restore",
                        "allocation_disposition": "live",
                        "profile_restored": True,
                    }
                )
            )
        if operation == "profile_checkpoint":
            self._record_evidence(
                operation,
                cookie_count=len(self._profile_state["cookies"]),
                origin_count=len(self._profile_state["origins"]),
                session_id=request["session_id"],
            )
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "13",
                        "playwright_version": "1.62.0",
                        "kind": "profile_checkpoint",
                        "allocation_disposition": "live",
                        "profile_state": copy.deepcopy(self._profile_state),
                    }
                )
            )
        if operation == "close":
            self._record_evidence(operation)
            return ExecResult(
                stdout=json.dumps(
                    {
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "13",
                        "playwright_version": "1.62.0",
                        "kind": "closed",
                        "allocation_disposition": "retired",
                    }
                )
            )
        delta: dict[str, Any] = {
            "created_page_ids": [],
            "admitted_page_ids": [],
            "closed_page_ids": [],
            "crashed_page_ids": [],
            "refused": [],
        }
        failure: str | None = None
        observation: dict[str, Any] | None = None
        title = "Acceptance fixture"
        if operation == "navigate":
            self._current_url = request["url"]
            upstream = urlsplit(self._upstream_origin)
            target = urlsplit(request["url"])
            assert upstream.hostname is not None
            assert upstream.port is not None
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=2)
            try:
                cookies = "; ".join(
                    f"{item['name']}={item['value']}"
                    for item in self._profile_state["cookies"]
                    if item["domain"] == target.hostname
                    and (item["expires"] == -1 or item["expires"] > time.time())
                )
                connection.request(
                    "GET",
                    target.path or "/",
                    headers=({"Cookie": cookies} if cookies else {}),
                )
                response = connection.getresponse()
                body = response.read().decode("utf-8")
                assert response.status < 500
                title_start = body.find("<title>")
                title_end = body.find("</title>", title_start + 7)
                if title_start >= 0 and title_end > title_start:
                    title = body[title_start + 7 : title_end]
                if target.path == "/auth/login":
                    self._profile_state = {
                        "cookies": [
                            {
                                "name": "cayu_fixture_session",
                                "value": "active",
                                "domain": "docs.browser.test",
                                "path": "/",
                                "expires": -1,
                                "httpOnly": True,
                                "secure": True,
                                "sameSite": "Lax",
                            }
                        ],
                        "origins": [
                            {
                                "origin": "https://docs.browser.test",
                                "localStorage": [
                                    {"name": "cayu_fixture_session", "value": "active"}
                                ],
                            }
                        ],
                    }
                elif target.path == "/auth/login-expired":
                    self._profile_state = {
                        "cookies": [],
                        "origins": [
                            {
                                "origin": "https://docs.browser.test",
                                "localStorage": [
                                    {"name": "cayu_fixture_session", "value": "expired"}
                                ],
                            }
                        ],
                    }
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
                "title": title,
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
                        "protocol_version": "cayu.browser-session.v4",
                        "worker_version": "13",
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
                    session_component = hashlib.sha256(
                        current["session_id"].encode("utf-8")
                    ).hexdigest()[:16]
                    current["revision"] = f"br_acceptance_{session_component}_{self._revision + 1}"
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
        evidence_page = self._pages.get(request.get("page_id"))
        self._record_evidence(
            operation,
            page_id=request.get("page_id"),
            revision=(
                observation["revision"]
                if observation is not None
                else (evidence_page or {}).get("revision")
            ),
            session_id=request["session_id"],
            title=(
                observation["title"]
                if observation is not None
                else (evidence_page or {}).get("title")
            ),
            url=(
                observation["url"]
                if observation is not None
                else (evidence_page or {}).get("url", self._current_url)
            ),
        )
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
            "protocol_version": "cayu.browser-session.v4",
            "worker_version": "13",
            "playwright_version": "1.62.0",
            "kind": "error" if failure is not None else "success",
            "allocation_disposition": "live",
            "page_set": page_set,
            "page_delta": delta,
            "artifacts": artifacts,
            "profile_output_protected": self._profile_active,
        }
        if failure is not None:
            payload["error"] = failure
        elif observation is not None:
            payload["observation"] = observation
        if operation == "upload":
            files = request["upload_files"]
            assert len(files) == 1
            assert base64.b64decode(files[0]["content_base64"]) == (
                b"bounded browser acceptance upload\n"
            )
            payload["operation_evidence"] = {
                "operation": "upload",
                "selection_state": "selected",
                "selected_file_count": 1,
            }
            upstream = urlsplit(self._upstream_origin)
            assert upstream.hostname is not None
            connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=2)
            try:
                connection.request("GET", "/effect/upload-selected")
                response = connection.getresponse()
                assert response.status == 204
                response.read()
            finally:
                connection.close()
        return ExecResult(stdout=json.dumps(payload))

    def _record_evidence(self, operation: str, **fields: object) -> None:
        if self._evidence_path is None:
            return
        document = {"operation": operation, **fields}
        with Path(self._evidence_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(document, sort_keys=True) + "\n")

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        # This protocol fixture implements the fixed worker below, not arbitrary
        # requested executables. Its evidence is simulated, never live Docker proof.
        worker = ToolExecutableRequirement(executable=PINNED_BROWSER_SESSION_WORKLOAD.command[0])
        now = datetime.now(UTC)
        fingerprint = "sha256:" + "e" * 64
        return ExecutionAdmissionCandidate(
            candidate="docker",
            evidence=ExecutionCapabilityEvidence(
                subject="docker",
                environment_fingerprint=fingerprint,
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=(
                        ExecutionExecutableEvidence(
                            executable=worker.executable,
                            requirement_fingerprint=worker.fingerprint,
                            state="live_verified",
                            observed_at=now,
                            valid_until=now + timedelta(seconds=300),
                        ),
                    ),
                ),
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

    def __init__(self, upstream_origin: str, evidence_path: str | None = None) -> None:
        self._upstream_origin = upstream_origin
        self._evidence_path = evidence_path

    def execution_admission_evidence_for(self, requirements):
        # The fixture supports only the protocol worker, regardless of requests.
        worker = ToolExecutableRequirement(executable=PINNED_BROWSER_SESSION_WORKLOAD.command[0])
        fingerprint = "sha256:" + "e" * 64
        return self.execution_capability_evidence().model_copy(
            update={
                "environment_fingerprint": fingerprint,
                "tool_requirements": ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=(
                        ExecutionExecutableEvidence(
                            executable=worker.executable,
                            requirement_fingerprint=worker.fingerprint,
                            state="declared",
                        ),
                    ),
                ),
            }
        )

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
        return _ProtocolBrowserRunner(self._upstream_origin, self._evidence_path)

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
    control_server_container_id: str | None = None,
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
        control_server_container_id,
    )


def _run_browser_profile_acceptance_process(
    upstream_origin: str,
    upstream_routes: dict[str, str],
    store_path: str,
    evidence_path: str,
    profile_id: str,
    navigation_path: str,
    clock_value: str,
    use_profile: bool,
) -> None:
    async def scenario() -> None:
        observed_at = datetime.fromisoformat(clock_value)
        profile_store = SQLiteBrowserProfileStore(
            store_path,
            store_id="browser-acceptance-profiles",
            clock=lambda: observed_at,
        )
        profile_binding: BrowserProfileBinding | None = None
        if use_profile:
            profile_binding = BrowserProfileBinding.build(
                scope=BrowserProfileScope.build(
                    application_id="browser-acceptance",
                    tenant_id="credential-free-fixture",
                    sharing_scope="browser-profile-v1",
                ),
                destination_policy=BrowserProfileDestinationPolicy.build(
                    ("https://docs.browser.test",)
                ),
                browser_protocol="cayu.browser-session.v4",
                browser_worker_version="13",
                store=profile_store,
                key_authority=AESGCMBrowserProfileKeyAuthority(
                    authority_id="browser-acceptance-key-v1",
                    key=b"browser-profile-fixture-key-0001",
                ),
                profile_id=profile_id,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                checkpoint_policy=(BrowserProfileCheckpointPolicy.AFTER_TERMINAL_OPERATION),
                lease_seconds=60,
            )
            await profile_binding.initialize()
        adapter = _ProtocolEgressAdapter(upstream_origin, evidence_path)
        factory = VirtualEgressEnvironmentFactory(
            policies={
                "browser-profile-acceptance": BrowserEgressPolicy(
                    name="browser-profile-acceptance",
                    allowed_hosts=("docs.browser.test",),
                    allowed_path_prefixes=("/",),
                )
            },
            approved_destinations=(
                ApprovedEgressDestination(
                    destination="docs.browser.test",
                    policy_name="browser-profile-acceptance",
                ),
            ),
            adapter=adapter,
            upstream=HttpxUpstream(routes=upstream_routes),
            image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            artifact_store=LocalArtifactStore(
                Path(store_path).with_suffix(".artifacts"),
                store_id="browser-profile-acceptance-artifacts",
            ),
        )
        bridge = WebBridge.sandboxed_browser(
            environment=factory,
            browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            interactive=True,
            interactive_options={
                "idle_timeout_seconds": 1,
                "max_wait_ms": 0,
                "max_operations": 4,
            },
            browser_profile=profile_binding,
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="browser-profile-call",
                        name="browser_session",
                        arguments={
                            "operation": "navigate",
                            "url": f"https://docs.browser.test{navigation_path}",
                            "operation_id": "browser-profile-navigation",
                        },
                    ),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("complete"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "stop",
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    ),
                ],
            ]
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
            AgentSpec(name="browser-agent", model="browser-profile-fixture-v1"),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="browser-agent",
                    session_id=(
                        f"browser-profile-{profile_id}-{navigation_path.rsplit('/', 1)[-1]}"
                    ),
                    messages=[Message.text("user", "Navigate once.")],
                    max_steps=2,
                    limits=RunLimits(max_tool_calls=1, max_elapsed_seconds=15),
                )
            )
        ]
        if not events or events[-1].type != "session.completed":
            raise AssertionError("browser profile acceptance session did not complete")
        terminal_tool = next(
            (
                event
                for event in events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ),
            None,
        )
        if terminal_tool is None:
            raise AssertionError("browser profile acceptance tool did not settle")
        result = terminal_tool.payload.get("result")
        if (
            type(result) is not dict
            or result.get("is_error") is not False
            or type(result.get("structured")) is not dict
            or result["structured"].get("access_state") != "available"
        ):
            error_code = (
                result.get("structured", {}).get("error")
                if type(result) is dict and type(result.get("structured")) is dict
                else "invalid_result"
            )
            raise AssertionError(
                f"browser profile acceptance result was not published: {error_code}"
            )
        await profile_store.close()

    asyncio.run(scenario())


def _block_process_scenario_until_killed(
    root_value: str,
    upstream_routes: dict[str, str],
    hosts: tuple[str, ...],
    case_document: dict[str, Any],
    seccomp_value: str,
    scenario_value: str,
    session_id: str,
    control_server_container_id: str | None = None,
) -> None:
    del control_server_container_id
    del upstream_routes, hosts, case_document, seccomp_value, scenario_value, session_id
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    root = Path(root_value)
    pid_path = root / "blocked-child.pid"
    staged_pid = pid_path.with_suffix(".tmp")
    staged_pid.write_text(str(os.getpid()), encoding="ascii")
    staged_pid.replace(pid_path)
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
    environment_profile_identity=None,
    session_store=None,
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
        case_id="unsupported-script",
        category=BrowserAcceptanceCaseCategory.CAPABILITY,
        expected_state=BrowserAcceptanceState.UNSUPPORTED,
        semantic_oracle=BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED,
        semantic_success_required=False,
        required=True,
        operations=("evaluate_script",),
        oracle_parameters={"operation": "evaluate_script"},
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
        execution_profile_identity=environment_profile_identity,
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
    app = CayuApp(enable_logging=False, session_store=session_store)
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
        recovered_tool_calls=result.recovered_tool_calls,
    )


@pytest.mark.parametrize("requirement_kind", ["worker", "unknown", "different_probe"])
def test_protocol_browser_fixture_proves_only_its_fixed_worker(requirement_kind: str) -> None:
    worker = PINNED_BROWSER_SESSION_WORKLOAD.command[0]
    dependency = ToolExecutableRequirement(
        executable="unknown_fixture_program" if requirement_kind == "unknown" else worker,
        probe_arguments=() if requirement_kind == "different_probe" else None,
    )

    class FixtureTool(Tool):
        spec = ToolSpec(
            name="fixture_tool",
            execution_requirements=(
                ToolExecutionRequirement(name="worker", alternatives=(dependency,)),
            ),
        )

        async def run(self, ctx, args):
            raise AssertionError("Admission must not execute a fixture tool")

    async def run():
        runner = _ProtocolBrowserRunner("http://unused.invalid")
        provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="fixture"), runner=runner), default=True
        )
        app.register_agent(AgentSpec(name="fixture", model="scripted"), tools=[FixtureTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="fixture",
                    session_id=f"fixed-worker-{requirement_kind}",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        admitted = requirement_kind == "worker"
        assert len(provider.requests) == int(admitted)
        assert any(event.type is EventType.SESSION_COMPLETED for event in events) is admitted
        assert runner.operations == []

    asyncio.run(run())


def test_browser_acceptance_rejects_factory_script_identity(tmp_path: Path) -> None:
    def provider_factory(events):
        return ScriptedModelProvider(response_factory=lambda request: events[0])

    with BrowserAcceptanceFixtureV1() as fixture:
        plan = _plan(tmp_path, fixture, provider_type=provider_factory)
        with pytest.raises(ValueError, match="requires positional scripted batches"):
            asyncio.run(inspect_browser_acceptance_runtime_identity(plan))


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
    assert unsupported.case_id == "unsupported-script"
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


def test_browser_profile_restores_authenticated_fixture_state_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    observed_at = datetime(2026, 1, 2, tzinfo=UTC)

    def run_phase(
        fixture: BrowserAcceptanceFixtureV1,
        *,
        profile_id: str,
        navigation_path: str,
        phase: str,
        seconds_after_start: int,
        use_profile: bool = True,
    ) -> list[dict[str, Any]]:
        evidence_path = tmp_path / f"{profile_id}-{phase}.jsonl"
        process = context.Process(
            target=_run_browser_profile_acceptance_process,
            args=(
                fixture.upstream_origin,
                fixture.upstream_routes,
                str(tmp_path / f"{profile_id}.sqlite"),
                str(evidence_path),
                profile_id,
                navigation_path,
                (observed_at + timedelta(seconds=seconds_after_start)).isoformat(),
                use_profile,
            ),
        )
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        exit_code = process.exitcode
        process.close()
        assert exit_code == 0
        return [json.loads(line) for line in evidence_path.read_text().splitlines()]

    with BrowserAcceptanceFixtureV1() as fixture:
        login = run_phase(
            fixture,
            profile_id="bprof_acceptance_active",
            navigation_path="/auth/login",
            phase="login",
            seconds_after_start=0,
        )
        restored = run_phase(
            fixture,
            profile_id="bprof_acceptance_active",
            navigation_path="/auth/account",
            phase="restored",
            # The first worker disappears without a positive remote-close
            # acknowledgement.  A fresh worker must wait for the finite
            # writer lease rather than overlap the possibly live allocation.
            seconds_after_start=61,
        )
        credentialless = run_phase(
            fixture,
            profile_id="bprof_acceptance_no_profile",
            navigation_path="/auth/account",
            phase="credentialless",
            seconds_after_start=31,
            use_profile=False,
        )
        expired_login = run_phase(
            fixture,
            profile_id="bprof_acceptance_expired",
            navigation_path="/auth/login-expired",
            phase="expired-login",
            seconds_after_start=0,
        )
        expired_restore = run_phase(
            fixture,
            profile_id="bprof_acceptance_expired",
            navigation_path="/auth/account",
            phase="expired-restore",
            seconds_after_start=61,
        )

    assert [item["operation"] for item in login] == [
        "profile_restore",
        "navigate",
        "profile_checkpoint",
    ]
    assert login[0]["cookie_count"] == 0
    assert login[2]["cookie_count"] == 1
    assert [item["operation"] for item in restored] == [
        "profile_restore",
        "navigate",
        "profile_checkpoint",
    ]
    assert restored[0]["cookie_count"] == 1
    assert restored[1]["title"] == "Authenticated fixture account"
    assert restored[0]["session_id"] == restored[1]["session_id"]
    assert login[0]["session_id"] != restored[0]["session_id"]
    assert login[1]["page_id"] != restored[1]["page_id"]
    assert login[1]["revision"] != restored[1]["revision"]

    assert [item["operation"] for item in credentialless] == ["navigate"]
    assert credentialless[0]["title"] == "Signed out fixture account"
    assert expired_login[2]["cookie_count"] == 0
    assert expired_restore[0]["cookie_count"] == 0
    assert expired_restore[1]["title"] == "Signed out fixture account"

    raw_profile_files = tuple(tmp_path.glob("bprof_acceptance_*.sqlite*"))
    assert raw_profile_files
    assert all(
        b"cayu_fixture_session" not in path.read_bytes()
        for path in raw_profile_files
        if path.is_file()
    )


def test_recovered_browser_planner_accepts_immutable_durable_json_wrappers() -> None:
    recovered = freeze_json_value(
        {
            "session_id": "bs_recovered",
            "page_id": "bp_recovered",
            "revision": "br_recovered",
            "refs": [{"ref": "ref_submit", "name": "Submit"}],
            "page_set": {
                "pages": [
                    {
                        "page_id": "bp_recovered",
                        "control_epoch": 3,
                        "lifecycle": "active",
                    }
                ]
            },
            "portable_result_evidence": {
                "structured": {
                    "session_id": "bs_recovered",
                    "page_id": "bp_recovered",
                    "revision": "br_recovered",
                    "allocation_disposition": "live",
                }
            },
        }
    )

    state = internal_acceptance._latest_browser_state((recovered,))
    compact = internal_acceptance._compact_recovered_browser_result(
        recovered,
        is_error=False,
    )

    assert state["control_epoch"] == 3
    assert (
        internal_acceptance._latest_page_set((recovered,))["pages"][0]["page_id"] == "bp_recovered"
    )
    assert internal_acceptance._ref(recovered, ("Submit",)) == "ref_submit"
    assert compact["structured"] == {
        "session_id": "bs_recovered",
        "page_id": "bp_recovered",
        "revision": "br_recovered",
        "allocation_disposition": "live",
    }


def test_cayu_owned_deterministic_target_binds_every_executable_manifest_case() -> None:
    async def raw_profile_fingerprints(plan: BrowserAcceptancePlanV1) -> tuple[str, ...]:
        assert plan.eval_plan.app is not None
        assert plan.eval_plan.suite is not None
        fingerprints: list[str] = []
        for case in plan.eval_plan.suite.cases:
            fingerprints.append(
                await plan.eval_plan.app.inspect_run_execution_profile(case.request)
            )
        return tuple(fingerprints)

    with BrowserAcceptanceFixtureV1() as fixture:
        plan = asyncio.run(build_internal_browser_acceptance(fixture))
        case_profile_fingerprints = asyncio.run(raw_profile_fingerprints(plan))
        runtime_identity = asyncio.run(inspect_browser_acceptance_runtime_identity(plan))

    assert plan.eval_plan.suite is not None
    assert len(set(case_profile_fingerprints)) > 1
    assert len(runtime_identity.execution_profile_fingerprint) == 64
    assert runtime_identity.execution_profile_fingerprint not in case_profile_fingerprints
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


def test_cayu_owned_deterministic_target_binds_each_case_execution_profile() -> None:
    with BrowserAcceptanceFixtureV1() as fixture:
        plan = asyncio.run(build_internal_browser_acceptance(fixture))
        assert plan.eval_plan.app is not None
        assert plan.eval_plan.suite is not None
        case_profiles = [
            {
                "case_id": eval_case.id,
                "case_revision": eval_case.metadata["browser_acceptance_case_revision"],
                "execution_profile_fingerprint": asyncio.run(
                    plan.eval_plan.app.inspect_run_execution_profile(eval_case.request)
                ),
            }
            for eval_case in plan.eval_plan.suite.cases
        ]
        identity = asyncio.run(inspect_browser_acceptance_runtime_identity(plan))

    assert len({item["execution_profile_fingerprint"] for item in case_profiles}) > 1
    assert identity.execution_profile_fingerprint == acceptance_module._identity_fingerprint(
        {"case_execution_profiles": case_profiles},
        "browser acceptance case execution profiles",
    )


@pytest.mark.parametrize(
    "action",
    [
        "back",
        "click",
        "download",
        "fill",
        "forward",
        "hover",
        "press",
        "reload",
        "screenshot",
        "scroll",
        "select",
        "upload",
        "wait",
    ],
)
def test_stale_reference_cases_reuse_pre_action_revision_and_reference(action: str) -> None:
    case = next(
        item
        for item in deterministic_browser_acceptance_manifest().cases
        if item.case_id == f"revision-stale-ref-after-{action}"
    )
    result_count = len(case.operations) - 1
    results = tuple(
        {
            "session_id": "browser-session",
            "page_id": "browser-page",
            "revision": f"revision-{index}",
            "control_epoch": index + 1,
            "refs": [
                {"name": "Save", "ref": f"save-{index}"},
                {"name": "Name", "ref": f"name-{index}"},
                {"name": "Region", "ref": f"region-{index}"},
                {"name": "Download report", "ref": f"download-{index}"},
                {"name": "Forward destination", "ref": f"forward-{index}"},
                {"name": "Back destination", "ref": f"back-{index}"},
                {"name": "Hover target", "ref": f"hover-{index}"},
                {"name": "Reload anchor", "ref": f"reload-{index}"},
                {"name": "Upload file", "ref": f"upload-{index}"},
            ],
        }
        for index in range(result_count)
    )
    stale_index = {"back": 1, "forward": 2}.get(action, 0)
    expected_ref = {
        "back": f"forward-{stale_index}",
        "download": f"download-{stale_index}",
        "forward": f"back-{stale_index}",
        "hover": f"hover-{stale_index}",
        "reload": f"reload-{stale_index}",
    }.get(action, f"save-{stale_index}")

    stale_arguments = internal_acceptance._operation_arguments(
        case_id=case.case_id,
        operation=case.operations[-1],
        operation_index=len(case.operations) - 1,
        fixture_route=case.fixture_route,
        results=results,
    )

    assert stale_arguments["expected_revision"] == f"revision-{stale_index}"
    assert stale_arguments["ref"] == expected_ref


@pytest.mark.parametrize(
    "case_id",
    [
        "artifact-upload",
        "artifact-upload-missing",
        "artifact-upload-wrong-session",
        "navigation-scroll-over-limit",
    ],
)
def test_public_acceptance_enforces_upload_authority_and_scroll_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_id: str
) -> None:
    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        manifest = deterministic_browser_acceptance_manifest()
        case = next(case for case in manifest.cases if case.case_id == case_id)
        material = {name: getattr(manifest, name) for name in type(manifest).model_fields}
        material.pop("revision")
        material["cases"] = (case,)
        selected = BrowserAcceptanceManifestV1.build(**material)
        monkeypatch.setattr(
            internal_acceptance, "deterministic_browser_acceptance_manifest", lambda: selected
        )
        monkeypatch.setattr(
            internal_acceptance,
            "DockerEgressAdapter",
            lambda **kwargs: _ProtocolEgressAdapter(fixture.upstream_origin),
        )
        plan = await build_internal_browser_acceptance(fixture)
        report = await run_browser_acceptance(
            plan, deterministic_fixture=fixture, receipt_directory=tmp_path / "receipts"
        )
        assert report.aggregate.overall_status.value == "passed", report.rows
        assert report.rows[0].completion_state.value == "complete"
        assert report.rows[0].observed_state is case.expected_state
        if case_id == "navigation-scroll-over-limit":
            assert report.rows[0].diagnostic.browser_dispatches == 1
            assert (
                report.rows[0].diagnostic.operations[-1].state.value == "operation_not_dispatched"
            )
        assert fixture.request_counts().get("/effect/upload-selected", 0) == (
            1 if case_id == "artifact-upload" else 0
        )

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize(
    ("phase", "scenario", "state", "error", "stage"),
    [
        (
            "disconnection",
            BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_DISCONNECTION,
            BrowserAcceptanceState.FAILED,
            "browser_crash",
            "upload_disconnection",
        ),
        (
            "acknowledgement-loss",
            BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_ACKNOWLEDGEMENT_LOSS,
            BrowserAcceptanceState.AMBIGUOUS,
            "outcome_ambiguous",
            "upload_acknowledgement_loss",
        ),
    ],
)
def test_upload_failure_corpus_binds_exact_fault_boundary_and_effect(
    phase: str,
    scenario: BrowserAcceptanceFaultScenario,
    state: BrowserAcceptanceState,
    error: str,
    stage: str,
) -> None:
    case = next(
        case
        for case in deterministic_browser_acceptance_manifest().cases
        if case.case_id == f"artifact-upload-{phase}"
    )
    assert case.operations == ("navigate", "upload")
    assert case.fault_scenario is scenario
    assert case.expected_state is state
    assert case.oracle_parameters["expected_browser_dispatches"] == 2
    assert case.oracle_parameters["error"] == error
    assert case.oracle_parameters["expected_effects"] == {"upload-selected": 1}
    assert case.oracle_parameters["allocation_disposition"] == "uncertain"
    assert internal_acceptance._scenario_stage(case.fault_scenario) == (
        "browser",
        stage,
    )


def test_upload_fixture_uses_isolated_trial_session_not_authored_session() -> None:
    from cayu.evals.runner import _isolated_trial_request

    case = next(
        case
        for case in deterministic_browser_acceptance_manifest().cases
        if case.case_id == "artifact-upload"
    )
    planned = internal_acceptance._case_request(case, session_id="authored-session")
    isolated = _isolated_trial_request(planned)
    bound = internal_acceptance.bind_trial_session("suite", case.case_id, 1, isolated)
    assert bound.session_id != planned.session_id
    assert bound.limits == planned.limits
    assert bound.max_steps == planned.max_steps
    assert bound.messages != planned.messages
    model_request = ModelRequest(model="scripted", messages=bound.messages)
    assert internal_acceptance._parent_session_id(model_request) == bound.session_id
    assert internal_acceptance._case_id(model_request) == case.case_id


def test_deterministic_planner_emits_closed_scroll_hover_and_upload_arguments() -> None:
    state = {
        "control_epoch": 1,
        "session_id": "browser-session",
        "page_id": "browser-page",
        "revision": "revision-1",
        "refs": [
            {"name": "Hover target", "ref": "hover-ref"},
            {"name": "Upload file", "ref": "upload-ref"},
        ],
    }
    scroll = internal_acceptance._operation_arguments(
        case_id="navigation-scroll-dependent-control",
        operation="scroll",
        operation_index=1,
        fixture_route="/scroll",
        results=(state,),
    )
    hover = internal_acceptance._operation_arguments(
        case_id="action-strict-hover",
        operation="hover",
        operation_index=1,
        fixture_route="/hover",
        results=(state,),
    )
    upload = internal_acceptance._operation_arguments(
        case_id="artifact-upload",
        operation="upload",
        operation_index=1,
        fixture_route="/upload",
        results=(state,),
        upload_artifact_id="art_0123456789abcdef0123456789abcdef",
    )

    assert scroll == {
        "expected_control_epoch": 1,
        "operation": "scroll",
        "session_id": "browser-session",
        "operation_id": "navigation-scroll-dependent-control:2:scroll",
        "page_id": "browser-page",
        "expected_revision": "revision-1",
        "direction": "down",
        "amount": "page",
        "repeat_count": 2,
    }
    assert hover["ref"] == "hover-ref"
    assert upload["ref"] == "upload-ref"
    assert upload["artifact_ids"] == ["art_0123456789abcdef0123456789abcdef"]


@pytest.mark.parametrize("operation", ["click_visual_target", "click_visual_point"])
def test_visual_recovery_preserves_authority_from_read_only_result(operation: str) -> None:
    state = {
        "session_id": "browser-session",
        "page_id": "browser-page",
        "revision": "revision-1",
        "control_epoch": 3,
        "visual": {
            "visual_revision": "visual-revision-1",
            "screenshot_sha256": "a" * 64,
            "targets": [
                {"ref": "visual-ref", "geometry": {"x": 10, "y": 20, "width": 8, "height": 6}}
            ],
        },
    }
    result = ToolResult(structured=state)
    assert result.structured is not None
    arguments = {
        "case_id": "visual-process-terminal-replay",
        "operation": operation,
        "operation_index": 2,
        "fixture_route": "/visual-popup",
    }
    plain = internal_acceptance._operation_arguments(**arguments, results=(state,))
    frozen = internal_acceptance._operation_arguments(
        **arguments, results=(dict(result.structured),)
    )
    assert frozen == plain
    assert frozen["expected_control_epoch"] == 3
    assert frozen["visual_revision"] == "visual-revision-1"
    if operation == "click_visual_target":
        assert frozen["visual_ref"] == "visual-ref"
    else:
        assert (frozen["x"], frozen["y"], frozen["screenshot_sha256"]) == (14, 23, "a" * 64)
    with pytest.raises(RuntimeError, match="lacks its visual target"):
        internal_acceptance._operation_arguments(
            **arguments, results=({**state, "visual": {"targets": []}},)
        )


def test_deterministic_planner_uses_browser_history_operations_not_navigation_substitutes() -> None:
    initial = internal_acceptance._operation_arguments(
        case_id="navigation-history-forward",
        operation="navigate",
        operation_index=0,
        fixture_route="/history-start",
        results=(),
    )
    start_state = {
        "control_epoch": 1,
        "session_id": "browser-session",
        "page_id": "browser-page",
        "revision": "revision-start",
        "refs": [{"name": "Next destination", "ref": "next-ref"}],
    }
    click = internal_acceptance._operation_arguments(
        case_id="navigation-history-forward",
        operation="click",
        operation_index=1,
        fixture_route="/history-start",
        results=(start_state,),
    )
    next_state = {
        **start_state,
        "revision": "revision-next",
        "refs": [{"name": "Forward destination", "ref": "forward-ref"}],
    }
    back = internal_acceptance._operation_arguments(
        case_id="navigation-history-forward",
        operation="back",
        operation_index=2,
        fixture_route="/history-start",
        results=(start_state, next_state),
    )
    forward = internal_acceptance._operation_arguments(
        case_id="navigation-history-forward",
        operation="forward",
        operation_index=3,
        fixture_route="/history-start",
        results=(start_state, next_state, start_state),
    )

    assert initial["url"] == "https://docs.browser.test/history-start"
    assert click["ref"] == "next-ref"
    assert back["operation"] == "back"
    assert "url" not in back
    assert forward["operation"] == "forward"
    assert "url" not in forward


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
        assert plan.eval_plan.app is not None
        assert plan.eval_plan.suite is not None
        eval_case = next(item for item in plan.eval_plan.suite.cases if item.id == case_id)
        assert eval_case.request.limits is not None
        assert (
            eval_case.request.limits.max_elapsed_seconds
            == plan.manifest.limits.max_wall_time_ms // 1000
        )
        assert (
            result.execution_profile_fingerprint
            == await plan.eval_plan.app.inspect_run_execution_profile(eval_case.request)
        )
        projected = await _project_scenario_execution(plan, case, result, fixture)
        assert projected.semantic_state.value == "passed", projected.model_dump(mode="json")
        assert projected.completion_state.value == "complete"

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


def test_cayu_owned_fault_executor_reconciles_acknowledgement_loss(
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
            item for item in plan.manifest.cases if item.case_id == "ambiguity-acknowledgement-loss"
        )
        executor = plan.scenario_executor
        assert executor is not None

        result = await executor(case, 1, 1, 30)

        assert result.fault.scenario is BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS
        assert result.fault.browser_dispatches == 1
        projected = await _project_scenario_execution(plan, case, result, fixture)
        assert projected.semantic_state.value == "passed", projected.model_dump(mode="json")
        assert projected.completion_state.value == "complete"

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
        projected = await _project_scenario_execution(plan, case, result, fixture)
        assert projected.semantic_state.value == "passed", projected.model_dump(mode="json")
        assert projected.completion_state.value == "complete"

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
        for _ in range(2000):
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
                suite=EvalSuite(id=manifest.suite_id, cases=list(suite_cases)),
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
                suite=EvalSuite(id=manifest.suite_id, cases=list(distinct_authority_cases)),
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
        await asyncio.wait_for(entered.wait(), timeout=10)
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
