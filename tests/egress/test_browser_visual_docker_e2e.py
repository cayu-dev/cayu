"""Real public browser-tool visual interaction over admitted Docker egress."""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import os
import secrets
import subprocess
import threading
from pathlib import Path

import pytest

from cayu import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    BrowserPopupPolicy,
    BrowserSessionTool,
    BrowserVisualPolicy,
    LocalArtifactStore,
    ToolContext,
)
from cayu.egress import HttpxUpstream
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryRequest
from cayu.evals.browser_acceptance_fixture import _fixture_address
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import SecretRedactor

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_VISUAL_BROWSER_ACCEPTANCE") != "1",
        reason="Set CAYU_RUN_VISUAL_BROWSER_ACCEPTANCE=1 with the pinned browser image installed.",
    ),
]


def _cleanup_fault_session_resources(session_id: str) -> None:
    """Reap only the networks explicitly labelled for this crashed test owner."""
    result = subprocess.run(
        ["docker", "network", "ls", "-q", "--filter", f"label=cayu.egress.session={session_id}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    for network_id in result.stdout.split():
        inspected = subprocess.run(
            ["docker", "network", "inspect", network_id],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        network = json.loads(inspected.stdout)[0]
        assert network["Labels"]["cayu.egress.session"] == session_id
        containers = list(network["Containers"])
        if containers:
            subprocess.run(
                ["docker", "rm", "-f", *containers], capture_output=True, timeout=20, check=True
            )
        subprocess.run(
            ["docker", "network", "rm", network_id], capture_output=True, timeout=10, check=True
        )


class _VisualFixture(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/mutate":
            self.server.mutation_gate.wait(timeout=10)
            body = b"ready"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/mutation-applied":
            self.server.mutation_applied.set()
            self.send_response(204)
            self.end_headers()
            return
        body = b"""<!doctype html><title>Visual fixture</title>
        <style id="rules">#overlay{display:none;position:fixed;inset:0;background:red}</style>
        <button id="overlay" onclick="document.getElementById('result').textContent='UNINTENDED'">Overlay</button>
        <canvas width="200" height="100" id="control"></canvas><p id="result">Clicks: 0</p>
        <script>
        const c=document.getElementById('control'), ctx=c.getContext('2d');
        ctx.fillStyle='green';ctx.fillRect(0,0,200,100);
        let count=0;
        c.addEventListener('click',()=>{count++;document.getElementById('result').textContent='Clicks: '+count;});
        </script>"""
        change = self.server.visual_change
        if change is not None:
            mutation = {
                "moved": "c.style.transform='translateX(50px)'",
                "covered": "const overlay=document.createElement('div');overlay.style='position:fixed;inset:0;background:black';document.body.appendChild(overlay)",
                "detached": "c.remove()",
                "scroll": "scrollTo(0,20)",
                "css_point": "document.getElementById('rules').sheet.insertRule('#overlay{display:block}',1)",
            }[change]
            body += (
                "<style>body{height:2000px}</style><script>fetch('/mutate').then(()=>{"
                + mutation
                + ";fetch('/mutation-applied')})</script>"
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.mark.parametrize("change", [None, "moved", "covered", "detached", "scroll", "css_point"])
def test_public_visual_canvas_target_point_replay_and_artifacts(
    tmp_path: Path, change: str | None
) -> None:
    async def scenario() -> None:
        address = _fixture_address()
        endpoint = http.server.ThreadingHTTPServer((address, 0), _VisualFixture)
        endpoint.visual_change = change
        endpoint.mutation_gate = threading.Event()
        endpoint.mutation_applied = threading.Event()
        thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
        thread.start()
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="visual-artifacts")
        factory = VirtualEgressEnvironmentFactory(
            policies={
                "visual": BrowserEgressPolicy(
                    name="visual",
                    allowed_hosts=("visual.browser.test",),
                    allowed_path_prefixes=("/",),
                )
            },
            approved_destinations=(
                ApprovedEgressDestination(destination="visual.browser.test", policy_name="visual"),
            ),
            credentials=[],
            adapter=DockerEgressAdapter(
                seccomp_profile=str(
                    Path(__file__).resolve().parents[2]
                    / "examples/browser_fetch/seccomp_profile.json"
                )
            ),
            image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            artifact_store=store,
            upstream=HttpxUpstream(
                routes={"visual.browser.test": f"http://{address}:{endpoint.server_port}"}
            ),
        )
        allocation = None
        try:
            allocation = await factory.create(
                EnvironmentFactoryRequest(
                    session_id="visual-e2e", agent_name="agent", environment_name="browser"
                )
            )
            environment = allocation.environment
            assert environment.runner is not None
            handle = InvocationRunnerHandle(
                environment.runner,
                redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(0, SecretRedactor()),
            )
            context = ToolContext(
                session_id="visual-e2e",
                agent_name="agent",
                environment_name="browser",
                runner=handle,
                artifact_store=store,
            )
            policy = BrowserVisualPolicy(
                artifact_store_id=store.id,
                allowed_origins=("https://visual.browser.test",),
                retention="application_managed",
                publish_to_model=True,
                allow_coordinate_fallback=True,
            )
            tool = BrowserSessionTool(
                expected_runner_candidate="docker",
                visual_policy=policy,
                multi_page=True,
                popup_policy=BrowserPopupPolicy(
                    mode="same_origin",
                    allowed_operations=("click_visual_point", "click_visual_target"),
                ),
            )
            opened = await tool.run(
                context,
                {
                    "operation": "navigate",
                    "url": "https://visual.browser.test/",
                    "operation_id": "open",
                },
            )
            assert not opened.is_error, opened.model_dump(mode="json")
            session = opened.structured["session_id"]
            page = opened.structured["page_id"]
            assert opened.structured["visual"] is None
            assert "canvas" not in opened.structured["snapshot"].lower()
            observed = await tool.run(
                context,
                {
                    "operation": "observe_visual",
                    "session_id": session,
                    "page_id": page,
                    "operation_id": "visual-1",
                },
            )
            assert not observed.is_error, observed.model_dump(mode="json")
            evidence = observed.structured["visual"]
            assert len(evidence["targets"]) == 1
            assert (
                len(observed.artifacts) == 1
                and observed.artifacts[0]["type"] == "cayu.file_attachment.v1"
            )
            image = await store.read_bytes(observed.artifacts[0]["artifact_id"])
            assert hashlib.sha256(image.content).hexdigest() == evidence["screenshot_sha256"]
            request = {
                "operation": "click_visual_target",
                "session_id": session,
                "page_id": page,
                "expected_revision": observed.structured["revision"],
                "expected_control_epoch": observed.structured["control_epoch"],
                "visual_revision": evidence["visual_revision"],
                "visual_ref": evidence["targets"][0]["ref"],
                "operation_id": "click-1",
            }
            if change is not None:
                if change == "css_point":
                    geometry = evidence["targets"][0]["geometry"]
                    request.pop("visual_ref")
                    request.update(
                        operation="click_visual_point",
                        screenshot_sha256=evidence["screenshot_sha256"],
                        x=geometry["x"] + geometry["width"] / 2,
                        y=geometry["y"] + geometry["height"] / 2,
                    )
                endpoint.mutation_gate.set()
                assert await asyncio.to_thread(endpoint.mutation_applied.wait, 5)
                rejected = await tool.run(context, request)
                assert rejected.is_error, rejected.model_dump(mode="json")
                assert rejected.structured["error"] == (
                    "visual_viewport_mismatch"
                    if change == "scroll"
                    else "visual_hit_test_mismatch"
                    if change == "css_point"
                    else "visual_evidence_expired"
                )
                refreshed = await tool.run(
                    context,
                    {
                        "operation": "observe",
                        "session_id": session,
                        "page_id": page,
                        "operation_id": "after-rejection",
                    },
                )
                assert "Clicks: 0" in refreshed.content
                assert "UNINTENDED" not in refreshed.content
                return
            for field, wrong, error in (
                ("session_id", "bs_other", "unknown_session"),
                ("page_id", "bp_other", "unknown_page"),
                ("expected_revision", "br_other", "stale_observation"),
                (
                    "expected_control_epoch",
                    observed.structured["control_epoch"] + 1,
                    "stale_observation",
                ),
                ("visual_revision", "vr_" + "0" * 32, "visual_evidence_expired"),
                ("visual_ref", "vt_" + "0" * 32, "unknown_visual_target"),
            ):
                refused = await tool.run(
                    context, {**request, field: wrong, "operation_id": "conflict-" + field}
                )
                assert refused.structured["error"] == error
                assert refused.structured["execution"]["dispatch"] == "not_started"
            clicked = await tool.run(context, request)
            assert not clicked.is_error, clicked.model_dump(mode="json")
            assert "Clicks: 1" in clicked.content
            assert (await tool.run(context, request)).model_dump(mode="json") == clicked.model_dump(
                mode="json"
            )
            stale = await tool.run(context, {**request, "operation_id": "stale"})
            assert stale.structured["error"] == "stale_observation"
            observed = await tool.run(
                context,
                {
                    "operation": "observe_visual",
                    "session_id": session,
                    "page_id": page,
                    "operation_id": "visual-2",
                },
            )
            assert not observed.is_error, observed.model_dump(mode="json")
            evidence = observed.structured["visual"]
            rectangle = evidence["targets"][0]["geometry"]
            point = {
                "operation": "click_visual_point",
                "session_id": session,
                "page_id": page,
                "expected_revision": observed.structured["revision"],
                "expected_control_epoch": observed.structured["control_epoch"],
                "visual_revision": evidence["visual_revision"],
                "screenshot_sha256": evidence["screenshot_sha256"],
                "x": rectangle["x"] + rectangle["width"] / 2,
                "y": rectangle["y"] + rectangle["height"] / 2,
                "operation_id": "point-1",
            }
            for field, wrong, error in (
                ("screenshot_sha256", "0" * 64, "visual_evidence_expired"),
                ("x", 1, "invalid_arguments"),
                ("y", -1, "invalid_arguments"),
            ):
                refused = await tool.run(
                    context, {**point, field: wrong, "operation_id": "conflict-" + field}
                )
                assert refused.structured["error"] == error
                assert refused.structured["execution"]["dispatch"] == "not_started"
            clicked = await tool.run(context, point)
            assert not clicked.is_error, clicked.model_dump(mode="json")
            assert "Clicks: 2" in clicked.content
            closed = await tool.run(
                context, {"operation": "close", "session_id": session, "operation_id": "close"}
            )
            assert not closed.is_error and closed.structured["allocation_disposition"] == "retired"
        finally:
            endpoint.mutation_gate.set()
            if allocation is not None:
                environment = allocation.environment
                if environment.runner is not None and environment.binding is not None:
                    bound = await environment.binding.bind(
                        None, environment.runner, session_id="visual-e2e"
                    )
                    await environment.binding.finalize(bound, outcome="completed")
            endpoint.shutdown()
            endpoint.server_close()
            thread.join(timeout=5)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case_id",
    [
        "visual-canvas-control",
        "visual-inaccessible-control",
        "visual-positioned-control",
        "visual-hostile-pixels-labels",
        "visual-popup-outcome",
        "visual-semantic-preference",
    ],
)
def test_visual_acceptance_reaches_model_and_durable_runtime(case_id: str) -> None:
    from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
    from cayu.evals.internal.browser_acceptance import build

    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        plan = await build(fixture)
        assert plan.eval_plan.suite is not None
        case = next(case for case in plan.eval_plan.suite.cases if case.id == case_id)
        events = [event async for event in plan.eval_plan.app.run(case.request)]
        failures = [
            event for event in events if event.type in {"session.failed", "tool.call.failed"}
        ]
        assert not failures, [event.model_dump(mode="json") for event in failures]
        assert events[-1].type == "session.completed"
        assert fixture.request_counts().get("/effect/visual-activated") == 1
        provider = plan.eval_plan.app.get_provider()
        image_published = any(
            request.options.get("cayu_file_attachments") for request in provider.requests
        )
        assert image_published is (case_id != "visual-semantic-preference")

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize(
    "case_id",
    [
        "visual-virtualized-movement",
        "visual-sticky-overlay",
        "visual-viewport-scroll-change",
        "visual-stale-screenshot",
        "visual-cross-origin-frame",
        "visual-opaque-host-refusal",
    ],
)
def test_visual_corpus_refusal_uses_real_page_change(case_id: str) -> None:
    from cayu.evals import EvaluationEvidencePolicySpec, project_assertion_evidence_view
    from cayu.evals.browser_acceptance import project_browser_acceptance_diagnostic
    from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
    from cayu.evals.internal.browser_acceptance import build
    from cayu.evals.trajectory import trajectory_from_session

    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        plan = await build(fixture)
        assert plan.eval_plan.suite is not None
        eval_case = next(case for case in plan.eval_plan.suite.cases if case.id == case_id)
        case = next(case for case in plan.manifest.cases if case.case_id == case_id)
        events = [event async for event in plan.eval_plan.app.run(eval_case.request)]
        assert events[-1].type == "session.completed"
        session_id = events[-1].session_id
        trajectory = await trajectory_from_session(plan.eval_plan.app, session_id)
        evidence = project_assertion_evidence_view(
            plan.eval_plan.app,
            trajectory,
            evidence_policy=EvaluationEvidencePolicySpec.create(
                include_tool_arguments=True,
                include_tool_results=True,
            ),
        )
        diagnostic = project_browser_acceptance_diagnostic(evidence)
        assert diagnostic.operations[-1].error_category == case.oracle_parameters["error"]
        assert fixture.request_counts().get("/effect/visual-activated", 0) == 0

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))


@pytest.mark.parametrize("case_id", ["visual-process-loss", "visual-acknowledgement-loss"])
def test_visual_effect_is_not_repeated_after_real_durable_fault(case_id: str) -> None:
    from cayu.evals.browser_acceptance import (
        BrowserAcceptanceCaseCategory,
        BrowserAcceptanceFaultScenario,
        BrowserAcceptanceState,
    )
    from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
    from cayu.evals.browser_acceptance_manifests import _case
    from cayu.evals.internal.browser_acceptance import _recover_scenario, build
    from cayu.runtime._tool_round_recovery import PENDING_TOOL_ROUND_CHECKPOINT_KEY

    attempt_number = secrets.randbelow(1_000_000) + 1
    session_id = f"browser-acceptance-{case_id}-1-{attempt_number}"

    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        plan = await build(fixture)
        # A crashed factory-backed invocation has an unknown dynamic secret
        # scope. Its request is deliberately archived, not forged into public
        # transcript evidence for the corpus's terminal-result scoring.
        case = _case(
            case_id,
            category=BrowserAcceptanceCaseCategory.RECOVERY,
            state=BrowserAcceptanceState.AMBIGUOUS,
            operations=("navigate", "observe_visual", "click_visual_target"),
            route="/visual-popup",
            fault_scenario=(
                BrowserAcceptanceFaultScenario.PROCESS_BEFORE_TERMINAL
                if case_id == "visual-process-loss"
                else BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS
            ),
        )
        assert plan.scenario_executor is not None
        result = await plan.scenario_executor(case, 1, attempt_number, 60)
        assert result.fault.boundary_observed
        assert result.fault.browser_dispatches == 3
        assert result.fault.process_loss_observed is (case_id == "visual-process-loss")
        assert result.fault.recovered_in_fresh_app is (case_id == "visual-process-loss")
        assert fixture.request_counts().get("/effect/visual-activated") == 1
        assert result.trial.trajectory is not None
        checkpoint = await result.app.session_store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert PENDING_TOOL_ROUND_CHECKPOINT_KEY not in checkpoint
        if case_id == "visual-process-loss":
            archived = checkpoint["abandoned_unreplayable_tool_round"]["tool_round"]
            assert archived["assistant_publication"]["state"] == "blocked"
            assert (
                archived["assistant_publication"]["reason"] == "incomplete_invocation_secret_scope"
            )
            assert [call["tool_call_id"] for call in archived["tool_calls"]] == [f"{case_id}-3"]
        else:
            terminal = [
                event
                for event in result.trial.trajectory.events
                if event.type == "tool.call.completed"
                and event.payload.get("tool_call_id") == f"{case_id}-3"
            ]
            assert len(terminal) == 1
        session = await result.app.session_store.load(session_id)
        expected_status = "interrupted" if case_id == "visual-process-loss" else "completed"
        assert session is not None and session.status.value == expected_status
        # Repeated public recovery must not reopen the archived native effect.
        await _recover_scenario(result.app, session_id)
        assert await result.app.session_store.load_checkpoint(session_id) == checkpoint
        assert fixture.request_counts().get("/effect/visual-activated") == 1

    try:
        with BrowserAcceptanceFixtureV1() as fixture:
            asyncio.run(scenario(fixture))
    finally:
        _cleanup_fault_session_resources(session_id)


@pytest.mark.parametrize(
    "case_id",
    [
        "visual-secret-refusal",
        "visual-terminal-acknowledgement-loss",
        "visual-process-terminal-replay",
    ],
)
def test_visual_fault_corpus_is_scored_from_real_runtime_evidence(case_id: str) -> None:
    from cayu.evals import EvaluationEvidencePolicySpec, project_assertion_evidence_view
    from cayu.evals.browser_acceptance import project_browser_acceptance_trial
    from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
    from cayu.evals.internal.browser_acceptance import build

    async def scenario(fixture: BrowserAcceptanceFixtureV1) -> None:
        plan = await build(fixture)
        case = next(case for case in plan.manifest.cases if case.case_id == case_id)
        assert plan.scenario_executor is not None
        result = await plan.scenario_executor(case, 1, 1, 60)
        assert result.fault.boundary_observed
        assert (
            result.fault.browser_dispatches == case.oracle_parameters["expected_browser_dispatches"]
        )
        assert result.trial.trajectory is not None
        evidence = project_assertion_evidence_view(
            result.app,
            result.trial.trajectory,
            evidence_policy=EvaluationEvidencePolicySpec.create(
                include_tool_arguments=True,
                include_tool_results=True,
            ),
        )
        receipt = project_browser_acceptance_trial(
            case=case,
            run_identity_revision="sha256:" + "9" * 64,
            trial=result.trial,
            evidence=evidence,
            fixture_route_observed=True,
            fixture_route_request_count=fixture.request_counts().get(case.fixture_route, 0),
            fixture_effects={
                "visual-activated": fixture.request_counts().get("/effect/visual-activated", 0)
            },
            fault=result.fault,
        )
        assert receipt.observed_state is case.expected_state, receipt.model_dump_json()
        assert receipt.semantic_state.value == "passed", receipt.model_dump_json()
        if case_id == "visual-secret-refusal":
            assert all(not operation.artifacts for operation in receipt.diagnostic.operations)
        if case_id == "visual-process-terminal-replay":
            for change in ({"quarantined_tool_calls": 0}, {"browser_dispatches": 2}):
                rejected = project_browser_acceptance_trial(
                    case=case,
                    run_identity_revision="sha256:" + "9" * 64,
                    trial=result.trial,
                    evidence=evidence,
                    fixture_route_observed=True,
                    fixture_route_request_count=fixture.request_counts().get(case.fixture_route, 0),
                    fixture_effects={"visual-activated": 1},
                    fault=result.fault.model_copy(update=change),
                )
                assert rejected.semantic_state.value != "passed"
                assert rejected.observed_state.value == "unavailable"
        assert "browser-acceptance-private-pixel-canary" not in receipt.model_dump_json()

    try:
        with BrowserAcceptanceFixtureV1() as fixture:
            asyncio.run(scenario(fixture))
    finally:
        _cleanup_fault_session_resources(f"browser-acceptance-{case_id}-1-1")
