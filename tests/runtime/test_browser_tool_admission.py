from __future__ import annotations

import asyncio
import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.runtime.test_egress_factory import (
    _available_untrusted_execution_evidence,
    _RecordingAdapter,
    _run_virtual_factory_lifecycle,
    _virtual_factory,
)
from tests.runtime.test_execution_admission_dispatch import (
    _completed_docker_probe_result,
    _EvidenceRunner,
    _RecordingProvider,
    _run,
)

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserSessionTool,
    BrowserWebFetchAdapter,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionRequirements,
    ExecutionToolRequirementEvidence,
    ScreenshotPageTool,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolRunnerCapabilityRequirement,
    ToolSpec,
    WebFetchTool,
)
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.runners import DockerRunner, ExecResult
from cayu.runners import docker as docker_module
from cayu.tools.browser_session import BrowserSessionBackend
from cayu.vaults import SecretRef, StaticVault


@pytest.mark.parametrize(
    "capability,proof",
    [
        ("workspace_text_search_v1", "verified"),
        ("workspace_text_search_v1", "missing"),
        ("workspace_text_search_v1", "declared"),
        ("workspace_text_search_v1", "outer_refusal"),
        ("deny_by_default_network", "outer_refusal"),
        ("deny_by_default_network", "outer_missing"),
    ],
)
def test_managed_native_admission_preserves_outer_refusal(capability, proof):
    def verified_claim():
        now = datetime.now(UTC)
        return ExecutionCapabilityClaim.live_verified(
            capability,
            observation="denied" if capability == "deny_by_default_network" else "supported",
            observed_at=now,
            valid_until=now + timedelta(seconds=60),
        )

    class NativeTool(Tool):
        spec = ToolSpec(
            name="native_fixture",
            execution_requirements=(
                ToolExecutionRequirement(
                    name="native_backend",
                    alternatives=(ToolRunnerCapabilityRequirement(capability=capability),),
                ),
            ),
        )

        async def run(self, ctx, args):
            raise AssertionError("Admission must not execute the fixture tool")

    class NativeRunner(_EvidenceRunner):
        def execution_admission_candidate(self):
            claims = (
                ()
                if proof == "missing"
                else (ExecutionCapabilityClaim.declared(capability),)
                if proof == "declared"
                else (verified_claim(),)
            )
            return ExecutionAdmissionCandidate(
                candidate="docker",
                evidence=ExecutionCapabilityEvidence(
                    subject="docker",
                    environment_fingerprint="sha256:" + "9" * 64,
                    claims=claims,
                    unclaimed_reason_code="native_missing" if not claims else None,
                ),
            )

    async def create_runner(request):
        return NativeRunner("docker")

    class NativeAdapter(_RecordingAdapter):
        def execution_capability_evidence(self, runner=None):
            base = _available_untrusted_execution_evidence(self.runner_kind)
            claims = [claim for claim in base.claims if claim.capability != capability]
            if runner is None:
                # Preflight permits allocation; final bound evidence is tested below.
                claims.append(verified_claim())
            elif proof == "outer_refusal":
                claims.append(
                    ExecutionCapabilityClaim.unsupported(
                        capability,
                        reason_code="adapter_refused",
                        remediation_code="choose_supported_adapter",
                    )
                )
            return base.model_copy(update={"claims": tuple(claims)})

    adapter = NativeAdapter("docker", runner_factory=create_runner)
    events, provider, _ = asyncio.run(
        _run_virtual_factory_lifecycle(
            _virtual_factory(
                adapter=adapter,
                credentials=[],
                approved_destinations=[
                    ApprovedEgressDestination(
                        destination="api.stripe.com",
                        policy_name="provider-example",
                        protocol="https",
                        port=443,
                    )
                ],
            ),
            session_id=f"managed_native_{capability}_{proof}",
            requirements=ExecutionRequirements.trusted(
                **({"network_access": "deny_by_default"} if proof == "outer_missing" else {})
            ),
            tools=[NativeTool()],
        )
    )
    admitted = proof == "verified"
    assert len(adapter.prepare_calls) == 1
    assert adapter.torn_down == 1
    assert len(provider.requests) == int(admitted)
    assert any(event.type is EventType.SESSION_COMPLETED for event in events) is admitted
    if not admitted:
        assert not any(str(event.type).startswith("model.") for event in events)
        failure = next(event for event in events if event.type is EventType.SESSION_FAILED)
        refusals = [
            refusal
            for refusal in failure.payload["execution_admission"]["refusals"]
            if refusal["tool_name"] == "native_fixture"
            and refusal["requirement_name"] == "native_backend"
            and refusal["capability"] == capability
        ]
        assert len(refusals) == 1
        if proof == "outer_refusal":
            assert refusals[0]["observed_state"] == "unsupported"
            assert refusals[0]["reason_code"] == "adapter_refused"
            assert refusals[0]["remediation_code"] == "choose_supported_adapter"


@pytest.mark.parametrize(
    "tool_kind,worker",
    [
        ("screenshot", "/usr/local/bin/python"),
        ("screenshot", "/opt/browser/worker"),
        ("browser_fetch", "/usr/local/bin/python"),
        ("browser_fetch", "/opt/browser/worker"),
        ("browser_session", "/usr/local/bin/python"),
    ],
)
@pytest.mark.parametrize("proof", ["missing", "basename", "exact"])
def test_browser_worker_requires_exact_executable_before_provider(tool_kind, worker, proof):
    worker_command = (worker,)
    if tool_kind == "browser_session":
        tool = BrowserSessionTool()
    elif tool_kind == "screenshot":
        tool = ScreenshotPageTool(worker_command=worker_command)
    else:
        tool = WebFetchTool(adapter=BrowserWebFetchAdapter(worker_command=worker_command))

    class WorkerRunner(_EvidenceRunner):
        def execution_admission_candidate(self):
            now = datetime.now(UTC)
            fingerprint = "sha256:" + "1" * 64
            executable = worker if proof == "exact" else worker.rsplit("/", 1)[-1]
            return ExecutionAdmissionCandidate(
                candidate="hosted",
                evidence=ExecutionCapabilityEvidence(
                    subject="hosted",
                    unclaimed_reason_code="security_unclaimed",
                    environment_fingerprint=fingerprint,
                    tool_requirements=ExecutionToolRequirementEvidence(
                        environment_fingerprint=fingerprint,
                        executables=(
                            ()
                            if proof == "missing"
                            else (
                                ExecutionExecutableEvidence(
                                    executable=executable,
                                    state="live_verified",
                                    observed_at=now,
                                    valid_until=now + timedelta(seconds=60),
                                    requirement_fingerprint=ToolExecutableRequirement(
                                        executable=executable
                                    ).fingerprint,
                                ),
                            )
                        ),
                    ),
                ),
            )

    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="hosted"), runner=WorkerRunner("hosted")), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        return await _run(app, "browser-worker-admission"), provider

    events, provider = asyncio.run(run())
    admitted = proof == "exact"
    assert len(provider.requests) == int(admitted)
    assert (events[-1].type is EventType.SESSION_COMPLETED) is admitted
    if not admitted:
        assert not any(str(event.type).startswith("model.") for event in events)
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert any(
            refusal["tool_name"] == tool.name and refusal["executable"] == worker
            for refusal in failed.payload["execution_admission"]["refusals"]
        )


@pytest.mark.parametrize("tool_kind", ["screenshot", "browser_fetch", "browser_session"])
@pytest.mark.parametrize("conflict", [False, True])
def test_browser_worker_declaration_preserves_caller_requirements(tool_kind, conflict):
    supplied = ToolExecutionRequirement(
        name="browser_worker" if conflict else "caller_dependency",
        alternatives=(ToolExecutableRequirement(executable="caller_program"),),
    )
    spec = ToolSpec(name="custom_browser", execution_requirements=(supplied,))

    def build():
        if tool_kind == "screenshot":
            return ScreenshotPageTool(spec=spec)
        if tool_kind == "browser_session":
            return BrowserSessionTool(spec=spec)
        return WebFetchTool(adapter=BrowserWebFetchAdapter(), spec=spec)

    if conflict:
        with pytest.raises(ValueError, match="conflicts"):
            build()
    else:
        tool = build()
        assert supplied in tool.spec.execution_requirements
        assert len(tool.spec.execution_requirements) == 2
        assert tool.spec.execution_requirements[0].alternatives == (
            ToolExecutableRequirement(executable="/usr/local/bin/python"),
        )


def test_default_http_web_fetch_needs_no_guest_executable():
    async def run():
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        tool = WebFetchTool()
        assert tool.spec.execution_requirements == ()
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        return await _run(app, "http-fetch-no-guest"), provider

    events, provider = asyncio.run(run())
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 1


def test_custom_browser_session_backend_keeps_application_requirements():
    class CustomBackend(BrowserSessionBackend):
        async def execute(self, ctx, request):
            raise AssertionError("Admission must not execute browser work")

    declaration = ToolExecutionRequirement(
        name="custom_backend",
        alternatives=(ToolExecutableRequirement(executable="custom_worker"),),
    )
    tool = BrowserSessionTool(
        _backend=CustomBackend(),
        spec=BrowserSessionTool.spec.model_copy(update={"execution_requirements": (declaration,)}),
    )
    assert tool.spec.execution_requirements == (declaration,)


@pytest.mark.parametrize("tool_kind", ["screenshot", "browser_fetch", "browser_session"])
@pytest.mark.parametrize("field", ["description", "executable"])
def test_browser_declaration_rejects_mutated_spec_without_diagnostic_leaks(
    tool_kind, field, caplog, capsys
):
    canary = "private-browser-requirement-860"
    sibling = "private-browser-sibling-860"

    class RejectedValue:
        def __repr__(self):
            return canary

        __str__ = __repr__

    spec = ToolSpec(
        name="custom_browser",
        description=sibling,
        input_schema={"description": sibling},
        execution_requirements=(
            ToolExecutionRequirement(
                name="caller",
                alternatives=(ToolExecutableRequirement(executable="caller_program"),),
            ),
        ),
    )
    target = spec if field == "description" else spec.execution_requirements[0].alternatives[0]
    object.__setattr__(target, field, RejectedValue())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises((TypeError, ValueError)) as failure:
            if tool_kind == "screenshot":
                ScreenshotPageTool(spec=spec)
            elif tool_kind == "browser_session":
                BrowserSessionTool(spec=spec)
            else:
                WebFetchTool(adapter=BrowserWebFetchAdapter(), spec=spec)
    output = capsys.readouterr()
    diagnostics = "\n".join(
        [str(item.message) for item in captured]
        + [str(failure.value), repr(failure.value), caplog.text, output.out, output.err]
    )
    assert canary not in diagnostics
    assert sibling not in diagnostics


def test_shared_docker_runner_keeps_admission_plans_request_local(monkeypatch):
    container_id = "9" * 64
    image_id = "sha256:" + "8" * 64
    first_dispatched = asyncio.Event()
    finish_first = asyncio.Event()
    probes = []
    observers = []

    class SharedDockerRunner(DockerRunner):
        def execution_admission_observer(self, requirements):
            observer = super().execution_admission_observer(requirements)
            observers.append(observer)
            return observer

    runner = SharedDockerRunner(
        "shared", docker_path="/usr/bin/docker", close_action="none", _container_id=container_id
    )

    async def docker_cli(path, args, **kwargs):
        if args[0] == "inspect":
            return ExecResult(
                stdout=json.dumps(
                    {"Id": container_id, "Image": image_id, "State": {"Running": True}}
                )
            )
        if any("read pid process_group" in value for value in args):
            finish_first.set()
            return ExecResult()
        executable = "/opt/first" if any("/opt/first" in value for value in args) else "/opt/second"
        probes.append(executable)
        if executable == "/opt/first":
            first_dispatched.set()
            await finish_first.wait()
        return _completed_docker_probe_result(args)

    monkeypatch.setattr(docker_module, "_run_docker", docker_cli)

    def application(executable):
        app = CayuApp(enable_logging=False)
        provider = _RecordingProvider()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="docker"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ScreenshotPageTool(worker_command=(executable,))],
        )
        return app, provider

    async def run():
        first, first_provider = application("/opt/first")
        second, second_provider = application("/opt/second")
        first_task = asyncio.create_task(_run(first, "shared_first"))
        try:
            await asyncio.wait_for(first_dispatched.wait(), timeout=10)
            refused = await _run(second, "shared_second_busy")
            assert refused[-1].type is EventType.SESSION_FAILED
            assert first_provider.requests == second_provider.requests == []
            assert probes == ["/opt/first"]
        finally:
            finish_first.set()
            first_events = await first_task
        assert first_events[-1].type is EventType.SESSION_COMPLETED
        assert len(first_provider.requests) == 1
        retried = await _run(second, "shared_second_retry")
        assert retried[-1].type is EventType.SESSION_COMPLETED
        assert len(second_provider.requests) == 1
        assert probes == ["/opt/first", "/opt/second"]
        assert len(observers) == 2
        assert observers[0] is not observers[1]
        for observer, expected in zip(observers, probes, strict=True):
            assert observer.requirements.executable_names() == (expected,)
            assert tuple(
                item.executable
                for item in observer.snapshot().evidence.tool_requirements.executables
            ) == (expected,)

    asyncio.run(run())


def test_custom_browser_adapter_requirements_remain_application_owned():
    class CustomBrowserAdapter(BrowserWebFetchAdapter):
        pass

    requirement = ToolExecutionRequirement(
        name="custom_backend", alternatives=(ToolExecutableRequirement(executable="custom_worker"),)
    )
    tool = WebFetchTool(
        adapter=CustomBrowserAdapter(),
        spec=ToolSpec(name="custom_fetch", execution_requirements=(requirement,)),
    )
    assert tool.spec.execution_requirements == (requirement,)


@pytest.mark.parametrize(
    "outcome",
    ["present", "missing", "stale", "image_drift", "environment_drift", "wrong_container"],
)
@pytest.mark.parametrize("private_overlay", [False, True])
@pytest.mark.parametrize("tool_kind", ["screenshot", "browser_session"])
def test_docker_egress_browser_admission_uses_supervised_guest_evidence(
    monkeypatch, outcome, private_overlay, tool_kind, caplog, capsys
):
    container_id = "e" * 64
    image_id = "sha256:" + "f" * 64
    probes = []
    env_files = []
    canary = "cayu-860-private-probe-overlay"
    runners = []
    inspections = 0
    observation_time = datetime.now(UTC)

    class ObservationClock(datetime):
        @classmethod
        def now(cls, tz=None):
            # Simulate a collection taking longer than the evidence window.
            # Timestamping after the probe would incorrectly admit this case.
            if outcome == "stale" and not probes:
                return observation_time - timedelta(seconds=301)
            return observation_time

    monkeypatch.setattr(docker_module, "datetime", ObservationClock)

    async def docker_cli(path, args, **kwargs):
        nonlocal inspections
        if args[0] == "inspect":
            inspections += 1
            return ExecResult(
                stdout=json.dumps(
                    {
                        "Id": "d" * 64 if outcome == "wrong_container" else container_id,
                        "Image": (
                            "sha256:" + "a" * 64
                            if outcome == "image_drift" and inspections > 1
                            else image_id
                        ),
                        "State": {"Running": True},
                    }
                )
            )
        if args[0] == "exec" and any("cayu-admission-probe-complete-" in item for item in args):
            probes.append(args)
            assert canary not in repr(args)
            if private_overlay:
                env_file = Path(args[args.index("--env-file") + 1])
                env_files.append(env_file)
                assert env_file.stat().st_mode & 0o777 == 0o600
                assert f"CAYU_TEST_PRIVATE={canary}\n" in env_file.read_text()
            if outcome == "environment_drift":
                runners[0].env_overlay["PATH"] = "/different/bin"
            return _completed_docker_probe_result(
                args, guest_exit_code=1 if outcome == "missing" else 0
            )
        return ExecResult()

    monkeypatch.setattr(docker_module, "_require_docker", lambda path=None: "docker")
    monkeypatch.setattr(docker_module, "_run_docker", docker_cli)

    async def create_runner(request):
        runner = DockerRunner(
            request.name,
            image=request.image,
            _container_id=container_id,
            env_overlay={
                **dict(request.env_overlay),
                **({"CAYU_TEST_PRIVATE": canary} if private_overlay else {}),
            },
            _env_overlay_secret_values_present=(
                private_overlay or request.env_overlay_secret_values_present
            ),
        )
        runners.append(runner)
        return runner

    class ProbeAdapter(_RecordingAdapter):
        execution_admission_evidence_for = DockerEgressAdapter.execution_admission_evidence_for

        def execution_capability_evidence(self, runner=None):
            return _available_untrusted_execution_evidence(self.runner_kind)

    adapter = ProbeAdapter("docker", runner_factory=create_runner)
    events, provider, _ = asyncio.run(
        _run_virtual_factory_lifecycle(
            _virtual_factory(
                adapter=adapter,
                credentials=[],
                approved_destinations=[
                    ApprovedEgressDestination(
                        destination="api.stripe.com",
                        policy_name="provider-example",
                        protocol="https",
                        port=443,
                    )
                ],
            ),
            session_id=f"docker_browser_probe_{outcome}",
            requirements=ExecutionRequirements.trusted(),
            tools=[
                BrowserSessionTool() if tool_kind == "browser_session" else ScreenshotPageTool()
            ],
        )
    )
    assert bool(probes) is (outcome != "wrong_container")
    assert all(container_id in args and "-w" in args for args in probes)
    admitted = outcome == "present"
    assert len(provider.requests) == int(admitted)
    assert (EventType.SESSION_COMPLETED in {event.type for event in events}) is admitted
    if not admitted:
        assert EventType.SESSION_FAILED in {event.type for event in events}
        assert not any(str(event.type).startswith("model.") for event in events)
    if outcome == "missing":
        failure = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert any(
            refusal["requirement_name"] == "browser_worker"
            and refusal["executable"] == "/usr/local/bin/python"
            and refusal["observed_state"] == "unsupported"
            for refusal in failure.payload["execution_admission"]["refusals"]
        ), failure.payload["execution_admission"]["refusals"]
    assert adapter.torn_down == 1
    assert all(not path.exists() for path in env_files)
    output = capsys.readouterr()
    assert canary not in repr([(event.type, event.payload) for event in events])
    assert canary not in caplog.text + output.out + output.err


@pytest.mark.parametrize(
    "outcome",
    [
        "present",
        "missing",
        "resolution_failure",
        "reference_drift",
        "resolver_drift",
        "cancel",
        "file_creation_failure",
        "file_unlink_failure",
        "file_unlink_persistent",
        "file_unlink_retry_cancel",
        "file_unlink_replaced",
        "overlay_wins",
        "credential_mode_drift",
        "child_cancel_before_dispatch",
    ],
)
def test_docker_tool_admission_resolves_private_environment_before_guest_dispatch(
    monkeypatch, outcome, caplog, capsys
):
    canary = "private-resolved-probe-860"
    overlay_canary = "private-overlay-winner-860"
    container_id = "c" * 64
    image_id = "sha256:" + "b" * 64
    dispatched = []
    paths = []
    resolution_started = asyncio.Event()
    allow_unlink = False
    drain_task = None
    cancellation_sent = False
    cancelled_children = []
    if outcome == "child_cancel_before_dispatch":
        original_create_task = asyncio.create_task

        def cancel_probe_child(coro, **kwargs):
            child = original_create_task(coro, **kwargs)
            if kwargs.get("name") == f"cayu-docker-admission-probe-{container_id[:12]}":
                child.cancel("cancel child before dispatch")
                assert child.cancelling() == 1
                cancelled_children.append(child)
            return child

        monkeypatch.setattr(asyncio, "create_task", cancel_probe_child)

    class ProbeVault(StaticVault):
        async def resolve(self, ref, *, scope=None):
            resolution_started.set()
            if outcome == "cancel":
                await asyncio.Event().wait()
            if outcome == "resolution_failure":
                raise RuntimeError("test lookup unavailable")
            return await super().resolve(ref, scope=scope)

    vault = ProbeVault({"probe_secret": canary})
    if outcome.startswith("file_unlink"):
        from cayu.runners import _secrets as runner_secrets

        original_unlink = runner_secrets.os.unlink
        unlink_failed = False

        def fail_probe_unlink(path, *args, **kwargs):
            nonlocal unlink_failed, cancellation_sent
            if Path(path) in paths and (
                not unlink_failed or (outcome != "file_unlink_failure" and not allow_unlink)
            ):
                unlink_failed = True
                if (
                    outcome == "file_unlink_retry_cancel"
                    and drain_task is not None
                    and not cancellation_sent
                ):
                    cancellation_sent = True
                    drain_task.cancel("cancel private file cleanup retry")
                raise OSError("test private file unlink failed")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(runner_secrets.os, "unlink", fail_probe_unlink)
    if outcome == "file_creation_failure":
        from cayu.runners import _secrets as runner_secrets

        original_mkstemp = runner_secrets.tempfile.mkstemp

        def fail_probe_file(*args, **kwargs):
            if kwargs.get("prefix") == "cayu-runner-env-":
                raise OSError("test private file allocation failed")
            return original_mkstemp(*args, **kwargs)

        monkeypatch.setattr(runner_secrets.tempfile, "mkstemp", fail_probe_file)
    runner = DockerRunner(
        "secret-probe",
        docker_path="/usr/bin/docker",
        close_action="none",
        _container_id=container_id,
        secret_env={"CAYU_TEST_PRIVATE": SecretRef(name="probe_secret")},
        secret_resolver=vault,
        env_overlay={"CAYU_TEST_PRIVATE": overlay_canary} if outcome == "overlay_wins" else None,
        _env_overlay_secret_values_present=outcome == "overlay_wins",
    )
    if outcome == "credential_mode_drift":
        runner.credential_mode = "trusted_tool"

    async def docker_cli(path, args, **kwargs):
        dispatched.append(args)
        assert canary not in repr(args)
        if args[0] == "inspect":
            return ExecResult(
                stdout=json.dumps(
                    {"Id": container_id, "Image": image_id, "State": {"Running": True}}
                )
            )
        if any("read pid process_group" in value for value in args):
            return ExecResult()
        path = Path(args[args.index("--env-file") + 1])
        paths.append(path)
        expected_value = overlay_canary if outcome == "overlay_wins" else canary
        assert f"CAYU_TEST_PRIVATE={expected_value}\n" in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert expected_value not in kwargs["output_redactor"].redact_text(expected_value)
        if outcome == "reference_drift":
            runner.secret_env["CAYU_TEST_PRIVATE"] = SecretRef(name="changed_reference")
        if outcome == "resolver_drift":
            runner.secret_resolver = StaticVault({"probe_secret": "different-value"})
        return _completed_docker_probe_result(args, guest_exit_code=int(outcome == "missing"))

    monkeypatch.setattr(docker_module, "_run_docker", docker_cli)

    async def run():
        nonlocal allow_unlink, drain_task
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="docker"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[ScreenshotPageTool()]
        )
        task = asyncio.create_task(_run(app, f"secret_probe_{outcome}"))
        if outcome == "cancel":
            await asyncio.wait_for(resolution_started.wait(), timeout=10)
            task.cancel("cancel before Docker probe dispatch")
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            events = []
        else:
            events = await task
            assert not task.cancelled()
            assert task.cancelling() == 0
            assert (events[-1].type is EventType.SESSION_COMPLETED) is (
                outcome in {"present", "overlay_wins"}
            )
        assert len(provider.requests) == int(outcome in {"present", "overlay_wins"})
        if outcome in {
            "file_unlink_persistent",
            "file_unlink_retry_cancel",
            "file_unlink_replaced",
        }:
            retained_backup = None
            try:
                assert paths and all(path.exists() for path in paths)
                with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
                    runner._ensure_exec_open()
                if outcome == "file_unlink_replaced":
                    retained_backup = paths[0].with_suffix(".retained-test")
                    paths[0].rename(retained_backup)
                    paths[0].write_text("replacement-owned-by-test")
                    allow_unlink = True
                    assert not await app.drain_environment_cleanups(timeout_s=0.1)
                    assert paths[0].read_text() == "replacement-owned-by-test"
                elif outcome == "file_unlink_retry_cancel":
                    drain_task = asyncio.create_task(app.drain_environment_cleanups(timeout_s=0.1))
                    with pytest.raises(asyncio.CancelledError):
                        await drain_task
                    assert cancellation_sent
                    assert drain_task.cancelling() == 1
                    assert drain_task.cancelled()
                else:
                    assert not await app.drain_environment_cleanups(timeout_s=0.1)
                assert all(path.exists() for path in paths)
                with pytest.raises(RuntimeError, match="acquisition or rollback is still pending"):
                    runner._ensure_exec_open()
            finally:
                if retained_backup is not None:
                    paths[0].unlink(missing_ok=True)
                    retained_backup.rename(paths[0])
                allow_unlink = True
                assert await app.drain_environment_cleanups(timeout_s=5)
            assert all(not path.exists() for path in paths)
            runner._ensure_exec_open()
        return events

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        events = asyncio.run(run())
    assert resolution_started.is_set() is (outcome != "credential_mode_drift")
    assert bool(dispatched) is (
        outcome not in {"cancel", "resolution_failure", "credential_mode_drift"}
    )
    if outcome == "file_creation_failure":
        assert all(args[0] == "inspect" for args in dispatched)
        runner._ensure_exec_open()
    if outcome == "child_cancel_before_dispatch":
        assert cancelled_children and all(child.cancelled() for child in cancelled_children)
        assert all(args[0] == "inspect" for args in dispatched)
        assert events[-1].type is EventType.SESSION_FAILED
        runner._ensure_exec_open()
    if outcome.startswith("file_unlink"):
        assert unlink_failed
        assert not any("read pid process_group" in value for args in dispatched for value in args)
        runner._ensure_exec_open()
    try:
        assert all(not path.exists() for path in paths)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
    output = capsys.readouterr()
    diagnostics = repr([(event.type, event.payload) for event in events])
    diagnostics += caplog.text + output.out + output.err
    diagnostics += "".join(str(item.message) for item in captured)
    assert canary not in diagnostics
    assert overlay_canary not in diagnostics
