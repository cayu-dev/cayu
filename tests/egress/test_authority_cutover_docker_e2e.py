"""Real Docker tracer for governed same-allocation egress-authority adoption."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import textwrap

import pytest
from tests.egress_authority_public_support import (
    AuthorizeEgressAdoptionPolicy,
    PublicEgressAdoptionHandler,
    completed_provider,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    ExecutionProfileAdoptionIntent,
    Message,
    ResumeRequest,
)
from cayu.core.events import Event, EventType
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpEgressPolicy,
)
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners.base import ExecCommand
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.runtime.egress_authority_transitions import (
    EgressAuthorityTransitionCoordinator,
    SessionCheckpointEgressAuthorityTransitionStore,
    authorized_egress_authority_transition,
    egress_authority_owner_fingerprint,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    _with_runtime_execution_profile_decision_authority,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_decision_payload,
    execution_profile_egress_authority_change,
    execution_profile_with_egress_authority,
)
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

_IMAGE = "python:3.12-slim"
_OLD_HOST = "old.example.com"
_NEW_HOST = "new.example.com"


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


_DOCKER_AVAILABLE = _docker_available()
if os.environ.get("CAYU_REQUIRE_DOCKER_EGRESS") == "1" and not _DOCKER_AVAILABLE:
    raise RuntimeError("CAYU_REQUIRE_DOCKER_EGRESS=1 but the Docker daemon is unavailable.")

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        not _DOCKER_AVAILABLE,
        reason="Docker daemon not available for egress-authority E2E.",
    ),
]


class _Backend:
    def prepare(
        self,
        request: CapturedRequest,
        *,
        limits: EgressUpstreamLimits,
    ) -> EgressUpstreamOperation:
        assert limits.max_response_bytes > 0

        async def send() -> CapturedResponse:
            return CapturedResponse(
                status_code=200,
                headers={"Content-Type": "text/plain"},
                body=f"allowed:{request.host}".encode(),
            )

        return EgressUpstreamOperation(send)


def _policy(name: str, host: str) -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name=name,
        allowed_hosts=(host,),
        allowed_endpoints=(("GET", "/allowed"),),
    )


def _profile(authority):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="docker-e2e",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="system",
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'0' * 64}",
        egress_authority=authority,
    )


def _authorized_record(
    *,
    current_factory: VirtualEgressEnvironmentFactory,
    target_factory: VirtualEgressEnvironmentFactory,
    owner_fingerprint: str,
    source_environment_fingerprint: str,
):
    expected = _profile(current_factory.egress_authority_identity)
    candidate = execution_profile_with_egress_authority(
        expected,
        target_factory.egress_authority_identity,
    )
    changed = changed_execution_profile_components(expected, candidate)
    egress_change = execution_profile_egress_authority_change(expected, candidate)
    actor = ResolutionActor(
        subject="docker-egress-e2e-policy",
        source=ResolutionActorSource.SYSTEM,
    )
    payload = execution_profile_decision_payload(
        kind=ExecutionProfileDecisionKind.ADOPTED,
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=changed,
        policy_identity="docker-egress-e2e-policy:v2",
        policy_reason="The trusted test policy authorized the target destination.",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        egress_authority_change=egress_change,
        idempotency_identity="docker-egress-cutover-e2e",
        actor=actor,
        reason="Adopt the second egress generation.",
    )
    event = Event(
        type=EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        session_id="docker-egress-cutover-e2e",
        agent_name="agent",
        environment_name="egress",
        payload=payload,
    )
    decision = _with_runtime_execution_profile_decision_authority(
        ExecutionProfileDecision(
            kind=ExecutionProfileDecisionKind.ADOPTED,
            expected_profile=expected,
            candidate_profile=candidate,
            changed_component_classes=changed,
            policy_identity="docker-egress-e2e-policy:v2",
            policy_reason="The trusted test policy authorized the target destination.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            egress_authority_change=egress_change,
            idempotency_identity="docker-egress-cutover-e2e",
            actor=actor,
            reason="Adopt the second egress generation.",
            event=event,
        )
    )
    return authorized_egress_authority_transition(
        decision=decision,
        transition_id="docker-egress-cutover-e2e",
        environment_name="egress",
        owner_fingerprint=owner_fingerprint,
        source_environment_fingerprint=source_environment_fingerprint,
    )


async def _wait_for_file(runner, path: str) -> None:
    for _ in range(80):
        result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                f"import os,sys; sys.exit(0 if os.path.exists({path!r}) else 1)",
            ),
            timeout_s=5,
        )
        if result.exit_code == 0:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {path} in the retained container.")


async def _drive_real_docker_cutover() -> dict[str, object]:
    backend = _Backend()
    current_factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"old": "sk_test_old_authority"}),
        policies={"old-policy": _policy("old-policy", _OLD_HOST)},
        credentials=(
            VirtualCredentialSpec(
                env_name="OLD_TOKEN",
                secret=SecretRef(name="old"),
                destination=_OLD_HOST,
                policy_name="old-policy",
            ),
        ),
        runner_kind="docker",
        image=_IMAGE,
        upstream=backend,
        egress_authority_generation=1,
        egress_authority_source="trusted-docker-e2e",
        egress_policy_version="v1",
    )
    target_factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"new": "sk_test_new_authority"}),
        policies={"new-policy": _policy("new-policy", _NEW_HOST)},
        credentials=(
            VirtualCredentialSpec(
                env_name="NEW_TOKEN",
                secret=SecretRef(name="new"),
                destination=_NEW_HOST,
                policy_name="new-policy",
            ),
        ),
        runner_kind="docker",
        image=_IMAGE,
        upstream=backend,
        egress_authority_generation=2,
        egress_authority_source="trusted-docker-e2e",
        egress_policy_version="v2",
    )
    result = await current_factory.create(
        EnvironmentFactoryRequest(
            session_id="docker-egress-cutover-e2e",
            agent_name="agent",
            environment_name="egress",
        )
    )
    environment = result.environment
    runner = environment.runner
    assert runner is not None
    inner = runner._runner
    container_name = inner.name
    adapter = runner._adapter
    container_id_before = await adapter._container_id(container_name)
    old_virtual_token = runner._authority_revoker._presented_values[0]

    request_script = textwrap.dedent(
        f"""
        import os, urllib.request
        request = urllib.request.Request(
            "https://{_OLD_HOST}/allowed",
            headers={{"Authorization": "Bearer " + os.environ["OLD_TOKEN"]}},
        )
        print(urllib.request.urlopen(request, timeout=20).read().decode())
        open("/workspace/authority-continuity", "w").write("same-workspace")
        """
    )
    before = await runner.exec(
        ExecCommand.process("python3", "-c", request_script),
        timeout_s=40,
    )
    assert before.exit_code == 0
    assert f"allowed:{_OLD_HOST}" in before.stdout

    stale_connection_script = textwrap.dedent(
        f"""
        import os, socket, ssl, time
        from urllib.parse import urlparse

        open("/workspace/stale-pid", "w").write(str(os.getpid()))
        proxy = urlparse(os.environ["HTTPS_PROXY"])
        raw = socket.create_connection((proxy.hostname, proxy.port), timeout=20)
        raw.sendall(b"CONNECT {_OLD_HOST}:443 HTTP/1.1\\r\\nHost: {_OLD_HOST}:443\\r\\n\\r\\n")
        head = b""
        while not head.endswith(b"\\r\\n\\r\\n"):
            head += raw.recv(1)
        context = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
        context.wrap_socket(raw, server_hostname={_OLD_HOST!r})
        open("/workspace/stale-ready", "w").write("ready")
        while True:
            time.sleep(1)
        """
    )
    launcher = textwrap.dedent(
        f"""
        import subprocess, sys
        subprocess.Popen(
            [sys.executable, "-c", {stale_connection_script!r}],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        """
    )
    launched = await runner.exec(
        ExecCommand.process("python3", "-c", launcher),
        timeout_s=10,
    )
    assert launched.exit_code == 0
    await _wait_for_file(runner, "/workspace/stale-ready")
    stale_pid_result = await runner.exec(
        ExecCommand.process(
            "python3",
            "-c",
            'print(open("/workspace/stale-pid").read())',
        ),
        timeout_s=5,
    )
    stale_pid = int(stale_pid_result.stdout.strip())

    session_store = InMemorySessionStore()
    await session_store.create(
        RunRequest(
            agent_name="agent",
            session_id="docker-egress-cutover-e2e",
            messages=[],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    coordinator = EgressAuthorityTransitionCoordinator(
        SessionCheckpointEgressAuthorityTransitionStore(session_store)
    )
    owner_token = "docker-e2e-owner"
    owner_fingerprint = egress_authority_owner_fingerprint(owner_token)
    authorized = _authorized_record(
        current_factory=current_factory,
        target_factory=target_factory,
        owner_fingerprint=owner_fingerprint,
        source_environment_fingerprint=result.reconnect_metadata["allocation_fingerprint"],
    )
    try:
        active_handoff = await target_factory.adopt_authority(
            factory_result=result,
            authorized=authorized,
            coordinator=coordinator,
            owner_token=owner_token,
            agent_name="agent",
        )
        reconciled_handoff = await target_factory.adopt_authority(
            factory_result=result,
            authorized=authorized,
            coordinator=EgressAuthorityTransitionCoordinator(
                SessionCheckpointEgressAuthorityTransitionStore(session_store)
            ),
            owner_token=owner_token,
            agent_name="agent",
        )
        active = active_handoff.transition
        reconciled_active = reconciled_handoff.transition
        container_id_after = await adapter._container_id(container_name)

        new_request = textwrap.dedent(
            f"""
            import os, urllib.request
            request = urllib.request.Request(
                "https://{_NEW_HOST}/allowed",
                headers={{"Authorization": "Bearer " + os.environ["NEW_TOKEN"]}},
            )
            print(urllib.request.urlopen(request, timeout=20).read().decode())
            print(open("/workspace/authority-continuity").read())
            """
        )
        new_result = await runner.exec(
            ExecCommand.process("python3", "-c", new_request),
            timeout_s=40,
        )

        old_denial = textwrap.dedent(
            f"""
            import urllib.error, urllib.request
            request = urllib.request.Request(
                "https://{_OLD_HOST}/allowed",
                headers={{"Authorization": "Bearer " + {old_virtual_token!r}}},
            )
            try:
                urllib.request.urlopen(request, timeout=20)
            except urllib.error.HTTPError as error:
                print(error.code)
            except urllib.error.URLError as error:
                if "Tunnel connection failed: 403 Forbidden" not in str(error.reason):
                    raise
                print(403)
            """
        )
        denied_result = await runner.exec(
            ExecCommand.process("python3", "-c", old_denial),
            timeout_s=40,
        )
        stale_process = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                f"import os,sys;\ntry: os.kill({stale_pid}, 0)\nexcept ProcessLookupError: sys.exit(0)\nsys.exit(1)",
            ),
            timeout_s=5,
        )
        evidence = await session_store.query_events()
        return {
            "active": active,
            "reconciled_active": reconciled_active,
            "container_id_before": container_id_before,
            "container_id_after": container_id_after,
            "new_result": new_result,
            "denied_result": denied_result,
            "stale_process": stale_process,
            "events": evidence,
        }
    finally:
        bound = await environment.binding.bind(
            None,
            runner,
            session_id="docker-egress-cutover-e2e",
        )
        await environment.binding.finalize(bound, outcome="completed")


def test_real_docker_rotates_authority_on_same_container_and_workspace() -> None:
    evidence = asyncio.run(_drive_real_docker_cutover())
    active = evidence["active"]
    assert active.state.value == "active"
    assert active.receipt.same_allocation is True
    assert active.receipt.workspace_continuity_verified is True
    assert evidence["reconciled_active"] == active
    assert evidence["container_id_before"] == evidence["container_id_after"]
    assert evidence["new_result"].exit_code == 0
    assert f"allowed:{_NEW_HOST}" in evidence["new_result"].stdout
    assert "same-workspace" in evidence["new_result"].stdout
    assert evidence["denied_result"].stdout.strip() == "403"
    assert evidence["stale_process"].exit_code == 0
    assert [record.event.type for record in evidence["events"]][-1] is (
        EventType.EGRESS_AUTHORITY_ACTIVATED
    )


def test_public_resume_rotates_the_runtime_retained_docker_allocation() -> None:
    async def exercise() -> None:
        session_id = "docker-egress-public-resume-e2e"
        backend = _Backend()
        current_factory = VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"old": "sk_test_old_public_authority"}),
            policies={"old-policy": _policy("old-policy", _OLD_HOST)},
            credentials=(
                VirtualCredentialSpec(
                    env_name="OLD_TOKEN",
                    secret=SecretRef(name="old"),
                    destination=_OLD_HOST,
                    policy_name="old-policy",
                ),
            ),
            runner_kind="docker",
            image=_IMAGE,
            upstream=backend,
            egress_authority_generation=1,
            egress_authority_source="trusted-docker-public-e2e",
            egress_policy_version="v1",
        )
        target_factory = VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"new": "sk_test_new_public_authority"}),
            policies={"new-policy": _policy("new-policy", _NEW_HOST)},
            credentials=(
                VirtualCredentialSpec(
                    env_name="NEW_TOKEN",
                    secret=SecretRef(name="new"),
                    destination=_NEW_HOST,
                    policy_name="new-policy",
                ),
            ),
            runner_kind="docker",
            image=_IMAGE,
            upstream=backend,
            egress_authority_generation=2,
            egress_authority_source="trusted-docker-public-e2e",
            egress_policy_version="v2",
        )
        store = InMemorySessionStore()
        provider = completed_provider()
        handler = PublicEgressAdoptionHandler(
            target_factory=target_factory,
            owner_token="docker-public-resume-owner",
        )
        source_app = CayuApp(
            session_store=store,
            egress_authority_adoption_handler=handler,
            enable_logging=False,
        )
        source_app.register_provider(provider, default=True)
        source_app.register_environment_factory(
            EnvironmentSpec(name="egress"),
            current_factory,
            default=True,
        )
        source_app.register_agent(AgentSpec(name="agent", model="fake-model"))
        await _collect_events(
            source_app.run(
                RunRequest(
                    agent_name="agent",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                )
            )
        )

        target_app = CayuApp(
            session_store=store,
            execution_profile_policy=AuthorizeEgressAdoptionPolicy(),
            egress_authority_adoption_handler=handler,
            enable_logging=False,
        )
        target_app.register_provider(provider, default=True)
        target_app.register_environment_factory(
            EnvironmentSpec(name="egress"),
            target_factory,
            default=True,
        )
        target_app.register_agent(AgentSpec(name="agent", model="fake-model"))
        try:
            events = await _collect_events(
                target_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="docker-public-egress-generation-2",
                            reason="Adopt the reviewed Docker egress generation.",
                            requested_by=ResolutionActor(
                                subject="docker-public-e2e-policy",
                                source=ResolutionActorSource.SYSTEM,
                            ),
                        ),
                    )
                )
            )
            assert handler.calls == 1
            assert handler.result is not None
            assert handler.factory_result is not None
            assert handler.result.transition.receipt is not None
            assert handler.result.transition.receipt.same_allocation is True
            assert (
                handler.result.transition.receipt.environment_fingerprint
                == handler.expected_environment_fingerprint
            )
            assert len(provider.requests) == 2
            assert EventType.SESSION_COMPLETED in {event.type for event in events}
        finally:
            if handler.factory_result is not None:
                managed = handler.factory_result.environment.runner
                if managed is not None:
                    await managed.finalize(outcome="completed")

    asyncio.run(exercise())


async def _collect_events(stream):
    return [event async for event in stream]
