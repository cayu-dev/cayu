"""Opt-in adversarial virtual-egress test for a real E2B sandbox.

The test needs a raw TCP tunnel command because E2B cannot reach the local
Cayu process directly. The command template receives ``{host}`` and ``{port}``;
``CAYU_E2B_PROXY_URL`` must advertise the tunnel as an IPv4-literal URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import textwrap
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from tests.egress_authority_public_support import (
    AuthorizeEgressAdoptionPolicy,
    PublicEgressAdoptionHandler,
    completed_provider,
)
from tests.egress_conformance import (
    EgressScenarioEvidence,
    egress_nightly_failure_boundary,
    emit_egress_nightly_evidence,
    registration_for,
)
from tests.egress_e2e_support import CapturingEgressAdapter, drive_adversarial_egress_contract

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
    EgressAuthorityBindingIdentity,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverStrategy,
    HttpEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
    build_egress_authority_identity,
)
from cayu.egress.e2b_adapter import E2BEgressAdapter, _e2b_environment_fingerprint
from cayu.egress.proxy_exposure import ExposedProxy
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
from cayu.workspaces import E2BWorkspace

pytest.importorskip("cryptography")
pytest.importorskip("e2b")

_TUNNEL_COMMAND = os.environ.get("CAYU_E2B_PROXY_EXPOSURE_COMMAND")
_PROXY_URL = os.environ.get("CAYU_E2B_PROXY_URL")
_NEXT_TUNNEL_COMMAND = os.environ.get("CAYU_E2B_PROXY_EXPOSURE_COMMAND_NEXT")
_NEXT_PROXY_URL = os.environ.get("CAYU_E2B_PROXY_URL_NEXT")

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_E2B_EGRESS_E2E") != "1"
    or not os.environ.get("E2B_API_KEY")
    or not _TUNNEL_COMMAND
    or not _PROXY_URL,
    reason=(
        "Set CAYU_RUN_E2B_EGRESS_E2E=1, E2B_API_KEY, "
        "CAYU_E2B_PROXY_EXPOSURE_COMMAND, and CAYU_E2B_PROXY_URL."
    ),
)

REAL_SECRET = "sk_test_51E2BRealSecretNeverInGuest"


class _CommandExposure:
    def __init__(self, command_template: str, proxy_url: str) -> None:
        self._command_template = command_template
        self._proxy_url = proxy_url

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        command = self._command_template.format(host=local_host, port=local_port)
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await self._wait_until_reachable(process)
        except BaseException:
            process.terminate()
            with contextlib.suppress(Exception):
                await process.wait()
            raise

        async def teardown() -> None:
            if process.returncode is not None:
                return
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()

        return ExposedProxy(proxy_url=self._proxy_url, teardown=teardown)

    async def _wait_until_reachable(self, process: asyncio.subprocess.Process) -> None:
        split = urlsplit(self._proxy_url)
        if split.scheme != "http" or split.hostname is None:
            raise RuntimeError("CAYU_E2B_PROXY_URL must be an absolute HTTP proxy URL.")
        port = split.port or 80
        for _ in range(60):
            if process.returncode is not None:
                stderr = await process.stderr.read() if process.stderr is not None else b""
                raise RuntimeError(f"E2B tunnel exited early: {stderr.decode()[:500]}")
            try:
                _reader, writer = await asyncio.open_connection(split.hostname, port)
            except OSError:
                await asyncio.sleep(0.5)
                continue
            writer.close()
            await writer.wait_closed()
            return
        raise RuntimeError("E2B proxy tunnel did not become reachable within 30 seconds.")


class _SequenceExposure:
    """Expose the current and target Cayu routes through distinct public endpoints."""

    def __init__(self, exposures: tuple[_CommandExposure, ...]) -> None:
        self._exposures = exposures
        self._next = 0

    async def expose(self, *, local_host: str, local_port: int) -> ExposedProxy:
        if self._next >= len(self._exposures):
            raise RuntimeError("E2B cutover requested an undeclared proxy exposure.")
        exposure = self._exposures[self._next]
        self._next += 1
        return await exposure.expose(local_host=local_host, local_port=local_port)


async def _drive() -> tuple[EgressScenarioEvidence, ...]:
    assert _TUNNEL_COMMAND is not None
    assert _PROXY_URL is not None
    adapter = CapturingEgressAdapter(
        E2BEgressAdapter(
            exposure=_CommandExposure(_TUNNEL_COMMAND, _PROXY_URL),
        )
    )
    return await drive_adversarial_egress_contract(
        registration=registration_for("e2b"),
        adapter=adapter,
        real_secret=REAL_SECRET,
        image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
        search_roots=("/home/user/workspace", "/tmp", "/etc/cayu", "/root"),
        response_id="cus_e2b",
        workspace_factory=E2BWorkspace,
    )


@pytest.fixture(scope="module")
def e2e() -> tuple[EgressScenarioEvidence, ...]:
    with egress_nightly_failure_boundary("e2b"):
        return asyncio.run(_drive())


def test_e2b_shared_real_boundary_security_contract(
    e2e: tuple[EgressScenarioEvidence, ...],
) -> None:
    with egress_nightly_failure_boundary("e2b"):
        assert all(item.adapter == "e2b" for item in e2e)
        assert all(item.status == "verified" for item in e2e)
        assert REAL_SECRET not in repr(e2e)
        emit_egress_nightly_evidence(e2e)


class _AuthorityBackend:
    async def send(self, request: CapturedRequest) -> CapturedResponse:
        return CapturedResponse(
            status_code=200,
            headers={"Content-Type": "text/plain"},
            body=f"allowed:{request.host}".encode(),
        )


def _authority(
    *,
    host: str,
    policy: HttpEgressPolicy,
    generation: int,
):
    return build_egress_authority_identity(
        policies={policy.name: policy},
        bindings=(
            EgressAuthorityBindingIdentity(
                destination=host,
                policy_name=policy.name,
                credential_kind="stripe_bearer",
                credential_authority_fingerprint=("1" * 64 if generation == 1 else "2" * 64),
            ),
        ),
        generation=generation,
        authority_source="trusted-e2b-e2e",
        authority_scope="session",
        policy_version=f"v{generation}",
        runner_kind="e2b",
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def _authority_profile(authority):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="e2b-e2e",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="system",
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'0' * 64}",
        egress_authority=authority,
    )


def _authorized_transition(
    *,
    expected_authority,
    target_authority,
    owner_fingerprint: str,
    source_environment_fingerprint: str,
):
    expected = _authority_profile(expected_authority)
    candidate = execution_profile_with_egress_authority(expected, target_authority)
    changed = changed_execution_profile_components(expected, candidate)
    egress_change = execution_profile_egress_authority_change(expected, candidate)
    actor = ResolutionActor(
        subject="e2b-egress-e2e-policy",
        source=ResolutionActorSource.SYSTEM,
    )
    payload = execution_profile_decision_payload(
        kind=ExecutionProfileDecisionKind.ADOPTED,
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=changed,
        policy_identity="e2b-egress-e2e-policy:v2",
        policy_reason="The trusted test policy authorized the target destination.",
        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        egress_authority_change=egress_change,
        idempotency_identity="e2b-egress-cutover-e2e",
        actor=actor,
        reason="Adopt the second E2B egress generation.",
    )
    event = Event(
        type=EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        session_id="e2b-egress-cutover-e2e",
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
            policy_identity="e2b-egress-e2e-policy:v2",
            policy_reason="The trusted test policy authorized the target destination.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            egress_authority_change=egress_change,
            idempotency_identity="e2b-egress-cutover-e2e",
            actor=actor,
            reason="Adopt the second E2B egress generation.",
            event=event,
        )
    )
    return authorized_egress_authority_transition(
        decision=decision,
        transition_id="e2b-egress-cutover-e2e",
        environment_name="egress",
        owner_fingerprint=owner_fingerprint,
        source_environment_fingerprint=source_environment_fingerprint,
    )


async def _wait_for_guest_file(runner, path: str) -> None:
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
    raise AssertionError(f"Timed out waiting for {path} in the retained E2B workspace.")


async def _drive_authority_cutover(ca_path: Path) -> dict[str, object]:
    assert _TUNNEL_COMMAND is not None
    assert _PROXY_URL is not None
    assert _NEXT_TUNNEL_COMMAND is not None
    assert _NEXT_PROXY_URL is not None
    old_host = "old.example.com"
    new_host = "new.example.com"
    old_policy = HttpEgressPolicy(
        name="old-policy",
        allowed_hosts=(old_host,),
        allowed_endpoints=(("GET", "/allowed"),),
    )
    new_policy = HttpEgressPolicy(
        name="new-policy",
        allowed_hosts=(new_host,),
        allowed_endpoints=(("GET", "/allowed"),),
    )
    old_authority = _authority(host=old_host, policy=old_policy, generation=1)
    target_authority = _authority(host=new_host, policy=new_policy, generation=2)
    old_registry = VirtualCredentialRegistry()
    old_grant = old_registry.mint(
        session_id="e2b-egress-cutover-e2e",
        env_name="OLD_TOKEN",
        secret=SecretRef(name="old"),
        destination=old_host,
        credential_kind="stripe_bearer",
        policy_name=old_policy.name,
    )
    target_registry = VirtualCredentialRegistry()
    target_grant = target_registry.mint(
        session_id="e2b-egress-cutover-e2e",
        env_name="NEW_TOKEN",
        secret=SecretRef(name="new"),
        destination=new_host,
        credential_kind="stripe_bearer",
        policy_name=new_policy.name,
    )
    backend = _AuthorityBackend()
    old_broker = TransparentEgressBroker(
        registry=old_registry,
        resolver=StaticVault({"old": "sk_test_old_e2b_authority"}),
        policies={old_policy.name: old_policy},
        upstream=backend,
    )
    target_broker = TransparentEgressBroker(
        registry=target_registry,
        resolver=StaticVault({"new": "sk_test_new_e2b_authority"}),
        policies={new_policy.name: new_policy},
        upstream=backend,
    )
    adapter = E2BEgressAdapter(
        exposure=_SequenceExposure(
            (
                _CommandExposure(_TUNNEL_COMMAND, _PROXY_URL),
                _CommandExposure(_NEXT_TUNNEL_COMMAND, _NEXT_PROXY_URL),
            )
        ),
    )
    old_binding = await adapter.prepare(
        session_id="e2b-egress-cutover-e2e",
        grants=(old_grant,),
        broker=old_broker,
    )
    old_binding.bind_authority(old_authority)
    ca_path.write_bytes(old_binding.ca_cert_pem or b"")
    runner = await adapter.create_runner(
        VirtualEgressRunnerRequest(
            name="e2b-egress-cutover-e2e",
            runner_kind="e2b",
            image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
            binding=old_binding,
            env_overlay={**old_binding.env, old_grant.env_name: old_grant.presented_value},
            env_overlay_secret_values_present=True,
            ca_cert_host_path=str(ca_path),
            guest_ca_path=old_binding.guest_ca_path or "/etc/cayu/ca.pem",
            setup_commands=(),
            egress_destinations=(old_host,),
            session_id="e2b-egress-cutover-e2e",
            environment_name="egress",
        )
    )
    sandbox_id_before = runner.sandbox_id
    active_result = None
    try:
        before = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                textwrap.dedent(
                    f"""
                    import os, urllib.request
                    request = urllib.request.Request(
                        "https://{old_host}/allowed",
                        headers={{"Authorization": "Bearer " + os.environ["OLD_TOKEN"]}},
                    )
                    print(urllib.request.urlopen(request, timeout=20).read().decode())
                    open("/home/user/workspace/authority-continuity", "w").write("same-workspace")
                    """
                ),
            ),
            timeout_s=40,
        )
        assert before.exit_code == 0

        stale_script = textwrap.dedent(
            f"""
            import os, socket, ssl, time
            from urllib.parse import urlparse
            open("/home/user/workspace/stale-pid", "w").write(str(os.getpid()))
            proxy = urlparse(os.environ["HTTPS_PROXY"])
            raw = socket.create_connection((proxy.hostname, proxy.port), timeout=20)
            raw.sendall(b"CONNECT {old_host}:443 HTTP/1.1\\r\\nHost: {old_host}:443\\r\\n\\r\\n")
            head = b""
            while not head.endswith(b"\\r\\n\\r\\n"):
                head += raw.recv(1)
            context = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
            context.wrap_socket(raw, server_hostname={old_host!r})
            open("/home/user/workspace/stale-ready", "w").write("ready")
            while True:
                time.sleep(1)
            """
        )
        launched = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                textwrap.dedent(
                    f"""
                    import subprocess, sys
                    subprocess.Popen(
                        [sys.executable, "-c", {stale_script!r}],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    """
                ),
            ),
            timeout_s=10,
        )
        assert launched.exit_code == 0
        await _wait_for_guest_file(runner, "/home/user/workspace/stale-ready")
        stale_pid_result = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                'print(open("/home/user/workspace/stale-pid").read())',
            ),
            timeout_s=5,
        )
        stale_pid = int(stale_pid_result.stdout.strip())

        session_store = InMemorySessionStore()
        await session_store.create(
            RunRequest(
                agent_name="agent",
                session_id="e2b-egress-cutover-e2e",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        coordinator = EgressAuthorityTransitionCoordinator(
            SessionCheckpointEgressAuthorityTransitionStore(session_store)
        )
        owner_token = "e2b-e2e-owner"
        owner_fingerprint = egress_authority_owner_fingerprint(owner_token)
        authorized = await coordinator.authorize(
            _authorized_transition(
                expected_authority=old_authority,
                target_authority=target_authority,
                owner_fingerprint=owner_fingerprint,
                source_environment_fingerprint=_e2b_environment_fingerprint(runner.sandbox_id),
            )
        )

        async def revoke_old() -> bool:
            await old_broker.revoke_authority_and_wait((old_grant.presented_value,))
            return False

        active, active_result = await coordinator.install(
            authorized=authorized,
            adapter=adapter,
            request=EgressAuthorityCutoverRequest(
                session_id="e2b-egress-cutover-e2e",
                environment_name="egress",
                owner_fingerprint=owner_fingerprint,
                environment_fingerprint=_e2b_environment_fingerprint(runner.sandbox_id),
                runner=runner,
                current_binding=old_binding,
                expected_authority=old_authority,
                target_authority=target_authority,
                target_broker=target_broker,
                target_grants=(target_grant,),
                target_env_overlay={target_grant.env_name: target_grant.presented_value},
                target_egress_destinations=(new_host,),
                revoke_current_authority=revoke_old,
                ca_cert_host_path=str(ca_path),
                guest_ca_path=old_binding.guest_ca_path or "/etc/cayu/ca.pem",
                invocation_quiescent=True,
            ),
            owner_token=owner_token,
        )
        assert active_result is not None
        after = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                textwrap.dedent(
                    f"""
                    import os, urllib.request
                    request = urllib.request.Request(
                        "https://{new_host}/allowed",
                        headers={{"Authorization": "Bearer " + os.environ["NEW_TOKEN"]}},
                    )
                    print(urllib.request.urlopen(request, timeout=20).read().decode())
                    print(open("/home/user/workspace/authority-continuity").read())
                    """
                ),
            ),
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
        old_route_denial = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                textwrap.dedent(
                    f"""
                    import urllib.error, urllib.request
                    request = urllib.request.Request(
                        "https://{old_host}/allowed",
                        headers={{"Authorization": "Bearer " + {old_grant.presented_value!r}}},
                    )
                    try:
                        urllib.request.urlopen(request, timeout=20)
                    except urllib.error.HTTPError as error:
                        print(error.code)
                    """
                ),
            ),
            timeout_s=40,
        )
        old_denial = await old_broker.handle_request(
            CapturedRequest(
                method="GET",
                host=old_host,
                path="/allowed",
                headers={"Authorization": f"Bearer {old_grant.presented_value}"},
            )
        )
        events = await session_store.query_events()
        return {
            "active": active,
            "after": after,
            "sandbox_id_before": sandbox_id_before,
            "sandbox_id_after": runner.sandbox_id,
            "stale_process": stale_process,
            "old_route_denial": old_route_denial,
            "old_denial": old_denial,
            "events": events,
        }
    finally:
        await target_broker.revoke_authority_and_wait((target_grant.presented_value,))
        if active_result is not None:
            await active_result.binding.close()
        else:
            await old_binding.close()
        await adapter.finalize_runner(runner, outcome="completed")


@pytest.mark.skipif(
    not _NEXT_TUNNEL_COMMAND or not _NEXT_PROXY_URL,
    reason=(
        "E2B authority cutover additionally requires "
        "CAYU_E2B_PROXY_EXPOSURE_COMMAND_NEXT and CAYU_E2B_PROXY_URL_NEXT."
    ),
)
def test_e2b_rotates_authority_on_same_sandbox_and_workspace(tmp_path: Path) -> None:
    evidence = asyncio.run(_drive_authority_cutover(tmp_path / "e2b-authority-ca.pem"))
    assert evidence["active"].state.value == "active"
    assert evidence["active"].receipt.same_allocation is True
    assert evidence["sandbox_id_before"] == evidence["sandbox_id_after"]
    assert evidence["after"].exit_code == 0
    assert "allowed:new.example.com" in evidence["after"].stdout
    assert "same-workspace" in evidence["after"].stdout
    assert evidence["stale_process"].exit_code == 0
    assert evidence["old_route_denial"].stdout.strip() == "403"
    assert evidence["old_denial"].status_code == 403
    assert evidence["events"][-1].event.type is EventType.EGRESS_AUTHORITY_ACTIVATED


@pytest.mark.skipif(
    not _NEXT_TUNNEL_COMMAND or not _NEXT_PROXY_URL,
    reason=(
        "E2B authority cutover additionally requires "
        "CAYU_E2B_PROXY_EXPOSURE_COMMAND_NEXT and CAYU_E2B_PROXY_URL_NEXT."
    ),
)
def test_public_resume_rotates_the_runtime_retained_e2b_sandbox() -> None:
    async def exercise() -> None:
        assert _TUNNEL_COMMAND is not None
        assert _PROXY_URL is not None
        assert _NEXT_TUNNEL_COMMAND is not None
        assert _NEXT_PROXY_URL is not None
        session_id = "e2b-egress-public-resume-e2e"
        old_host = "old.example.com"
        new_host = "new.example.com"
        old_policy = HttpEgressPolicy(
            name="old-policy",
            allowed_hosts=(old_host,),
            allowed_endpoints=(("GET", "/allowed"),),
        )
        new_policy = HttpEgressPolicy(
            name="new-policy",
            allowed_hosts=(new_host,),
            allowed_endpoints=(("GET", "/allowed"),),
        )
        adapter = E2BEgressAdapter(
            exposure=_SequenceExposure(
                (
                    _CommandExposure(_TUNNEL_COMMAND, _PROXY_URL),
                    _CommandExposure(_NEXT_TUNNEL_COMMAND, _NEXT_PROXY_URL),
                )
            )
        )
        current_factory = VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"old": "sk_test_old_public_e2b_authority"}),
            policies={old_policy.name: old_policy},
            credentials=(
                VirtualCredentialSpec(
                    env_name="OLD_TOKEN",
                    secret=SecretRef(name="old"),
                    destination=old_host,
                    policy_name=old_policy.name,
                ),
            ),
            adapter=adapter,
            image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
            workspace_factory=E2BWorkspace,
            upstream=_AuthorityBackend(),
            egress_authority_generation=1,
            egress_authority_source="trusted-e2b-public-e2e",
            egress_policy_version="v1",
        )
        target_factory = VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"new": "sk_test_new_public_e2b_authority"}),
            policies={new_policy.name: new_policy},
            credentials=(
                VirtualCredentialSpec(
                    env_name="NEW_TOKEN",
                    secret=SecretRef(name="new"),
                    destination=new_host,
                    policy_name=new_policy.name,
                ),
            ),
            adapter=adapter,
            image=os.environ.get("CAYU_E2B_TEMPLATE", "base"),
            workspace_factory=E2BWorkspace,
            upstream=_AuthorityBackend(),
            egress_authority_generation=2,
            egress_authority_source="trusted-e2b-public-e2e",
            egress_policy_version="v2",
        )
        store = InMemorySessionStore()
        provider = completed_provider()
        handler = PublicEgressAdoptionHandler(
            target_factory=target_factory,
            owner_token="e2b-public-resume-owner",
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
        await _collect_public_events(
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
            events = await _collect_public_events(
                target_app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="e2b-public-egress-generation-2",
                            reason="Adopt the reviewed E2B egress generation.",
                            requested_by=ResolutionActor(
                                subject="e2b-public-e2e-policy",
                                source=ResolutionActorSource.SYSTEM,
                            ),
                        ),
                    )
                )
            )
            assert handler.calls == 1
            assert handler.result is not None
            assert handler.factory_result is not None
            managed = handler.factory_result.environment.runner
            workspace = handler.factory_result.environment.workspace
            assert managed is not None
            assert workspace is not None
            assert handler.result.transition.receipt is not None
            assert handler.result.transition.receipt.same_allocation is True
            assert (
                handler.result.transition.receipt.environment_fingerprint
                == handler.expected_environment_fingerprint
            )
            assert managed._runner.sandbox_id in workspace.id
            assert len(provider.requests) == 2
            assert EventType.SESSION_COMPLETED in {event.type for event in events}
        finally:
            if handler.factory_result is not None:
                managed = handler.factory_result.environment.runner
                if managed is not None:
                    await managed.finalize(outcome="completed")

    asyncio.run(exercise())


async def _collect_public_events(stream):
    return [event async for event in stream]
