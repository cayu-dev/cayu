from __future__ import annotations

import asyncio
from typing import Any

from tests.runtime.test_egress_authority_transitions import (
    _create_session,
    _decision_between,
)

from cayu.egress.adapter import (
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
)
from cayu.egress.authority import (
    EgressAuthorityCutoverStrategy,
    _build_adapter_verified_egress_authority_cutover_receipt,
)
from cayu.egress.capabilities import EgressCapabilityEvidence
from cayu.egress.policy import HttpEgressPolicy
from cayu.egress.runtime import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.egress.transitions import (
    EgressAuthorityTransitionCoordinator,
    SessionCheckpointEgressAuthorityTransitionStore,
    authorized_egress_authority_transition,
    egress_authority_owner_fingerprint,
)
from cayu.environments.factory import EnvironmentFactoryRequest
from cayu.events import Event, EventType
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.sessions.memory import InMemorySessionStore
from cayu.vaults import SecretRef, StaticVault

_SESSION = "handoff-session"
_ENVIRONMENT = "egress"
_FINGERPRINT = "d" * 64
_SECRET = "sk_test_handoff_secret"
_POLICY = "provider"


class _HandoffRunner(Runner):
    isolation = "docker"

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        del command, kwargs
        raise AssertionError("handoff tests do not execute guest work")


class _HandoffAdapter(SandboxEgressAdapter):
    runner_kind = "docker"
    process_external_allocation = False
    egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

    async def prepare(self, *, session_id, grants, broker) -> EgressBinding:  # type: ignore[no-untyped-def]
        del session_id, grants, broker
        return EgressBinding(
            ca_cert_pem=b"certificate",
            runner_kind=self.runner_kind,
            guest_ca_path="/etc/cayu/ca.pem",
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        del request
        return _HandoffRunner()

    def capability_evidence(self, runner: Runner) -> EgressCapabilityEvidence:
        del runner
        return EgressCapabilityEvidence.unclaimed(self.runner_kind)

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        del runner
        return _FINGERPRINT

    async def cutover_authority(self, request) -> Any:  # type: ignore[no-untyped-def]
        replacement = EgressBinding(
            ca_cert_pem=b"certificate",
            runner_kind=self.runner_kind,
            guest_ca_path="/etc/cayu/ca.pem",
        )
        replacement.bind_authority(request.target_authority)
        receipt = _build_adapter_verified_egress_authority_cutover_receipt(
            expected=request.expected_authority,
            target=request.target_authority,
            environment_fingerprint=_FINGERPRINT,
        )
        from cayu.egress.adapter import EgressAuthorityCutoverResult

        return EgressAuthorityCutoverResult(binding=replacement, receipt=receipt)

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        del outcome
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


def _factory(
    adapter: _HandoffAdapter,
    *,
    generation: int,
    allow_post: bool,
) -> VirtualEgressEnvironmentFactory:
    endpoints = [("GET", "/v1/items")]
    if allow_post:
        endpoints.append(("POST", "/v1/items"))
    policy = HttpEgressPolicy(
        name=_POLICY,
        allowed_hosts=["api.example.com"],
        allowed_endpoints=endpoints,
    )
    return VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"handoff_secret": _SECRET}),
        policies={_POLICY: policy},
        credentials=[
            VirtualCredentialSpec(
                env_name="HANDOFF_TOKEN",
                secret=SecretRef(name="handoff_secret"),
                destination="api.example.com",
                policy_name=_POLICY,
            )
        ],
        adapter=adapter,
        egress_authority_generation=generation,
    )


async def _adopted_environment(*, adapter: _HandoffAdapter, events: list[Event]):
    async def emit(event: Event) -> Event:
        events.append(event)
        return event

    source_factory = _factory(adapter, generation=1, allow_post=False)
    target_factory = _factory(adapter, generation=2, allow_post=True)
    result = await source_factory.create(
        EnvironmentFactoryRequest(
            session_id=_SESSION,
            agent_name="assistant",
            environment_name=_ENVIRONMENT,
        )
    )
    # The source result has already emitted its grant events through its own
    # bridge.  The target bridge is the one whose adoption events we inspect.
    target_factory._emitter = emit
    result.environment.binding._emitter = emit  # type: ignore[union-attr]

    session_store = InMemorySessionStore()
    await _create_session(session_store, _SESSION)
    expected = source_factory.egress_authority_identity
    target = target_factory.egress_authority_identity
    decision = _decision_between(
        expected,
        target,
        session_id=_SESSION,
        idempotency_identity="handoff-transition",
    )
    authorized = authorized_egress_authority_transition(
        decision=decision,
        transition_id="handoff-transition",
        environment_name=_ENVIRONMENT,
        owner_fingerprint=egress_authority_owner_fingerprint("handoff-owner"),
        source_environment_fingerprint=_FINGERPRINT,
    )
    coordinator = EgressAuthorityTransitionCoordinator(
        SessionCheckpointEgressAuthorityTransitionStore(session_store)
    )
    adoption = await target_factory.adopt_authority(
        factory_result=result,
        authorized=authorized,
        coordinator=coordinator,
        owner_token="handoff-owner",
        agent_name="assistant",
    )
    assert adoption.environment is result.environment
    managed = result.environment.runner
    assert managed is not None
    target_grant = managed._egress_grants[0]
    binding = result.environment.binding
    assert binding is not None
    bound = await binding.bind(None, managed, session_id=_SESSION)
    return binding, bound, target_grant


def test_authority_adoption_tracks_target_grant_lifecycle_events() -> None:
    async def run() -> tuple[set[str], set[str], str]:
        events: list[Event] = []
        adapter = _HandoffAdapter()
        binding, bound, target_grant = await _adopted_environment(
            adapter=adapter,
            events=events,
        )
        await binding.finalize(bound, outcome="completed")
        minted = {
            event.payload["grant_id"]
            for event in events
            if event.type is EventType.EGRESS_GRANT_MINTED
        }
        revoked = {
            event.payload["grant_id"]
            for event in events
            if event.type is EventType.EGRESS_GRANT_REVOKED
        }
        return minted, revoked, target_grant.grant_id

    minted, revoked, target_id = asyncio.run(run())
    assert (target_id in minted, target_id in revoked) == (True, True)
