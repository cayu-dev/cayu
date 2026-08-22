"""Shared public-runtime support for real egress-authority tracers."""

from __future__ import annotations

from cayu import (
    EgressAuthorityAdoptionHandler,
    EgressAuthorityAdoptionResult,
    EgressAuthorityTransitionCoordinator,
    EnvironmentFactoryResult,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    ModelStreamEvent,
    VirtualEgressEnvironmentFactory,
    authorized_egress_authority_transition,
    egress_authority_owner_fingerprint,
)
from tests.core._workload_secret_support import FakeProvider


class AuthorizeEgressAdoptionPolicy(ExecutionProfilePolicy):
    """Trusted application policy used by public backend tracers."""

    @property
    def identity(self) -> str:
        return "tests:public-egress-adoption-policy:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        del request
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Authorize the exact backend egress generation transition.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class PublicEgressAdoptionHandler(EgressAuthorityAdoptionHandler):
    """Install a target factory on the exact runtime-parked allocation."""

    def __init__(
        self,
        *,
        target_factory: VirtualEgressEnvironmentFactory,
        owner_token: str,
    ) -> None:
        self._target_factory = target_factory
        self._owner_token = owner_token
        self.calls = 0
        self.factory_result: EnvironmentFactoryResult | None = None
        self.result: EgressAuthorityAdoptionResult | None = None
        self.expected_environment_fingerprint: str | None = None

    async def adopt(
        self,
        decision: ExecutionProfileDecision,
        *,
        coordinator: EgressAuthorityTransitionCoordinator,
        expected_environment_fingerprint: str,
        factory_result: EnvironmentFactoryResult,
    ) -> EgressAuthorityAdoptionResult:
        self.calls += 1
        self.factory_result = factory_result
        self.expected_environment_fingerprint = expected_environment_fingerprint
        authorized = authorized_egress_authority_transition(
            decision=decision,
            transition_id=(
                f"{decision.event.session_id}:egress:"
                f"{decision.candidate_profile.egress_authority.generation}"
            ),
            environment_name=decision.event.environment_name or "egress",
            owner_fingerprint=egress_authority_owner_fingerprint(self._owner_token),
            source_environment_fingerprint=expected_environment_fingerprint,
        )
        self.result = await self._target_factory.adopt_authority(
            factory_result=factory_result,
            authorized=authorized,
            coordinator=coordinator,
            owner_token=self._owner_token,
            agent_name=decision.event.agent_name or "agent",
            execution_profile_fingerprint=decision.candidate_profile.fingerprint,
        )
        return self.result


def completed_provider(*, batches: int = 2) -> FakeProvider:
    return FakeProvider(
        [
            [ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"})]
            for _ in range(batches)
        ]
    )
