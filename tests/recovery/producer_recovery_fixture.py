"""Explicitly versioned deterministic provider for cross-process producer recovery."""

import asyncio

from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_peer_content import QualifiedPeerProvider

from cayu import BeforeStopDecision, CayuConfig, LoopPolicy, Message, OperationsConfig
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.cleanup import RecoveryCleanupPolicy


def producer_recovery_config():
    return CayuConfig(
        operations=OperationsConfig(
            recovery_cleanup_policy=RecoveryCleanupPolicy(
                step_timeout_seconds=2, overall_timeout_seconds=5
            )
        )
    )


class ProducerRecoveryProvider(QualifiedPeerProvider):
    @property
    def execution_profile_identity(self):
        return versioned_test_provider_identity(self)


class ProducerRecoveryPolicy(LoopPolicy):
    """Stable policy whose external decision may change between observations."""

    def __init__(self, *, continue_requested=False, block=False):
        self.continue_requested = continue_requested
        self.block = block
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:producer-recovery-policy", behavior_version="1", implementation_version="1"
        )

    async def before_stop(self, context):
        self.calls += 1
        if self.block:
            self.entered.set()
            await self.release.wait()
        if self.continue_requested:
            return BeforeStopDecision.continue_with(
                Message.text("user", "Additional work needs a separate execution owner."),
                reason="external review requires another step",
            )
        return BeforeStopDecision.complete()
