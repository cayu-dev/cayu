"""Application permission boundary; not full authenticated-channel acceptance."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_browser_control import identity

from cayu.runtime._browser_control_authorization import (
    BrowserControlPermissionDenied,
    authorize_browser_control,
)
from cayu.runtime.browser_control import (
    BrowserControlPolicy,
    BrowserControlPolicyResult,
    BrowserControlPrincipal,
    BrowserControlRecord,
)


class Policy(BrowserControlPolicy):
    identity = "browser-permissions:v1"

    def __init__(self, allowed):
        self.allowed = allowed
        self.requests = []

    async def decide(self, request):
        self.requests.append(request)
        return BrowserControlPolicyResult(allowed=self.allowed)


async def authorize(policy, *, action="view", tenant=None):
    return await authorize_browser_control(
        policy=policy,
        principal=BrowserControlPrincipal(subject="authenticated-user", tenant=tenant),
        record=BrowserControlRecord(identity=identity()),
        operator_session_id="server-issued-session",
        action=action,
    )


@pytest.mark.parametrize("tenant", [None, "same-tenant"])
def test_authentication_and_tenant_without_policy_cannot_authorize(tenant) -> None:
    async def scenario():
        with pytest.raises(BrowserControlPermissionDenied):
            await authorize(None, tenant=tenant)
        with pytest.raises(BrowserControlPermissionDenied):
            await authorize(Policy(False), tenant=tenant)

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["view", "takeover", "renew", "handback", "checkpoint"])
def test_authorization_binds_exact_record_and_action(action) -> None:
    async def scenario():
        policy = Policy(True)
        result = await authorize(policy, action=action)
        assert result.action == action
        assert result.record == BrowserControlRecord(identity=identity())
        assert result.operator.subject == "authenticated-user"
        assert result.operator.tenant is None
        assert result.operator.operator_session_id == "server-issued-session"
        request = policy.requests[0]
        assert request.identity == identity()
        assert request.record_revision == result.record.revision
        assert request.control_epoch == result.record.control_epoch
        assert request.state == result.record.state

    asyncio.run(scenario())


def test_policy_cannot_change_the_scope_it_was_asked_to_authorize() -> None:
    class MutatingPolicy(Policy):
        async def decide(self, request):
            object.__setattr__(request.identity, "session_id", "another-session")
            object.__setattr__(request.principal, "subject", "another-operator")
            object.__setattr__(request, "action", "takeover")
            return BrowserControlPolicyResult(allowed=True)

    async def scenario():
        result = await authorize(MutatingPolicy(True))
        assert result.record.identity.session_id == "session"
        assert result.operator.subject == "authenticated-user"
        assert result.action == "view"

    asyncio.run(scenario())


def test_policy_identity_change_during_decision_fails_closed() -> None:
    class ChangingPolicy(Policy):
        async def decide(self, request):
            self.identity = "browser-permissions:v2"
            return BrowserControlPolicyResult(allowed=True)

    async def scenario():
        with pytest.raises(BrowserControlPermissionDenied):
            await authorize(ChangingPolicy(True))

    asyncio.run(scenario())


def test_caller_cancellation_does_not_become_permission() -> None:
    async def scenario():
        entered = asyncio.Event()

        class WaitingPolicy(Policy):
            async def decide(self, request):
                entered.set()
                await asyncio.Event().wait()
                return BrowserControlPolicyResult(allowed=True)

        task = asyncio.create_task(authorize(WaitingPolicy(True)))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()

    asyncio.run(scenario())


def test_truthy_policy_result_is_not_permission() -> None:
    class WrongPolicy(Policy):
        async def decide(self, request):
            return {"allowed": True}

    async def scenario():
        with pytest.raises(BrowserControlPermissionDenied):
            await authorize(WrongPolicy(True))

    asyncio.run(scenario())


def test_child_cancellation_is_permission_failure_not_owner_cancellation() -> None:
    class CancelledPolicy(Policy):
        async def decide(self, request):
            raise asyncio.CancelledError()

    async def scenario():
        task = asyncio.create_task(authorize(CancelledPolicy(True)))
        with pytest.raises(BrowserControlPermissionDenied):
            await task
        assert task.cancelling() == 0
        assert not task.cancelled()

    asyncio.run(scenario())


def test_swallowed_caller_cancellation_does_not_authorize() -> None:
    async def scenario():
        entered = asyncio.Event()

        class SwallowingPolicy(Policy):
            async def decide(self, request):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return BrowserControlPolicyResult(allowed=True)

        task = asyncio.create_task(authorize(SwallowingPolicy(True)))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()

    asyncio.run(scenario())
