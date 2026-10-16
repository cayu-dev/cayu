"""Durable reusable authorization grants through the public approval path."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from tests.runtime.authorization_grant_support import (
    INTRUDER,
    OWNER,
    GatedPolicy,
    Harness,
    _command_turn,
)

from cayu.agents import AgentSpec
from cayu.approvals import (
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    AllowedActorsGrantAuthority,
    InMemoryToolAuthorizationGrantStore,
    ToolApprovalDecision,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantDenied,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantService,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantUnavailable,
    ToolAuthorizationGrantUseDecision,
    tool_authorization_grant_sample_bucket,
)
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.tools.policy import (
    AlwaysRequireApprovalToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
)


def _types(events) -> list[str]:
    return [str(event.type) for event in events]


def _session_grant(**kwargs) -> ToolAuthorizationGrantRequest:
    return ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.SESSION, **kwargs)


def test_remember_and_approve_reuses_the_exact_command_in_the_same_session(tmp_path) -> None:
    turns = _command_turn("call_1", ["git", "status"]) + _command_turn("call_2", ["git", "status"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        first = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(first)
        pending = await harness.pending("s1")
        offer = pending.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]
        assert offer["grantable"] is True and offer["approval_gate"] == "commands"

        resolved = await harness.remember("s1", _session_grant(justification="routine"))
        assert str(EventType.TOOL_CALL_APPROVED) in _types(resolved)
        assert harness.tool.calls == [{"argv": ["git", "status"]}]
        (grant,) = await harness.service.list_grants()
        assert grant.status is ToolAuthorizationGrantStatus.ACTIVE
        assert grant.scope.kind is ToolAuthorizationGrantScopeKind.SESSION
        assert grant.scope.id == "s1"
        assert grant.created_by is not None and grant.created_by.subject == "owner"
        assert grant.resolved_by is not None and grant.resolved_by.subject == "owner"
        assert grant.origin.approval_id == pending.approval_id
        assert grant.approval_gate == "commands"

        second = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(second)
        assert harness.tool.calls == [{"argv": ["git", "status"]}] * 2
        (use,) = await harness.service.list_uses(grant.grant_id)
        assert use.decision is ToolAuthorizationGrantUseDecision.ALLOWED
        assert use.tool_call_id == "call_2" and use.scope == grant.scope

    asyncio.run(scenario())


def test_approve_once_does_not_create_reusable_authority(tmp_path) -> None:
    turns = _command_turn("call_1", ["git", "status"]) + _command_turn("call_2", ["git", "status"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1")
        _ = [event async for event in harness.app.resolve_tool_approval(approval)]
        assert await harness.service.list_grants() == ()
        second = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(second)
        assert len(harness.tool.calls) == 1

    asyncio.run(scenario())


def test_near_match_and_shell_text_require_approval(tmp_path) -> None:
    turns = (
        _command_turn("call_1", ["git", "status"])
        + _command_turn("call_2", ["git", "status", "--porcelain"])
        + [
            [
                ModelStreamEvent.tool_call(
                    id="call_3", name="exec_command", arguments={"shell": "git status"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant())
        near = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(near)
        deny = await harness.approval("s1", decision=ToolApprovalDecision.DENY)
        _ = [event async for event in harness.app.resolve_tool_approval(deny)]
        shell = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(shell)

    asyncio.run(scenario())


def test_workspace_grant_applies_to_a_later_session_and_respects_boundaries(tmp_path) -> None:
    turns = (
        _command_turn("call_1", ["make", "test"])
        + _command_turn("call_2", ["make", "test"])
        + _command_turn("call_3", ["make", "test"])
    )
    harness = Harness(tmp_path, turns=turns, boundary_labels=("tenant",))

    async def scenario() -> None:
        await harness.run("s1", labels={"tenant": "acme"})
        await harness.remember(
            "s1",
            ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.WORKSPACE),
        )
        (grant,) = await harness.service.list_grants()
        assert grant.scope.id == "ws-1" and grant.boundary.labels == {"tenant": "acme"}

        later = await harness.run("s2", labels={"tenant": "acme"})
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(later)
        assert len(harness.tool.calls) == 2

        other_tenant = await harness.run("s3", labels={"tenant": "globex"})
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(other_tenant)
        assert len(harness.tool.calls) == 2
        applicable = await harness.service.list_applicable(session_id="s3", workspace_id="ws-1")
        assert [item.grant_id for item in applicable] == [grant.grant_id]

    asyncio.run(scenario())


def test_session_grant_does_not_cross_sessions(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant())
        other = await harness.run("s2")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(other)

    asyncio.run(scenario())


def test_expiry_and_revocation_stop_authorizing(tmp_path) -> None:
    turns = (
        _command_turn("call_1", ["ls"])
        + _command_turn("call_2", ["ls"])
        + _command_turn("call_3", ["ls"])
        + _command_turn("call_4", ["ls"])
    )
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant(expires_in_seconds=60))
        (grant,) = await harness.service.list_grants()
        assert grant.expires_at == harness.clock.now + timedelta(seconds=60)
        allowed = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(allowed)

        harness.clock.now += timedelta(seconds=60)
        assert grant.effective_status(harness.clock.now) is ToolAuthorizationGrantStatus.EXPIRED
        expired = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(expired)
        assert await harness.service.list_applicable(session_id="s1") == ()

    asyncio.run(scenario())


def test_revocation_requires_authority_and_is_idempotent(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant())
        (grant,) = await harness.service.list_grants()
        with pytest.raises(ToolAuthorizationGrantDenied):
            await harness.service.revoke(grant.grant_id, actor=INTRUDER)
        revoked = await harness.service.revoke(grant.grant_id, actor=OWNER, reason="drift")
        again = await harness.service.revoke(grant.grant_id, actor=OWNER, reason="later")
        assert revoked == again
        assert revoked.status is ToolAuthorizationGrantStatus.REVOKED
        assert revoked.revocation_reason == "drift"
        later = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(later)

    asyncio.run(scenario())


def test_unauthorized_creation_leaves_approval_and_grants_unchanged(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))

    async def scenario() -> None:
        await harness.run("s1")
        before = await harness.sessions.load_checkpoint("s1")
        approval = await harness.approval("s1", actor=INTRUDER)
        with pytest.raises(ToolAuthorizationGrantDenied):
            _ = [e async for e in harness.service.approve_and_grant(approval, _session_grant())]
        assert await harness.service.list_grants() == ()
        assert await harness.sessions.load_checkpoint("s1") == before
        assert harness.tool.calls == []

    asyncio.run(scenario())


def test_mandatory_controls_are_never_grantable(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"])
    harness = Harness(tmp_path, turns=turns, then=GatedPolicy(mandatory=True))
    named = Harness(
        tmp_path,
        turns=_command_turn("call_1", ["ls"]),
        mandatory_controls=(
            ToolAuthorizationGrantControl(tool_name="exec_command", approval_gate="commands"),
        ),
    )
    whole_tool = Harness(
        tmp_path,
        turns=_command_turn("call_1", ["ls"]),
        mandatory_controls=(ToolAuthorizationGrantControl(tool_name="exec_command"),),
    )

    async def scenario() -> None:
        for item in (harness, named, whole_tool):
            await item.run("s1")
            pending = await item.pending("s1")
            offer = pending.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]
            assert offer == {
                "schema_version": 1,
                "policy_id": "commands",
                "grantable": False,
                "reason": "mandatory_control",
            }
            approval = await item.approval("s1")
            with pytest.raises(ToolAuthorizationGrantUnavailable):
                await item.service.prepare(approval, _session_grant())
            assert await item.service.list_grants() == ()

    asyncio.run(scenario())


def test_denied_approval_abandons_a_prepared_grant(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        prepared = await harness.service.prepare(await harness.approval("s1"), _session_grant())
        assert prepared.grant.status is ToolAuthorizationGrantStatus.STAGED
        assert (
            prepared.approval_request.metadata[TOOL_AUTHORIZATION_GRANT_METADATA_KEY]["grant_id"]
            == prepared.grant.grant_id
        )
        # A different (approve-once) request resolves the approval instead.
        approve_once = await harness.approval("s1")
        _ = [event async for event in harness.app.resolve_tool_approval(approve_once)]
        settled = await harness.service.reconcile(prepared.grant.grant_id)
        assert settled.status is ToolAuthorizationGrantStatus.ABANDONED
        assert settled.abandonment_reason == "approved_without_grant"
        later = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(later)

    asyncio.run(scenario())


def test_duplicate_submission_is_idempotent_and_conflicts_change_nothing(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["ls"]))

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1")
        first = await harness.service.prepare(approval, _session_grant())
        second = await harness.service.prepare(approval, _session_grant())
        assert first.grant == second.grant
        assert first.approval_request == second.approval_request
        with pytest.raises(ToolAuthorizationGrantConflict):
            await harness.service.prepare(approval, _session_grant(sampled_review_rate_bps=10))
        assert (await harness.service.get(first.grant.grant_id)) == first.grant

        events = [e async for e in harness.service.approve_and_grant(approval, _session_grant())]
        assert str(EventType.TOOL_CALL_APPROVED) in _types(events)
        # The retry replays the runtime's durable closure; the call does not run again.
        replay = [e async for e in harness.service.approve_and_grant(approval, _session_grant())]
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(replay)
        assert harness.tool.calls == [{"argv": ["ls"]}]
        (grant,) = await harness.service.list_grants()
        assert grant.status is ToolAuthorizationGrantStatus.ACTIVE

    asyncio.run(scenario())


def test_crash_between_grant_write_and_approval_is_reconstructable(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1")
        prepared = await harness.service.prepare(approval, _session_grant())
        # The caller stops here: the grant is staged and inert, the approval pending.
        assert prepared.grant.status is ToolAuthorizationGrantStatus.STAGED
        assert await harness.pending("s1") is not None
        staged_policy = await harness.policy.applicable_grants(
            ToolPolicyRequest(
                session=await harness.sessions.load("s1"),
                agent=AgentSpec(name="assistant", model="fake-model"),
                tool_name="exec_command",
                tool_call_id="probe",
                arguments={"argv": ["ls"]},
                workspace_id="ws-1",
                environment_name="local",
            ),
            approval_gate="commands",
        )
        assert staged_policy == ()

        # A new service over the same durable stores resumes the same grant.
        restarted = ToolAuthorizationGrantService(
            harness.app,
            store=harness.grants,
            authority=AllowedActorsGrantAuthority(creators=["owner"]),
            clock=harness.clock,
        )
        events = [e async for e in restarted.approve_and_grant(approval, _session_grant())]
        assert str(EventType.TOOL_CALL_APPROVED) in _types(events)
        assert harness.tool.calls == [{"argv": ["ls"]}]
        (grant,) = await restarted.list_grants()
        assert grant.grant_id == prepared.grant.grant_id
        assert grant.status is ToolAuthorizationGrantStatus.ACTIVE
        assert grant.resolution_request_digests == prepared.grant.resolution_request_digests

    asyncio.run(scenario())


def test_crash_after_approval_activates_on_reconcile(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        prepared = await harness.service.prepare(await harness.approval("s1"), _session_grant())
        # The approval is resolved with the prepared request, then the caller
        # stops before activating the grant.
        _ = [e async for e in harness.app.resolve_tool_approval(prepared.approval_request)]
        assert (await harness.service.get(prepared.grant.grant_id)).status is (
            ToolAuthorizationGrantStatus.STAGED
        )
        (settled,) = await harness.service.reconcile_staged()
        assert settled.status is ToolAuthorizationGrantStatus.ACTIVE
        later = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(later)

    asyncio.run(scenario())


def test_policy_mandatory_controls_refuse_creation_and_default_sampled_rate(tmp_path) -> None:
    refused = Harness(
        tmp_path,
        turns=_command_turn("call_1", ["ls"]),
        mandatory_controls=(
            ToolAuthorizationGrantControl(tool_name="exec_command", approval_gate="commands"),
        ),
    )
    defaulted = Harness(
        tmp_path, turns=_command_turn("call_1", ["ls"]), default_sampled_review_rate_bps=500
    )

    async def scenario() -> None:
        await refused.run("s1")
        with pytest.raises(ToolAuthorizationGrantUnavailable):
            await refused.service.prepare(await refused.approval("s1"), _session_grant())
        assert await refused.service.list_grants() == ()

        await defaulted.run("s1")
        approval = await defaulted.approval("s1")
        prepared = await defaulted.service.prepare(approval, _session_grant())
        assert prepared.grant.sampled_review_rate_bps == 500
        explicit = await defaulted.service.prepare(
            approval, _session_grant(sampled_review_rate_bps=500)
        )
        assert explicit.grant == prepared.grant

    asyncio.run(scenario())


def test_sampled_review_is_deterministic_and_recorded(tmp_path) -> None:
    turns = _command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant(sampled_review_rate_bps=10_000))
        (grant,) = await harness.service.list_grants()
        sampled = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(sampled)
        pending = await harness.pending("s1")
        evidence = pending.metadata[TOOL_AUTHORIZATION_GRANT_METADATA_KEY]
        (use,) = await harness.service.list_uses(grant.grant_id)
        assert use.tool_round_id is not None
        bucket = tool_authorization_grant_sample_bucket(
            grant_id=grant.grant_id,
            session_id="s1",
            tool_call_id="call_2",
            tool_round_id=use.tool_round_id,
        )
        assert evidence["decision"] == "sampled_review"
        assert evidence["sample_bucket"] == bucket
        assert pending.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]["grantable"] is False
        assert use.decision is ToolAuthorizationGrantUseDecision.SAMPLED_REVIEW
        assert use.sample_bucket == bucket

    asyncio.run(scenario())


def test_grant_never_overrides_allow_or_deny(tmp_path) -> None:
    harness = Harness(
        tmp_path,
        turns=_command_turn("call_1", ["ls"]),
        then=AlwaysRequireApprovalToolPolicy(tools=["other"]),
    )

    async def scenario() -> None:
        events = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(events)
        assert harness.tool.calls == [{"argv": ["ls"]}]

    asyncio.run(scenario())


class _BrokenStore(InMemoryToolAuthorizationGrantStore):
    async def list(self, query):
        raise RuntimeError("store unavailable")


def _policy_request(session) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name="exec_command",
        tool_call_id="call_1",
        arguments={"argv": ["ls"]},
        workspace_id="ws-1",
    )


def test_policy_offers_fail_closed_and_carry_the_configured_gate(tmp_path) -> None:
    gated = AlwaysRequireApprovalToolPolicy(tools=["exec_command"], approval_gate="commands")
    harness = Harness(
        tmp_path, turns=_command_turn("call_1", ["ls"]) + _command_turn("call_2", ["ls"])
    )

    async def scenario() -> None:
        await harness.run("labeled", labels={"tenant": "acme"})
        await harness.run("unlabeled")
        labeled = await harness.sessions.load("labeled")
        unlabeled = await harness.sessions.load("unlabeled")
        policy = ToolAuthorizationGrantPolicy(
            policy_id="p",
            store=InMemoryToolAuthorizationGrantStore(),
            grantable={"exec_command": "process_command"},
            then=gated,
            boundary_labels=("tenant",),
        )
        offered = await policy.authorize(_policy_request(labeled))
        offer = offered.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]
        assert offer["grantable"] is True and offer["approval_gate"] == "commands"
        assert offer["boundary"]["labels"] == {"tenant": "acme"}

        missing_label = await policy.authorize(_policy_request(unlabeled))
        assert missing_label.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]["reason"] == (
            "boundary_unavailable"
        )

        broken = ToolAuthorizationGrantPolicy(
            policy_id="p",
            store=_BrokenStore(),
            grantable={"exec_command": "process_command"},
            then=gated,
        )
        result = await broken.authorize(_policy_request(labeled))
        assert result.decision is ToolPolicyDecision.REQUIRE_APPROVAL
        assert result.metadata[TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY]["reason"] == (
            "grant_store_unavailable"
        )

    asyncio.run(scenario())
