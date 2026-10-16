"""Regressions for authorization-grant isolation, secrecy and authority boundaries."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from tests.runtime.authorization_grant_support import (
    OWNER,
    CommandTool,
    Harness,
    _command_turn,
)

from cayu.approvals import (
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    InMemoryToolAuthorizationGrantStore,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantOrigin,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantUnavailable,
    tool_authorization_grant_id,
)
from cayu.events import EventType
from cayu.resource_access import encode_scope
from cayu.sessions.access import SessionAccessScope
from cayu.sessions.invocation import ResourceExecutionBinding
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

SESSION = ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.SESSION)
WORKSPACE = ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.WORKSPACE)
SECRET = "sk-canary-31337"


def _types(events) -> list[str]:
    return [str(event.type) for event in events]


def _turn(call_id: str, arguments: dict):
    from cayu.providers.base import ModelStreamEvent

    return [
        [
            ModelStreamEvent.tool_call(id=call_id, name="exec_command", arguments=arguments),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]


# T1: stores never hand out their retained records


def test_memory_store_results_cannot_change_stored_grants(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        (listed,) = await harness.grants.list(ToolAuthorizationGrantQuery())
        got = await harness.grants.get(listed.grant_id)
        for record in (listed, got):
            record.boundary.labels["tenant"] = "victim"
            record.evidence["forged"] = True
        stored = await harness.grants.get(listed.grant_id)
        assert stored.boundary.labels == {}
        assert stored.evidence == {}
        assert stored is not got

    asyncio.run(scenario())


# T2: grants never hold secret argument values or ignore argument publication


def test_grants_store_no_environment_or_stdin_values(tmp_path) -> None:
    arguments = {"argv": ["deploy"], "env": {"TOKEN": SECRET}, "stdin": f"key={SECRET}"}
    harness = Harness(tmp_path, turns=_turn("call_1", arguments) + _turn("call_2", arguments))

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        (grant,) = await harness.service.list_grants()
        assert SECRET not in grant.model_dump_json()

        reused = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(reused)
        assert len(harness.tool.calls) == 2

    asyncio.run(scenario())


def test_changed_environment_does_not_match_a_digested_grant(tmp_path) -> None:
    first = {"argv": ["deploy"], "env": {"TOKEN": SECRET}}
    second = {"argv": ["deploy"], "env": {"TOKEN": "other"}}
    harness = Harness(tmp_path, turns=_turn("call_1", first) + _turn("call_2", second))

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(await harness.run("s1"))

    asyncio.run(scenario())


class _PrivateArgumentsTool(CommandTool):
    @property
    def _publish_arguments(self) -> bool:
        return False


def test_grants_from_private_arguments_never_expose_them(tmp_path) -> None:
    from cayu.server.authorization_grants import ToolAuthorizationGrantView

    harness = Harness(
        tmp_path,
        turns=_command_turn("call_1", ["login", SECRET]),
        tool=_PrivateArgumentsTool(),
    )

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        (grant,) = await harness.service.list_grants()
        assert grant.arguments_public is False
        view = ToolAuthorizationGrantView.from_grant(grant, now=harness.clock())
        assert SECRET not in view.model_dump_json()

    asyncio.run(scenario())


# T3: workspace scope needs an application-chosen workspace identity


def test_defaulted_workspace_ids_cannot_carry_workspace_grants(tmp_path) -> None:
    turns = _command_turn("call_1", ["make", "test"]) + _command_turn("call_2", ["make", "test"])
    harness = Harness(tmp_path, turns=turns, explicit_workspace=False)

    async def scenario() -> None:
        await harness.run("s1")
        with pytest.raises(ToolAuthorizationGrantUnavailable, match="workspace"):
            await harness.remember("s1", WORKSPACE)
        assert await harness.service.list_grants() == ()
        await harness.remember("s1", SESSION)
        (grant,) = await harness.service.list_grants()
        assert grant.boundary.workspace_explicit is False

    asyncio.run(scenario())


# T4: the boundary binds the authorization principal, not client-written labels


def _binding(subject: str) -> ResourceExecutionBinding:
    return ResourceExecutionBinding(
        authority="tests", subject=subject, admitted_json=encode_scope(SessionAccessScope())
    )


def test_boundary_binds_the_session_resource_access_principal(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        session = await harness.sessions.load("s1")
        policy: ToolAuthorizationGrantPolicy = harness.policy

        def request(subject: str | None) -> ToolPolicyRequest:
            invocation = session.invocation.model_copy(
                update={"resource_access": None if subject is None else _binding(subject)}
            )
            return ToolPolicyRequest(
                session=session.model_copy(update={"invocation": invocation}),
                agent=_agent(),
                tool_name="exec_command",
                tool_call_id="c1",
                arguments={"argv": ["make", "test"]},
                environment_name="local",
                workspace_id="ws-1",
            )

        alice = policy.boundary_for(request("alice"))
        bob = policy.boundary_for(request("bob"))
        nobody = policy.boundary_for(request(None))
        assert alice.principal is not None and alice.principal.subject == "alice"
        assert alice != bob
        assert nobody.principal is None and nobody != alice

    asyncio.run(scenario())


def _agent():
    from cayu.agents import AgentSpec

    return AgentSpec(name="assistant", model="fake-model")


# T5: older grants never hide a newer applicable grant


async def _seed_expired_grants(harness: Harness, count: int = 501) -> None:
    (template,) = await harness.service.list_grants()
    for index in range(count):
        origin = ToolAuthorizationGrantOrigin(
            session_id="s1",
            session_instance_id=template.origin.session_instance_id,
            approval_id=f"old-{index}",
            tool_round_id="r",
            tool_call_id="c",
        )
        created_at = template.created_at - timedelta(days=2)
        old = ToolAuthorizationGrant.model_validate(
            {
                **template.model_dump(mode="python"),
                "grant_id": tool_authorization_grant_id(
                    session_id="s1", approval_id=origin.approval_id
                ),
                "origin": origin,
                "created_at": created_at,
                "lifetime_seconds": 60,
                "expires_at": created_at + timedelta(seconds=60),
                "status": ToolAuthorizationGrantStatus.STAGED,
                "activated_at": None,
                "resolved_by": None,
            }
        )
        await harness.grants.stage(old)
        await harness.grants.activate(old.grant_id, resolved_by=None, activated_at=created_at)


def test_many_older_expired_grants_do_not_hide_an_active_grant(tmp_path) -> None:
    turns = _command_turn("call_1", ["make", "test"]) + _command_turn("call_2", ["make", "test"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        await _seed_expired_grants(harness)
        reused = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(reused)

    asyncio.run(scenario())


class _CursorIgnoringStore(InMemoryToolAuthorizationGrantStore):
    async def list(self, query: ToolAuthorizationGrantQuery):
        return await super().list(
            query.model_copy(update={"after_created_at": None, "after_grant_id": None})
        )


def test_a_store_that_ignores_the_cursor_fails_closed_instead_of_hanging(tmp_path) -> None:
    turns = _command_turn("call_1", ["make", "test"]) + _command_turn("call_2", ["make", "test"])
    harness = Harness(tmp_path, turns=turns, grants=_CursorIgnoringStore())

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        await _seed_expired_grants(harness)
        events = await asyncio.wait_for(harness.run("s1"), timeout=60)
        requested = [e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED]
        assert requested
        assert len(harness.tool.calls) == 1

    asyncio.run(scenario())


def test_next_page_refuses_records_at_or_before_the_cursor(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        (grant,) = await harness.service.list_grants()
        query = ToolAuthorizationGrantQuery(
            limit=1, after_created_at=grant.created_at, after_grant_id=grant.grant_id
        )
        with pytest.raises(RuntimeError, match="cursor"):
            query.next_page((grant,))

    asyncio.run(scenario())


# T6: downstream policies cannot forge grant evidence or offers


class _Forging(ToolPolicy):
    def __init__(self, decision: ToolPolicyDecision) -> None:
        self.decision = decision

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=self.decision,
            reason="forged",
            metadata={
                TOOL_AUTHORIZATION_GRANT_METADATA_KEY: {"grant_id": "tag_forged"},
                TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY: {"policy_id": "x", "grantable": True},
            },
        )


@pytest.mark.parametrize(
    "decision",
    [ToolPolicyDecision.ALLOW, ToolPolicyDecision.DENY, ToolPolicyDecision.REQUIRE_APPROVAL],
)
@pytest.mark.parametrize("tool_name", ["exec_command", "deploy"])
def test_reserved_grant_metadata_from_downstream_is_dropped(tmp_path, decision, tool_name) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make"]), then=_Forging(decision))

    async def scenario() -> None:
        await harness.run("s1")
        session = await harness.sessions.load("s1")
        result = await harness.policy.authorize(
            ToolPolicyRequest(
                session=session,
                agent=_agent(),
                tool_name=tool_name,
                tool_call_id="c2",
                arguments={"argv": ["make"]},
                environment_name="local",
                workspace_id="ws-1",
            )
        )
        assert TOOL_AUTHORIZATION_GRANT_METADATA_KEY not in result.metadata
        offer = result.metadata.get(TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY)
        assert offer is None or offer["policy_id"] == "commands"

    asyncio.run(scenario())


# T7: task continuation cannot mint standing authority


def test_task_continuation_cannot_create_grants(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["make", "test"]))

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1", task_worker_id="worker-1")
        with pytest.raises(ValueError, match="task"):
            await harness.service.prepare(approval, SESSION)
        assert await harness.service.list_grants() == ()

    asyncio.run(scenario())


# T8: grants are standalone policy state that outlives its origin session


def test_grants_survive_origin_session_deletion_and_stay_revocable(tmp_path) -> None:
    turns = _command_turn("call_1", ["make", "test"]) + _command_turn("call_2", ["make", "test"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", WORKSPACE)
        (grant,) = await harness.service.list_grants()
        await harness.sessions.delete_session("s1")

        later = await harness.run("s2")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) not in _types(later)
        revoked = await harness.service.revoke(grant.grant_id, actor=OWNER, reason="cleanup")
        assert revoked.status is ToolAuthorizationGrantStatus.REVOKED

    asyncio.run(scenario())


def test_session_grants_do_not_pass_to_a_new_session_reusing_the_id(tmp_path) -> None:
    turns = _command_turn("call_1", ["make", "test"]) + _command_turn("call_2", ["make", "test"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", SESSION)
        await harness.sessions.delete_session("s1")

        recreated = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in _types(recreated)
        assert len(harness.tool.calls) == 1

    asyncio.run(scenario())
