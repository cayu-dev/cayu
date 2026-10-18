"""Changed arguments with grants and manual recovery on the public approval entrances."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.authorization_grant_support import Harness, _command_turn
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore
from tests.runtime.runtime_session_support import (
    collect_tool_approval_events,
    collect_tool_approval_recovery_events,
)
from tests.runtime.runtime_supervision_support import (
    pause_for_approval,
    resolve_approval,
    supervision_page,
)
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import (
    TOOL_AUTHORIZATION_GRANT_METADATA_KEY,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
)
from cayu.approvals.tools import (
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
)
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._tool_execution import APPROVED_ARGUMENTS_NOTICE
from cayu.sessions import SupervisionEventKind, SupervisionGrantRole
from cayu.sessions import _pending_approval_reader as pending_approval_reader


def _session_grant() -> ToolAuthorizationGrantRequest:
    return ToolAuthorizationGrantRequest(scope=ToolAuthorizationGrantScopeKind.SESSION)


async def _still_pending(harness: Harness, session_id: str) -> None:
    checkpoint = await harness.sessions.load_checkpoint(session_id)
    assert pending_approval_reader.pending_approval_from_checkpoint(checkpoint) is not None
    assert pending_approval_reader.approval_resolution_intent_from_checkpoint(checkpoint) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {TOOL_AUTHORIZATION_GRANT_METADATA_KEY: {"grant_id": "tag_forged"}},
        {TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY: {"grantable": True}},
    ],
    ids=["grant", "offer"],
)
def test_public_resolution_refuses_caller_supplied_grant_metadata(tmp_path, metadata) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["git", "status"]))

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1", metadata=metadata)
        with pytest.raises(ValueError, match="reserved for the grant service"):
            _ = [event async for event in harness.app.resolve_tool_approval(approval)]
        assert harness.tool.calls == []
        await _still_pending(harness, "s1")
        assert await harness.service.list_grants() == ()

    asyncio.run(scenario())


def test_service_created_grant_still_resolves_and_reports_its_role(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["git", "status"]))

    async def scenario() -> None:
        await harness.run("s1")
        prepared = await harness.service.prepare(await harness.approval("s1"), _session_grant())
        # The service's own marked request passes the public entrance, also
        # after an ordinary copy that keeps the issued marker.
        request = prepared.approval_request.model_copy(update={"reason": "routine"})
        events = [event async for event in harness.app.resolve_tool_approval(request)]
        assert str(EventType.TOOL_CALL_APPROVED) in [str(event.type) for event in events]
        assert harness.tool.calls == [{"argv": ["git", "status"]}]
        await harness.service.reconcile(prepared.grant.grant_id)
        page = await supervision_page(harness.sessions)
        (touch,) = page.events
        assert touch.kind is SupervisionEventKind.APPROVE
        assert touch.authorization_grant_role is SupervisionGrantRole.CREATED
        assert touch.authorization_grant_id == prepared.grant.grant_id

    asyncio.run(scenario())


def test_approve_and_grant_refuses_changed_arguments(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["git", "status"]))

    async def scenario() -> None:
        await harness.run("s1")
        approval = await harness.approval("s1", arguments={"argv": ["git", "log"]})
        with pytest.raises(ToolApprovalArgumentsRefused, match="grant cannot be created"):
            _ = [e async for e in harness.service.approve_and_grant(approval, _session_grant())]
        assert harness.tool.calls == []
        await _still_pending(harness, "s1")
        assert await harness.service.list_grants() == ()

    asyncio.run(scenario())


def test_reauthorizing_an_edit_never_uses_or_samples_a_grant(tmp_path) -> None:
    turns = _command_turn("call_1", ["git", "status"]) + _command_turn("call_2", ["git", "log"])
    harness = Harness(tmp_path, turns=turns)

    async def scenario() -> None:
        await harness.run("s1")
        await harness.remember("s1", _session_grant())
        (grant,) = await harness.service.list_grants()
        # The second call is not covered by the grant, so it pauses.
        second = await harness.run("s1")
        assert str(EventType.TOOL_CALL_APPROVAL_REQUESTED) in [str(e.type) for e in second]
        # Changing it to the granted command is accepted on the person's
        # approval, and the grant records no use for the re-check.
        approval = await harness.approval("s1", arguments={"argv": ["git", "status"]})
        _ = [event async for event in harness.app.resolve_tool_approval(approval)]
        assert harness.tool.calls == [{"argv": ["git", "status"]}] * 2
        assert await harness.service.list_uses(grant.grant_id) == ()
        page = await supervision_page(harness.sessions, kinds=["edit"])
        (edit,) = page.events
        assert edit.arguments_revision is not None
        assert edit.arguments_revision.changed_fields == ("argv",)
        # Digests are keyed by the store's alias keyring, never a plain hash.
        assert edit.arguments_revision.digest_key_id == "memory"
        assert edit.arguments_revision.proposed_arguments_digest != (
            edit.arguments_revision.approved_arguments_digest
        )

    asyncio.run(scenario())


def test_manual_recovery_closes_an_edit_approved_call_without_rerunning_it() -> None:
    session_id = "sess_edit_recovery"
    store = FailingTerminalToolEventStore()
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    tool = PureRecordingTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=RequireApprovalPolicy(),
    )

    async def run():
        pending = await pause_for_approval(app, store, session_id)
        edited = resolve_approval(
            session_id, pending, ToolApprovalDecision.APPROVE, arguments={"note": "edited"}
        )
        await collect_tool_approval_events(app, edited)
        assert tool.calls == [{"note": "edited"}]
        events = await collect_tool_approval_recovery_events(
            app,
            ToolApprovalRecoveryRequest(
                session_id=session_id,
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="done outside",
            ),
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"note": "edited"}]
        results = [
            part
            for message in await store.load_transcript(session_id)
            for part in message.content
            if type(part) is ToolResultPart
        ]
        assert [part.content for part in results] == [
            f"{APPROVED_ARGUMENTS_NOTICE}\n\ndone outside"
        ]
        return await supervision_page(store)

    page = asyncio.run(run())
    assert {event.kind for event in page.events} == {
        SupervisionEventKind.EDIT,
        SupervisionEventKind.RECOVER,
    }


def test_a_prepared_grant_request_cannot_be_edited_afterwards(tmp_path) -> None:
    harness = Harness(tmp_path, turns=_command_turn("call_1", ["git", "status"]))

    async def scenario() -> None:
        await harness.run("s1")
        prepared = await harness.service.prepare(await harness.approval("s1"), _session_grant())
        edited = prepared.approval_request.model_copy(
            update={"arguments": {"argv": ["git", "log"]}}
        )
        with pytest.raises(ToolApprovalArgumentsRefused, match="grant"):
            _ = [event async for event in harness.app.resolve_tool_approval(edited)]
        assert harness.tool.calls == []
        await _still_pending(harness, "s1")
        (grant,) = await harness.service.list_grants()
        assert (await harness.service.reconcile(grant.grant_id)).status.value == "staged"

    asyncio.run(scenario())
