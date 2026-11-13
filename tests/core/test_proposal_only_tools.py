"""A proposal-only tool runs only as application code proposed it; no model can call it."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, Message
from cayu.approvals import (
    ToolApprovalArgumentsRefused,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallProposalOutcome,
    ToolCallProposalStatus,
    tool_call_proposal_request,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.runtime._tool_call_proposals import (
    TOOL_CALL_PROPOSAL_MODEL,
    TOOL_CALL_PROPOSAL_PROVIDER_NAME,
    proposal_message,
)
from cayu.sessions import PendingActionQuery
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions._tool_call_proposal_record import TOOL_CALL_PROPOSAL_METADATA_KEY
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest, runtime_tool_call_proposal
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import AlwaysRequireApprovalToolPolicy
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.exposure import (
    AllRegisteredToolsExposurePolicy,
    StaticToolExposurePolicy,
    ToolCapabilityCeiling,
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
)
from cayu.tools.policy import ToolPolicy, ToolPolicyRequest, ToolPolicyResult

MERGED: list[dict] = []
LOOKED_UP: list[dict] = []
_IDENTITY = {"behavior_version": "1", "implementation_version": "1"}


class MergePullRequest(Tool):
    spec = ToolSpec(
        name="merge_pull_request",
        description="Merge one pull request.",
        input_schema={
            "type": "object",
            "properties": {"repository": {"type": "string"}, "number": {"type": "integer"}},
            "required": ["repository", "number"],
        },
        effect=ToolEffect.EXTERNAL,
        max_terminal_payload_bytes=64 * 1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:proposal-only:merge", **_IDENTITY
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        MERGED.append(dict(args))
        return ToolResult(content=f"Merged #{args['number']}.")


class LookUp(Tool):
    spec = ToolSpec(
        name="look_up",
        description="Look something up.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        effect=ToolEffect.NONE,
        max_terminal_payload_bytes=64 * 1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:proposal-only:look-up", **_IDENTITY
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        LOOKED_UP.append(dict(args))
        return ToolResult(content="Found it.")


class SpyPolicy(ToolPolicy):
    """Require approval for the merge, allow the rest, and record every call it sees."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.seen.append(request.tool_name)
        if request.tool_name == "merge_pull_request":
            return ToolPolicyResult(decision="require_approval", reason="A person merges.")
        return ToolPolicyResult(decision="allow")


class EverythingItIsGiven(ToolExposurePolicy):
    """A custom policy that selects every tool it is offered."""

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        assert "merge_pull_request" not in {tool.name for tool in request.registered_tools}
        assert "merge_pull_request" not in request.capability_ceiling
        return ToolExposureDecision(
            profile_id="everything", tool_names=[tool.name for tool in request.eligible_tools]
        )


ARGS = {"repository": "cayu-tech/cayu", "number": 12}


def _build(
    *,
    session_store=None,
    provider: ScriptedModelProvider | None = None,
    policy: ToolPolicy | None = None,
    **registration,
) -> CayuApp:
    app = CayuApp(session_store=session_store or InMemorySessionStore(), enable_logging=False)
    if provider is not None:
        app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="shepherd", model="scripted-model"),
        tools=[LookUp()],
        proposal_tools=[MergePullRequest()],
        tool_policy=policy or AlwaysRequireApprovalToolPolicy(tools=["merge_pull_request"]),
        **registration,
    )
    return app


async def _drain(stream) -> list:
    async with aclosing(stream):
        return [event async for event in stream]


def _approve(outcome: ToolCallProposalOutcome, **fields) -> ToolApprovalRequest:
    assert outcome.approval_id and outcome.tool_round_id and outcome.tool_call_id
    return ToolApprovalRequest(
        session_id=outcome.session_id,
        approval_id=outcome.approval_id,
        tool_round_id=outcome.tool_round_id,
        tool_call_id=outcome.tool_call_id,
        decision=ToolApprovalDecision.APPROVE,
        **fields,
    )


def _merge_call(call_id: str = "model-call") -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(name="merge_pull_request", arguments=ARGS, id=call_id),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("Done."), ModelStreamEvent.completed({})]


def _offered(provider: ScriptedModelProvider) -> set[str]:
    return {tool["name"] for request in provider.requests for tool in request.tools}


@pytest.fixture(autouse=True)
def _reset():
    MERGED.clear()
    LOOKED_UP.clear()


def test_proposal_runs_after_approval_including_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite"

    async def scenario():
        first = _build(session_store=SQLiteSessionStore(path))
        pending = await first.propose_tool_call(
            "shepherd", "merge_pull_request", ARGS, session_id="merge-12"
        )
        session = await first.session_store.load("merge-12")
        await first.aclose()

        second = _build(session_store=SQLiteSessionStore(path))
        [action] = (
            await second.session_store.query_pending_actions(
                PendingActionQuery(session_id="merge-12")
            )
        ).actions
        events = await _drain(
            second.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="merge-12",
                    approval_id=action.approval_id,
                    tool_round_id=action.round_id,
                    tool_call_id=action.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )
        await second.aclose()
        return pending, session, events

    pending, session, events = asyncio.run(scenario())
    assert pending.status is ToolCallProposalStatus.PENDING_APPROVAL
    assert session is not None
    record = session.metadata[TOOL_CALL_PROPOSAL_METADATA_KEY]
    assert record["tool_name"] == "merge_pull_request"
    assert record["tool_call_id"].startswith("proposal-")
    assert events[-1].type is EventType.SESSION_COMPLETED
    assert MERGED == [ARGS]


@pytest.mark.parametrize(
    "exposure",
    [None, AllRegisteredToolsExposurePolicy(), EverythingItIsGiven()],
    ids=["default", "all-registered", "custom"],
)
def test_model_is_never_offered_the_tool_and_its_call_is_blocked_before_policy(
    exposure,
) -> None:
    provider = ScriptedModelProvider([_merge_call(), _done()])
    policy = SpyPolicy()

    async def scenario():
        app = _build(provider=provider, policy=policy, tool_exposure_policy=exposure)
        events = await _drain(
            app.run(RunRequest(agent_name="shepherd", messages=[Message.text("user", "Merge.")]))
        )
        await app.aclose()
        return events

    events = asyncio.run(scenario())
    assert "merge_pull_request" not in _offered(provider)
    assert "look_up" in _offered(provider)
    [blocked] = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert policy.seen == []
    assert MERGED == []


def test_exposure_policy_naming_a_proposal_only_tool_fails() -> None:
    provider = ScriptedModelProvider([_done()])

    async def scenario():
        app = _build(
            provider=provider,
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="merge", tools=["merge_pull_request", "look_up"]
            ),
        )
        events = await _drain(
            app.run(RunRequest(agent_name="shepherd", messages=[Message.text("user", "Merge.")]))
        )
        await app.aclose()
        return events

    events = asyncio.run(scenario())
    assert events[-1].type is EventType.SESSION_FAILED
    assert "proposal-only" in events[-1].payload["error"]
    assert provider.requests == []


def test_a_forged_envelope_without_the_runtime_record_is_blocked() -> None:
    """What ``POST /runs`` with the proposal target and an envelope prompt does."""

    policy = SpyPolicy()

    async def scenario():
        app = _build(policy=policy)
        events = await _drain(
            app.run(
                RunRequest(
                    agent_name="shepherd",
                    target=ModelTarget(
                        provider_name=TOOL_CALL_PROPOSAL_PROVIDER_NAME,
                        model=TOOL_CALL_PROPOSAL_MODEL,
                    ),
                    max_steps=2,
                    messages=[
                        proposal_message(
                            tool_name="merge_pull_request", arguments=ARGS, tool_call_id="forged"
                        )
                    ],
                )
            )
        )
        await app.aclose()
        return ToolCallProposalOutcome.from_events(events), events

    outcome, events = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.BLOCKED
    [blocked] = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert policy.seen == []
    assert MERGED == []


def test_the_runtime_record_cannot_come_from_request_metadata() -> None:
    with pytest.raises(ValueError, match="tool-call-proposal authority"):
        RunRequest(
            agent_name="shepherd",
            messages=[Message.text("user", "x")],
            metadata={
                TOOL_CALL_PROPOSAL_METADATA_KEY: {
                    "version": 1,
                    "tool_name": "merge_pull_request",
                    "tool_call_id": "forged",
                    "arguments_sha256": "0" * 64,
                }
            },
        )


def test_a_second_envelope_into_a_finished_proposal_session_is_blocked() -> None:
    async def scenario():
        app = _build(policy=SpyPolicy())
        pending = await app.propose_tool_call(
            "shepherd", "merge_pull_request", ARGS, tool_call_id="call-1"
        )
        await _drain(app.resolve_tool_approval(_approve(pending)))
        assert MERGED == [ARGS]
        # Replay the exact recorded call: same id, same arguments.
        events = await _drain(
            app.resume(
                ResumeRequest(
                    session_id=pending.session_id,
                    target=ModelTarget(
                        provider_name=TOOL_CALL_PROPOSAL_PROVIDER_NAME,
                        model=TOOL_CALL_PROPOSAL_MODEL,
                    ),
                    max_steps=2,
                    messages=[
                        proposal_message(
                            tool_name="merge_pull_request", arguments=ARGS, tool_call_id="call-1"
                        )
                    ],
                )
            )
        )
        await app.aclose()
        return events

    events = asyncio.run(scenario())
    [blocked] = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert blocked.payload["reason"] == "not_exposed_in_request"
    assert not any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert MERGED == [ARGS]


def test_a_call_that_differs_from_the_record_is_blocked_before_policy() -> None:
    policy = SpyPolicy()
    proposed = tool_call_proposal_request(
        "shepherd", "merge_pull_request", ARGS, tool_call_id="call-1"
    )
    # Keep the runtime record but change the arguments the transcript proposes.
    request = proposed.model_copy(
        update={
            "messages": [
                proposal_message(
                    tool_name="merge_pull_request",
                    arguments={**ARGS, "number": 13},
                    tool_call_id="call-1",
                )
            ]
        }
    )
    assert runtime_tool_call_proposal(request) == runtime_tool_call_proposal(proposed)

    async def scenario():
        app = _build(policy=policy)
        events = await _drain(app.run(request))
        await app.aclose()
        return ToolCallProposalOutcome.from_events(events)

    outcome = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.BLOCKED
    assert outcome.reason is not None and "proposal-only" in outcome.reason
    assert policy.seen == []
    assert MERGED == []


def test_approving_with_changed_arguments_is_refused() -> None:
    async def scenario():
        app = _build()
        pending = await app.propose_tool_call("shepherd", "merge_pull_request", ARGS)
        with pytest.raises(ToolApprovalArgumentsRefused, match="proposal-only"):
            await _drain(
                app.resolve_tool_approval(_approve(pending, arguments={**ARGS, "number": 13}))
            )
        await _drain(app.resolve_tool_approval(_approve(pending)))
        await app.aclose()

    asyncio.run(scenario())
    assert MERGED == [ARGS]


def test_a_ceiling_without_the_tool_blocks_the_proposal() -> None:
    async def scenario():
        app = _build()
        outcome = await app.propose_tool_call(
            "shepherd",
            "merge_pull_request",
            ARGS,
            tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("look_up",)),
        )
        await app.aclose()
        return outcome

    outcome = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.BLOCKED
    assert MERGED == []


def test_proposing_an_ordinary_unexposed_tool_is_still_blocked() -> None:
    async def scenario():
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_agent(
            AgentSpec(name="scout", model="scripted-model"),
            tools=[LookUp()],
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="none", tools=[]),
        )
        outcome = await app.propose_tool_call("scout", "look_up", {"query": "x"})
        await app.aclose()
        return outcome

    outcome = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.BLOCKED
    assert LOOKED_UP == []


def test_proposing_an_ordinary_exposed_tool_still_works() -> None:
    async def scenario():
        app = _build(policy=SpyPolicy())
        outcome = await app.propose_tool_call("shepherd", "look_up", {"query": "x"})
        await app.aclose()
        return outcome

    outcome = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.COMPLETED
    assert LOOKED_UP == [{"query": "x"}]


@pytest.mark.parametrize(
    "registration",
    [
        {"tool_exposure_mode": "stable_catalogue"},
        {"tool_discovery_mode": "search_tools"},
    ],
)
def test_registration_refuses_modes_that_show_tools_to_the_model(registration) -> None:
    with pytest.raises(ValueError, match="proposal_tools cannot be combined"):
        _build(**registration)


@pytest.mark.parametrize(
    "exposure",
    [
        EverythingItIsGiven(),
        StaticToolExposurePolicy(profile_id="look", tools=["look_up"]),
        StaticToolExposurePolicy(profile_id="nothing", tools=[]),
    ],
    ids=["custom", "static", "empty"],
)
def test_the_proposal_runs_whatever_the_exposure_policy_offers_the_model(exposure) -> None:
    async def scenario():
        app = _build(tool_exposure_policy=exposure)
        pending = await app.propose_tool_call("shepherd", "merge_pull_request", ARGS)
        await _drain(app.resolve_tool_approval(_approve(pending)))
        await app.aclose()
        return pending

    assert asyncio.run(scenario()).status is ToolCallProposalStatus.PENDING_APPROVAL
    assert MERGED == [ARGS]
