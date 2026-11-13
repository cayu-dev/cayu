"""A managed setting decides a tool's calls, routed to that tool alone."""

from __future__ import annotations

import asyncio

import pytest
from tests._session_provenance import fixture_session_invocation
from tests.core.test_managed_settings_controller import FakeCloud, _controller

from cayu import AgentSpec, CayuApp
from cayu.approvals import (
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallProposalOutcome,
    ToolCallProposalStatus,
)
from cayu.events import EventType
from cayu.managed_settings import (
    ManagedSetting,
    ManagedSettingChoice,
    ManagedSettingReading,
    map_setting,
    namespaced_settings,
)
from cayu.runtime import check_manifest
from cayu.runtime._execution_profile_admission import _cayu_policy_material
from cayu.sessions.records import Session, SessionStatus
from cayu.tools import (
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    GuardedToolPolicy,
    SettingToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    ToolPolicyRouter,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

ALLOW = ToolPolicyDecision.ALLOW
APPROVE = ToolPolicyDecision.REQUIRE_APPROVAL
DENY = ToolPolicyDecision.DENY
MODES = {"auto": ALLOW, "approve": APPROVE, "report": DENY}
REPOSITORY = r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+"
MERGED: list[dict] = []


class Merge(Tool):
    spec = ToolSpec(
        name="merge_pull_request",
        description="Merge one pull request.",
        input_schema={
            "type": "object",
            "properties": {"repository": {"type": "string"}, "number": {"type": "integer"}},
            "required": ["number"],
        },
        effect=ToolEffect.EXTERNAL,
        max_terminal_payload_bytes=64 * 1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        MERGED.append(dict(args))
        return ToolResult(content="Merged.")


class Comment(Tool):
    spec = ToolSpec(
        name="post_comment",
        description="Comment on a pull request.",
        input_schema={"type": "object", "properties": {"body": {"type": "string"}}},
        effect=ToolEffect.EXTERNAL,
        max_terminal_payload_bytes=64 * 1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="Commented.")


class Opaque(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=APPROVE)


def _mode_setting(**fields) -> ManagedSetting:
    return ManagedSetting(
        key="merge_mode",
        type="enum",
        label="Merge mode",
        choices=tuple(ManagedSettingChoice(value=value, label=value) for value in MODES),
        **fields,
    )


def _modes_setting(**fields) -> ManagedSetting:
    return map_setting(
        "merge_modes",
        label="Merge modes",
        item_pattern=REPOSITORY,
        value_choices=tuple(MODES),
        **fields,
    )


def _app(*, settings, policy, cloud: FakeCloud | None = None) -> CayuApp:
    app = CayuApp(
        enable_logging=False, managed_settings=None if cloud is None else _controller(cloud)
    )
    app.declare_settings(settings)
    app.register_agent(
        AgentSpec(name="shepherd", model="unconfigured-model"),
        tools=[Merge(), Comment()],
        tool_policy=policy(app),
    )
    return app


def _merge_only(app, setting: str, decisions, **fields) -> ToolPolicy:
    return ToolPolicyRouter(
        {"merge_pull_request": SettingToolPolicy(app, setting, decisions, **fields)},
        default=AlwaysRequireApprovalToolPolicy(),
    )


def _enum_policy(app) -> ToolPolicy:
    return _merge_only(app, "app.merge_mode", MODES)


def _map_policy(app) -> ToolPolicy:
    return _merge_only(app, "app.merge_modes", MODES, key_argument="repository")


async def _propose(app: CayuApp, arguments: dict) -> tuple[ToolCallProposalOutcome, list]:
    outcome = await app.propose_tool_call("shepherd", "merge_pull_request", arguments)
    events = await app.session_store.load_events(outcome.session_id)
    return outcome, events


def _find_setting(value):
    if isinstance(value, dict):
        if "cayu:setting" in value:
            return value["cayu:setting"]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            found = _find_setting(item)
            if found is not None:
                return found
    return None


def _setting_record(events: list) -> dict | None:
    """The reading a pending or blocked call's policy decision recorded."""

    for event in events:
        if event.type in {EventType.TOOL_CALL_APPROVAL_REQUESTED, EventType.TOOL_CALL_BLOCKED}:
            return _find_setting(event.payload)
    return None


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    MERGED.clear()
    for name in ("APP_MERGE_MODE", "APP_MERGE_MODES"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("value", "status"),
    [
        ("auto", ToolCallProposalStatus.COMPLETED),
        ("approve", ToolCallProposalStatus.PENDING_APPROVAL),
        ("report", ToolCallProposalStatus.BLOCKED),
    ],
)
def test_an_enum_setting_decides_each_proposal(monkeypatch, value, status) -> None:
    monkeypatch.setenv("APP_MERGE_MODE", value)

    async def scenario():
        app = _app(
            settings=[_mode_setting(env="APP_MERGE_MODE", applies="live")], policy=_enum_policy
        )
        result = await _propose(app, {"number": 12})
        await app.aclose()
        return result

    outcome, events = asyncio.run(scenario())
    assert outcome.status is status
    merged = [{"number": 12}] if value == "auto" else []
    assert merged == MERGED
    if value != "auto":
        # An allowed call's policy metadata is not published in any event today.
        record = _setting_record(events)
        assert record == {"key": "app.merge_mode", "source": "environment", "value": value}


def test_a_map_setting_decides_per_repository_and_fails_closed() -> None:
    async def scenario():
        app = _app(
            settings=[
                _modes_setting(applies="live", default={"o/auto": "auto", "o/manual": "approve"})
            ],
            policy=_map_policy,
        )
        results = {}
        for name, arguments in {
            "auto": {"repository": "o/auto", "number": 1},
            "manual": {"repository": "o/manual", "number": 2},
            "unlisted": {"repository": "o/other", "number": 3},
            "case": {"repository": "O/Auto", "number": 4},
            "missing": {"number": 5},
        }.items():
            results[name] = await _propose(app, arguments)
        await app.aclose()
        return results

    results = asyncio.run(scenario())
    statuses = {name: outcome.status for name, (outcome, _events) in results.items()}
    assert statuses == {
        "auto": ToolCallProposalStatus.COMPLETED,
        "manual": ToolCallProposalStatus.PENDING_APPROVAL,
        "unlisted": ToolCallProposalStatus.BLOCKED,
        "case": ToolCallProposalStatus.BLOCKED,
        "missing": ToolCallProposalStatus.BLOCKED,
    }
    assert MERGED == [{"repository": "o/auto", "number": 1}]
    assert _setting_record(results["manual"][1]) == {
        "key": "app.merge_modes",
        "source": "default",
        "entry": "o/manual",
        "value": "approve",
    }
    assert _setting_record(results["case"][1])["value"] is None


def _declared(*settings: ManagedSetting) -> tuple[ManagedSetting, ...]:
    return namespaced_settings(settings, prefix="app", owner="app")


class _Reader:
    def __init__(
        self,
        reading=None,
        error: Exception | None = None,
        declared: tuple[ManagedSetting, ...] | None = None,
    ) -> None:
        self.reading = reading
        self.error = error
        self.declared = _declared(_modes_setting(applies="live")) if declared is None else declared

    def managed_settings(self) -> tuple[ManagedSetting, ...]:
        return self.declared

    def read_managed_setting(self, key: str):
        if self.error is not None:
            raise self.error
        return self.reading


def _request(arguments: dict, *, metadata: dict | None = None) -> ToolPolicyRequest:
    return ToolPolicyRequest(
        session=Session(
            id="s",
            agent_name="shepherd",
            provider_name="fake",
            model="fake-model",
            invocation=fixture_session_invocation("s"),
            status=SessionStatus.RUNNING,
            metadata=metadata or {},
        ),
        agent=AgentSpec(name="shepherd", model="fake-model"),
        tool_name="merge_pull_request",
        tool_call_id="call_1",
        arguments=arguments,
    )


def test_unusable_values_take_the_fallback_and_never_allow() -> None:
    def decide(reader, arguments, **fields) -> ToolPolicyDecision:
        policy = SettingToolPolicy(reader, "app.merge_modes", MODES, **fields)
        return asyncio.run(policy.authorize(_request(arguments))).decision

    modes = ManagedSettingReading(key="app.merge_modes", value={"o/a": "auto"}, source="cloud")
    keyed = {"key_argument": "repository"}
    assert decide(_Reader(modes), {"repository": "o/a"}, **keyed) is ALLOW
    assert decide(_Reader(modes), {"repository": 7}, **keyed) is DENY
    assert decide(_Reader(modes), {}, **keyed) is DENY
    refused = ManagedSettingReading(
        key="app.merge_modes", value={"o/a": "auto"}, source="default", refused="bad"
    )
    assert decide(_Reader(refused), {"repository": "o/a"}, **keyed) is DENY
    undeclared = _Reader(error=KeyError("app.merge_modes"), declared=())
    assert decide(undeclared, {"repository": "o/a"}, **keyed) is DENY
    assert decide(undeclared, {}, otherwise=APPROVE) is APPROVE
    with pytest.raises(ValueError, match="never ALLOW"):
        SettingToolPolicy(_Reader(modes), "app.merge_modes", MODES, otherwise=ALLOW)


def test_a_declaration_the_policy_cannot_map_never_allows() -> None:
    # A plain string setting whose current value happens to be a mapped one.
    text = ManagedSetting(
        key="merge_mode", type="string", label="Merge mode", default="auto", max_length=20
    )
    auto = ManagedSettingReading(key="app.merge_mode", value="auto", source="default")

    def decide(declared, decisions) -> ToolPolicyResult:
        policy = SettingToolPolicy(_Reader(auto, declared=declared), "app.merge_mode", decisions)
        return asyncio.run(policy.authorize(_request({})))

    refused = decide(_declared(text), {"auto": ALLOW})
    assert refused.decision is DENY
    assert refused.reason == (
        "The policy can't map app.merge_mode as declared, so the policy's fallback applies."
    )
    # An enum whose choices aren't all mapped is refused too, even for a mapped value.
    assert decide(_declared(_mode_setting()), {"auto": ALLOW}).decision is DENY
    assert decide(_declared(_mode_setting()), MODES).decision is ALLOW


def test_a_next_session_setting_keeps_the_value_the_session_started_with(monkeypatch) -> None:
    monkeypatch.setenv("APP_MERGE_MODES", "o/a=approve")
    arguments = {"repository": "o/a", "number": 1}

    async def scenario():
        app = _app(settings=[_modes_setting(env="APP_MERGE_MODES")], policy=_map_policy)
        first, _ = await _propose(app, arguments)
        monkeypatch.setenv("APP_MERGE_MODES", "o/a=auto")
        second, _ = await _propose(app, {**arguments, "number": 2})
        session = await app.session_store.load(first.session_id)
        assert session is not None
        policy = app._get_registered_agent("shepherd").tool_policy
        # The first session authorizes a later call with the value it started with.
        later = await policy.authorize(_request(arguments).model_copy(update={"session": session}))
        await app.aclose()
        return first, second, session, later

    first, second, session, later = asyncio.run(scenario())
    assert session.metadata["cayu:session_settings"] == {
        "app.merge_modes": {"value": {"o/a": "approve"}, "source": "environment", "refused": None}
    }
    assert first.status is ToolCallProposalStatus.PENDING_APPROVAL
    assert second.status is ToolCallProposalStatus.COMPLETED
    assert later.decision is APPROVE
    assert later.metadata["cayu:setting"]["value"] == "approve"
    assert [{**arguments, "number": 2}] == MERGED


@pytest.mark.parametrize("same_agent", [True, False])
def test_a_fork_keeps_or_reads_the_next_session_settings_of_its_agent(
    monkeypatch, same_agent
) -> None:
    from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.messages import Message
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime.execution_profiles import (
        ExecutionProfileAdoptionIntent,
        ExecutionProfileAuthorityDecision,
        ExecutionProfilePolicy,
        ExecutionProfilePolicyAction,
        ExecutionProfilePolicyResult,
    )
    from cayu.sessions.requests import ForkSessionRequest, RunRequest

    monkeypatch.setenv("APP_MERGE_MODES", "o/a=approve")

    class Adopt(ExecutionProfilePolicy):
        identity = "test:setting-policy-fork-adoption:v1"

        async def decide(self, request):
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Fork to the selected agent.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )

    async def scenario():
        app = CayuApp(enable_logging=False, execution_profile_policy=Adopt())
        reply = [
            ModelStreamEvent.text_delta("Done."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
        app.register_provider(ScriptedModelProvider(reply, name="fake"), default=True)
        app.declare_settings([_modes_setting(env="APP_MERGE_MODES")])
        app.register_agent(
            AgentSpec(name="shepherd", model="fake-model"),
            tools=[Merge(), Comment()],
            tool_policy=_map_policy(app),
        )
        app.register_agent(
            AgentSpec(name="helper", model="fake-model"),
            tools=[Comment()],
            tool_policy=AllowAllToolPolicy(),
        )
        source_agent = "shepherd" if same_agent else "helper"
        async for _ in app.run(
            RunRequest(
                agent_name=source_agent,
                session_id="source",
                messages=[Message.text("user", "Look at o/a.")],
            )
        ):
            pass
        # The value changes after the source started and before the fork.
        monkeypatch.setenv("APP_MERGE_MODES", "o/a=auto")
        request = ForkSessionRequest(
            source_session_id="source",
            session_id="fork",
            agent_name="shepherd",
            execution_profile_selection="current_child",
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="fork",
                reason="Fork to the shepherd.",
                requested_by=ResolutionActor(
                    subject="operator", source=ResolutionActorSource.REQUEST
                ),
            ),
        )
        [_ async for _ in app.fork_session(request)]
        fork = await app.session_store.load("fork")
        await app.aclose()
        return fork

    fork = asyncio.run(scenario())
    assert fork is not None
    # A fork of the same agent keeps the source's value; one that selects another agent
    # reads that agent's settings as it starts, like any new session.
    expected = {"o/a": "approve"} if same_agent else {"o/a": "auto"}
    assert fork.metadata["cayu:session_settings"] == {
        "app.merge_modes": {"value": expected, "source": "environment", "refused": None}
    }


def test_a_live_setting_is_not_kept_by_the_session(monkeypatch) -> None:
    monkeypatch.setenv("APP_MERGE_MODE", "approve")

    async def scenario():
        app = _app(
            settings=[_mode_setting(env="APP_MERGE_MODE", applies="live")], policy=_enum_policy
        )
        outcome, _ = await _propose(app, {"number": 1})
        session = await app.session_store.load(outcome.session_id)
        await app.aclose()
        return session

    session = asyncio.run(scenario())
    assert session is not None
    assert "cayu:session_settings" not in session.metadata


def test_a_recorded_value_wins_over_the_current_one_only_for_next_session() -> None:
    now = ManagedSettingReading(key="app.merge_modes", value={"o/a": "auto"}, source="cloud")
    started = {
        "cayu:session_settings": {
            "app.merge_modes": {"value": {"o/a": "report"}, "source": "cloud", "refused": None}
        }
    }
    call = {"repository": "o/a"}

    def decide(applies) -> ToolPolicyDecision:
        reader = _Reader(now, declared=_declared(_modes_setting(applies=applies)))
        policy = SettingToolPolicy(reader, "app.merge_modes", MODES, key_argument="repository")
        return asyncio.run(policy.authorize(_request(call, metadata=started))).decision

    assert decide("next_session") is DENY
    assert decide("live") is ALLOW


def test_callers_cannot_supply_session_settings() -> None:
    from cayu.sessions.requests import RunRequest

    with pytest.raises(ValueError, match="session-settings authority"):
        RunRequest(
            agent_name="shepherd",
            input="hi",
            metadata={"cayu:session_settings": {"app.merge_modes": {}}},
        )


def test_a_refused_cloud_value_does_not_fall_back_to_the_default() -> None:
    cloud = FakeCloud()
    cloud.values = {"app.merge_modes": "o/a=merge"}

    async def scenario():
        app = _app(
            settings=[_modes_setting(applies="live", default={"o/a": "auto"})],
            policy=_map_policy,
            cloud=cloud,
        )
        await app.start_managed_settings(role="worker")
        result = await _propose(app, {"repository": "o/a", "number": 1})
        await app.aclose()
        return result

    outcome, events = asyncio.run(scenario())
    assert outcome.status is ToolCallProposalStatus.BLOCKED
    assert MERGED == []
    assert _setting_record(events) == {"key": "app.merge_modes", "source": "default", "value": None}


def test_a_value_change_applies_to_the_next_call_and_leaves_a_pending_one(monkeypatch) -> None:
    monkeypatch.setenv("APP_MERGE_MODE", "approve")

    async def scenario():
        app = _app(
            settings=[_mode_setting(env="APP_MERGE_MODE", applies="live")], policy=_enum_policy
        )
        pending, _ = await _propose(app, {"number": 1})
        monkeypatch.setenv("APP_MERGE_MODE", "report")
        blocked, _ = await _propose(app, {"number": 2})
        assert pending.approval_id and pending.tool_round_id and pending.tool_call_id
        async for _event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=pending.session_id,
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
        ):
            pass
        await app.aclose()
        return pending, blocked

    pending, blocked = asyncio.run(scenario())
    assert pending.status is ToolCallProposalStatus.PENDING_APPROVAL
    assert blocked.status is ToolCallProposalStatus.BLOCKED
    # The pending approval was not revoked by the later value.
    assert MERGED == [{"number": 1}]


def test_the_profile_identity_covers_configuration_not_the_live_value() -> None:
    reader = _Reader()
    first = _cayu_policy_material(SettingToolPolicy(reader, "app.merge_mode", MODES))
    again = _cayu_policy_material(SettingToolPolicy(_Reader(), "app.merge_mode", MODES))
    changed = _cayu_policy_material(
        SettingToolPolicy(reader, "app.merge_mode", {**MODES, "auto": APPROVE})
    )
    assert first == again and first is not None
    assert changed != first
    routed = _cayu_policy_material(
        ToolPolicyRouter(
            {"merge_pull_request": SettingToolPolicy(reader, "app.merge_mode", MODES)},
            default=AllowAllToolPolicy(),
        )
    )
    assert routed is not None
    assert (
        _cayu_policy_material(ToolPolicyRouter({"x": Opaque()}, default=AllowAllToolPolicy()))
        is None
    )


def test_the_router_routes_each_tool_and_refuses_two_taint_policies() -> None:
    router = ToolPolicyRouter({"merge_pull_request": Opaque()}, default=AllowAllToolPolicy())
    assert asyncio.run(router.authorize(_request({}))).decision is APPROVE
    other = _request({}).model_copy(update={"tool_name": "post_comment"})
    assert asyncio.run(router.authorize(other)).decision is ALLOW
    taint = {
        "protected_tools": {"merge_pull_request": ["web"]},
        "taint_sources": {"fetch": ["web"]},
    }
    with pytest.raises(ValueError, match="at most one TaintAwareToolPolicy"):
        ToolPolicyRouter(
            {"merge_pull_request": TaintAwareToolPolicy(**taint)},
            default=TaintAwareToolPolicy(**taint),
        )


def _codes(app: CayuApp) -> list[str]:
    return [
        item.code
        for item in check_manifest(app.describe()).diagnostics
        if item.code.startswith(("TOOL_POLICY_SETTING", "EXTERNAL_TOOL"))
    ]


def test_check_reads_the_router_and_the_setting_policy() -> None:
    app = _app(settings=[_modes_setting(default={"o/a": "approve"})], policy=_map_policy)
    manifest = app.describe()
    tools = {tool.name: tool for tool in manifest.agents[0].tools}
    assert tools["merge_pull_request"].policy_coverage == "conditional"
    assert tools["merge_pull_request"].policy_setting == "app.merge_modes"
    assert tools["post_comment"].policy_coverage == "approval_required"
    report = check_manifest(manifest)
    [info] = [item for item in report.diagnostics if item.code == "TOOL_POLICY_SETTING_ALLOWS"]
    assert info.message == (
        "merge_pull_request runs without approval when app.merge_modes is auto for its repository."
    )
    assert _codes(app) == ["TOOL_POLICY_SETTING_ALLOWS"]

    allowing = _app(settings=[_modes_setting(default={"o/a": "auto"})], policy=_map_policy)
    assert "TOOL_POLICY_SETTING_DEFAULT_ALLOWS" in _codes(allowing)


@pytest.mark.parametrize(
    ("settings", "policy", "problem"),
    [
        ([], _map_policy, "is not a declared managed setting"),
        ([_mode_setting()], _map_policy, "is not a declared managed setting"),
        (
            [map_setting("merge_modes", label="Merge modes", value_pattern="[a-z]+")],
            _map_policy,
            "must be an enum setting, or a map setting",
        ),
        (
            [_modes_setting()],
            lambda app: _merge_only(app, "app.merge_modes", MODES),
            "so key_argument must name",
        ),
        (
            [_mode_setting()],
            lambda app: _merge_only(app, "app.merge_mode", {"auto": ALLOW}),
            "unmapped choices: approve, report",
        ),
        (
            [_mode_setting()],
            lambda app: _merge_only(app, "app.merge_mode", {**MODES, "merge": ALLOW}),
            "mapped values that aren't choices: merge",
        ),
        (
            [_modes_setting()],
            lambda app: _merge_only(app, "app.merge_modes", MODES, key_argument="number"),
            "key_argument number is not a string property",
        ),
    ],
)
def test_check_refuses_a_setting_the_policy_cannot_map(settings, policy, problem) -> None:
    app = _app(settings=settings, policy=policy)
    report = check_manifest(app.describe())
    invalid = [item for item in report.diagnostics if item.code == "TOOL_POLICY_SETTING_INVALID"]
    assert len(invalid) == 1
    assert invalid[0].severity == "error"
    assert problem in invalid[0].message


def test_a_custom_policy_routed_to_one_tool_leaves_only_that_tool_unknown() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="shepherd", model="unconfigured-model"),
        tools=[Merge(), Comment()],
        tool_policy=ToolPolicyRouter(
            {"merge_pull_request": Opaque()}, default=AlwaysRequireApprovalToolPolicy()
        ),
    )
    tools = {tool.name: tool for tool in app.describe().agents[0].tools}
    assert tools["merge_pull_request"].policy_coverage == "unknown"
    assert tools["post_comment"].policy_coverage == "approval_required"
    [finding] = [
        item
        for item in check_manifest(app.describe()).diagnostics
        if item.code == "EXTERNAL_TOOL_COVERAGE_UNKNOWN"
    ]
    assert finding.parameters["tools"] == ("merge_pull_request",)
    assert finding.parameters["policy"].endswith("Opaque")


def test_an_opaque_policy_is_reported_once_for_all_its_tools() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="shepherd", model="unconfigured-model"),
        tools=[Merge(), Comment()],
        tool_policy=GuardedToolPolicy(guards=[], then=Opaque()),
    )
    [finding] = [
        item
        for item in check_manifest(app.describe()).diagnostics
        if item.code == "EXTERNAL_TOOL_COVERAGE_UNKNOWN"
    ]
    assert finding.subject == "agent:shepherd"
    assert finding.parameters["tools"] == ("merge_pull_request", "post_comment")
    assert "merge_pull_request" in finding.message and "post_comment" in finding.message
