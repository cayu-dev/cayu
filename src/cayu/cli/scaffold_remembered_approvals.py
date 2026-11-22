"""The `remembered-approvals` capability: tool authorization grants for the approvals.

A person approving a paused call can have Cayu remember it, so later calls of the same
kind run without asking (`docs/tool-authorization-grants.md`). This module adds the
generated project's side: the grant store beside the sessions, the grant policy around
each agent's tool policy, who may grant, and the hosting `cayu serve` mounts.
"""

from __future__ import annotations

import ast

from cayu.cli.scaffold_convention import _replace_once, capability_card
from cayu.cli.scaffold_plan import REMEMBERED_APPROVALS, ApplicationPlan

_GRANTS_PY = '''"""Remembered approvals: when a grant may stand in for a person's approval.

A person approving a paused call can have it remembered as a tool authorization
grant, so the same kind of call in that session runs without asking again.
`cayu inbox approve --remember` makes one and `cayu grants` lists and revokes
them. See `uv run --no-sync cayu guide long-running#remembered-approvals`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from cayu import CayuApp
from cayu.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantAuthority,
    ToolAuthorizationGrantControl,
    ToolAuthorizationGrantCreation,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantService,
    ToolAuthorizationGrantStore,
)
from cayu.evals import load_project_objectives
from cayu.server import ControlPlaneRoles
from cayu.tools import ToolPolicy

from configuration.settings import configured_grant_owners

if TYPE_CHECKING:
    from cayu.server import AuthContext

# Every grant records this id and only matches under it, so changing it retires
# every grant made so far.
POLICY_ID = "__AGENT_NAME__.external-effects"
# The class of decision the external-effect approvals name (`cayu:approval_gate`).
APPROVAL_GATE = "external-effect"
# Questions and knowledge proposals are never granted.
NEVER_GRANTABLE = frozenset({"ask_user", "remember_knowledge"})
# Mandatory controls and the default sampled-review rate come from the project's
# objectives document; these apply without one.
OBJECTIVES = Path(__file__).resolve().parents[1] / "cayu-objectives.toml"
MANDATORY_CONTROLS: tuple[ToolAuthorizationGrantControl, ...] = ()
SAMPLED_REVIEW_RATE_BPS = 500


def grant_terms() -> tuple[tuple[ToolAuthorizationGrantControl, ...], int]:
    """The mandatory controls and the default sampled-review rate (500 = 5%)."""

    if not OBJECTIVES.is_file():
        return MANDATORY_CONTROLS, SAMPLED_REVIEW_RATE_BPS
    objectives = load_project_objectives(OBJECTIVES)
    controls = tuple(
        ToolAuthorizationGrantControl(
            tool_name=control.tool_name, approval_gate=control.approval_gate
        )
        for control in objectives.mandatory_controls
    )
    return controls, objectives.sampled_review_rate_bps


def remember_approvals(
    policy: ToolPolicy, *, store: ToolAuthorizationGrantStore, tools: Iterable[str]
) -> ToolPolicy:
    """Let an active grant stand in for a person's approval of ``tools``.

    A grant only turns ``policy``'s REQUIRE_APPROVAL into ALLOW, never a denial,
    and a tool under a mandatory control always asks. Without a tool a grant
    could cover, ``policy`` is returned as it is.
    """

    grantable = {name: "tool_call" for name in tools if name not in NEVER_GRANTABLE}
    if not grantable:
        return policy
    controls, _ = grant_terms()
    return ToolAuthorizationGrantPolicy(
        policy_id=POLICY_ID,
        store=store,
        grantable=grantable,
        then=policy,
        mandatory_controls=controls,
    )


class GrantOwners(ToolAuthorizationGrantAuthority):
    """Who may remember an approval, revoke a grant, and read grants.

    Only an actor the grant routes authenticated (through `cayu serve`'s auth, or
    locally as the CLI's `--reviewer`), and then only with a full-access role or a subject listed in ``owners``
    (CAYU_GRANT_OWNERS). ``roles`` are the ones [tool.cayu.serve] auth applies,
    whose defaults give admin and developer full access, never member. A person
    who may approve calls one at a time may not grant unless listed.
    """

    def __init__(
        self, owners: Iterable[str] = (), *, roles: ControlPlaneRoles | None = None
    ) -> None:
        self.owners = frozenset(owners)
        self.roles = ControlPlaneRoles() if roles is None else roles

    def allows(self, subject: str, claims: Mapping[str, object]) -> bool:
        role = claims.get(self.roles.role_claim)
        return (
            type(role) is str and role in self.roles.full_access
        ) or subject in self.owners

    def permits(self, actor: ResolutionActor | None) -> bool:
        return (
            actor is not None
            and actor.source is ResolutionActorSource.HTTP_AUTH
            and self.allows(actor.subject, actor.claims)
        )

    async def authorize_creation(
        self, creation: ToolAuthorizationGrantCreation
    ) -> bool:
        return self.permits(creation.actor)

    async def authorize_revocation(
        self, actor: ResolutionActor | None, grant: ToolAuthorizationGrant
    ) -> bool:
        return self.permits(actor)

    async def authorize_read(
        self, caller: AuthContext, grant: ToolAuthorizationGrant
    ) -> bool:
        # The grant routes pass only callers their authentication accepted.
        return self.allows(caller.subject, caller.claims)


def host_remembered_approvals(
    app: CayuApp, store: ToolAuthorizationGrantStore
) -> ToolAuthorizationGrantService:
    """Have `cayu serve` serve the grant routes and `cayu worker` settle staged grants."""

    owners = GrantOwners(configured_grant_owners())
    _, sampled_review_rate_bps = grant_terms()
    service = ToolAuthorizationGrantService(
        app,
        store=store,
        authority=owners,
        default_sampled_review_rate_bps=sampled_review_rate_bps,
    )
    app.host_tool_authorization_grants(service, authorize_read=owners.authorize_read)
    return service
'''

_SETTINGS_APPEND = '''

def configured_grant_owners() -> frozenset[str]:
    """CAYU_GRANT_OWNERS: exact subjects who may grant besides full-access roles.

    Comma-separated and empty by default (see policies/grants.py).
    """

    value = os.environ.get("CAYU_GRANT_OWNERS") or ""
    return frozenset(item.strip() for item in value.split(",") if item.strip())
'''

_TEST_PY = '''"""Remembered approvals: approve once, and the same call runs without asking.

The test adds one external tool, `send_update`, under the project's own tool
policy and the grants `build_app` hosts.
"""

from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.approvals import (
    PendingToolApprovalEventView,
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolAuthorizationGrantDenied,
    ToolAuthorizationGrantRequest,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantUnavailable,
)
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.sessions import InMemorySessionStore
__STORE_IMPORTS__
import policies.grants
from app import build_app
from configuration.settings import configured_grant_owners
from policies.grants import GrantOwners
from policies.tools import build_tool_policy

SESSION = "weekly-update"
DEVELOPER = ResolutionActor(
    subject="dev@example.test",
    source=ResolutionActorSource.HTTP_AUTH,
    claims={"role": "developer"},
)
MEMBER = ResolutionActor(
    subject="member@example.test",
    source=ResolutionActorSource.HTTP_AUTH,
    claims={"role": "member"},
)
# No sampled review, so no remembered call is drawn to go to a person anyway.
REMEMBER = ToolAuthorizationGrantRequest(
    scope=ToolAuthorizationGrantScopeKind.SESSION, sampled_review_rate_bps=0
)


class SendUpdate(Tool):
    spec = ToolSpec(
        name="send_update",
        effect=ToolEffect.EXTERNAL,
        description="Send a status update.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": 200}},
            "required": ["text"],
            "additionalProperties": False,
        },
        max_terminal_payload_bytes=64 * 1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests.send_update", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self, sent: list[str]) -> None:
        super().__init__()
        self.sent = sent

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.sent.append(args["text"])
        return ToolResult(content="Sent.")


def send(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(name="send_update", arguments={"text": text}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def done() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta("Done."),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def build(turns, sessions, sent):
    """The project's app, as one process builds it, with the update agent added."""

    provider = ScriptedModelProvider(turns)
    app = build_app(
        provider=provider,
        session_store=sessions,__STORE_ARGUMENTS__
    )
    hosting = app.tool_authorization_grant_hosting
    assert hosting is not None
    app.register_agent(
        AgentSpec(name="updates", model="scripted", provider_name=provider.name),
        tools=[SendUpdate(sent)],
        tool_policy=build_tool_policy(
            ("send_update",), grant_store=hosting.service.store
        ),
    )
    return app, hosting.service


async def events(stream) -> list:
    return [event async for event in stream]


async def approval(sessions, actor: ResolutionActor) -> ToolApprovalRequest:
    """The paused call's approval, read from the session's durable events."""

    requested = next(
        event
        for event in await sessions.load_events(SESSION)
        if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    pending = PendingToolApprovalEventView.from_event(requested)
    return ToolApprovalRequest(
        session_id=SESSION,
        approval_id=pending.approval_id,
        tool_round_id=pending.tool_round_id,
        tool_call_id=pending.tool_call_id,
        decision=ToolApprovalDecision.APPROVE,
        resolved_by=actor,
    )


def test_a_remembered_approval_lets_the_same_call_run_without_asking() -> None:
    sessions, sent = InMemorySessionStore(), []

    async def scenario() -> None:
        turns = [send("Shipped."), done(), send("Shipped again."), done()]
        app, service = build(turns, sessions, sent)
        await events(
            app.run(
                RunRequest(
                    agent_name="updates",
                    session_id=SESSION,
                    messages=[Message.text("user", "Send the update.")],
                )
            )
        )
        assert sent == []
__RESTART__        async with app:
            with pytest.raises(ToolAuthorizationGrantDenied):
                await events(
                    service.approve_and_grant(
                        await approval(sessions, MEMBER), REMEMBER
                    )
                )
            await events(
                service.approve_and_grant(await approval(sessions, DEVELOPER), REMEMBER)
            )
            assert sent == ["Shipped."]
            again = await events(
                app.resume(
                    ResumeRequest(
                        session_id=SESSION,
                        messages=[Message.text("user", "Send it again.")],
                    )
                )
            )
            assert EventType.TOOL_CALL_APPROVAL_REQUESTED not in {
                event.type for event in again
            }
            assert sent == ["Shipped.", "Shipped again."]
            [grant] = await service.list_grants()
            assert grant.created_by is not None
            assert grant.created_by.subject == "dev@example.test"
            [use] = await service.list_uses(grant.grant_id)
            assert use.tool_name == "send_update"

    asyncio.run(scenario())


def test_only_full_access_roles_and_listed_owners_may_grant(monkeypatch) -> None:
    owners = GrantOwners(["local:me"])
    assert owners.allows("someone@example.test", {"role": "admin"})
    assert owners.allows("local:me", {})
    assert not owners.allows("member@example.test", {"role": "member"})
    assert not owners.allows("someone@example.test", {})
    # A subject only claimed in a request, not authenticated, never grants.
    claimed = ResolutionActor(subject="local:me", source=ResolutionActorSource.REQUEST)
    assert not owners.permits(claimed)
    monkeypatch.setenv("CAYU_GRANT_OWNERS", "local:me, ops@example.test")
    assert configured_grant_owners() == {"local:me", "ops@example.test"}


def test_a_tool_under_a_mandatory_control_is_never_remembered(
    tmp_path, monkeypatch
) -> None:
    objectives = tmp_path / "cayu-objectives.toml"
    objectives.write_text(
        "schema_version = 1\\n"
        "sampled_review_rate_bps = 0\\n"
        "[[correctness_floor]]\\n"
        'suite_id = "agent-output"\\n'
        "[[mandatory_controls]]\\n"
        'tool_name = "send_update"\\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(policies.grants, "OBJECTIVES", objectives)
    sessions, sent = InMemorySessionStore(), []

    async def scenario() -> None:
        app, service = build([send("Shipped.")], sessions, sent)
        async with app:
            await events(
                app.run(
                    RunRequest(
                        agent_name="updates",
                        session_id=SESSION,
                        messages=[Message.text("user", "Send the update.")],
                    )
                )
            )
            with pytest.raises(ToolAuthorizationGrantUnavailable, match="mandatory"):
                await events(
                    service.approve_and_grant(
                        await approval(sessions, DEVELOPER), REMEMBER
                    )
                )
            assert sent == []

    asyncio.run(scenario())
'''

_CARD_APPEND = """
## Who may grant

Remembering an approval, revoking a grant and reading grants are allowed only
for a caller the grant routes authenticated (through `cayu serve`'s auth, or
locally as the CLI's `--reviewer`) who either has a role `cayu serve` gives full
access (admin or developer on Cayu Cloud, never member) or
is listed in `CAYU_GRANT_OWNERS`, comma-separated exact subjects, empty by
default. A person who may approve calls one at a time may not grant unless
listed. The rule is `GrantOwners` in `policies/grants.py`.

Locally, add yourself as an owner, then approve and remember:

```console
CAYU_GRANT_OWNERS=local:me uv run --no-sync cayu inbox approve ITEM --remember --reviewer local:me
```

The agents' external tools are grantable (`policies/grants.py`); questions and
knowledge proposals never are. Mandatory controls and the default sampled-review
rate come from `cayu-objectives.toml` when the project has it. See
`uv run --no-sync cayu guide long-running#remembered-approvals`.
"""


def remembered_approvals_files(files: dict[str, str], plan: ApplicationPlan) -> dict[str, str]:
    """The files `--with remembered-approvals` adds or changes, from the rendered project."""

    if REMEMBERED_APPROVALS not in plan.capabilities:
        return {}
    restart = (
        # With recovery, a paused session can be approved by another process.
        "        # Another process approves it, as after a restart.\n"
        "        await app.aclose()\n"
        "        app, service = build(turns[1:], sessions, sent)\n"
        if "recovery" in plan.capabilities
        else ""
    )
    # The test injects the stores the selected capabilities use, as tests/test_agent.py does.
    stored = [
        (argument, module, store)
        for capability, argument, module, store in (
            ("tasks", "task_store", "cayu.tasks", "InMemoryTaskStore"),
            ("knowledge", "knowledge_store", "cayu.storage", "InMemoryKnowledgeStore"),
        )
        if capability in plan.capabilities
    ]
    changed = {
        "policies/grants.py": _GRANTS_PY.replace("__AGENT_NAME__", plan.agent_name),
        "policies/REMEMBERED_APPROVALS.md": capability_card(plan, REMEMBERED_APPROVALS)
        + _CARD_APPEND,
        "policies/tools.py": _tool_policy(files["policies/tools.py"]),
        "configuration/settings.py": files["configuration/settings.py"].rstrip()
        + "\n"
        + _SETTINGS_APPEND,
        "configuration/storage.py": _storage(files["configuration/storage.py"]),
        "agents/registration.py": _registration(files["agents/registration.py"]),
        "app.py": _app(files["app.py"]),
        "tests/test_remembered_approvals.py": _TEST_PY.replace("__RESTART__", restart)
        .replace(
            "__STORE_IMPORTS__",
            "".join(f"from {module} import {store}\n" for _, module, store in stored),
        )
        .replace(
            "__STORE_ARGUMENTS__",
            "".join(f"\n        {argument}={store}()," for argument, _, store in stored),
        ),
    }
    standard = files.get("tests/test_standard_capabilities.py")
    if standard is not None:
        # The grant policy wraps the agent's policy once it has a tool a grant can cover.
        changed["tests/test_standard_capabilities.py"] = _replace_once(
            standard,
            '    assert agent.tool_policy == "ParameterConstrainedToolPolicy"\n',
            "    assert agent.tool_policy in {\n"
            '        "ParameterConstrainedToolPolicy",\n'
            '        "ToolAuthorizationGrantPolicy",\n'
            "    }\n",
        )
    return changed


def _tool_policy(source: str) -> str:
    source = _add_imports(
        source,
        "from cayu.approvals import ToolAuthorizationGrantStore\n\n"
        "from policies.grants import APPROVAL_GATE, remember_approvals\n",
    )
    source = _replace_once(
        source,
        "def build_tool_policy(external_tool_names: tuple[str, ...]) -> ToolPolicy:\n",
        "def build_tool_policy(\n"
        "    external_tool_names: tuple[str, ...],\n"
        "    *,\n"
        "    grant_store: ToolAuthorizationGrantStore | None = None,\n"
        ") -> ToolPolicy:\n",
    )
    remember = (
        "    if grant_store is None:\n"
        "        return policy\n"
        "    return remember_approvals(policy, store=grant_store, tools=external_tool_names)\n"
    )
    if "GuardedToolPolicy(" in source:
        # The long-running preset: approval comes from AlwaysRequireApprovalToolPolicy.
        source = _replace_once(
            source, "    return GuardedToolPolicy(\n", "    policy = GuardedToolPolicy(\n"
        )
        source = _replace_once(
            source,
            "            expires_in_seconds=APPROVAL_EXPIRES_AFTER.total_seconds(),\n"
            "        ),\n"
            "    )\n",
            "            expires_in_seconds=APPROVAL_EXPIRES_AFTER.total_seconds(),\n"
            "            # Remembered approvals match on this gate.\n"
            "            approval_gate=APPROVAL_GATE,\n"
            "        ),\n"
            "    )\n" + remember,
        )
        return source
    # The approval gate is part of the policy's behavior, so its declared identity moves.
    source = source.replace('    behavior_version="2",\n', '    behavior_version="3",\n', 1)
    return _replace_once(
        source,
        "        return ParameterConstrainedToolPolicy(\n"
        "            rules,\n"
        "            decision=ToolPolicyDecision.REQUIRE_APPROVAL,\n"
        "            execution_profile_identity=_TOOL_POLICY_IDENTITY,\n"
        "        )\n",
        "        policy = ParameterConstrainedToolPolicy(\n"
        "            rules,\n"
        "            decision=ToolPolicyDecision.REQUIRE_APPROVAL,\n"
        "            execution_profile_identity=_TOOL_POLICY_IDENTITY,\n"
        "            # Remembered approvals match on this gate.\n"
        "            approval_gate=APPROVAL_GATE,\n"
        "        )\n" + _indent(remember),
    )


def _storage(source: str) -> str:
    source = _add_imports(
        source,
        "from cayu.approvals import (\n"
        "    InMemoryToolAuthorizationGrantStore,\n"
        "    ToolAuthorizationGrantStore,\n"
        ")\n",
    )
    source = _replace_once(
        source,
        "    configured: ApplicationStores | None\n",
        "    configured: ApplicationStores | None\n"
        "    # Remembered approvals, beside the sessions: in the configured database, or\n"
        "    # in memory when the caller injects its own session store, as tests do.\n"
        "    grant_store: ToolAuthorizationGrantStore\n",
    )
    source = _replace_once(
        source,
        "            configured=None,\n            product_store=product_store,\n",
        "            configured=None,\n"
        "            grant_store=InMemoryToolAuthorizationGrantStore(),\n"
        "            product_store=product_store,\n",
    )
    source = _replace_once(
        source,
        "        product_operations=build_product,\n",
        "        product_operations=build_product,\n"
        "        tool_authorization_grants=session_store is None,\n",
    )
    return _replace_once(
        source,
        "        configured=configured,\n",
        "        configured=configured,\n"
        "        grant_store=configured.tool_authorization_grant_store\n"
        "        or InMemoryToolAuthorizationGrantStore(),\n",
    )


def _registration(source: str) -> str:
    source = _add_imports(source, "from cayu.approvals import ToolAuthorizationGrantStore\n")
    source = _replace_once(
        source,
        "    context_policy: ContextPolicy | None = None,\n",
        "    context_policy: ContextPolicy | None = None,\n"
        "    grant_store: ToolAuthorizationGrantStore | None = None,\n",
    )
    return _replace_once(
        source,
        "        tool_policy=build_tool_policy(tuple(starter_external_tool_names)),\n",
        "        tool_policy=build_tool_policy(\n"
        "            tuple(starter_external_tool_names), grant_store=grant_store\n"
        "        ),\n",
    )


def _app(source: str) -> str:
    source = _add_imports(source, "from policies.grants import host_remembered_approvals\n")
    source = _replace_once(
        source,
        "        register_agents(\n            app,\n",
        "        register_agents(\n            app,\n            grant_store=stores.grant_store,\n",
    )
    return _replace_once(
        source,
        "        return app\n",
        "        # `cayu serve` serves the grant routes; `cayu worker` settles staged grants.\n"
        "        host_remembered_approvals(app, stores.grant_store)\n"
        "        return app\n",
    )


def _add_imports(source: str, imports: str) -> str:
    """Insert ``imports`` after the module's last top-level import."""

    last = max(
        node.end_lineno or node.lineno
        for node in ast.parse(source).body
        if isinstance(node, ast.Import | ast.ImportFrom)
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[:last]) + imports + "".join(lines[last:])


def _indent(block: str) -> str:
    return "".join("    " + line if line.strip() else line for line in block.splitlines(True))
