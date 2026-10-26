"""Session read tools never cross a resource-access tenant boundary."""

from __future__ import annotations

import asyncio
import base64
import json
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from pydantic import SecretStr

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu.messages import Message, MessageRole, StructuredOutputPart, ToolResultPart
from cayu.resource_access import (
    ResourceAccessPolicy,
    _runtime_owned_store_access,
    encode_scope,
    execution_access,
    model_data_access,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.access import (
    SessionAccessDenied,
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
)
from cayu.sessions.queries import SessionOrder, encode_session_cursor
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore
from cayu.storage import SQLiteSessionStore
from cayu.tools import (
    ListSessionsTool,
    MatchingSessionAccess,
    OwnTreeSessionAccess,
    ReadSessionTool,
    SessionReadAccess,
)
from cayu.tools.base import ToolContext, ToolResult

ORG = "organization"
# Every foreign session id, title, label value and transcript carries this marker.
FOREIGN = "globex"
_IDENTITY = SessionIdentity(provider_name="fake", model="fake")
_CODEC = PublicAuthorityAliasCodec(
    PublicAuthorityAliasKeyring(
        active_key_id="tenancy",
        keys={"tenancy": SecretStr(base64.urlsafe_b64encode(bytes([9]) * 32).decode().rstrip("="))},
    )
)
# Every raw session id these tests create; none may appear in any tool output.
_RAW_IDS: set[str] = set()


async def _alias(store: SessionStore, caller: str, session_id: str) -> str:
    """The alias a caller would hold for a session, registered as if it had been shown."""

    alias = _CODEC.encode(session_id, field_name="session_ref", session_id=caller)
    async with _runtime_owned_store_access():
        await store.register_public_authority_alias(
            alias, field_name="session_ref", private_value=session_id, scope_session_id=caller
        )
    return alias


async def _raw(store: SessionStore, caller: str, alias: str) -> str:
    async with _runtime_owned_store_access():
        resolved = await store.resolve_public_authority_alias(
            alias, field_name="session_ref", scope_session_id=caller
        )
    assert resolved is not None
    return resolved


def _scope(org: str) -> SessionAccessScope:
    rule = SessionAccessRule(selectors=(SessionAccessSelector(ORG, values=(org,)),))
    return SessionAccessScope(
        read=(rule,),
        execute=(rule,),
        create=(rule,),
        update_labels=(rule,),
        protected_label_keys=(ORG,),
    )


@asynccontextmanager
async def _as_org(org: str) -> AsyncIterator[None]:
    scope = _scope(org)

    class Policy(ResourceAccessPolicy):
        authority = "session-read-tenancy"

        async def resolve(self, subject):
            return scope

    policy = Policy()
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject=org, admitted_json=encode_scope(scope)
    )
    async with execution_access(binding, policy, {ORG: org}), model_data_access():
        yield


class _AdmitAll:
    """A policy that relies entirely on the run's scope."""

    async def narrow(self, caller, query):
        return query

    async def allows(self, caller, session):
        return session.id != caller.id


def _policies(world: World) -> dict[str, SessionReadAccess]:
    return {
        "admit_all": _AdmitAll(),
        "agents": MatchingSessionAccess(agent_names=["scout", "hunter", "filter"]),
        "foreign_label": MatchingSessionAccess(labels={ORG: world.foreign_org}),
        "own_label": MatchingSessionAccess(labels={ORG: world.own_org}),
        "own_tree": OwnTreeSessionAccess(),
    }


@dataclass
class World:
    prefix: str
    own_org: str
    foreign_org: str
    caller: str
    own: list[Session] = field(default_factory=list)
    foreign: list[Session] = field(default_factory=list)


async def _session(
    store: SessionStore,
    session_id: str,
    *,
    org: str,
    agent: str,
    parent: str | None = None,
    team: str = "red",
    title: str | None = None,
    messages: list[Message] | None = None,
) -> Session:
    session = await store.create(
        RunRequest(
            agent_name=agent,
            session_id=session_id,
            parent_session_id=parent,
            labels={ORG: org, "team": team},
            title=title,
            messages=[],
        ),
        identity=_IDENTITY,
    )
    if messages:
        await store.append_transcript_messages(session_id, messages)
    _RAW_IDS.add(session_id)
    return session


def _answer(text: str) -> list[Message]:
    return [
        Message.text("user", f"Investigate {text}"),
        Message(
            role=MessageRole.TOOL,
            content=(
                ToolResultPart(tool_call_id="call-1", tool_name="submit", content=text),
                StructuredOutputPart(name="Report", output={"finding": text}),
            ),
        ),
        Message.text("assistant", f"Finished {text}"),
    ]


async def _world(store: SessionStore) -> World:
    prefix = "t" + uuid4().hex[:10]
    world = World(
        prefix=prefix,
        own_org=f"{prefix}-acme",
        foreign_org=f"{prefix}-{FOREIGN}",
        caller=f"{prefix}-acme-caller",
    )
    caller = await _session(store, world.caller, org=world.own_org, agent="scout")
    for index in range(3):
        world.own.append(
            await _session(
                store,
                f"{prefix}-acme-{index}",
                org=world.own_org,
                agent="hunter",
                parent=caller.id,
                team="red" if index % 2 else "blue",
                title=f"acme hunt {index}",
                messages=_answer(f"acme finding {index}"),
            )
        )
    world.own.append(
        await _session(
            store,
            f"{prefix}-acme-child",
            org=world.own_org,
            agent="filter",
            parent=world.own[0].id,
            title="acme child",
        )
    )
    # More, newer foreign sessions share agents, teams, titles and tree shape.
    foreign_root = await _session(
        store, f"{prefix}-{FOREIGN}-root", org=world.foreign_org, agent="scout"
    )
    world.foreign.append(foreign_root)
    # An own session whose parent belongs to the other organization.
    world.own.append(
        await _session(
            store,
            f"{prefix}-acme-orphan",
            org=world.own_org,
            agent="hunter",
            parent=foreign_root.id,
            title="acme orphan",
        )
    )
    for index in range(8):
        world.foreign.append(
            await _session(
                store,
                f"{prefix}-{FOREIGN}-{index}",
                org=world.foreign_org,
                agent="hunter",
                parent=foreign_root.id,
                team="red" if index % 2 else "blue",
                title=f"{FOREIGN} hunt {index} acme",
                messages=_answer(f"{FOREIGN} finding {index}"),
            )
        )
    return world


def _ctx(session_id: str) -> ToolContext:
    return ToolContext(session_id=session_id)


def _assert_clean(result: ToolResult) -> None:
    published = result.content + json.dumps(
        None if result.structured is None else dict(result.structured), default=str
    )
    assert FOREIGN not in published
    assert not [raw for raw in _RAW_IDS if raw in published]


async def _row_ids(store: SessionStore, caller: str, result: ToolResult) -> list[str]:
    assert result.structured is not None
    return [
        await _raw(store, caller, row["session_id"])
        for row in result.structured.get("sessions", [])
    ]


async def _list_all(
    tool: ListSessionsTool, store: SessionStore, caller: str, **args: object
) -> list[str]:
    seen: list[str] = []
    cursor = None
    for _ in range(20):
        call: dict[str, object] = {"limit": 1, **args}
        if cursor is not None:
            call["cursor"] = cursor
        result = await tool.run(_ctx(caller), call)
        _assert_clean(result)
        assert result.structured is not None and not result.is_error, result.content
        assert result.structured["scan_limited"] is False
        seen.extend(await _row_ids(store, caller, result))
        cursor = result.structured["next_cursor"]
        if cursor is None:
            return seen
    raise AssertionError("listing did not finish")


def _run(
    backend: str, sqlite_resources, scenario: Callable[[SessionStore], Awaitable[None]]
) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            if backend == "memory":
                await scenario(InMemorySessionStore(public_authority_alias_codec=_CODEC))
            else:
                await scenario(
                    resources.own(
                        SQLiteSessionStore(resources.path(), public_authority_alias_codec=_CODEC)
                    )
                )

    asyncio.run(run())


async def _check_every_path_stays_in_the_callers_organization(store: SessionStore) -> None:
    world = await _world(store)
    own_ids = {session.id for session in world.own}
    orphan = world.own[-1]
    child = world.own[3]
    foreign = world.foreign[1]
    async with _as_org(world.own_org):
        for name, access in _policies(world).items():
            list_tool = ListSessionsTool(store, access=access)
            read_tool = ReadSessionTool(store, access=access)
            listed = await _list_all(list_tool, store, world.caller)
            expected = (
                set()
                if name == "foreign_label"
                else own_ids - {orphan.id}
                if name == "own_tree"
                else own_ids
            )
            assert set(listed) == expected, name

            denied = await read_tool.run(
                _ctx(world.caller), {"session_id": await _alias(store, world.caller, foreign.id)}
            )
            missing = await read_tool.run(
                _ctx(world.caller),
                {"session_id": await _alias(store, world.caller, f"{world.prefix}-missing")},
            )
            genuine = _CODEC.encode(
                world.own[0].id, field_name="session_ref", session_id=world.caller
            )
            # The tag's last base64 character carries 4 bits, so it is already
            # "A" one time in 16; swap in a different canonical final character.
            forged = await read_tool.run(
                _ctx(world.caller),
                {"session_id": genuine[:-1] + ("E" if genuine.endswith("A") else "A")},
            )
            other_caller = await read_tool.run(
                _ctx(world.caller),
                {"session_id": await _alias(store, world.own[1].id, world.own[0].id)},
            )
            for result in (forged, other_caller):
                _assert_clean(result)
                assert result.structured == {"error": "session_not_found"}
            _assert_clean(denied)
            assert denied.is_error and denied.content == missing.content
            assert denied.structured == missing.structured == {"error": "session_not_found"}
            assert await read_tool._final_structured_output(foreign.id) == ("unavailable", None)

            if name != "foreign_label":
                own_alias = await _alias(store, world.caller, world.own[0].id)
                own = await read_tool.run(_ctx(world.caller), {"session_id": own_alias})
                _assert_clean(own)
                assert own.structured is not None
                outcome = own.structured["session"]["outcome"]
                assert outcome["structured_output_status"] == "found"
                assert "acme finding 0" in own.content
                # The caller itself is hidden by every policy here, so it reads as no parent.
                assert own.structured["session"]["parent_session_id"] is None
                child_read = await read_tool.run(
                    _ctx(world.caller),
                    {"session_id": await _alias(store, world.caller, child.id)},
                )
                _assert_clean(child_read)
                assert child_read.structured is not None
                assert child_read.structured["session"]["parent_session_id"] == own_alias

                rows = await list_tool.run(_ctx(world.caller), {"limit": 50})
                _assert_clean(rows)
                assert rows.structured is not None
                parents = {
                    await _raw(store, world.caller, row["session_id"]): row["parent_session_id"]
                    for row in rows.structured["sessions"]
                }
                assert parents[child.id] == own_alias
                assert parents[world.own[0].id] is None
                if orphan.id in parents:
                    # A parent in another organization is indistinguishable from none.
                    assert parents[orphan.id] is None

        with pytest.raises(SessionAccessDenied):
            await store.load_bounded_transcript_window(
                foreign.id, start_index=0, limit=5, max_record_bytes=4096, max_total_bytes=4096
            )
        window = await store.load_bounded_transcript_window(
            world.own[0].id, start_index=0, limit=5, max_record_bytes=4096, max_total_bytes=4096
        )
        assert len(window.records) == 3
        assert await store.load(foreign.id) is None


async def _check_model_arguments_cannot_reach_another_organization(store: SessionStore) -> None:
    world = await _world(store)
    own_ids = {session.id for session in world.own}
    foreign = world.foreign[2]
    async with _as_org(world.own_org):
        list_tool = ListSessionsTool(store, access=_AdmitAll())
        read_tool = ReadSessionTool(store, access=_AdmitAll())
        for labels in ({ORG: world.foreign_org}, {ORG: world.own_org}):
            refused = await list_tool.run(_ctx(world.caller), {"labels": labels})
            _assert_clean(refused)
            assert refused.is_error and refused.structured is not None
            assert refused.structured["error"] == "invalid_arguments"
        for args in (
            {"labels": {"team": "red"}},
            {"text": FOREIGN},
            {"text": "hunt"},
            {"text": world.own[0].id},
            {"agent_name": "hunter"},
            {"agent_name": "scout"},
        ):
            result = await list_tool.run(_ctx(world.caller), {"limit": 50, **args})
            _assert_clean(result)
            assert not result.is_error
            assert set(await _row_ids(store, world.caller, result)) <= own_ids
        assert await _list_all(list_tool, store, world.caller, text=FOREIGN, limit=50) == []
        assert await _list_all(list_tool, store, world.caller, text=world.own[0].id, limit=50) == []
        for cursor in (
            encode_session_cursor(foreign, SessionOrder.CREATED_AT_DESC),
            encode_session_cursor(world.own[0], SessionOrder.CREATED_AT_DESC),
            "after." + await _alias(store, world.caller, foreign.id),
            "after." + _CODEC.encode(foreign.id, field_name="session_ref", session_id=world.caller),
            "after." + world.own[0].id,
        ):
            forged = await list_tool.run(_ctx(world.caller), {"cursor": cursor})
            _assert_clean(forged)
            assert forged.is_error and forged.structured is not None
            assert forged.structured["error"] == "invalid_arguments"
        for args in (
            {"session_id": await _alias(store, world.caller, foreign.id)},
            {"session_id": await _alias(store, world.caller, foreign.id), "offset": 1},
            {"session_id": await _alias(store, world.caller, world.foreign[0].id), "offset": 2**40},
            {"session_id": foreign.id},
            {"session_id": world.own[1].id},
            {"session_id": foreign.id.upper()},
        ):
            result = await read_tool.run(_ctx(world.caller), args)
            _assert_clean(result)
            assert result.structured == {"error": "session_not_found"}
        offset = await read_tool.run(
            _ctx(world.caller),
            {
                "session_id": await _alias(store, world.caller, world.own[1].id),
                "offset": 2,
                "limit": 5,
            },
        )
        _assert_clean(offset)
        assert offset.structured is not None
        assert [message["index"] for message in offset.structured["messages"]] == [2]


async def _check_runs_cannot_move_sessions_into_another_organization(store: SessionStore) -> None:
    world = await _world(store)
    own = world.own[0]
    async with _as_org(world.own_org):
        with pytest.raises(SessionAccessDenied):
            await store.update_labels(own.id, {ORG: world.foreign_org, "team": "red"})
        with pytest.raises(SessionAccessDenied):
            await store.update_labels(own.id, {"team": "red"})
        with pytest.raises(SessionAccessDenied):
            await store.create(
                RunRequest(
                    agent_name="hunter",
                    session_id=f"{world.prefix}-moved",
                    labels={ORG: world.foreign_org},
                    messages=[],
                ),
                identity=_IDENTITY,
            )
        relabeled = await store.update_labels(own.id, {ORG: world.own_org, "team": "green"})
        assert relabeled.labels == {ORG: world.own_org, "team": "green"}
    loaded = await store.load(own.id)
    assert loaded is not None and loaded.labels[ORG] == world.own_org
    assert await store.load(f"{world.prefix}-moved") is None


async def _check_randomized_reads_never_show_another_organization(
    store: SessionStore, *, seed: int = 2761, operations: int = 60
) -> None:
    rng = random.Random(seed)
    prefix = "r" + uuid4().hex[:10]
    orgs = {"own": f"{prefix}-acme", FOREIGN: f"{prefix}-{FOREIGN}-a", "b": f"{prefix}-{FOREIGN}-b"}
    caller = f"{prefix}-acme-caller"
    await _session(store, caller, org=orgs["own"], agent="scout")
    words = ["alpha", "beta", "gamma", "acme", "hunt"]
    queries = [*words, FOREIGN]
    agents = ["scout", "hunter", "filter"]
    teams = ["red", "blue", "green"]
    by_org: dict[str, list[Session]] = {key: [] for key in orgs}
    for index in range(30):
        key = rng.choice(list(orgs))
        marker = "acme" if key == "own" else FOREIGN
        session_id = f"{prefix}-{marker}-{key}-{index}"
        messages = (
            _answer(f"{marker} result {index}")
            if rng.random() < 0.5
            else [Message.text("assistant", f"{marker} note {index} {rng.choice(words)}")]
        )
        by_org[key].append(
            await _session(
                store,
                session_id,
                org=orgs[key],
                agent=rng.choice(agents),
                team=rng.choice(teams),
                title=f"{marker} {rng.choice(words)} {index}",
                messages=messages,
            )
        )
    own_ids = {session.id for session in by_org["own"]}
    every = [session for sessions in by_org.values() for session in sessions]
    foreign_sessions = [session for session in every if session.id not in own_ids]
    policies: list[SessionReadAccess] = [
        _AdmitAll(),
        MatchingSessionAccess(agent_names=agents),
        MatchingSessionAccess(labels={"team": "red"}),
    ]
    async with _as_org(orgs["own"]):
        assert (
            set(await _list_all(ListSessionsTool(store, access=_AdmitAll()), store, caller))
            == own_ids
        )
        for _ in range(operations):
            access = rng.choice(policies)
            if rng.random() < 0.6:
                args: dict[str, object] = {"limit": rng.randint(1, 4)}
                if rng.random() < 0.4:
                    args["agent_name"] = rng.choice(agents)
                if rng.random() < 0.4:
                    args["labels"] = {"team": rng.choice(teams)}
                if rng.random() < 0.4:
                    args["text"] = rng.choice(queries)
                if rng.random() < 0.3 and foreign_sessions:
                    forged = rng.choice(foreign_sessions)
                    args["cursor"] = rng.choice(
                        [
                            encode_session_cursor(forged, SessionOrder.CREATED_AT_DESC),
                            "after." + await _alias(store, caller, forged.id),
                        ]
                    )
                tool = ListSessionsTool(store, access=access)
                for _page in range(3):
                    result = await tool.run(_ctx(caller), args)
                    _assert_clean(result)
                    if result.is_error:
                        break
                    assert set(await _row_ids(store, caller, result)) <= own_ids
                    assert result.structured is not None
                    cursor = result.structured.get("next_cursor")
                    if cursor is None:
                        break
                    args["cursor"] = cursor
            else:
                target = rng.choice(every)
                args = {
                    "session_id": await _alias(store, caller, target.id),
                    "offset": rng.randint(0, 3),
                    "limit": rng.randint(1, 3),
                    "max_message_chars": rng.randint(5, 200),
                }
                result = await ReadSessionTool(store, access=access).run(_ctx(caller), args)
                _assert_clean(result)
                if target.id not in own_ids:
                    assert result.structured == {"error": "session_not_found"}


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "check",
    [
        _check_every_path_stays_in_the_callers_organization,
        _check_model_arguments_cannot_reach_another_organization,
        _check_runs_cannot_move_sessions_into_another_organization,
        _check_randomized_reads_never_show_another_organization,
    ],
    ids=["paths", "arguments", "relabel", "randomized"],
)
def test_session_reads_stay_inside_the_callers_organization(
    backend: str, check, sqlite_resources
) -> None:
    _run(backend, sqlite_resources, check)


def test_postgres_session_reads_stay_inside_the_callers_organization(postgres_dsn) -> None:
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run() -> None:
        store = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_CODEC
        )
        try:
            await _check_every_path_stays_in_the_callers_organization(store)
            await _check_model_arguments_cannot_reach_another_organization(store)
            await _check_runs_cannot_move_sessions_into_another_organization(store)
            await _check_randomized_reads_never_show_another_organization(
                store, seed=27610, operations=30
            )
        finally:
            await store.close()

    asyncio.run(run())


def test_tools_refuse_to_run_without_a_resource_access_scope() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore(public_authority_alias_codec=_CODEC)
        world = await _world(store)
        for tool, args in (
            (ListSessionsTool(store, access=_AdmitAll()), {}),
            (
                ReadSessionTool(store, access=_AdmitAll()),
                {"session_id": await _alias(store, world.caller, world.own[0].id)},
            ),
        ):
            refused = await tool.run(_ctx(world.caller), args)
            assert refused.is_error
            assert refused.structured == {"error": "resource_scope_required"}
            assert "unscoped=True" in refused.content
            assert "single-tenant" in refused.content
        opted_out = ListSessionsTool(store, access=OwnTreeSessionAccess(), unscoped=True)
        result = await opted_out.run(_ctx(world.caller), {})
        assert set(await _row_ids(store, world.caller, result)) == {
            session.id for session in world.own[:-1]
        }

        class UnaliasedStore(InMemorySessionStore):
            @property
            def public_authority_alias_codec(self):  # ty: ignore[invalid-method-override]
                return None

        with pytest.raises(ValueError, match="alias keyring"):
            ListSessionsTool(UnaliasedStore(), access=_AdmitAll(), unscoped=True)
        with pytest.raises(TypeError, match="unscoped"):
            ReadSessionTool(store, access=_AdmitAll(), unscoped="yes")  # ty: ignore[invalid-argument-type]

    asyncio.run(scenario())
