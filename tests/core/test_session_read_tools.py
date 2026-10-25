"""Scoped, read-only session tools for agents."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable

import pytest
from pydantic import SecretStr

from cayu.messages import (
    Message,
    MessageRole,
    StructuredOutputPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.resource_access import _runtime_owned_store_access
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import InMemorySessionStore, RunRequest, SessionIdentity, SessionQuery
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
from cayu.tools.base import ToolContext, ToolEffect, ToolResult
from cayu.vaults import SecretRedactor

_IDENTITY = SessionIdentity(provider_name="fake", model="fake")
_CODEC = PublicAuthorityAliasCodec(
    PublicAuthorityAliasKeyring(
        active_key_id="session-reads",
        keys={
            "session-reads": SecretStr(
                base64.urlsafe_b64encode(bytes([7]) * 32).decode().rstrip("=")
            )
        },
    )
)


def _memory_store() -> InMemorySessionStore:
    return InMemorySessionStore(public_authority_alias_codec=_CODEC)


async def _alias(store: SessionStore, caller: str, session_id: str) -> str:
    """The alias a caller holds for a session after the tools have shown it."""

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


async def _session(
    store: SessionStore,
    session_id: str,
    *,
    agent: str = "assistant",
    parent: str | None = None,
    labels: dict[str, str] | None = None,
    title: str | None = None,
    messages: list[Message] | None = None,
) -> Session:
    session = await store.create(
        RunRequest(
            agent_name=agent,
            session_id=session_id,
            parent_session_id=parent,
            labels=labels or {},
            title=title,
            messages=[],
        ),
        identity=_IDENTITY,
    )
    if messages:
        await store.append_transcript_messages(session_id, messages)
    return session


def _ctx(session_id: str, *secrets: str) -> ToolContext:
    redactor = SecretRedactor(list(secrets))
    return ToolContext(session_id=session_id, invocation_secret_redactor=lambda: redactor)


async def _listed(
    store: SessionStore, access: SessionReadAccess, caller: str, **args: object
) -> list[str]:
    result = await ListSessionsTool(store, unscoped=True, access=access).run(
        _ctx(caller), dict(args)
    )
    assert not result.is_error
    assert result.structured is not None
    return [await _raw(store, caller, row["session_id"]) for row in result.structured["sessions"]]


async def _read(
    store: SessionStore, access: SessionReadAccess, caller: str, **args: object
) -> ToolResult:
    if type(args.get("session_id")) is str:
        args["session_id"] = await _alias(store, caller, str(args["session_id"]))
    return await ReadSessionTool(store, unscoped=True, access=access).run(_ctx(caller), dict(args))


def _run_with_store(
    backend: str,
    sqlite_resources,
    scenario: Callable[[SessionStore], Awaitable[None]],
) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            if backend == "memory":
                await scenario(_memory_store())
            else:
                await scenario(
                    resources.own(
                        SQLiteSessionStore(resources.path(), public_authority_alias_codec=_CODEC)
                    )
                )

    asyncio.run(run())


def test_tools_need_an_explicit_read_only_policy() -> None:
    store = _memory_store()
    with pytest.raises(TypeError, match="SessionReadAccess"):
        ListSessionsTool(store, access=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="SessionReadAccess"):
        ReadSessionTool(store, access=object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="agent_names or labels"):
        MatchingSessionAccess(exclude_labels={"role": "hunter"})
    with pytest.raises(TypeError, match="collection"):
        MatchingSessionAccess(agent_names="hunter")
    list_tool = ListSessionsTool(store, access=OwnTreeSessionAccess())
    read_tool = ReadSessionTool(store, access=OwnTreeSessionAccess())
    assert list_tool.spec.name == "list_sessions"
    assert read_tool.spec.name == "read_session"
    assert list_tool.spec.effect is ToolEffect.NONE
    assert read_tool.spec.effect is ToolEffect.NONE


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_tools_reject_stores_that_cannot_register_aliases(tmp_path, backend, diagnostic) -> None:
    from contextlib import nullcontext

    from cayu.storage import PostgresSessionStore
    from cayu.storage._diagnostic_inspection import diagnostic_store_inspection
    from cayu.storage.migrations import SchemaMode

    async def scenario() -> None:
        path = tmp_path / "aliases.sqlite"
        if backend == "sqlite":
            writer = SQLiteSessionStore(path, public_authority_alias_codec=_CODEC)
            assert writer.can_register_public_authority_aliases
            await writer.close()
        with diagnostic_store_inspection() if diagnostic else nullcontext():
            store = (
                SQLiteSessionStore(
                    path,
                    schema_mode=SchemaMode.VALIDATE,
                    read_only=not diagnostic,
                    public_authority_alias_codec=_CODEC,
                )
                if backend == "sqlite"
                else PostgresSessionStore(
                    "postgresql://localhost/unused",
                    schema_mode=SchemaMode.VALIDATE,
                    read_only=not diagnostic,
                    public_authority_alias_codec=_CODEC,
                )
            )
            try:
                assert not store.can_register_public_authority_aliases
                for tool_type in (ListSessionsTool, ReadSessionTool):
                    with pytest.raises(ValueError, match="writable session store.*alias"):
                        tool_type(store, access=OwnTreeSessionAccess())
            finally:
                await store.close()

    asyncio.run(scenario())


def test_a_policy_that_admits_nothing_denies_everything() -> None:
    class DenyAll:
        async def narrow(self, caller, query):
            return None

        async def allows(self, caller, session):
            return False

    async def scenario() -> None:
        store = _memory_store()
        await _session(store, "caller")
        await _session(store, "other")
        assert await _listed(store, DenyAll(), "caller") == []
        result = await _read(store, DenyAll(), "caller", session_id="other")
        assert result.is_error
        assert result.structured == {"error": "session_not_found"}

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_own_tree_scope_covers_the_root_tree_and_hides_the_caller(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root", agent="scout")
        await _session(store, "hunter", agent="hunter", parent="root")
        await _session(store, "filter", agent="filter", parent="hunter")
        await _session(store, "elsewhere", agent="hunter")
        await _session(store, "elsewhere-child", agent="filter", parent="elsewhere")

        access = OwnTreeSessionAccess()
        assert sorted(await _listed(store, access, "hunter")) == ["filter", "root"]
        assert sorted(await _listed(store, access, "filter")) == ["hunter", "root"]
        assert sorted(await _listed(store, OwnTreeSessionAccess(include_caller=True), "root")) == [
            "filter",
            "hunter",
            "root",
        ]
        assert await _listed(store, access, "hunter", agent_name="filter") == ["filter"]
        assert await _listed(store, access, "elsewhere") == ["elsewhere-child"]

        visible = await _read(store, access, "hunter", session_id="filter")
        assert not visible.is_error
        assert visible.structured is not None
        alias = visible.structured["session"]["session_id"]
        assert await _raw(store, "hunter", alias) == "filter"
        assert "filter" not in alias
        for hidden in ("elsewhere", "hunter"):
            result = await _read(store, access, "hunter", session_id=hidden)
            assert result.is_error
            assert result.structured == {"error": "session_not_found"}

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_matching_scope_filters_by_agents_and_labels_and_excludes_labels(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "hunt-1", agent="hunter", labels={"app": "scout", "role": "hunter"})
        await _session(store, "hunt-2", agent="hunter", labels={"app": "scout", "role": "hunter"})
        await _session(store, "triage", agent="triager", labels={"app": "scout", "role": "triage"})
        await _session(store, "verify", agent="verifier", labels={"app": "scout", "role": "verify"})
        await _session(store, "verify-old", agent="verifier", labels={"app": "scout"})
        await _session(store, "foreign", agent="hunter", labels={"app": "other"})

        verifier = MatchingSessionAccess(labels={"app": "scout"}, exclude_labels={"role": "hunter"})
        assert sorted(await _listed(store, verifier, "verify")) == ["triage", "verify-old"]
        assert await _listed(store, verifier, "verify", labels={"role": "hunter"}) == []
        assert await _listed(store, verifier, "verify", labels={"app": "other"}) == []
        hidden = await _read(store, verifier, "verify", session_id="hunt-1")
        assert hidden.structured == {"error": "session_not_found"}

        hunters = MatchingSessionAccess(agent_names=["hunter", "triager"], labels={"app": "scout"})
        assert sorted(await _listed(store, hunters, "hunt-1")) == ["hunt-2", "triage"]
        assert await _listed(store, hunters, "hunt-1", agent_name="triager") == ["triage"]
        assert await _listed(store, hunters, "hunt-1", agent_name="verifier") == []
        assert sorted(
            await _listed(store, MatchingSessionAccess(agent_names=["hunter"]), "verify")
        ) == ["foreign", "hunt-1", "hunt-2"]
        mine = MatchingSessionAccess(agent_names=["hunter"], include_caller=True)
        assert "hunt-1" in await _listed(store, mine, "hunt-1")
        readable = await _read(store, mine, "hunt-1", session_id="hunt-1")
        assert not readable.is_error

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_sessions_outside_the_policy_look_exactly_like_missing_ones(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "caller", labels={"team": "a"})
        await _session(store, "secret-plan", labels={"team": "b"}, title="Secret plan")
        access = MatchingSessionAccess(labels={"team": "a"})

        denied = await _read(store, access, "caller", session_id="secret-plan")
        missing = await _read(store, access, "caller", session_id="no-such-session")
        assert denied.is_error and missing.is_error
        assert denied.content == missing.content
        assert denied.structured is not None and missing.structured is not None
        assert {**denied.structured, "session_id": "x"} == {**missing.structured, "session_id": "x"}
        assert await _listed(store, access, "caller", text="Secret") == []

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_listing_pages_with_a_cursor_and_filters_by_time_status_and_title(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        caller = await _session(store, "root", title="Coordinator")
        for index in range(5):
            await _session(store, f"hunt-{index}", parent="root", title=f"Hunt module {index}")
        tool = ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess())
        seen: list[str] = []
        cursor = None
        pages = 0
        while True:
            args: dict[str, object] = {"limit": 2, "max_outcome_chars": 0}
            if cursor is not None:
                args["cursor"] = cursor
            result = await tool.run(_ctx("root"), args)
            assert result.structured is not None
            rows = result.structured["sessions"]
            assert len(rows) <= 2
            assert all("outcome" not in row and "summary" not in row for row in rows)
            seen.extend([await _raw(store, "root", row["session_id"]) for row in rows])
            cursor = result.structured["next_cursor"]
            pages += 1
            if cursor is None:
                break
            assert f'cursor="{cursor}"' in result.content
        assert pages == 3
        assert sorted(seen) == [f"hunt-{index}" for index in range(5)]
        assert seen == sorted(seen, reverse=True)

        assert await _listed(store, OwnTreeSessionAccess(), "root", text="MODULE 3") == ["hunt-3"]
        assert await _listed(store, OwnTreeSessionAccess(), "root", status="running") == []
        assert len(await _listed(store, OwnTreeSessionAccess(), "root", status="pending")) == 5
        later = (caller.created_at.replace(year=caller.created_at.year + 1)).isoformat()
        assert await _listed(store, OwnTreeSessionAccess(), "root", created_after=later) == []

        invalid = await tool.run(_ctx("root"), {"cursor": "not-a-cursor"})
        assert invalid.is_error and invalid.structured is not None
        assert invalid.structured["error"] == "invalid_arguments"
        naive = await tool.run(_ctx("root"), {"created_after": "2026-01-01T00:00:00"})
        assert naive.is_error
        unknown = await tool.run(_ctx("root"), {"secret_field_name": 1})
        assert unknown.is_error and "secret_field_name" not in unknown.content

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_read_session_pages_the_transcript_and_caps_each_message(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root")
        messages = [Message.text("user", "Find the overflow in parser.c")]
        for index in range(4):
            messages.append(Message.text("assistant", f"Step {index}: " + "x" * 50))
        await _session(store, "hunt", parent="root", messages=messages)
        access = OwnTreeSessionAccess()

        first = await _read(store, access, "root", session_id="hunt", limit=2)
        assert first.structured is not None
        assert [m["index"] for m in first.structured["messages"]] == [0, 1]
        assert first.structured["total_messages"] == 5
        assert first.structured["next_offset"] == 2
        assert "offset=2" in first.content
        last = await _read(store, access, "root", session_id="hunt", offset=4, limit=2)
        assert last.structured is not None
        assert [m["index"] for m in last.structured["messages"]] == [4]
        assert last.structured["next_offset"] is None
        beyond = await _read(store, access, "root", session_id="hunt", offset=9)
        assert beyond.structured is not None and beyond.structured["messages"] == []

        capped = await _read(store, access, "root", session_id="hunt", max_message_chars=10)
        assert capped.structured is not None
        message = capped.structured["messages"][1]
        assert message["parts"] == [{"type": "text", "text": "Step 0: xx"}]
        assert message["truncated"] is True
        outcome = capped.structured["session"]["outcome"]
        assert outcome["text"] == "Step 3: xx" and outcome["text_truncated"] is True

        listed = await ListSessionsTool(store, unscoped=True, access=access).run(
            _ctx("root"), {"max_outcome_chars": 6}
        )
        assert listed.structured is not None
        row = listed.structured["sessions"][0]
        assert row["outcome"] == {"available": True, "text": "Step 3", "text_truncated": True}

        invalid = await _read(store, access, "root", session_id="hunt", limit=51)
        assert invalid.is_error and invalid.structured is not None
        assert invalid.structured["error"] == "invalid_arguments"

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_read_session_masks_secrets_and_shows_tool_activity(backend: str, sqlite_resources) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root", title="Deploy with tok-12345")
        await _session(store, "worker", parent="root")
        await store.append_transcript_messages(
            "worker",
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        ThinkingPart(text="private reasoning"),
                        TextPart(text="Using token tok-12345 now."),
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="deploy",
                            arguments={"token": "tok-12345"},
                        ),
                        ToolCallPart(
                            tool_call_id="call-2",
                            tool_name="deploy",
                            arguments_state="unavailable",
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.TOOL,
                    content=(
                        ToolResultPart(
                            tool_call_id="call-1",
                            tool_name="deploy",
                            content="deployed with tok-12345",
                            structured={"token": "tok-12345"},
                            is_error=True,
                        ),
                        StructuredOutputPart(name="Report", output={"status": "done"}),
                    ),
                ),
            ],
        )
        tool = ReadSessionTool(store, unscoped=True, access=OwnTreeSessionAccess())
        result = await tool.run(
            _ctx("worker", "tok-12345"), {"session_id": await _alias(store, "worker", "root")}
        )
        assert result.structured is not None
        assert "tok-12345" not in result.content
        assert "tok-12345" not in str(result.structured)
        assert result.structured["session"]["title"] != "Deploy with tok-12345"

        read = await tool.run(
            _ctx("root", "tok-12345"), {"session_id": await _alias(store, "root", "worker")}
        )
        assert read.structured is not None
        assert "tok-12345" not in read.content
        assert "tok-12345" not in str(read.structured)
        assert "private reasoning" not in read.content
        assistant = read.structured["messages"][0]
        assert assistant["omitted_parts"] == 1
        assert [part["type"] for part in assistant["parts"]] == ["text", "tool_call", "tool_call"]
        assert assistant["parts"][2]["arguments"] is None
        assert assistant["parts"][2]["arguments_state"] == "unavailable"
        tool_message = read.structured["messages"][1]
        assert tool_message["parts"][0]["is_error"] is True
        assert tool_message["parts"][1] == {
            "type": "structured_output",
            "name": "Report",
            "output": '{"status":"done"}',
        }
        outcome = read.structured["session"]["outcome"]
        assert outcome["structured_output_name"] == "Report"
        assert outcome["structured_output"] == '{"status":"done"}'
        assert "tool_call deploy (arguments unavailable)" in read.content

        listed = await ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess()).run(
            _ctx("root", "tok-12345"), {}
        )
        assert listed.structured is not None
        assert (
            listed.structured["sessions"][0]["outcome"]["text"]
            == "Using token [REDACTED_SECRET] now."
        )
        assert "structured_output" not in listed.structured["sessions"][0]["outcome"]

    _run_with_store(backend, sqlite_resources, scenario)


class _LabelledOnly:
    """Admits everything when narrowing, but only labelled sessions when deciding."""

    async def narrow(self, caller, query):
        return query

    async def allows(self, caller, session):
        return session.id != caller.id and session.labels.get("visible") == "yes"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_broad_narrow_never_exposes_hidden_sessions_through_rows_or_cursors(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "caller")
        for index in range(3):
            await _session(store, f"shown-{index}", labels={"visible": "yes"})
            await _session(store, f"hidden-{index}")
        tool = ListSessionsTool(store, unscoped=True, access=_LabelledOnly())
        seen: list[str] = []
        cursor = None
        while True:
            args: dict[str, object] = {"limit": 1, "max_outcome_chars": 0}
            if cursor is not None:
                args["cursor"] = cursor
            result = await tool.run(_ctx("caller"), args)
            assert result.structured is not None
            assert "hidden" not in result.content
            assert result.structured["scan_limited"] is False
            aliases = [row["session_id"] for row in result.structured["sessions"]]
            seen.extend([await _raw(store, "caller", alias) for alias in aliases])
            cursor = result.structured["next_cursor"]
            if cursor is None:
                break
            assert cursor == "after." + aliases[-1]
        assert seen == ["shown-2", "shown-1", "shown-0"]

        for index in range(3, 3 + 6):
            await _session(store, f"hidden-{index}")
        stalled = await tool.run(_ctx("caller"), {"limit": 1, "max_outcome_chars": 0})
        assert stalled.structured is not None
        assert stalled.structured["sessions"] == []
        assert stalled.structured["next_cursor"] is None
        assert stalled.structured["scan_limited"] is True
        assert "hidden" not in stalled.content and "narrow the filters" in stalled.content

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_structured_output_search_stays_within_its_message_and_byte_limits(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root")
        oversized = Message(
            role=MessageRole.TOOL,
            content=(
                ToolResultPart(tool_call_id="call-1", tool_name="submit", content="ok"),
                StructuredOutputPart(name="Report", output={"body": "y" * (600 * 1024)}),
            ),
        )
        await _session(store, "large", parent="root", messages=[oversized])
        early = Message(
            role=MessageRole.TOOL,
            content=(
                ToolResultPart(tool_call_id="call-1", tool_name="submit", content="ok"),
                StructuredOutputPart(name="Report", output={"status": "early"}),
            ),
        )
        chatter = [Message.text("assistant", f"note {index}") for index in range(8)]
        await _session(store, "buried", parent="root", messages=[early, *chatter])
        recent = [early, *chatter[:7]]
        await _session(store, "recent", parent="root", messages=recent)
        access = OwnTreeSessionAccess()

        large = await _read(store, access, "root", session_id="large", max_message_chars=100)
        assert large.structured is not None
        outcome = large.structured["session"]["outcome"]
        assert outcome["structured_output_status"] == "too_large"
        assert "structured_output" not in outcome
        assert "read limit" in large.content
        shown = large.structured["messages"][0]
        assert shown["truncated"] is True
        assert len(shown["parts"][0]["content"]) + len(shown["parts"][1]["output"]) == 100

        buried = await _read(store, access, "root", session_id="buried")
        assert buried.structured is not None
        outcome = buried.structured["session"]["outcome"]
        assert outcome["structured_output_status"] == "not_found_within_limit"
        assert "no structured output in the last 8 messages" in buried.content

        found = await _read(store, access, "root", session_id="recent")
        assert found.structured is not None
        outcome = found.structured["session"]["outcome"]
        assert outcome["structured_output_status"] == "found"
        assert outcome["structured_output"] == '{"status":"early"}'
        assert outcome["text"] == "note 6"

        plain = await _read(store, access, "root", session_id="root")
        assert plain.structured == {"error": "session_not_found"}

    _run_with_store(backend, sqlite_resources, scenario)


async def _check_bounded_transcript_window(store: SessionStore) -> None:
    messages = [Message.text("user", "a" * 100)]
    messages.append(Message.text("assistant", "b" * 5000))
    messages.extend(Message.text("assistant", f"{index} " + "c" * 100) for index in range(7))
    await _session(store, "bounded", messages=messages)

    window = await store.load_bounded_transcript_window(
        "bounded", start_index=0, limit=20, max_record_bytes=1000, max_total_bytes=1000
    )
    assert window.cursor == 9
    first, huge, *rest = window.records
    assert first.message is not None and first.message.role is MessageRole.USER
    assert huge.index == 1 and huge.message is None and huge.role is MessageRole.ASSISTANT
    assert huge.stored_bytes > 5000
    assert rest and all(record.message is not None for record in rest)
    loaded = [first, *rest]
    assert sum(record.stored_bytes for record in loaded) <= 1000
    assert [record.index for record in window.records] == list(range(len(window.records)))
    assert len(window.records) < 9

    following = await store.load_bounded_transcript_window(
        "bounded",
        start_index=window.records[-1].index + 1,
        limit=20,
        max_record_bytes=1000,
        max_total_bytes=1000,
    )
    assert following.records[0].index == window.records[-1].index + 1
    assert following.records[-1].index == 8
    starting_large = await store.load_bounded_transcript_window(
        "bounded", start_index=1, limit=2, max_record_bytes=1000, max_total_bytes=1000
    )
    assert [record.index for record in starting_large.records] == [1, 2]
    assert starting_large.records[0].message is None
    with pytest.raises(KeyError):
        await store.load_bounded_transcript_window(
            "missing", start_index=0, limit=1, max_record_bytes=10, max_total_bytes=10
        )
    with pytest.raises(ValueError, match="max_record_bytes"):
        await store.load_bounded_transcript_window(
            "bounded", start_index=0, limit=1, max_record_bytes=20, max_total_bytes=10
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_bounded_transcript_windows_measure_rows_before_loading_them(
    backend: str, sqlite_resources
) -> None:
    _run_with_store(backend, sqlite_resources, _check_bounded_transcript_window)


def test_postgres_bounded_transcript_windows_measure_rows_before_loading_them(
    postgres_dsn,
) -> None:
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run() -> None:
        store = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_CODEC
        )
        try:
            await _check_bounded_transcript_window(store)
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_read_session_lists_an_oversized_stored_message_without_loading_it(
    backend: str, sqlite_resources
) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root")
        messages = [
            Message.text("user", "Scan the dump"),
            Message.text("assistant", "z" * (1200 * 1024)),
            Message.text("assistant", "Done"),
        ]
        await _session(store, "dump", parent="root", messages=messages)
        result = await _read(store, OwnTreeSessionAccess(), "root", session_id="dump")
        assert result.structured is not None
        user, huge, done = result.structured["messages"]
        assert user["parts"][0]["text"] == "Scan the dump"
        assert huge["index"] == 1 and huge["omitted"] is True and huge["parts"] == []
        assert huge["stored_bytes"] > 1200 * 1024
        assert done["parts"][0]["text"] == "Done"
        assert result.structured["next_offset"] is None
        assert "stored bytes exceed the 1024 KiB" in result.content
        assert "zzzz" not in result.content

        paged = await _read(store, OwnTreeSessionAccess(), "root", session_id="dump", limit=2)
        assert paged.structured is not None
        assert [m["index"] for m in paged.structured["messages"]] == [0, 1]
        assert paged.structured["next_offset"] == 2

    _run_with_store(backend, sqlite_resources, scenario)


class _AdmitAll:
    async def narrow(self, caller, query):
        return query

    async def allows(self, caller, session):
        return session.id != caller.id


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_structured_output_search_applies_the_runs_resource_access_scope(
    backend: str, sqlite_resources
) -> None:
    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu.resource_access import (
        ResourceAccessPolicy,
        encode_scope,
        execution_access,
        model_data_access,
    )
    from cayu.sessions.access import SessionAccessRule, SessionAccessScope, SessionAccessSelector

    acme = SessionAccessRule(selectors=(SessionAccessSelector("organization", values=("acme",)),))
    scope = SessionAccessScope(read=(acme,), execute=(acme,))

    class Policy(ResourceAccessPolicy):
        authority = "session-read-tools-test"

        async def resolve(self, subject):
            return scope

    policy = Policy()
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject="alice", admitted_json=encode_scope(scope)
    )

    def answer(status: str) -> list[Message]:
        return [
            Message(
                role=MessageRole.TOOL,
                content=(
                    ToolResultPart(tool_call_id="call-1", tool_name="submit", content="ok"),
                    StructuredOutputPart(name="Report", output={"status": status}),
                ),
            )
        ]

    async def scenario(store: SessionStore) -> None:
        await _session(store, "caller", labels={"organization": "acme"})
        await _session(store, "own", labels={"organization": "acme"}, messages=answer("own"))
        await _session(store, "foreign", labels={"organization": "other"}, messages=answer("x"))
        list_tool = ListSessionsTool(store, access=_AdmitAll())
        read_tool = ReadSessionTool(store, access=_AdmitAll())

        status, part = await read_tool._final_structured_output("foreign")
        assert status == "found" and part is not None

        async with (
            execution_access(binding, policy, {"organization": "acme"}),
            model_data_access(),
        ):
            listed = await list_tool.run(_ctx("caller"), {})
            assert listed.structured is not None
            assert [
                await _raw(store, "caller", row["session_id"])
                for row in listed.structured["sessions"]
            ] == ["own"]
            outcome = listed.structured["sessions"][0]["outcome"]
            assert outcome["structured_output"] == '{"status":"own"}'

            read = await read_tool.run(
                _ctx("caller"), {"session_id": await _alias(store, "caller", "own")}
            )
            assert read.structured is not None
            outcome = read.structured["session"]["outcome"]
            assert outcome["structured_output_status"] == "found"

            foreign = await read_tool.run(
                _ctx("caller"), {"session_id": await _alias(store, "caller", "foreign")}
            )
            assert foreign.structured == {"error": "session_not_found"}
            assert await read_tool._final_structured_output("foreign") == ("unavailable", None)

    _run_with_store(backend, sqlite_resources, scenario)


async def _check_tree_and_agent_queries(store: SessionStore) -> None:
    await _session(store, "root", agent="scout")
    await _session(store, "child", agent="hunter", parent="root")
    await _session(store, "grandchild", agent="filter", parent="child")
    await _session(store, "other", agent="hunter")
    tree = await store.list_sessions(SessionQuery(root_session_id="root"))
    assert sorted(session.id for session in tree.sessions) == ["child", "grandchild", "root"]
    agents = await store.list_sessions(SessionQuery(agent_names=("hunter", "filter")))
    assert sorted(session.id for session in agents.sessions) == ["child", "grandchild", "other"]
    both = await store.list_sessions(
        SessionQuery(root_session_id="root", agent_names=("hunter",), include_total_count=True)
    )
    assert [session.id for session in both.sessions] == ["child"]
    assert both.total_count == 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_session_queries_filter_by_root_tree_and_agent_names(
    backend: str, sqlite_resources
) -> None:
    _run_with_store(backend, sqlite_resources, _check_tree_and_agent_queries)


def test_postgres_session_queries_filter_by_root_tree_and_agent_names(postgres_dsn) -> None:
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run() -> None:
        store = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_CODEC
        )
        try:
            await _check_tree_and_agent_queries(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_session_query_agent_names_are_validated() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SessionQuery(agent_names=("a", "a"))
    with pytest.raises(ValueError, match="sequence"):
        SessionQuery(agent_names="a")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        SessionQuery(agent_names=tuple(f"agent-{index}" for index in range(101)))
    assert SessionQuery(agent_names=["a", "b"]).agent_names == ("a", "b")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("secret", ['secret"quoted', "secret\\slash", "secret\nnewline"])
def test_structured_secrets_are_redacted_before_json_encoding(
    backend: str, sqlite_resources, secret: str
) -> None:
    import json

    async def scenario(store: SessionStore) -> None:
        await _session(store, "root")
        await _session(
            store,
            "worker",
            parent="root",
            messages=[
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        ToolCallPart(
                            tool_call_id="call", tool_name="deploy", arguments={secret: secret}
                        ),
                    ),
                ),
                Message(
                    role=MessageRole.TOOL,
                    content=(
                        ToolResultPart(
                            tool_call_id="call",
                            tool_name="deploy",
                            content="Done",
                            structured={secret: secret},
                        ),
                        StructuredOutputPart(name="Report", output={secret: secret}),
                    ),
                ),
            ],
        )
        ctx = _ctx("root", secret)
        read = await ReadSessionTool(store, unscoped=True, access=OwnTreeSessionAccess()).run(
            ctx, {"session_id": await _alias(store, "root", "worker")}
        )
        assert read.structured is not None
        messages = read.structured["messages"]
        encoded = [
            messages[0]["parts"][0]["arguments"],
            messages[1]["parts"][0]["structured"],
            messages[1]["parts"][1]["output"],
            read.structured["session"]["outcome"]["structured_output"],
        ]
        listed = await ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess()).run(
            ctx, {}
        )
        assert listed.structured is not None
        encoded.append(listed.structured["sessions"][0]["outcome"]["structured_output"])
        for value in encoded:
            assert json.loads(value) == {"[REDACTED_SECRET]": "[REDACTED_SECRET]"}
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in read.content
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in listed.content

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_text_search_matches_only_redacted_session_fields(backend: str, sqlite_resources) -> None:
    async def scenario(store: SessionStore) -> None:
        await _session(store, "root")
        await _session(
            store, "worker", parent="root", title="Deploy tok-12345", labels={"token": "tok-12345"}
        )
        tool = ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess())
        ctx = _ctx("root", "tok-12345")
        results = []
        for guess in ["tok-1", "tok-12", "tok-19", "tok-12345"]:
            result = await tool.run(ctx, {"text": guess, "limit": 1})
            assert result.structured is not None
            assert result.structured["sessions"] == []
            assert result.structured["next_cursor"] is None
            results.append(result.structured)
        assert all(result == results[0] for result in results)
        for visible in ["deploy", "assistant", "token", "[REDACTED_SECRET]"]:
            result = await tool.run(ctx, {"text": visible})
            assert result.structured is not None
            assert [
                await _raw(store, "root", row["session_id"])
                for row in result.structured["sessions"]
            ] == ["worker"]
        assert (await tool.run(ctx, {"text": "worker"})).structured["sessions"] == []

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_structured_outcome_budget_counts_trailing_messages_once(backend, sqlite_resources):
    async def scenario(store):
        await _session(store, "root")
        await _session(
            store,
            "worker",
            parent="root",
            messages=[
                Message(
                    role=MessageRole.TOOL,
                    content=(StructuredOutputPart(name="Report", output={"answer": 42}),),
                ),
                Message.text("assistant", "x" * (300 * 1024)),
            ],
        )
        result = await _read(store, OwnTreeSessionAccess(), "root", session_id="worker", limit=1)
        assert result.structured is not None
        outcome = result.structured["session"]["outcome"]
        assert outcome["structured_output_status"] == "found"
        assert outcome["structured_output"] == '{"answer":42}'
        assert 'structured output Report: {"answer":42}' in result.content

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_structured_outcome_reads_newest_suffix_before_large_tool_history(
    backend, sqlite_resources
):
    async def scenario(store):
        await _session(store, "root")
        history = [
            Message(
                role=MessageRole.TOOL,
                content=(
                    ToolResultPart(
                        tool_call_id=f"read-{index}",
                        tool_name="read_file",
                        content="x" * (100 * 1024),
                    ),
                ),
            )
            for index in range(6)
        ]
        answer = Message(
            role=MessageRole.TOOL,
            content=(StructuredOutputPart(name="Report", output={"answer": 42}),),
        )
        await _session(store, "worker", parent="root", messages=[*history, answer])
        read = await _read(store, OwnTreeSessionAccess(), "root", session_id="worker", limit=1)
        listed = await ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess()).run(
            _ctx("root"), {}
        )
        assert read.structured is not None and listed.structured is not None
        for outcome in [
            read.structured["session"]["outcome"],
            listed.structured["sessions"][0]["outcome"],
        ]:
            assert outcome["structured_output_status"] == "found"
            assert outcome["structured_output"] == '{"answer":42}'
        assert 'structured output Report: {"answer":42}' in read.content
        assert 'structured output Report: {"answer":42}' in listed.content

    _run_with_store(backend, sqlite_resources, scenario)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_session_search_cannot_probe_private_identifiers_or_exact_secret_filters(
    backend, sqlite_resources
):
    async def scenario(store):
        await _session(store, "root")
        await _session(
            store,
            "private-tok-12345",
            parent="root",
            agent="agent-tok-12345",
            labels={"token": "tok-12345", "app": "scout"},
        )
        tool = ListSessionsTool(store, unscoped=True, access=OwnTreeSessionAccess())
        ctx = _ctx("root", "tok-12345")
        for args in (
            {"text": "tok-123"},
            {"text": "tok-999"},
            {"text": "private-tok-12345"},
            {"agent_name": "agent-tok-12345"},
            {"agent_name": "agent-tok-99999"},
            {"labels": {"token": "tok-12345"}},
            {"labels": {"token": "tok-99999"}},
        ):
            result = await tool.run(ctx, {**args, "max_outcome_chars": 0})
            assert not result.is_error
            assert result.structured is not None
            assert result.structured["sessions"] == []
            assert result.structured["next_cursor"] is None
        for args in ({"text": "agent"}, {"labels": {"app": "scout"}}):
            result = await tool.run(ctx, {**args, "max_outcome_chars": 0})
            assert result.structured is not None
            [row] = result.structured["sessions"]
            assert row["agent_name"] == "agent-[REDACTED_SECRET]"
            assert "tok-12345" not in result.content
            assert "tok-12345" not in str(result.structured)
            assert await _raw(store, "root", row["session_id"]) == "private-tok-12345"

    _run_with_store(backend, sqlite_resources, scenario)
