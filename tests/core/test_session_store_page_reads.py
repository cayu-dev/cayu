"""Page reads answer exactly what the single-session reads answer, on every store.

Each native page read is compared with the base ``SessionStore`` default, which
loops over the single-session methods, and with those methods themselves.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, cast

import psycopg
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _drain, _pending, _start

from cayu._validation import DurableValueError
from cayu.approvals import UserInputResponse
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.artifacts.base import ArtifactListResult, ArtifactStore, ArtifactStoreUnavailableError
from cayu.messages import Message, MessageRole
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.sessions import ForkSessionRequest
from cayu.sessions._browser_usage import BROWSER_USAGE_KEY
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions._page_reads import MAX_PAGE_READ_SESSIONS
from cayu.sessions._supervision_projection import RELEASE_MARKER_EVENT_TYPES
from cayu.sessions.checkpoints import (
    _SELF_CONTAINED_ROOT_FIELDS,
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY,
    BROWSER_CONTROLS_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
    CheckpointCompatibilityError,
    decode_runtime_checkpoint,
    decode_runtime_checkpoint_fields,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import TranscriptQuery

__all__ = ["conformance_postgres_dsn", "session_store_case"]

SESSIONS = ("written", "asked", "approval", "child", "fork")
PAGE = (*SESSIONS, "missing", "written")
# Longer than the size from which a tool result is stored only in its tool event.
LONG = "answer " * 200


async def _populate(app) -> None:
    await _start(app, "written", "writer")
    await _start(app, "asked", "asker")
    await _start(app, "approval", "approver")
    await _drain(
        app.run(
            RunRequest(
                session_id="child",
                parent_session_id="written",
                agent_name="writer",
                messages=[Message.text("user", "go")],
            )
        )
    )
    await _drain(
        app.fork_session(ForkSessionRequest(source_session_id="written", session_id="fork"))
    )

    def record(_session, checkpoint, _persisted):
        return SessionOperationPublication(
            checkpoint={} if checkpoint is None else dict(checkpoint),
            operation_records={"page-read-operation": {"status": "completed"}},
        )

    await app.session_store.publish_session_operation(
        "asked", idempotency_key="page-read-operation", operation_transform=record, events=[]
    )


@pytest.mark.store_semantics
def test_page_reads_match_single_session_reads(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _populate(app)

            sessions = await store.load_sessions(PAGE)
            assert list(sessions) == list(SESSIONS)
            assert sessions == {sid: await store.load(sid) for sid in SESSIONS}
            assert sessions == await SessionStore.load_sessions(store, PAGE)

            instances = await store.load_session_instance_ids(PAGE)
            assert instances == {sid: session.instance_id for sid, session in sessions.items()}
            assert list(instances) == list(SESSIONS)
            assert instances == await SessionStore.load_session_instance_ids(store, PAGE)

            checkpoints = await store.load_checkpoints(PAGE)
            single = {sid: await store.load_checkpoint(sid) for sid in SESSIONS}
            assert checkpoints == {sid: value for sid, value in single.items() if value}
            assert checkpoints == await SessionStore.load_checkpoints(store, PAGE)
            keys = (*sorted(checkpoints["written"])[:2], "absent")
            fields = await store.load_checkpoint_fields(PAGE, keys)
            assert fields == {
                sid: {key: value[key] for key in keys if key in value}
                for sid, value in checkpoints.items()
            }
            assert all(fields.values())
            assert fields == await SessionStore.load_checkpoint_fields(store, PAGE, keys)

            outcomes = await store.summarize_outcomes(PAGE)
            assert outcomes == {sid: await store.summarize_outcome(sid) for sid in SESSIONS}
            assert outcomes == await SessionStore.summarize_outcomes(store, PAGE)

            markers = await store.query_latest_events(PAGE, RELEASE_MARKER_EVENT_TYPES)
            expected = {}
            for sid in SESSIONS:
                latest = await store.query_events(
                    EventQuery(
                        session_id=sid,
                        event_types=RELEASE_MARKER_EVENT_TYPES,
                        order_by=EventOrder.SEQUENCE_DESC,
                        limit=1,
                    )
                )
                if latest:
                    expected[sid] = latest[0]
            assert expected and markers == expected
            assert markers == await SessionStore.query_latest_events(
                store, PAGE, RELEASE_MARKER_EVENT_TYPES
            )

            for key in ("page-read-operation", BROWSER_USAGE_KEY):
                operations = await store.load_session_operations(PAGE, key)
                assert operations == {
                    sid: await store.load_session_operation(sid, key) for sid in SESSIONS
                }
                assert operations == await SessionStore.load_session_operations(store, PAGE, key)
            assert (await store.load_session_operations(PAGE, "page-read-operation"))["asked"]

            parents = [sessions[sid] for sid in SESSIONS]
            children = await store.count_child_runs(parents)
            # The fork's source is its parent too, but a fork is not a child run.
            assert children == {"written": 1, "asked": 0, "approval": 0, "child": 0, "fork": 0}
            assert children == await SessionStore.count_child_runs(store, parents)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_first_transcript_records_match_single_transcript_reads(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _populate(app)
            await store.create(
                RunRequest(agent_name="writer", session_id="silent", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            await _drain(
                app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id="twice",
                        messages=[Message.text("user", "first"), Message.text("user", "second")],
                    )
                )
            )
            # A long tool result is stored once, in its tool event, so the tool
            # message is read back through its reference.
            question = await _pending(app, "asked")
            assert question.input_id is not None
            await _drain(
                app.resolve_user_input(
                    UserInputResponse(session_id="asked", input_id=question.input_id, answer=LONG)
                )
            )
            page = (*PAGE, "silent", "twice")
            for role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL):
                expected = {}
                for sid in dict.fromkeys(page):
                    if await store.load(sid) is None:
                        continue
                    single = await store.query_transcript(
                        TranscriptQuery(session_id=sid, role=role, limit=1)
                    )
                    if single.records:
                        expected[sid] = single.records[0]
                first = await store.query_first_transcript_records(page, role)
                assert first == expected, role
                assert list(first) == [sid for sid in dict.fromkeys(page) if sid in expected]
                assert first == await SessionStore.query_first_transcript_records(store, page, role)
            users = await store.query_first_transcript_records(page, MessageRole.USER)
            assert set(users) == {*SESSIONS, "twice"}
            assert users["twice"].message.content == Message.text("user", "first").content
            assert users["fork"].message == users["written"].message
            tools = await store.query_first_transcript_records(page, MessageRole.TOOL)
            assert list(tools) == ["asked"]
            assert LONG in tools["asked"].message.model_dump_json()
            assert await store.query_first_transcript_records((), MessageRole.USER) == {}
            with pytest.raises(TypeError, match="MessageRole"):
                await store.query_first_transcript_records(page, cast("Any", "user"))
            with pytest.raises(ValueError, match="at most"):
                await store.query_first_transcript_records(
                    [f"s{index}" for index in range(MAX_PAGE_READ_SESSIONS + 1)], MessageRole.USER
                )
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_checkpoint_field_reads_keep_each_json_type(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            for sid in ("typed", "empty", "unwritten"):
                await store.create(
                    RunRequest(agent_name="writer", session_id=sid, messages=[]),
                    identity=SessionIdentity(provider_name="test", model="test"),
                )
            typed = {
                "object": {"nested": [1, {"deep": None}]},
                "array": ["a", 2, False],
                "true": True,
                "false": False,
                "null": None,
                "integer": 7,
                "real": 1.5,
                "text": "null",
                "unselected": "x" * 1000,
            }
            await store.checkpoint("typed", typed)
            await store.checkpoint("empty", {"unselected": 1})
            page = ("typed", "empty", "unwritten", "missing")
            keys = ("object", "array", "true", "false", "null", "integer", "real", "text")
            fields = await store.load_checkpoint_fields(page, keys)
            assert fields == {"typed": {key: typed[key] for key in keys}, "empty": {}}
            assert {key: type(value) for key, value in fields["typed"].items()} == {
                key: type(typed[key]) for key in keys
            }
            assert fields == await SessionStore.load_checkpoint_fields(store, page, keys)
            assert await store.load_checkpoint_fields((), keys) == {}
            with pytest.raises(TypeError):
                await store.load_checkpoint_fields(page, "object")
            with pytest.raises(ValueError):
                await store.load_checkpoint_fields(page, ())
        finally:
            await _close_store(store)

    asyncio.run(scenario())


class _StoredCheckpoints(InMemorySessionStore):
    """Stored checkpoints as given, which the runtime boundary must decode."""

    def __init__(self, checkpoints: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.stored = checkpoints

    async def load_checkpoints(self, session_ids):
        return {sid: dict(self.stored[sid]) for sid in session_ids if sid in self.stored}


def _whole_checkpoint(version: int | None) -> dict[str, Any]:
    """Every root field of every schema version a migration reads or writes."""

    checkpoint: dict[str, Any] = {
        root: {"schema_version": 1, "value": root}
        for root in (
            BROWSER_CONTROLS_CHECKPOINT_KEY,
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
            SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
            AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY,
            WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
            MODEL_FAILOVER_CHECKPOINT_KEY,
            "last_model_step_publication",
            "pending_tool_round",
            "pending_user_input",
            "pending_session_interrupt",
            "user_input_resolution_intent",
            "workspace_checkpoints",
            "execution_snapshots",
            "pending_wake",
            "unselected",
        )
    }
    if version is not None:
        checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] = version
    return checkpoint


def test_runtime_checkpoint_fields_decode_as_their_whole_checkpoint():
    async def scenario():
        versions = {
            "versionless": None,
            **{
                f"v{version}": version
                for version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
            },
        }
        stored = _StoredCheckpoints({sid: _whole_checkpoint(v) for sid, v in versions.items()})
        keys = tuple(sorted(_SELF_CONTAINED_ROOT_FIELDS))
        # Each selectable field decodes from itself and the version alone exactly as
        # inside its whole checkpoint, at every schema version. A migration that
        # derives one from another field fails here.
        for sid, checkpoint in stored.stored.items():
            whole = decode_runtime_checkpoint(checkpoint, session_id=sid)
            assert whole is not None
            projection = {
                key: checkpoint[key]
                for key in (*keys, CHECKPOINT_SCHEMA_VERSION_KEY)
                if key in checkpoint
            }
            assert decode_runtime_checkpoint_fields(projection, keys, session_id=sid) == {
                key: whole[key] for key in keys if key in whole
            }, sid
        runtime = runtime_checkpoint_session_store(stored)
        page = (*versions, "missing")
        whole = await runtime.load_checkpoints(page)
        fields = await runtime.load_checkpoint_fields(page, keys)
        assert fields == {
            sid: {key: value[key] for key in keys if key in value} for sid, value in whole.items()
        }
        # Version 9 began owning browser controls; an older value is application data.
        assert {sid for sid, value in fields.items() if value} == {
            f"v{version}" for version in range(9, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
        }
        # A field another field's migration can write needs its whole checkpoint.
        # The refusal holds whatever is stored, including no checkpoint at all.
        for refused_page in (page, (), ("missing",)):
            with pytest.raises(ValueError, match="self-contained"):
                await runtime.load_checkpoint_fields(refused_page, ("unselected",))
        stored.stored["v1"][CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
        with pytest.raises(CheckpointCompatibilityError):
            await runtime.load_checkpoint_fields(page, keys)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_checkpoint_field_reads_refuse_a_checkpoint_that_is_not_an_object(session_store_case):
    kind, path, dsn = session_store_case
    if kind == "memory":
        pytest.skip("The in-memory store holds checkpoints only as objects.")

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            await store.create(
                RunRequest(agent_name="writer", session_id="listed", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            await store.checkpoint("listed", {"unselected": 1})
            if kind == "sqlite":
                with sqlite3.connect(path / "sessions.sqlite") as connection:
                    connection.execute(
                        "UPDATE cayu_checkpoints SET state_json = '[1]' WHERE session_id = ?",
                        ("listed",),
                    )
            else:
                with psycopg.connect(dsn) as connection:
                    connection.execute(
                        "UPDATE cayu_checkpoints SET state = '[1]'::jsonb WHERE session_id = %s",
                        ("listed",),
                    )
            # The whole-checkpoint read refuses it, and so does the field read.
            with pytest.raises(DurableValueError, match="JSON object"):
                await store.load_checkpoints(("listed",))
            with pytest.raises(DurableValueError, match="JSON object"):
                await store.load_checkpoint_fields(("listed",), ("unselected",))
        finally:
            await _close_store(store)

    asyncio.run(scenario())


async def _all_pending(read) -> list:
    actions, issues, cursor, cursors = [], [], None, set()
    while True:
        page = await read(cursor)
        actions.extend(page.actions)
        issues.extend(page.issues)
        if not page.has_more:
            return sorted((action.session.id, action.id) for action in actions) + issues
        assert page.next_cursor is not None and page.next_cursor not in cursors
        cursors.add(page.next_cursor)
        cursor = page.next_cursor


@pytest.mark.store_semantics
def test_pending_actions_for_sessions_pages_through_exactly_those_sessions(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _populate(app)
            await _start(app, "unrequested", "asker")
            single = []
            for sid in PAGE[:-1]:
                single.extend(
                    await _all_pending(
                        lambda cursor, sid=sid: store.query_pending_actions(
                            PendingActionQuery(session_id=sid, limit=1, cursor=cursor)
                        )
                    )
                )
            assert sorted(sid for sid, _ in single) == ["approval", "asked"]
            for read in (store.query_pending_actions_for_sessions, _base_pending(store)):
                paged = await _all_pending(
                    lambda cursor, read=read: read(PAGE, PendingActionQuery(limit=1, cursor=cursor))
                )
                assert paged == sorted(single)
            assert (await store.query_pending_actions_for_sessions(())).actions == []
            with pytest.raises(ValueError, match="must not name one session"):
                await store.query_pending_actions_for_sessions(
                    PAGE, PendingActionQuery(session_id="asked")
                )
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def _base_pending(store):
    async def read(session_ids, query):
        return await SessionStore.query_pending_actions_for_sessions(store, session_ids, query)

    return read


@pytest.mark.store_semantics
def test_page_reads_validate_their_session_ids(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            assert await store.load_sessions(()) == {}
            assert await store.load_checkpoints(()) == {}
            assert await store.summarize_outcomes(()) == {}
            assert await store.count_child_runs(()) == {}
            with pytest.raises(TypeError):
                await store.load_sessions("written")
            with pytest.raises(ValueError):
                await store.load_sessions([" padded "])
            with pytest.raises(ValueError, match="at most"):
                await store.load_sessions(
                    [f"s{index}" for index in range(MAX_PAGE_READ_SESSIONS + 1)]
                )
            with pytest.raises(ValueError, match="event_types"):
                await store.query_latest_events(["written"], ())
        finally:
            await _close_store(store)

    asyncio.run(scenario())


class _ListOnlyArtifacts(LocalArtifactStore):
    """A store that keeps the base default, which asks ``list`` once per session."""

    count_session_artifacts = ArtifactStore.count_session_artifacts


def test_artifact_counts_match_listing_totals(tmp_path):
    async def scenario():
        native = LocalArtifactStore(tmp_path / "artifacts")
        for index in range(3):
            await native.put_bytes(b"x", filename=f"{index}.txt", session_id="owner")
        await native.put_bytes(b"y", filename="other.txt", session_id="other")
        page = ("owner", "other", "empty", "owner")
        expected = {}
        for sid in ("owner", "other", "empty"):
            listed = await native.list(scope=ArtifactScope.SESSION, session_id=sid, limit=1)
            expected[sid] = listed.total_count
        assert expected == {"owner": 3, "other": 1, "empty": 0}
        assert await native.count_session_artifacts(page) == expected
        default = _ListOnlyArtifacts(tmp_path / "artifacts")
        assert await default.count_session_artifacts(page) == expected
        assert await native.count_session_artifacts(()) == {}

        async def incomplete(**kwargs):
            return ArtifactListResult(artifacts=(), total_count=None, truncated=True)

        default.list = incomplete  # type: ignore[method-assign]
        with pytest.raises(ArtifactStoreUnavailableError, match="complete count"):
            await default.count_session_artifacts(page)

    asyncio.run(scenario())
