from __future__ import annotations

import asyncio
import io
import json

import pytest

import cayu.runtime.sessions as sessions_module
import cayu.storage.jsonl_export as jsonl_export_module
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    MIN_DURABLE_JSON_INTEGER,
    DurableValueError,
)
from cayu.core import Event, EventType, Message
from cayu.runtime import (
    InMemorySessionStore,
    InMemoryTaskStore,
    RunRequest,
    SessionIdentity,
    TaskCreate,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
)
from cayu.storage import SQLiteSessionStore
from cayu.storage.jsonl_export import (
    ImportedSession,
    export_sessions,
    export_tasks,
    import_sessions,
    import_tasks,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _lines(stream: io.StringIO) -> list[dict]:
    text = stream.getvalue()
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def test_export_sessions_writes_one_line_per_session_with_nested_state():
    async def run() -> None:
        store = InMemorySessionStore()
        # A rich session with events, transcript, and a checkpoint.
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_rich",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_rich",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_rich")],
        )
        await store.append_transcript_messages(
            "sess_rich",
            [Message.text("assistant", "building")],
        )
        await store.checkpoint("sess_rich", {"step": 3})
        # A bare session with no events/transcript/checkpoint.
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_bare",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)

        assert count == 2
        lines = _lines(stream)
        assert len(lines) == 2
        assert all(line["type"] == "session" for line in lines)
        assert all(
            {
                "session",
                "events",
                "transcript_records",
                "checkpoint",
                "deferred_interaction_input",
            }
            <= line.keys()
            for line in lines
        )

        by_id = {line["session"]["id"]: line for line in lines}
        assert set(by_id) == {"sess_rich", "sess_bare"}

        rich = by_id["sess_rich"]
        assert rich["session"]["agent_name"] == "builder"
        assert len(rich["events"]) == 1
        assert rich["events"][0]["type"] == EventType.SESSION_STARTED.value
        assert len(rich["transcript_records"]) == 1
        assert rich["transcript_records"][0]["message"]["role"] == "assistant"
        assert rich["checkpoint"] == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "step": 3,
        }

        bare = by_id["sess_bare"]
        assert bare["events"] == []
        assert bare["transcript_records"] == []
        assert bare["checkpoint"] is None
        assert bare["deferred_interaction_input"] is None

    asyncio.run(run())


def test_export_sessions_empty_store_returns_zero():
    async def run() -> None:
        store = InMemorySessionStore()
        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)
        assert count == 0
        assert stream.getvalue() == ""

    asyncio.run(run())


def test_export_sessions_preserves_private_deferred_interaction_input():
    async def run() -> None:
        store = InMemorySessionStore()
        source = [Message.text("user", "build the feature")]
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_deferred",
                messages=source,
            ),
            identity=_identity(),
            interaction_started_event=Event(
                id="evt_interaction_started",
                type=EventType.INTERACTION_STARTED,
                session_id="sess_deferred",
                interaction_id="interaction-deferred",
            ),
            interaction_source_messages=source,
        )
        try:
            stream = io.StringIO()
            assert await export_sessions(store, stream=stream) == 1

            [line] = _lines(stream)
            assert line["transcript_records"] == []
            assert line["deferred_interaction_input"] == {
                "interaction_id": "interaction-deferred",
                "source_messages": [source[0].model_dump(mode="json")],
                "initial_transcript_messages": None,
            }

            [imported] = list(import_sessions(io.StringIO(stream.getvalue())))
            assert imported.deferred_interaction_input is not None
            assert imported.deferred_interaction_input.interaction_id == "interaction-deferred"
            assert imported.deferred_interaction_input.source_messages == source
            assert imported.deferred_interaction_input.initial_transcript_messages is None
        finally:
            await store.release_run_fence("sess_deferred")

    asyncio.run(run())


def test_export_sessions_preserves_materialized_transcript_interaction_attribution():
    async def run() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_interaction_transcript",
                messages=[],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            "sess_interaction_transcript",
            [Message.text("user", "first"), Message.text("assistant", "one")],
            interaction_id="interaction-one",
        )
        await store.append_transcript_messages(
            "sess_interaction_transcript",
            [Message.text("user", "second"), Message.text("assistant", "two")],
            interaction_id="interaction-two",
        )

        stream = io.StringIO()
        assert await export_sessions(store, stream=stream) == 1

        [line] = _lines(stream)
        assert [record["index"] for record in line["transcript_records"]] == [0, 1, 2, 3]
        assert [record["interaction_id"] for record in line["transcript_records"]] == [
            "interaction-one",
            "interaction-one",
            "interaction-two",
            "interaction-two",
        ]

        [imported] = list(import_sessions(io.StringIO(stream.getvalue())))
        assert [record.interaction_id for record in imported.transcript_records] == [
            "interaction-one",
            "interaction-one",
            "interaction-two",
            "interaction-two",
        ]
        assert [record.message for record in imported.transcript_records] == imported.transcript

    asyncio.run(run())


def test_export_sessions_round_trips_retained_absolute_transcript_indices(tmp_path):
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id="sess_compacted",
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.append_transcript_messages(
                "sess_compacted",
                [Message.text("assistant", f"message {index}") for index in range(5)],
            )
            assert await store.compact_transcript("sess_compacted", keep_last=2) == 3

            stream = io.StringIO()
            assert await export_sessions(store, stream=stream) == 1
            [line] = _lines(stream)
            assert [record["index"] for record in line["transcript_records"]] == [3, 4]

            [imported] = list(import_sessions(io.StringIO(stream.getvalue())))
            assert [record.index for record in imported.transcript_records] == [3, 4]
        finally:
            await store.close()

    asyncio.run(run())


def test_export_sessions_pages_past_default_list_limit():
    async def run() -> None:
        store = InMemorySessionStore()
        for index in range(1001):
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=f"sess_{index:03}",
                    messages=[Message.text("user", "build")],
                ),
                identity=_identity(),
            )

        stream = io.StringIO()
        count = await export_sessions(store, stream=stream)

        assert count == 1001
        lines = _lines(stream)
        assert len(lines) == 1001
        assert {line["session"]["id"] for line in lines} == {
            f"sess_{index:03}" for index in range(1001)
        }

    asyncio.run(run())


def test_export_tasks_writes_one_line_per_task():
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="task_a",
                type="process",
                title="Process A",
                input={"value": 1},
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="task_b",
                type="process",
                input={"value": 2},
            )
        )

        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)

        assert count == 2
        lines = _lines(stream)
        assert len(lines) == 2
        assert all(line["type"] == "task" for line in lines)
        assert all("task" in line for line in lines)

        by_id = {line["task"]["id"]: line["task"] for line in lines}
        assert set(by_id) == {"task_a", "task_b"}
        assert by_id["task_a"]["input"] == {"value": 1}
        assert by_id["task_a"]["title"] == "Process A"
        assert by_id["task_b"]["status"] == "pending"

    asyncio.run(run())


def test_export_tasks_pages_past_default_list_limit():
    async def run() -> None:
        store = InMemoryTaskStore()
        for index in range(1001):
            await store.create_task(
                TaskCreate(
                    task_id=f"task_{index:03}",
                    type="process",
                    input={"index": index},
                )
            )

        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)

        assert count == 1001
        lines = _lines(stream)
        assert len(lines) == 1001
        assert {line["task"]["id"] for line in lines} == {
            f"task_{index:03}" for index in range(1001)
        }

    asyncio.run(run())


def test_export_tasks_empty_store_returns_zero():
    async def run() -> None:
        store = InMemoryTaskStore()
        stream = io.StringIO()
        count = await export_tasks(store, stream=stream)
        assert count == 0
        assert stream.getvalue() == ""

    asyncio.run(run())


def test_export_sessions_keyset_paging_survives_concurrent_delete():
    # An offset walk skips a live session when one ahead of the cursor is
    # deleted mid-export (the window shifts down by one). Keyset paging anchors
    # each page to the last emitted (created_at, id), so it stays correct.
    async def run() -> None:
        store = InMemorySessionStore()
        for index in range(1002):
            await store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=f"sess_{index:04}",
                    messages=[Message.text("user", "build")],
                ),
                identity=_identity(),
            )

        # A store wrapper that deletes a not-yet-reached session on the second
        # page, mid-export, to model a concurrent writer.
        class _DeletingStore:
            def __init__(self, inner):
                self._inner = inner
                self._calls = 0

            async def list_sessions(self, query):
                self._calls += 1
                if self._calls == 2:
                    del self._inner._sessions["sess_1001"]
                return await self._inner.list_sessions(query)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        stream = io.StringIO()
        count = await export_sessions(_DeletingStore(store), stream=stream)

        ids = {line["session"]["id"] for line in _lines(stream)}
        # Exactly the sessions that survived are emitted, none skipped, none dup.
        assert len(ids) == count
        assert "sess_1001" not in ids
        assert ids == {f"sess_{index:04}" for index in range(1001)}

    asyncio.run(run())


def test_import_sessions_round_trips_export():
    async def run() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_rich",
                messages=[Message.text("user", "build")],
            ),
            identity=_identity(),
        )
        await store.append_events(
            "sess_rich",
            [Event(type=EventType.SESSION_STARTED, session_id="sess_rich")],
        )
        await store.append_transcript_messages(
            "sess_rich",
            [Message.text("assistant", "building")],
        )
        await store.checkpoint("sess_rich", {"step": 3})
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_bare",
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        stream = io.StringIO()
        await export_sessions(store, stream=stream)

        imported = list(import_sessions(io.StringIO(stream.getvalue())))
        assert all(isinstance(record, ImportedSession) for record in imported)
        by_id = {record.session.id: record for record in imported}
        assert set(by_id) == {"sess_rich", "sess_bare"}

        rich = by_id["sess_rich"]
        assert rich.session == await store.load("sess_rich")
        assert rich.events == await store.load_events("sess_rich")
        assert rich.transcript == await store.load_transcript("sess_rich")
        assert rich.checkpoint == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "step": 3,
        }
        assert rich.deferred_interaction_input is None

        bare = by_id["sess_bare"]
        assert bare.events == []
        assert bare.transcript == []
        assert bare.checkpoint is None
        assert bare.deferred_interaction_input is None

    asyncio.run(run())


def test_session_export_and_import_reject_future_root_checkpoint_versions() -> None:
    async def build_export() -> str:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_future_export",
                messages=[],
            ),
            identity=_identity(),
        )
        with sessions_module._invocation_lifecycle_authority_mutation_scope():
            await store.checkpoint(
                "sess_future_export",
                {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
                    "private": "must-not-appear",
                },
            )
        stream = io.StringIO()
        with pytest.raises(CheckpointCompatibilityError) as caught:
            await export_sessions(store, stream=stream)
        assert caught.value.observed_version == CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
        assert "must-not-appear" not in str(caught.value)
        assert stream.getvalue() == ""

        portable_store = InMemorySessionStore()
        await portable_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_future_export",
                messages=[],
            ),
            identity=_identity(),
        )
        await portable_store.checkpoint("sess_future_export", {"portable": True})
        await export_sessions(portable_store, stream=stream)
        return stream.getvalue()

    record = _lines(io.StringIO(asyncio.run(build_export())))[0]
    record["checkpoint"][CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    record["checkpoint"]["private"] = "must-not-appear"

    with pytest.raises(CheckpointCompatibilityError) as caught:
        list(import_sessions([json.dumps(record)]))

    assert caught.value.observed_version == CURRENT_CHECKPOINT_SCHEMA_VERSION + 1
    assert "must-not-appear" not in str(caught.value)


@pytest.mark.parametrize(
    "missing_field",
    [
        "transcript_records",
        "deferred_interaction_input",
        "targeted_tool_grant_state",
    ],
)
def test_import_sessions_rejects_incomplete_session_export(missing_field: str):
    session = SessionIdentity(provider_name="fake", model="fake-model")

    async def build_line() -> str:
        store = InMemorySessionStore()
        created = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_incomplete_export",
                messages=[],
            ),
            identity=session,
        )
        return json.dumps(
            {
                "type": "session",
                "session": created.model_dump(mode="json"),
                "events": [],
                "transcript_records": [],
                "checkpoint": None,
                "deferred_interaction_input": None,
                "targeted_tool_grant_state": {
                    "schema_version": 1,
                    "records": [],
                    "uses": [],
                },
            }
        )

    record = json.loads(asyncio.run(build_line()))
    record.pop(missing_field)
    with pytest.raises(ValueError, match=missing_field):
        list(import_sessions([json.dumps(record)]))


def test_import_sessions_rejects_removed_bare_transcript_field():
    async def build_line() -> str:
        store = InMemorySessionStore()
        created = await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_removed_transcript_field",
                messages=[],
            ),
            identity=_identity(),
        )
        return json.dumps(
            {
                "type": "session",
                "session": created.model_dump(mode="json"),
                "events": [],
                "transcript": [],
                "transcript_records": [],
                "checkpoint": None,
                "deferred_interaction_input": None,
                "targeted_tool_grant_state": {
                    "schema_version": 1,
                    "records": [],
                    "uses": [],
                },
            }
        )

    with pytest.raises(ValueError, match="unsupported fields"):
        list(import_sessions([asyncio.run(build_line())]))


def test_import_tasks_round_trips_export():
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(task_id="task_a", type="process", title="Process A", input={"value": 1})
        )
        await store.create_task(TaskCreate(task_id="task_b", type="process", input={"value": 2}))

        stream = io.StringIO()
        await export_tasks(store, stream=stream)

        imported = list(import_tasks(io.StringIO(stream.getvalue())))
        by_id = {task.id: task for task in imported}
        assert set(by_id) == {"task_a", "task_b"}
        assert by_id["task_a"] == await store.load_task("task_a")
        assert by_id["task_b"] == await store.load_task("task_b")

    asyncio.run(run())


def test_import_skips_blank_lines_and_rejects_wrong_type():
    imported = list(import_sessions(["", "  \n"]))
    assert imported == []

    task_lines = io.StringIO('{"type": "task", "task": {}}\n')
    try:
        list(import_sessions(task_lines))
    except ValueError as exc:
        assert "session record" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError on wrong record type")

    session_lines = io.StringIO('{"type": "session", "session": {}}\n')
    try:
        list(import_tasks(session_lines))
    except ValueError as exc:
        assert "task record" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError on wrong record type")


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ('{"type":"task","task":{"metadata":{"bad":NaN}}}', "non_finite_number"),
        (
            f'{{"type":"task","task":{{"metadata":{{"bad":{MAX_DURABLE_JSON_INTEGER + 1}}}}}}}',
            "integer_out_of_range",
        ),
        ('{"type":"task","task":{"metadata":{"bad":"value\\u0000"}}}', "nul_character"),
        ('{"type":"task","task":{"metadata":{"bad":"value\\ud800"}}}', "unicode_surrogate"),
        ('["not", "an", "object"]', "invalid_json_object"),
    ],
)
def test_import_rejects_nonportable_json_before_typed_record_validation(line, code):
    with pytest.raises(DurableValueError) as raised:
        list(import_tasks([line]))

    assert raised.value.code == code
    assert raised.value.field_name == "JSONL record"


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        (str(MIN_DURABLE_JSON_INTEGER), MIN_DURABLE_JSON_INTEGER),
        (str(MAX_DURABLE_JSON_INTEGER), MAX_DURABLE_JSON_INTEGER),
    ],
    ids=["minimum", "maximum"],
)
def test_import_json_parser_accepts_exact_signed_int64_boundaries(literal, expected):
    records = list(jsonl_export_module._iter_json_lines([f'{{"value":{literal}}}']))

    assert records == [{"value": expected}]
    assert type(records[0]["value"]) is int


@pytest.mark.parametrize(
    "literal",
    [
        str(MIN_DURABLE_JSON_INTEGER - 1),
        str(MAX_DURABLE_JSON_INTEGER + 1),
        "9" * 5000,
        "-" + "9" * 5000,
    ],
    ids=["below-minimum", "above-maximum", "huge-positive", "huge-negative"],
)
def test_import_json_parser_rejects_out_of_range_integers_with_typed_error(literal):
    with pytest.raises(DurableValueError) as raised:
        list(jsonl_export_module._iter_json_lines([f'{{"value":{literal}}}']))

    assert raised.value.code == "integer_out_of_range"
    assert raised.value.field_name == "JSONL record"


@pytest.mark.parametrize(
    "line",
    [
        '{"type":"task","secret-key":"secret-a","secret-key":"secret-b"}',
        ('{"type":"task","task":{"nested":{"secret-key":"secret-a","secret-key":"secret-b"}}}'),
    ],
    ids=["top-level", "nested"],
)
def test_import_rejects_duplicate_keys_without_echoing_key_or_values(line):
    with pytest.raises(DurableValueError) as raised:
        list(import_tasks([line]))

    assert raised.value.code == "duplicate_json_key"
    rendered = str(raised.value)
    assert "secret-key" not in rendered
    assert "secret-a" not in rendered
    assert "secret-b" not in rendered


def test_import_rejects_non_string_lines_before_invoking_string_methods():
    class HostileLine:
        strip_called = False

        def strip(self):
            self.strip_called = True
            raise AssertionError("strip must not be invoked")

    hostile = HostileLine()

    with pytest.raises(DurableValueError) as raised:
        list(import_tasks([hostile]))  # type: ignore[list-item]

    assert raised.value.code == "invalid_text_type"
    assert hostile.strip_called is False


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ('{"type":"task","workload-secret":"literal\x00value"}', "nul_character"),
        ('{"type":"task","workload-secret":"literal\ud800value"}', "unicode_surrogate"),
    ],
)
def test_import_rejects_nonportable_raw_line_text_before_json_parsing(line, code):
    with pytest.raises(DurableValueError) as raised:
        list(import_tasks([line]))

    assert raised.value.code == code
    rendered = str(raised.value)
    assert "workload-secret" not in rendered
    assert "literal" not in rendered


def test_import_normalizes_parser_recursion_failure(monkeypatch):
    def recurse(*args, **kwargs):
        del args, kwargs
        raise RecursionError("parser detail must not escape")

    monkeypatch.setattr(jsonl_export_module.json, "loads", recurse)

    with pytest.raises(DurableValueError) as raised:
        list(import_tasks(['{"type":"task"}']))

    assert raised.value.code == "nesting_too_deep"
    assert raised.value.field_name == "JSONL record"
    assert "parser detail" not in str(raised.value)


def test_import_normalizes_json_number_representations_before_model_validation():
    async def export_base_task() -> str:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="task_numbers", type="process"))
        stream = io.StringIO()
        await export_tasks(store, stream=stream)
        record = json.loads(stream.getvalue())
        record["task"]["input"] = {
            "ordinary": 1.0,
            "negative_zero": -0.0,
            "large": 1e18,
            "fractional": 1e-7,
        }
        return json.dumps(record)

    imported = list(import_tasks([asyncio.run(export_base_task())]))

    assert len(imported) == 1
    numbers = imported[0].input
    assert numbers == {
        "ordinary": 1,
        "negative_zero": 0,
        "large": 1_000_000_000_000_000_000,
        "fractional": 1e-7,
    }
    assert type(numbers["ordinary"]) is int
    assert type(numbers["negative_zero"]) is int
    assert type(numbers["large"]) is int
    assert type(numbers["fractional"]) is float


def test_jsonl_writer_rejects_the_whole_record_before_writing_any_bytes():
    stream = io.StringIO()

    with pytest.raises(DurableValueError) as raised:
        jsonl_export_module._write_line(
            stream,
            {"type": "task", "task": {"result": {"bad": float("nan")}}},
        )

    assert raised.value.code == "non_finite_number"
    assert stream.getvalue() == ""
