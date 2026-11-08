"""Checkpoint work per store publication, pinned as counts rather than time.

A runtime publication decodes the stored checkpoint once and checks the new
document once. A session-operation write (the tool effect journal) that leaves
the checkpoint as it was keeps the stored row instead of rewriting it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import Counter, defaultdict
from contextvars import ContextVar
from functools import wraps
from typing import Any

import pytest

import cayu._validation as validation
import cayu.storage.postgres as postgres_store
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import checkpoints
from cayu.storage import SQLiteSessionStore
from cayu.storage import _receipt_ledger_rows as receipt_ledger_rows
from cayu.storage import _sqlite_records as sqlite_records
from cayu.tools.base import ToolContext

_ROUNDS = 2


class _NoteTool(Tool):
    spec = ToolSpec(
        name="note",
        parallel_safe=False,
        max_terminal_payload_bytes=1024,
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"noted {args['index']}")


def _canonical(checkpoint: dict[str, Any] | None) -> str | None:
    return None if checkpoint is None else json.dumps(checkpoint, sort_keys=True)


class _Meter:
    """Count checkpoint decodes, whole-checkpoint checks and row writes per store call."""

    def __init__(self, store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.store = store
        self.calls: Counter[str] = Counter()
        self.counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.effect_writes_that_changed = 0
        self._current: ContextVar[str | None] = ContextVar("store_call", default=None)
        self._walking = False
        self._patch(monkeypatch)

    def _count(self, metric: str) -> None:
        call = self._current.get()
        if call is not None:
            self.counts[call][metric] += 1

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        meter = self
        decode = checkpoints.CheckpointMigrator.decode

        def counted_decode(self: Any, checkpoint: Any, *, session_id: str) -> Any:
            if checkpoint is not None:
                meter._count("decodes")
            return decode(self, checkpoint, session_id=session_id)

        monkeypatch.setattr(checkpoints.CheckpointMigrator, "decode", counted_decode)

        walk = validation._walk_bounded_durable_json

        def counted_walk(value: Any, field_name: str, **kwargs: Any) -> Any:
            # A whole checkpoint is a root object stamped with its schema
            # version; nested walks inside one are not counted again.
            if (
                not meter._walking
                and type(value) is dict
                and checkpoints.CHECKPOINT_SCHEMA_VERSION_KEY in value
            ):
                meter._count("checkpoint_walks")
                ledger = value.get(checkpoints.INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
                if type(ledger) is dict and ledger.get("record_type") == (
                    checkpoints.INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE
                ):
                    meter._count("walks_with_receipt_ledger")
                meter._walking = True
                try:
                    return walk(value, field_name, **kwargs)
                finally:
                    meter._walking = False
            return walk(value, field_name, **kwargs)

        monkeypatch.setattr(validation, "_walk_bounded_durable_json", counted_walk)

        sqlite_row = sqlite_records.checkpoint_row_values
        postgres_row = postgres_store._checkpoint_row_values

        def counted_sqlite_row(*args: Any, **kwargs: Any) -> Any:
            meter._count("row_writes")
            return sqlite_row(*args, **kwargs)

        def counted_postgres_row(*args: Any, **kwargs: Any) -> Any:
            meter._count("row_writes")
            return postgres_row(*args, **kwargs)

        monkeypatch.setattr(sqlite_records, "checkpoint_row_values", counted_sqlite_row)
        monkeypatch.setattr(postgres_store, "_checkpoint_row_values", counted_postgres_row)

        split = receipt_ledger_rows.split_receipt_ledger

        def counted_split(checkpoint: Any) -> Any:
            row, ledger = split(checkpoint)
            if ledger is not None:
                meter._count("ledger_writes")
            return row, ledger

        monkeypatch.setattr(receipt_ledger_rows, "split_receipt_ledger", counted_split)

        store = self.store

        def recorded(name: str, method: Any) -> Any:
            @wraps(method)
            async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
                if self is not store or meter._current.get() is not None:
                    return await method(self, *args, **kwargs)
                label = name
                if name == "publish_runtime_publication":
                    label = f"{name} {kwargs['request'].kind}"
                session_id = args[0] if args else kwargs.get("session_id")
                before = None
                if name == "publish_session_operation":
                    before = _canonical(await originals["load_checkpoint"](self, session_id))
                meter.calls[label] += 1
                token = meter._current.set(label)
                try:
                    result = await method(self, *args, **kwargs)
                finally:
                    meter._current.reset(token)
                if name == "publish_session_operation":
                    after = _canonical(await originals["load_checkpoint"](self, session_id))
                    meter.effect_writes_that_changed += int(after != before)
                return result

            return call

        originals: dict[str, Any] = {}

        store_type = type(store)
        for name in dir(store_type):
            method = getattr(store_type, name)
            if not name.startswith("_") and inspect.iscoroutinefunction(method):
                originals[name] = method
                monkeypatch.setattr(store_type, name, recorded(name, method))

    def per_call(self, label: str, metric: str) -> float:
        return self.counts[label][metric] / self.calls[label]


def _store(backend: str, request: pytest.FixtureRequest, tmp_path: Any) -> Any:
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_publications_decode_once_and_skip_unchanged_rewrites(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = f"checkpoint-cost-{backend}"
    batches = [
        [
            ModelStreamEvent.tool_call(id=f"call-{index}", name="note", arguments={"index": index}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        for index in range(_ROUNDS)
    ] + [
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]
    ]

    async def scenario() -> _Meter:
        store = _store(backend, request, tmp_path)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(ScriptedModelProvider(batches), default=True)
            app.register_agent(
                AgentSpec(name="worker", model="scripted-model"), tools=[_NoteTool()]
            )
            meter = _Meter(store, monkeypatch)
            run = RunRequest(
                session_id=session_id,
                agent_name="worker",
                max_steps=_ROUNDS + 1,
                messages=[Message.text("user", "take notes")],
            )
            last = [event.type async for event in app.run(run)][-1]
            assert last == EventType.SESSION_COMPLETED
            return meter
        finally:
            await store.close()

    meter = asyncio.run(scenario())

    publication = "publish_runtime_publication tool-round"
    assert meter.calls[publication] == _ROUNDS
    # The stored checkpoint is decoded once; the only other decode is the
    # whole-document check of the new checkpoint before it is written.
    assert meter.per_call(publication, "decodes") == 2
    # Those two, plus the receipt's checkpoint digest. A backend that copies the
    # row as it reads it (PostgreSQL) adds that read.
    assert meter.per_call(publication, "checkpoint_walks") == (3 if backend == "sqlite" else 4)
    assert meter.per_call(publication, "row_writes") == 1
    # The receipt ledger stays in its own row: a publication neither checks
    # nor rewrites it.
    assert meter.counts[publication]["walks_with_receipt_ledger"] == 0
    assert meter.counts[publication]["ledger_writes"] == 0
    # Lifecycle commands store new ledger versions; publications and the effect
    # journal never do.
    assert sum(counts["ledger_writes"] for counts in meter.counts.values()) > 0
    assert not {
        label
        for label, counts in meter.counts.items()
        if counts["ledger_writes"]
        and label.startswith(("publish_runtime_publication", "publish_session_operation"))
    }

    operation = "publish_session_operation"
    assert meter.calls[operation] >= _ROUNDS
    # Only an effect-journal write that changed the checkpoint rewrites its row,
    # and most of them leave it unchanged.
    assert meter.counts[operation]["row_writes"] == meter.effect_writes_that_changed
    assert meter.effect_writes_that_changed < meter.calls[operation]
