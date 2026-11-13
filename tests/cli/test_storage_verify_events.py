"""``cayu storage verify-events`` reruns full validation over stored events."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._validation import DURABLE_DOCUMENT_LIMITS
from cayu.cli import main
from cayu.cli import storage as storage_cli
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
)
from cayu.sessions.event_queries import EventQuery
from cayu.storage import SQLiteSessionStore

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated_target_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CAYU_DATABASE_URL",
        storage_cli.DIRECT_DATABASE_URL_ENV,
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    del request
    return [
        ModelStreamEvent.text_delta("Done."),
        ModelStreamEvent.completed(
            {"finish_reason": "stop", "usage": {"input_tokens": 10, "output_tokens": 2}}
        ),
    ]


def _store_with_history(path: Path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(path)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(ScriptedModelProvider(name="scripted", response_factory=_respond))
            app.register_agent(AgentSpec(name="writer", model="scripted-model"))
            for index in range(2):
                async for _event in app.run(
                    RunRequest(
                        agent_name="writer",
                        session_id=f"verify-{index}",
                        messages=[Message.text("user", "Write a line.")],
                    )
                ):
                    pass
            await app.aclose()
        finally:
            await store.close()

    asyncio.run(run())


def _verify(path: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, dict]:
    code = main(["storage", "verify-events", "--sqlite", str(path), "--json", *extra])
    return code, json.loads(capsys.readouterr().out)


def test_an_untouched_store_verifies_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "sessions.db"
    _store_with_history(path)
    with sqlite3.connect(path) as connection:
        stored = connection.execute("SELECT COUNT(*), MAX(sequence) FROM cayu_events").fetchone()

    code, result = _verify(path, capsys)

    assert code == 0
    assert result["checked"] == stored[0]
    assert result["through_sequence"] == stored[1]
    assert result["complete"] is True
    assert result["failures"] == []

    code, page = _verify(path, capsys, "--max-events", "3")
    assert code == 0
    assert page["checked"] == 3
    assert page["complete"] is False
    code, rest = _verify(path, capsys, "--after-sequence", str(page["through_sequence"]))
    assert code == 0
    assert rest["checked"] == stored[0] - 3
    assert rest["complete"] is True


def test_a_row_edited_outside_cayu_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "sessions.db"
    _store_with_history(path)
    with sqlite3.connect(path) as connection:
        sequence, session_id = connection.execute(
            "SELECT sequence, session_id FROM cayu_events "
            "WHERE event_type = 'model.completed' ORDER BY sequence LIMIT 1"
        ).fetchone()
        # Nesting past the durable depth limit: reads admit it, since the
        # store's ingestion never writes one, so only verification finds the edit.
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM cayu_events WHERE sequence = ?", (sequence,)
            ).fetchone()[0]
        )
        nested: Any = "edited-marker"
        for _ in range(DURABLE_DOCUMENT_LIMITS.max_nesting + 1):
            nested = [nested]
        payload["edited"] = nested
        connection.execute(
            "UPDATE cayu_events SET payload_json = ? WHERE sequence = ?",
            (json.dumps(payload, separators=(",", ":")), sequence),
        )

    async def read_back() -> Any:
        store = SQLiteSessionStore(path)
        try:
            records = await store.query_events(EventQuery(session_id=session_id, limit=1000))
        finally:
            await store.close()
        return next(r.event.payload["edited"] for r in records if r.sequence == sequence)

    edited = asyncio.run(read_back())
    for _ in range(DURABLE_DOCUMENT_LIMITS.max_nesting + 1):
        edited = edited[0]
    assert edited == "edited-marker"

    code, result = _verify(path, capsys)

    assert code == storage_cli.VERIFY_EVENTS_EXIT_INVALID
    assert result["invalid"] == 1
    (failure,) = result["failures"]
    assert failure["sequence"] == sequence
    assert failure["session_id"] == session_id
    assert failure["event_type"] == "model.completed"
    assert "edited-marker" not in json.dumps(failure)
