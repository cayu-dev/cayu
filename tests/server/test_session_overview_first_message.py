"""`include=first_message` returns each listed session's bounded first user text."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.core.test_cloud_reporting import _app, _drain

from cayu import RunRequest
from cayu.messages import Message, MessageRole, TextPart
from cayu.server import ServerConfig, create_server
from cayu.server.contracts import MAX_SESSION_FIRST_MESSAGE_LENGTH
from cayu.sessions.overview import OVERVIEW_READ_CONCURRENCY
from cayu.sessions.records import SessionIdentity
from cayu.vaults.redaction import SecretRedactor

SECRET = "PRIVATE-FIRST-MESSAGE-CANARY"
SPLIT = "SPLITCANARYVALUE"


async def _run(app, session_id: str, message: Message) -> None:
    await _drain(
        app.run(RunRequest(agent_name="writer", session_id=session_id, messages=[message]))
    )


async def _populate(app) -> None:
    await _run(app, "plain", Message.text("user", "  Summarize the release notes\nthen stop  "))
    await _run(app, "secret", Message.text("user", f"Use key {SECRET} to deploy"))
    await _run(app, "long", Message.text("user", "word " * 200))
    await _run(
        app,
        "parts",
        Message(
            role=MessageRole.USER,
            content=(TextPart(text="Use SPLIT"), TextPart(text="CANARYVALUE.")),
        ),
    )
    await app.session_store.create(
        RunRequest(agent_name="writer", session_id="empty", messages=[]),
        identity=SessionIdentity(provider_name="test", model="test"),
    )


def _overviews(client, *include: str) -> dict[str, dict]:
    response = client.get(
        "/api/sessions",
        params=[("include", "overview"), *(("include", item) for item in include)],
    )
    assert response.status_code == 200, response.text
    return {row["session_id"]: row for row in response.json()["overviews"]}


def _transcript_text(client, session_id: str) -> str:
    response = client.get(
        f"/api/sessions/{session_id}/transcript", params={"role": "user", "limit": 1}
    )
    assert response.status_code == 200
    (message,) = response.json()["messages"]
    return "".join(part["text"] for part in message["content"] if part["type"] == "text")


def test_first_message_is_opt_in_bounded_and_redacted_like_the_transcript(monkeypatch):
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        portal = client.portal
        assert portal is not None
        portal.call(_populate, app)
        # Stored text keeps the raw values; redaction happens on every read.
        monkeypatch.setattr(app, "_secret_redactor", SecretRedactor([SECRET, SPLIT]))

        assert all(row["first_message"] is None for row in _overviews(client).values())

        rows = _overviews(client, "first_message")
        assert rows["plain"]["first_message"] == "Summarize the release notes\nthen stop"
        assert rows["secret"]["first_message"] == _transcript_text(client, "secret")
        assert SECRET not in rows["secret"]["first_message"]
        assert "deploy" in rows["secret"]["first_message"]
        long = rows["long"]["first_message"]
        assert len(long) == MAX_SESSION_FIRST_MESSAGE_LENGTH
        assert long.endswith("…")
        assert _transcript_text(client, "long").strip().startswith(long[:-1])
        # The transcript redacts per part; the joined text is redacted again.
        assert SPLIT not in rows["parts"]["first_message"]
        assert rows["empty"]["first_message"] is None

        surface = client.get("/api/contract").json()["capabilities"]["surfaces"][
            "session_overview_first_message"
        ]
        assert surface["read"]["enabled"] is True
        assert surface["mutate"]["enabled"] is False


def test_first_message_requires_the_overview_include():
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.get("/api/sessions", params={"include": "first_message"})
        assert response.status_code == 422
        both = client.get(
            "/api/sessions", params=[("include", "usage"), ("include", "first_message")]
        )
        assert both.status_code == 422


def test_first_messages_are_one_page_read_and_drop_a_session_deleted_meanwhile(monkeypatch):
    app = _app()
    store = app.session_store

    async def populate():
        for index in range(12):
            await _run(app, f"row-{index}", Message.text("user", f"question {index}"))

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        portal = client.portal
        assert portal is not None
        portal.call(populate)
        original = store.query_first_transcript_records
        pages = []

        async def deleting(session_ids, role):
            pages.append(tuple(session_ids))
            await store.delete_session("row-3")
            return await original(session_ids, role)

        monkeypatch.setattr(store, "query_first_transcript_records", deleting)
        rows = _overviews(client, "first_message")
        assert len(pages) == 1 and len(pages[0]) == 12
        assert "row-3" not in rows
        assert rows["row-0"]["first_message"] == "question 0"
        assert rows["row-0"]["first_message_unavailable"] is False
        assert len(rows) == 11


def test_a_failed_page_read_falls_back_to_one_read_per_row(monkeypatch):
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.portal is not None
        client.portal.call(_populate, app)
        original = app.session_store.query_transcript
        running = peak = 0

        async def unavailable(session_ids, role):
            raise OSError("page read unavailable")

        async def tracked(query):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                await asyncio.sleep(0.01)
                return await original(query)
            finally:
                running -= 1

        monkeypatch.setattr(app.session_store, "query_first_transcript_records", unavailable)
        monkeypatch.setattr(app.session_store, "query_transcript", tracked)
        rows = _overviews(client, "first_message")
        assert 1 < peak <= OVERVIEW_READ_CONCURRENCY
        assert rows["plain"]["first_message"] == "Summarize the release notes\nthen stop"
        assert all(row["first_message_unavailable"] is False for row in rows.values())
        assert rows["empty"]["first_message"] is None


@pytest.mark.parametrize(
    "failure", [OSError("read unavailable"), KeyError("bad record"), ValueError("cannot decode")]
)
def test_first_message_failure_degrades_only_its_row(monkeypatch, failure):
    app = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.portal is not None
        client.portal.call(_populate, app)
        original = app.session_store.query_transcript

        async def failing(query):
            if query.session_id == "plain":
                raise failure
            return await original(query)

        monkeypatch.setattr(app.session_store, "query_transcript", failing)
        rows = _overviews(client, "first_message")
        assert len(rows) == 5
        assert rows["plain"]["first_message"] is None
        assert rows["plain"]["first_message_unavailable"] is True
        assert rows["long"]["first_message"]
        assert rows["long"]["first_message_unavailable"] is False
        assert rows["empty"]["first_message"] is None
        assert rows["empty"]["first_message_unavailable"] is False
