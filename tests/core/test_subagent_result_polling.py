"""SubagentResultTool retrieval efficiency: tail-limited reads, skip-terminal,
backoff polling, and full (un-truncated) child enumeration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.tools import ToolContext
from cayu.runtime.sessions import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStatus,
)
from cayu.tools.subagents import (
    SUBAGENT_RESULT_POLL_MAX_INTERVAL_S,
    SUBAGENT_RESULT_POLL_MIN_INTERVAL_S,
    SubagentResultTool,
    _list_background_subagent_children,
    _wait_for_all_subagents_terminal,
    _wait_for_subagent_terminal,
)


async def _seed_background_child(store, *, parent_id, child_id, assistant_texts, status):
    identity = SessionIdentity(provider_name="fake", model="fake-model")
    await store.create(
        RunRequest(
            agent_name="reviewer",
            session_id=child_id,
            parent_session_id=parent_id,
            messages=[Message.text("user", "child task")],
            metadata={"subagent": {"agent": "reviewer", "mode": "background"}},
        ),
        identity=identity,
    )
    messages = [Message.text("user", "child task")]
    for text in assistant_texts:
        messages.append(Message.text("assistant", text))
    await store.append_transcript_messages(child_id, messages)
    await store.append_events(
        child_id,
        [
            Event(type=EventType.SESSION_STARTED, session_id=child_id),
            Event(type=EventType.SESSION_COMPLETED, session_id=child_id),
        ],
    )
    await store.update_status(child_id, status)


def test_summary_returns_last_assistant_text_and_event_count():
    async def run():
        store = InMemorySessionStore()
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="parent",
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await _seed_background_child(
            store,
            parent_id="parent",
            child_id="child",
            assistant_texts=["first draft", "final answer"],
            status=SessionStatus.COMPLETED,
        )
        tool = SubagentResultTool(store)
        return await tool.run(
            ToolContext(session_id="parent"),
            {"child_session_id": "child", "wait": False},
        )

    result = asyncio.run(run())
    assert result.is_error is not True
    # Only the LAST assistant message is materialized, not the whole transcript.
    assert result.content == "final answer"
    assert result.structured["result_text"] == "final answer"
    assert result.structured["result_truncated"] is False
    # Event count is the full total from summarize_events, not a loaded list.
    assert result.structured["events"] == 2
    assert result.structured["terminal_event_type"] == str(EventType.SESSION_COMPLETED)


def test_summary_truncates_long_last_assistant_text():
    async def run():
        store = InMemorySessionStore()
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="parent",
                messages=[Message.text("user", "parent")],
            ),
            identity=identity,
        )
        await _seed_background_child(
            store,
            parent_id="parent",
            child_id="child",
            assistant_texts=["x" * 500],
            status=SessionStatus.COMPLETED,
        )
        tool = SubagentResultTool(store)
        return await tool.run(
            ToolContext(session_id="parent"),
            {"child_session_id": "child", "wait": False, "max_chars": 10},
        )

    result = asyncio.run(run())
    assert result.structured["result_text"] == "x" * 10
    assert result.structured["result_truncated"] is True


class _FakeChild:
    def __init__(self, child_id: str, status: SessionStatus) -> None:
        self.id = child_id
        self.status = status


class _FlippingStore:
    """load() reports RUNNING until the Nth call, then COMPLETED."""

    def __init__(self, *, flip_after: int) -> None:
        self.flip_after = flip_after
        self.loads: dict[str, int] = {}

    async def load(self, session_id: str) -> _FakeChild:
        self.loads[session_id] = self.loads.get(session_id, 0) + 1
        status = (
            SessionStatus.COMPLETED
            if self.loads[session_id] >= self.flip_after
            else SessionStatus.RUNNING
        )
        return _FakeChild(session_id, status)


def test_wait_for_one_uses_exponential_backoff(monkeypatch):
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    store = _FlippingStore(flip_after=5)
    child = _FakeChild("child", SessionStatus.RUNNING)
    result = asyncio.run(_wait_for_subagent_terminal(store, child, wait=True, timeout_s=100.0))
    assert result.status is SessionStatus.COMPLETED
    # First interval is the min; delays are non-decreasing and never exceed the cap.
    assert sleeps[0] == SUBAGENT_RESULT_POLL_MIN_INTERVAL_S
    assert sleeps == sorted(sleeps)
    assert all(delay <= SUBAGENT_RESULT_POLL_MAX_INTERVAL_S for delay in sleeps)
    # Genuine backoff, not a fixed interval.
    assert sleeps[1] > sleeps[0]


def test_wait_for_all_skips_already_terminal_children(monkeypatch):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    store = _FlippingStore(flip_after=3)
    children = [
        _FakeChild("done", SessionStatus.COMPLETED),
        _FakeChild("running", SessionStatus.RUNNING),
    ]
    result = asyncio.run(_wait_for_all_subagents_terminal(store, children, timeout_s=100.0))
    statuses = {child.id: child.status for child in result}
    assert statuses["done"] is SessionStatus.COMPLETED
    assert statuses["running"] is SessionStatus.COMPLETED
    # The already-terminal child is never reloaded; only the running one is polled.
    assert "done" not in store.loads
    assert store.loads["running"] >= 1


class _PagingListStore:
    """Minimal store exposing list_sessions across multiple keyset pages."""

    def __init__(self, pages) -> None:
        self._pages = pages
        self.queries = []

    async def list_sessions(self, query):
        self.queries.append(query)
        return self._pages[len(self.queries) - 1]


def _bg_session(session_id: str, *, background: bool = True):
    mode = "background" if background else "foreground"
    return SimpleNamespace(id=session_id, metadata={"subagent": {"mode": mode}})


def test_list_children_paginates_beyond_a_single_page():
    page1 = SimpleNamespace(
        sessions=[_bg_session("c1"), _bg_session("c2")],
        next_cursor="cursor-1",
    )
    page2 = SimpleNamespace(
        sessions=[_bg_session("c3"), _bg_session("skip", background=False)],
        next_cursor=None,
    )
    store = _PagingListStore([page1, page2])
    children = asyncio.run(_list_background_subagent_children(store, parent_session_id="parent"))
    assert [child.id for child in children] == ["c1", "c2", "c3"]
    # Two pages were fetched, and the second carried the first page's cursor.
    assert len(store.queries) == 2
    assert store.queries[0].cursor is None
    assert store.queries[1].cursor == "cursor-1"
