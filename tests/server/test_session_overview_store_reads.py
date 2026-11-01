"""An overview page costs a fixed number of store reads, however many sessions it holds."""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from contextvars import ContextVar

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _drain, _start

from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.server import ServerConfig, create_server
from cayu.sessions import ForkSessionRequest
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]

_nested: ContextVar[bool] = ContextVar("overview_store_call_nested", default=False)


def _count_calls(target: object, calls: Counter, prefix: str) -> None:
    """Count each public coroutine call made from outside the target itself."""

    for name in dir(type(target)):
        if name.startswith("_"):
            continue
        static = inspect.getattr_static(target, name)
        if isinstance(static, property):
            continue
        method = getattr(target, name)
        if not inspect.iscoroutinefunction(method):
            continue

        async def counted(*args, __method=method, __name=name, **kwargs):
            if _nested.get():
                return await __method(*args, **kwargs)
            calls[f"{prefix}.{__name}"] += 1
            token = _nested.set(True)
            try:
                return await __method(*args, **kwargs)
            finally:
                _nested.reset(token)

        setattr(target, name, counted)


async def _family(app, artifacts, index: int, page: str) -> None:
    """A root with a child run and a fork, a question, and an approval."""

    root = f"{page}-root-{index}"
    await _start(app, root, "writer")
    await _drain(
        app.run(
            RunRequest(
                session_id=f"{page}-child-{index}",
                parent_session_id=root,
                agent_name="writer",
                messages=[Message.text("user", "go")],
            )
        )
    )
    await _drain(
        app.fork_session(
            ForkSessionRequest(source_session_id=root, session_id=f"{page}-fork-{index}")
        )
    )
    await _start(app, f"{page}-ask-{index}", "asker")
    await _start(app, f"{page}-approve-{index}", "approver")
    await artifacts.put_bytes(b"content", filename="out.txt", session_id=root)


@pytest.mark.store_semantics
def test_overview_page_store_reads_stay_fixed_as_the_page_grows(session_store_case, tmp_path):
    async def page_reads(families: int) -> tuple[Counter, dict]:
        kind, _, dsn = session_store_case
        (tmp_path / f"store-{families}").mkdir()
        store = await _open_store((kind, tmp_path / f"store-{families}", dsn))
        try:
            app = _app(store)
            artifacts = LocalArtifactStore(tmp_path / f"artifacts-{families}")
            app.register_environment(
                Environment(EnvironmentSpec(name="files"), artifact_store=artifacts)
            )
            for index in range(families):
                await _family(app, artifacts, index, f"p{families}")
            calls: Counter = Counter()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                params = {"include": "overview", "include_forks": "true", "limit": "100"}
                before = await client.get("/api/sessions", params=params)
                assert before.status_code == 200, before.text
                _count_calls(store, calls, "sessions")
                _count_calls(artifacts, calls, "artifacts")
                response = await client.get("/api/sessions", params=params)
            assert response.status_code == 200, response.text
            body = response.json()
            # Counting the calls changes nothing the page reports.
            assert body == before.json()
            assert len(body["overviews"]) == 5 * families
            prefix = f"p{families}-"
            return calls, {row["session_id"].removeprefix(prefix): row for row in body["overviews"]}
        finally:
            await _close_store(store)

    async def scenario():
        small, small_rows = await page_reads(2)
        large, large_rows = await page_reads(6)
        assert small == large, (small, large)
        # Each read the overview needs is made once for the whole page.
        for name in (
            "sessions.load_sessions",
            "sessions.summarize_outcomes",
            "sessions.query_pending_actions_for_sessions",
            "sessions.query_latest_events",
            "sessions.load_checkpoints",
            "sessions.load_session_operations",
            "sessions.count_child_runs",
            "artifacts.count_session_artifacts",
        ):
            assert large[name] >= 1, (name, large)
        for name in (
            "sessions.load",
            "sessions.summarize_outcome",
            "sessions.query_pending_actions",
            "sessions.load_checkpoint",
            "sessions.load_session_operation",
            "artifacts.list",
        ):
            assert large[name] == 0, (name, large)
        for rows in (small_rows, large_rows):
            assert rows["root-0"]["uses"] == {"browser": False, "subagents": "1", "artifacts": "1"}
            assert rows["root-0"]["forked"] is False and rows["fork-0"]["forked"] is True
            assert rows["ask-0"]["state"] == "waiting_input"
            assert rows["approve-0"]["state"] == "waiting_approval"
            assert rows["child-0"]["state"] == "completed"

    asyncio.run(scenario())
