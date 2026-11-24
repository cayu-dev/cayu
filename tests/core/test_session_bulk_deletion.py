"""``delete_sessions``, ``POST /api/sessions/deletions`` and ``cayu session delete``."""

from __future__ import annotations

import asyncio
import json

import pytest

from cayu import AgentSpec, CayuApp
from cayu.messages import Message
from cayu.sessions.bulk_deletion import delete_sessions
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest


async def _seed(app: CayuApp) -> None:
    for session_id, labels in (
        ("ana-1", {"person": "ana"}),
        ("ana-2", {"person": "ana"}),
        ("ben-1", {"person": "ben"}),
    ):
        await app.session_store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                labels=labels,
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )


def _app() -> CayuApp:
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    # Closing a session settles it through its registered agent.
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def test_unfinished_matches_stay_without_a_close_reason_and_go_with_one() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _seed(app)
            ana = SessionQuery(labels={"person": "ana"})
            kept = await delete_sessions(app, ana)
            assert sorted(item.session_id for item in kept.refused) == ["ana-1", "ana-2"]
            assert kept.deleted == () and not kept.done
            assert "give a close reason" in kept.refused[0].reason

            gone = await delete_sessions(app, ana, close_reason="Reset")
            assert sorted(gone.deleted) == ["ana-1", "ana-2"]
            assert sorted(gone.closed) == ["ana-1", "ana-2"] and gone.done
            assert await app.session_store.load("ana-1") is None
            ben = await app.session_store.load("ben-1")
            assert ben is not None and ben.status == SessionStatus.PENDING

        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_an_unfinished_tree_is_closed_once_and_deleted_recursively() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            for session_id, parent in (("p1", None), ("c1", "p1"), ("g1", "c1")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"job": "close"} if parent is None else {},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            outcome = await delete_sessions(
                app, SessionQuery(labels={"job": "close"}), close_reason="Reset"
            )
            # One close closes the open children too; the recursive closure takes them.
            assert outcome.deleted == ("p1",) and outcome.closed == ("p1",), outcome
            assert outcome.done
            for session_id in ("p1", "c1", "g1"):
                assert await app.session_store.load(session_id) is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_refused_parent_leaves_its_children_alone() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            for session_id, parent in (("p1", None), ("c1", "p1")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"job": "close"} if parent is None else {},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            await app.close_session("c1", reason="Done with it.")
            outcome = await delete_sessions(app, SessionQuery(labels={"job": "close"}))
            assert outcome.deleted == () and [r.session_id for r in outcome.refused] == ["p1"]
            assert await app.session_store.load("c1") is not None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_running_parent_that_refuses_to_close_keeps_its_children() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            for session_id, parent in (("p1", None), ("c1", "p1")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"job": "close"} if parent is None else {},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            await app.session_store.update_status("p1", SessionStatus.RUNNING)
            outcome = await delete_sessions(
                app, SessionQuery(labels={"job": "close"}), close_reason="Reset"
            )
            assert outcome.deleted == () and outcome.closed == ()
            assert [item.session_id for item in outcome.refused] == ["p1"]
            child = await app.session_store.load("c1")
            assert child is not None and child.status is SessionStatus.PENDING
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_finished_parent_with_a_finished_subagent_is_deleted_recursively() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            for session_id, parent in (("p1", None), ("c1", "p1")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"job": "close"} if parent is None else {},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            # Both end, with their terminal evidence, before the deletion runs.
            await app.close_session("p1", reason="Finished with it.")
            outcome = await delete_sessions(app, SessionQuery(labels={"job": "close"}))
            assert outcome.deleted == ("p1",) and outcome.closed == () and outcome.done
            assert await app.session_store.load("c1") is None
            # A match below another match is reported with it.
            for session_id, parent in (("p2", None), ("c2", "p2")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"job": "both"},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            await app.close_session("p2", reason="Finished with it.")
            both = await delete_sessions(app, SessionQuery(labels={"job": "both"}))
            assert sorted(both.deleted) == ["c2", "p2"] and both.done, both
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_more_matches_than_max_sessions_say_so() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _seed(app)
            query = SessionQuery(agent_name="assistant")
            first = await delete_sessions(app, query, close_reason="Reset", max_sessions=2)
            assert len(first.deleted) == 2 and first.has_more and not first.done
            rest = await delete_sessions(app, query, close_reason="Reset", max_sessions=2)
            assert len(rest.deleted) == 1 and rest.done
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _finished_parent_pending_child(app: CayuApp) -> None:
    """A finished parent and a pending child, both labelled."""

    async def create(session_id: str, parent: str | None) -> None:
        await app.session_store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                labels={"job": "tree"},
                parent_session_id=parent,
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )

    await create("p1", None)
    await app.close_session("p1", reason="Finished with it.")
    await create("c1", "p1")


def test_an_unfinished_descendant_needs_a_close_reason_too() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _finished_parent_pending_child(app)
            query = SessionQuery(labels={"job": "tree"})
            assert await app.session_descendant_ids("p1") == ("c1",)
            assert await app.session_descendant_ids("c1") == ()
            kept = await delete_sessions(app, query)
            assert kept.deleted == () and kept.closed == ()
            assert sorted(item.session_id for item in kept.refused) == ["c1", "p1"]
            for session_id in ("p1", "c1"):
                assert await app.session_store.load(session_id) is not None
            child = await app.session_store.load("c1")
            assert child is not None and child.status is SessionStatus.PENDING

            gone = await delete_sessions(app, query, close_reason="Reset")
            assert gone.closed == ("c1",) and sorted(gone.deleted) == ["c1", "p1"], gone
            assert gone.done
            for session_id in ("p1", "c1"):
                assert await app.session_store.load(session_id) is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _pending_parent_finished_child(app: CayuApp) -> None:
    for session_id, parent in (("p1", None), ("c1", "p1")):
        await app.session_store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                labels={"job": "pages"},
                parent_session_id=parent,
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )
    await app.close_session("c1", reason="Finished with it.")


async def _every_page(app: CayuApp, **options) -> tuple[list[str], list[str]]:
    deleted, refused, after = [], [], None
    while True:
        page = await delete_sessions(
            app, SessionQuery(labels={"job": "pages"}), max_sessions=1, after=after, **options
        )
        deleted += list(page.deleted)
        refused += [item.session_id for item in page.refused]
        if not page.has_more:
            return deleted, refused
        after = page.next_after


def test_a_match_below_one_refused_on_an_earlier_page_is_kept() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _pending_parent_finished_child(app)
            deleted, refused = await _every_page(app)
            assert deleted == [] and refused == ["p1", "c1"]
            assert await app.session_store.load("c1") is not None
            parent = await app.session_store.load("p1")
            assert parent is not None and parent.status is SessionStatus.PENDING
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_match_below_one_deleted_on_an_earlier_page_goes_with_it() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _pending_parent_finished_child(app)
            deleted, refused = await _every_page(app, close_reason="Reset")
            assert deleted == ["p1"] and refused == []
            for session_id in ("p1", "c1"):
                assert await app.session_store.load(session_id) is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_matches", [False, True])
def test_store_only_filters_decide_an_ancestor_by_the_store(parent_matches: bool) -> None:
    async def scenario() -> None:
        app = _app()
        try:
            for session_id, parent in (("p1", None), ("c1", "p1")):
                await app.session_store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                        labels={"cleanup": "yes"} if parent or parent_matches else {},
                        parent_session_id=parent,
                    ),
                    identity=SessionIdentity(provider_name="provider", model="model"),
                )
            await app.close_session("c1", reason="Finished with it.")
            # inactive_for_seconds is resolved by the store only.
            query = SessionQuery(labels={"cleanup": "yes"}, inactive_for_seconds=0)
            after, deleted, refused = None, [], []
            while True:
                page = await delete_sessions(app, query, max_sessions=1, after=after)
                deleted += list(page.deleted)
                refused += [item.session_id for item in page.refused]
                if not page.has_more:
                    break
                after = page.next_after
            if parent_matches:
                # The pending parent is refused (no close reason) and keeps its child.
                assert deleted == [] and refused == ["p1", "c1"]
                assert await app.session_store.load("c1") is not None
            else:
                # The parent never matched, so the child is a root of its own.
                assert deleted == ["c1"] and refused == []
                assert await app.session_store.load("c1") is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


async def _refused_then_finished(app: CayuApp) -> None:
    """Three unfinished matches, oldest, then two finished ones."""

    for session_id in ("a1", "a2", "a3", "b1", "b2"):
        await app.session_store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                labels={"job": "mixed"},
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )
    for session_id in ("b1", "b2"):
        await app.close_session(session_id, reason="Finished with it.")


def test_refused_matches_are_examined_once_and_later_pages_still_run() -> None:
    async def scenario() -> None:
        app = _app()
        try:
            await _refused_then_finished(app)
            query = SessionQuery(labels={"job": "mixed"})
            refused, deleted, after, calls = [], [], None, 0
            while True:
                calls += 1
                page = await delete_sessions(app, query, max_sessions=2, after=after)
                refused += [item.session_id for item in page.refused]
                deleted += list(page.deleted)
                if not page.has_more:
                    break
                assert page.next_after is not None
                after = page.next_after
            assert sorted(refused) == ["a1", "a2", "a3"] and sorted(deleted) == ["b1", "b2"]
            assert calls == 3
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_the_command_pages_past_refused_sessions(served, capsys, monkeypatch) -> None:
    from cayu.cli import main
    from cayu.cli import session as session_cli

    app, _client = served
    asyncio.run(_refused_then_finished(app))
    monkeypatch.setattr(session_cli, "_DELETE_BATCH", 2)
    code = main(["session", "delete", "--label", "job=mixed", "--application", "agent", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1  # Three were refused.
    assert sorted(out["deleted"]) == ["b1", "b2"]
    assert sorted(item["session_id"] for item in out["refused"]) == ["a1", "a2", "a3"]
    assert out["has_more"] is False


@pytest.mark.parametrize("arguments", [{"max_sessions": 0}, {"close_reason": " "}])
def test_bad_arguments_are_refused(arguments: dict) -> None:
    with pytest.raises(ValueError):
        asyncio.run(delete_sessions(_app(), SessionQuery(), **arguments))


@pytest.fixture
def served(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from cayu.cli import _cloud_agent_api
    from cayu.cli._cloud_agent_api import CayuApi
    from cayu.server import ServerConfig, create_server

    app = _app()
    asyncio.run(_seed(app))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        monkeypatch.setattr(
            _cloud_agent_api,
            "open_cloud_agent_api",
            lambda *args, **kwargs: CayuApi("http://testserver", http=client),
        )
        yield app, client


def test_the_route_returns_public_session_ids(served, monkeypatch) -> None:
    from cayu.vaults.redaction import SecretRedactor

    app, client = served
    secret = "redaction-test-value"
    monkeypatch.setattr(app, "_secret_redactor", SecretRedactor([secret]))

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                session_id=f"s-{secret}",
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                labels={"person": "cara"},
            ),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )

    asyncio.run(seed())
    answer = client.post("/api/sessions/deletions", json={"labels": {"person": "cara"}})
    assert answer.status_code == 200, answer.text
    assert secret not in answer.text
    [refusal] = answer.json()["refused"]
    assert refusal["session_id"] == app.project_session_id_for_exposure(f"s-{secret}")


def test_the_route_needs_a_filter_and_deletes_what_it_matches(served) -> None:
    app, client = served
    assert client.post("/api/sessions/deletions", json={}).status_code == 422
    answer = client.post(
        "/api/sessions/deletions",
        json={"labels": {"person": "ben"}, "close_reason": "Reset"},
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["deleted"] == ["ben-1"]
    assert asyncio.run(app.session_store.load("ben-1")) is None


def test_the_command_deletes_a_cloud_agents_sessions_by_label(served, capsys) -> None:
    from cayu.cli import main

    app, _client = served
    code = main(
        [
            "session",
            "delete",
            "--label",
            "person=ana",
            "--close-reason",
            "Reset",
            "--application",
            "agent",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    assert sorted(out["deleted"]) == ["ana-1", "ana-2"] and out["where"] == "cloud"
    assert asyncio.run(app.session_store.load("ana-2")) is None
    assert main(["session", "delete", "--application", "agent", "--json"]) == 1
    assert "Give at least one" in capsys.readouterr().out
