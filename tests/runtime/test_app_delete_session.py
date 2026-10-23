"""A shared app can delete finished sessions and keep in-memory state bounded."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import EnvironmentSpec, LocalWorkspaceEnvironmentFactory
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import session_closure
from cayu.sessions._closure_records import ClosureProgressIndex
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.tools import ReadFileTool


def _response(request):
    if any(message.role == "tool" for message in request.messages):
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
    return [
        ModelStreamEvent.tool_call(id="read", name="read_file", arguments={"path": "a.txt"}),
        ModelStreamEvent.completed(),
    ]


def _shared_app(tmp_path, store, roots):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(response_factory=_response), default=True)
    app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[ReadFileTool()])
    app.register_environment_factory(
        EnvironmentSpec(name="local"),
        LocalWorkspaceEnvironmentFactory(
            allowed_roots=[tmp_path],
            root_from_request=lambda request: roots[request.session_id],
        ),
        default=True,
    )
    return app


def _container_sizes(store):
    return {
        name: len(value)
        for name, value in vars(store).items()
        if isinstance(value, dict | set | list | tuple)
    }


def test_shared_app_deletes_finished_sessions_with_bounded_store_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session_closure, "_COMPLETED_PLAN_ID_LIMIT", 3)
    store = InMemorySessionStore(closure_receipt_limit=3)
    roots = {}
    app = _shared_app(tmp_path, store, roots)

    async def serve(index):
        session_id = f"request-{index}"
        root = tmp_path / session_id
        root.mkdir()
        (root / "a.txt").write_text("hello")
        roots[session_id] = root
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="reader",
                    messages=[Message.text("user", "read a.txt")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        report = await app.delete_session(session_id)
        assert report is not None and report.complete
        del roots[session_id]
        assert await store.load(session_id) is None

    async def scenario():
        for index in range(6):
            await serve(index)
        warmed_up = _container_sizes(store)
        for index in range(6, 30):
            await serve(index)
        assert _container_sizes(store) == warmed_up
        assert len(store._session_closure_receipts) == 3
        assert all(len(receipt) < 16_384 for receipt in store._session_closure_receipts.values())
        assert not store._session_closure_progress
        assert not store._public_authority_aliases
        assert not store._sessions
        closure = app._environment_registry.session_closure
        assert len(closure._completed_plan_ids) == 3
        await app.aclose()

    asyncio.run(scenario())


def _create(store, session_id):
    return store.create(
        RunRequest(session_id=session_id, agent_name="worker", messages=[]),
        identity=SessionIdentity(provider_name="scripted", model="test"),
    )


def test_delete_session_refuses_sessions_that_can_still_run():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def scenario():
        assert await app.delete_session("missing") is None
        await _create(store, "pending")
        with pytest.raises(ValueError, match="Only completed or failed sessions"):
            await app.delete_session("pending")
        assert await store.load("pending") is not None
        await store.update_status("pending", SessionStatus.RUNNING)
        with pytest.raises(ValueError, match="terminal sessions"):
            await app.delete_session("pending", finished_only=False)
        assert await store.load("pending") is not None
        await _create(store, "never-ran")
        report = await app.delete_session("never-ran", finished_only=False)
        assert report is not None and report.complete
        assert await store.load("never-ran") is None
        await app.aclose()

    asyncio.run(scenario())


def test_evicted_closure_receipt_frees_the_session_id():
    store = InMemorySessionStore(closure_receipt_limit=1)
    app = CayuApp(session_store=store, enable_logging=False)

    async def scenario():
        await _create(store, "first")
        await app.delete_session("first", finished_only=False)
        with pytest.raises(ValueError, match="retired by closure"):
            await _create(store, "first")
        await _create(store, "second")
        deleted = await app.delete_session("second", finished_only=False)
        replayed = await app.erase_session_closure("second")
        assert replayed.complete and replayed.already_absent
        assert replayed.plan_id == deleted.plan_id
        # The second receipt evicted the first, so its id can be used again.
        await _create(store, "first")
        with pytest.raises(ValueError, match="retired by closure"):
            await _create(store, "second")
        await app.aclose()

    asyncio.run(scenario())


def test_closure_receipt_limit_must_be_positive():
    with pytest.raises(ValueError, match="closure_receipt_limit"):
        InMemorySessionStore(closure_receipt_limit=0)


def test_closure_progress_index_finds_owners_by_target():
    index = ClosureProgressIndex()
    first = {
        "root_session_id": "root",
        "plan_id": "plan",
        "descendants": [{"session_id": "child", "parent_session_id": "root"}],
    }
    second = {"root_session_id": "other", "plan_id": "plan", "descendants": []}
    index["root", "plan"] = first
    index["other", "plan"] = second
    assert index.owners(("child",)) == (first,)
    assert index.owners(("child", "other")) == (first, second)
    assert index.owns("root") and not index.owns("unrelated")
    assert index.targets() == {"root", "child", "other"}
    assert index.pop(("root", "plan")) is first
    assert index.owners(("child",)) == ()
    assert index.targets() == {"other"}
    del index["other", "plan"]
    assert not index and not index.targets()


def test_receipt_eviction_cannot_reopen_a_retired_task_namespace():
    from cayu.sessions.requests import ForkSessionRequest
    from cayu.tasks import InMemoryTaskStore, TaskCreate

    store = InMemorySessionStore(closure_receipt_limit=1)
    tasks = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(response_factory=lambda request: [ModelStreamEvent.completed()]),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker", model="scripted"))

    async def run(session_id):
        return [
            event
            async for event in app.run(
                RunRequest(session_id=session_id, agent_name="worker", messages=[])
            )
        ]

    async def scenario():
        for session_id in ("first", "second"):
            await run(session_id)
            await app.delete_session(session_id)
        assert "first" not in store._session_closure_receipt_counts
        claim = await tasks.load_session_closure_claim("first")
        assert claim is not None and claim.purpose == "deletion"
        with pytest.raises(ValueError, match="retired by task closure"):
            await run("first")
        assert await store.load("first") is None
        await run("source")
        with pytest.raises(ValueError, match="retired by task closure"):
            async for _ in app.fork_session(
                ForkSessionRequest(source_session_id="source", session_id="first")
            ):
                pass
        assert await store.load("first") is None
        with pytest.raises(ValueError, match="owned by closure"):
            await tasks.create_task(TaskCreate(type="test", session_id="first"))
        assert await tasks.load_session_closure_claim("first") == claim
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_finished_only_is_checked_under_closure_claim_after_inventory_race(
    backend, tmp_path, request, monkeypatch
):
    from uuid import uuid4

    from tests.core.session_closure_conformance import create_closure_session
    from tests.core.test_tool_effect_store_conformance import _stores

    from cayu.events import Event

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "finished-race.db", dsn) as open_store:
            store, other = open_store(), open_store()
            session_id = uuid4().hex
            await create_closure_session(store, session_id)
            original = store.load_session_closure_records
            raced = False

            async def inventory(*args, **kwargs):
                nonlocal raced
                if not raced:
                    raced = True
                    await other.update_status(session_id, SessionStatus.RUNNING)
                    await other.update_status(session_id, SessionStatus.INTERRUPTED)
                    await other.append_event(
                        session_id,
                        Event(
                            id=uuid4().hex,
                            session_id=session_id,
                            type=EventType.SESSION_INTERRUPTED,
                            payload={},
                        ),
                    )
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, "load_session_closure_records", inventory)
            app = CayuApp(session_store=store, enable_logging=False)
            with pytest.raises(ValueError, match="Only completed or failed sessions"):
                await app.delete_session(session_id)
            assert raced
            assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
            # Refusal published no claim: the resumable session is still mutable.
            await other.update_status(session_id, SessionStatus.COMPLETED)
            await other.append_event(
                session_id,
                Event(
                    id=uuid4().hex,
                    session_id=session_id,
                    type=EventType.SESSION_COMPLETED,
                    payload={},
                ),
            )
            report = await app.delete_session(session_id)
            assert report.complete
            assert await store.load(session_id) is None

    asyncio.run(scenario())


def test_finished_only_refuses_store_without_atomic_admission_support(monkeypatch):
    from tests.core.session_closure_conformance import create_closure_session

    store = InMemorySessionStore()
    monkeypatch.setattr(store, "supports_session_closure_finished_only", False)
    app = CayuApp(session_store=store, enable_logging=False)

    async def scenario():
        await create_closure_session(store, "unsupported")
        with pytest.raises(ValueError, match="cannot enforce finished-only"):
            await app.delete_session("unsupported")
        assert await store.load("unsupported") is not None
        # Existing unconstrained callers do not need the new optional keyword.
        assert (await app.erase_session_closure("unsupported")).complete
        await app.aclose()

    asyncio.run(scenario())


def test_detached_fork_retains_source_alias_until_last_child_is_deleted():
    from cayu.runtime.session_closure import SessionClosureChildPolicy, SessionClosurePolicy
    from cayu.sessions.requests import ForkSessionRequest
    from cayu.vaults.redaction import SecretRedactor

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store, secret_redactor=SecretRedactor("private-source"), enable_logging=False
    )
    app.register_provider(ScriptedModelProvider(events=[]), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted"))

    async def scenario():
        source_id = "legacy-private-source"
        source_app = CayuApp(session_store=store, enable_logging=False)
        source_app.register_provider(
            ScriptedModelProvider(events=[ModelStreamEvent.completed()]), default=True
        )
        source_app.register_agent(AgentSpec(name="worker", model="scripted"))
        [
            event
            async for event in source_app.run(
                RunRequest(session_id=source_id, agent_name="worker", messages=[])
            )
        ]
        alias = app.project_session_id_for_exposure(source_id)
        assert alias != source_id
        await store.register_public_authority_alias(
            alias, field_name="session_id", private_value=source_id
        )
        requests = [
            ForkSessionRequest(source_session_id=alias, session_id=f"child-{index}")
            for index in range(2)
        ]
        originals = [[event async for event in app.fork_session(request)] for request in requests]
        assert "invocation_configuration" in (await store.load_checkpoint(source_id))
        for fork_request in requests:
            assert "invocation_configuration" not in (
                await store.load_checkpoint(fork_request.session_id) or {}
            )
        report = await app.erase_session_closure(
            source_id, policy=SessionClosurePolicy(child_policy=SessionClosureChildPolicy.DETACH)
        )
        assert report.complete
        for request, original in zip(requests, originals, strict=True):
            replay = [event async for event in app.fork_session(request)]
            assert [event.id for event in replay] == [event.id for event in original]
        with pytest.raises(KeyError, match="Fork source session was not found"):
            [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(source_session_id=alias, session_id="new-child")
                )
            ]
        for index, request in enumerate(requests):
            await app.erase_session_closure(request.session_id)
            resolved = await store.resolve_public_authority_alias(alias, field_name="session_id")
            assert resolved == (source_id if index == 0 else None)
        await app.aclose()
        await source_app.aclose()

    asyncio.run(scenario())
