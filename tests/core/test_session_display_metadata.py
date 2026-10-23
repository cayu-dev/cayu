from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from cayu import AgentSpec, Message
from cayu.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage import SQLiteSessionStore


def run_async(func):
    @wraps(func)
    def run(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return run


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    result = (
        InMemorySessionStore()
        if request.param == "memory"
        else SQLiteSessionStore(tmp_path / "sessions.db")
    )
    yield result


async def create(store, **kwargs):
    return await store.create(
        RunRequest(agent_name="assistant", messages=[Message.text("user", "Hello")], **kwargs),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )


@run_async
async def test_title_survives_creation_and_metadata_replacement(store):
    session = await create(store, title="Quarterly review")
    assert session.metadata["cayu:title"] == "Quarterly review"
    await store.update_metadata(session.id, {"owner": "accounting"})
    loaded = await store.load(session.id)
    assert loaded.metadata["cayu:title"] == "Quarterly review"


@run_async
async def test_summary_and_annotation_updates_compose(store):
    session = await create(store)
    await asyncio.gather(
        store.set_summary(session.id, "Review required", tone="attention"),
        store.update_metadata(session.id, {"ticket": "123"}),
    )
    loaded = await store.load(session.id)
    assert loaded.metadata["ticket"] == "123"
    assert loaded.metadata["cayu:summary"] == "Review required"
    assert loaded.metadata["cayu:summary_tone"] == "attention"
    assert loaded.last_activity_at == session.last_activity_at


@pytest.mark.parametrize("value", [True, 7, "", " ", "x" * 257, "\ud800"])
def test_invalid_title_rejected(value):
    with pytest.raises(ValueError):
        RunRequest(agent_name="assistant", messages=[], title=value)


@pytest.mark.parametrize("value", [True, 7, "", " ", "x" * 1025, "\ud800"])
def test_invalid_description_rejected(value):
    with pytest.raises(ValueError):
        AgentSpec(name="assistant", model="fake", metadata={"cayu:description": value})


@run_async
async def test_summary_invalid_tone_is_atomic(store):
    session = await create(store)
    await store.set_summary(session.id, "Done")
    with pytest.raises(ValueError):
        await store.set_summary(session.id, "Changed", tone="warning")
    loaded = await store.load(session.id)
    assert loaded.metadata["cayu:summary"] == "Done"
    assert loaded.metadata["cayu:summary_tone"] == "neutral"


@pytest.mark.parametrize("key", ["cayu:title", "cayu:summary", "cayu:summary_tone"])
def test_raw_display_metadata_requires_typed_api(key):
    with pytest.raises(ValueError):
        RunRequest(agent_name="assistant", messages=[], metadata={key: "forged"})


@run_async
async def test_sdk_completion_summary_and_sqlite_reopen(tmp_path):
    from tests.server.server_test_support import OneShotProvider

    from cayu import CayuApp
    from cayu.sessions import ForkSessionRequest

    path = tmp_path / "persistent.db"
    store = SQLiteSessionStore(path)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="display-metadata-run",
                title="Review",
                messages=[Message.text("user", "hello")],
            )
        )
    ]
    assert events
    await app.session_store.set_summary("display-metadata-run", "Reviewed", tone="attention")
    await store.update_metadata("display-metadata-run", {"ticket": "42"})
    from cayu.sessions import ResumeRequest

    resumed = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="display-metadata-run",
                messages=[Message.text("user", "Follow up")],
            )
        )
    ]
    assert resumed
    assert (await store.load("display-metadata-run")).metadata["cayu:summary"] == "Reviewed"
    fork_events = [
        event
        async for event in app.fork_session(
            ForkSessionRequest(
                source_session_id="display-metadata-run",
                session_id="display-metadata-fork",
            )
        )
    ]
    assert fork_events
    fork = await store.load("display-metadata-fork")
    assert fork.metadata["cayu:title"] == "Review"
    assert fork.metadata["cayu:summary"] == "Reviewed"
    await app.aclose()
    reopened = SQLiteSessionStore(path)
    try:
        session = await reopened.load("display-metadata-run")
        assert session.metadata["cayu:title"] == "Review"
        assert session.metadata["cayu:summary"] == "Reviewed"
        assert session.metadata["cayu:summary_tone"] == "attention"
        assert session.metadata["ticket"] == "42"
    finally:
        await reopened.close()


@run_async
async def test_cancel_summary_waiting_for_owner_preserves_metadata(store):
    session = await create(store)
    await store._lock.acquire()
    task = asyncio.create_task(store.set_summary(session.id, "Never committed"))
    try:
        await asyncio.sleep(0)
        task.cancel()
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
    finally:
        store._lock.release()
    loaded = await store.load(session.id)
    assert "cayu:summary" not in loaded.metadata


@run_async
async def test_mutated_title_rejected_without_diagnostics(store, caplog, capsys):
    import warnings

    class Hostile:
        def __repr__(self):
            return "DISPLAY_SECRET_CANARY"

        __str__ = __repr__

    request = RunRequest(agent_name="assistant", messages=[])
    request.title = Hostile()
    with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError) as error:
        await store.create(request, identity=SessionIdentity(provider_name="fake", model="fake"))
    captured = capsys.readouterr()
    assert "DISPLAY_SECRET_CANARY" not in str(error.value) + repr(
        error.value
    ) + caplog.text + captured.out + captured.err + str(caught)


@run_async
async def test_stale_run_cannot_overwrite_summary(store):
    from cayu.sessions import SessionRunFenced, SessionStatus

    session = await create(store)
    await store.update_status(session.id, SessionStatus.COMPLETED)
    running, release = asyncio.Event(), asyncio.Event()

    async def stale():
        await store.transition_status(
            session.id, from_statuses={SessionStatus.COMPLETED}, to_status=SessionStatus.RUNNING
        )
        running.set()
        await release.wait()
        with pytest.raises(SessionRunFenced):
            await store.set_summary(session.id, "Stale")

    task = asyncio.create_task(stale())
    await running.wait()
    assert await store.fence_stalled_run(
        session.id, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
    )
    await store.transition_status(
        session.id, from_statuses={SessionStatus.RUNNING}, to_status=SessionStatus.INTERRUPTED
    )
    await store.release_run_fence(session.id)
    await store.set_summary(session.id, "Authoritative")
    release.set()
    await task
    assert (await store.load(session.id)).metadata["cayu:summary"] == "Authoritative"


@run_async
async def test_scoped_summary_requires_modify_and_visibility(store):
    from cayu import CayuApp
    from cayu.resource_access import ResourceAccessPolicy
    from cayu.sessions._access_rules import (
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
    )
    from cayu.sessions.access import SessionAccessDenied

    rule = SessionAccessRule(selectors=(SessionAccessSelector(key="org", values=("mine",)),))

    class Policy(ResourceAccessPolicy):
        authority = "metadata-test"
        current = SessionAccessScope(read=(rule,), modify=(rule,))

        async def resolve(self, subject):
            return self.current

    policy = Policy()
    app = CayuApp(session_store=store, resource_access_policy=policy, enable_logging=False)
    own = await create(store, labels={"org": "mine"})
    other = await create(store, labels={"org": "other"})
    access = await app.access("alice")
    await access.sessions.set_summary(own.id, "Visible")
    with pytest.raises((SessionAccessDenied, KeyError)):
        await access.sessions.set_summary(other.id, "Forbidden")
    policy.current = SessionAccessScope(read=(rule,))
    with pytest.raises((SessionAccessDenied, KeyError)):
        await access.sessions.set_summary(own.id, "Forbidden")
    assert (await store.load(own.id)).metadata["cayu:summary"] == "Visible"
    assert "cayu:summary" not in (await store.load(other.id)).metadata


@run_async
async def test_postgres_summary_and_title_persist(postgres_dsn):
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    try:
        session = await create(store, title="Database review")
        await asyncio.gather(
            store.set_summary(session.id, "Done"), store.update_metadata(session.id, {"ticket": 1})
        )
        loaded = await store.load(session.id)
        assert loaded.metadata["cayu:title"] == "Database review"
        assert loaded.metadata["cayu:summary"] == "Done"
        assert loaded.metadata["ticket"] == 1
    finally:
        await store.close()


@run_async
async def test_model_tool_summary_and_annotations_keep_resource_bounds(store):
    from cayu import CayuApp
    from cayu.evals import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent
    from cayu.resource_access import ResourceAccessPolicy
    from cayu.sessions._access_rules import (
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
    )
    from cayu.sessions.access import SessionAccessDenied
    from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec

    rule = SessionAccessRule(selectors=(SessionAccessSelector(key="org", values=("mine",)),))

    class Policy(ResourceAccessPolicy):
        authority = "display-tool-test"

        async def resolve(self, subject):
            return SessionAccessScope(read=(rule,), create=(rule,), execute=(rule,), modify=(rule,))

    foreign = await create(store, labels={"org": "foreign"})
    observations = []

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Summarize outcome",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.IDEMPOTENT,
        )

        async def run(self, ctx, args):
            await store.set_summary(ctx.session_id, "Tool completed")
            await store.update_metadata(ctx.session_id, {"tool": True})
            for write in (
                lambda: store.set_summary(foreign.id, "Forbidden"),
                lambda: store.update_metadata(foreign.id, {"forged": True}),
            ):
                with pytest.raises((SessionAccessDenied, KeyError)):
                    await write()
            observations.append(True)
            return ToolResult(content="Done")

    app = CayuApp(session_store=store, resource_access_policy=Policy(), enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call-summary", name="summarize", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=[Summarize()])
    access = await app.access("alice")
    events = [
        event
        async for event in access.run(
            RunRequest(
                agent_name="assistant",
                session_id="display-tool-run",
                messages=[Message.text("user", "Review")],
                labels={"org": "mine"},
                title="Tool review",
            )
        )
    ]
    assert events and observations == [True]
    completed = await store.load("display-tool-run")
    assert completed.metadata["cayu:summary"] == "Tool completed"
    assert completed.metadata["tool"] is True
    assert "cayu:summary" not in (await store.load(foreign.id)).metadata
    await app.aclose()


@run_async
async def test_display_boundaries_and_invalid_summary_are_atomic(store):
    session = await create(store, title="t" * 256)
    await store.set_summary(session.id, "s" * 4096)
    assert AgentSpec(name="assistant", model="fake", metadata={"cayu:description": "d" * 1024})
    for invalid in (True, 1, "", " ", "s" * 4097, "\ud800", "two\nlines"):
        with pytest.raises(ValueError):
            await store.set_summary(session.id, invalid)
    loaded = await store.load(session.id)
    assert loaded.metadata["cayu:summary"] == "s" * 4096
    assert loaded.metadata["cayu:title"] == "t" * 256


@run_async
async def test_readonly_sqlite_summary_cannot_mutate(tmp_path):
    import sqlite3

    from cayu.storage.migrations import SchemaMode

    path = tmp_path / "readonly.db"
    writable = SQLiteSessionStore(path)
    session = await create(writable, title="Read only")
    await writable.close()
    readonly = SQLiteSessionStore(path, read_only=True, schema_mode=SchemaMode.VALIDATE)
    try:
        with pytest.raises(sqlite3.OperationalError):
            await readonly.set_summary(session.id, "Forbidden")
        assert "cayu:summary" not in (await readonly.load(session.id)).metadata
    finally:
        await readonly.close()


def test_mutated_description_registration_rejects_without_diagnostics(caplog, capsys):
    import warnings

    from cayu import CayuApp

    class Hostile:
        def __repr__(self):
            return "DESCRIPTION_SECRET_CANARY"

        __str__ = __repr__

    spec = AgentSpec(name="assistant", model="fake")
    spec.metadata["cayu:description"] = Hostile()
    app = CayuApp(enable_logging=False)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError) as error:
        app.register_agent(spec)
    captured = capsys.readouterr()
    assert "DESCRIPTION_SECRET_CANARY" not in str(error.value) + repr(
        error.value
    ) + caplog.text + captured.out + captured.err + str(caught)


def test_fork_display_metadata_rejection_hides_sibling_values(caplog, capsys):
    import warnings

    from cayu.sessions import ForkSessionRequest

    with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError) as error:
        ForkSessionRequest(
            source_session_id="source",
            metadata={
                "cayu:title": "FORBIDDEN_TITLE_CANARY",
                "sibling": "SIBLING_SECRET_CANARY",
            },
        )
    captured = capsys.readouterr()
    diagnostics = (
        str(error.value)
        + repr(error.value)
        + caplog.text
        + captured.out
        + captured.err
        + str(caught)
    )
    assert "CANARY" not in diagnostics


@pytest.mark.parametrize("entrance", ["create", "run"])
@run_async
async def test_scoped_creation_validates_title_before_copying(store, caplog, capsys, entrance):
    import warnings

    from tests.core.test_resource_execution_access import Policy

    from cayu import CayuApp

    class Hostile:
        def __deepcopy__(self, memo):
            raise RuntimeError("COPY_SECRET_CANARY")

        def __repr__(self):
            return "COPY_SECRET_CANARY"

    app = CayuApp(session_store=store, resource_access_policy=Policy(), enable_logging=False)
    access = await app.access("alice")
    request = RunRequest(agent_name="assistant", messages=[], labels={"organization": "acme"})
    request.title = Hostile()
    with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError) as error:
        if entrance == "create":
            await access.sessions.create(
                request, identity=SessionIdentity(provider_name="fake", model="fake")
            )
        else:
            _ = [event async for event in access.run(request)]
    captured = capsys.readouterr()
    assert "COPY_SECRET_CANARY" not in str(
        error.value
    ) + caplog.text + captured.out + captured.err + str(caught)


@pytest.mark.parametrize("title_fields", [{}, {"title": None}, {"title": "Quarterly review"}])
def test_current_request_title_preserves_durable_digest_and_portable_source(title_fields):
    from cayu.runtime._durable_subagents import durable_subagent_request_sha256
    from cayu.runtime._session_request_boundary import work_attempt_source_document

    request = RunRequest(
        agent_name="assistant", messages=[Message.text("user", "Hello")], **title_fields
    )
    digest = durable_subagent_request_sha256(request)
    restored = RunRequest.model_validate_json(request.model_dump_json())
    assert restored.title == request.title
    assert durable_subagent_request_sha256(restored) == digest
    document = work_attempt_source_document(request)
    assert document["title"] == request.title
    assert work_attempt_source_document(RunRequest.model_validate(document)) == document
    changed = request.model_copy(update={"title": "Different title"})
    assert durable_subagent_request_sha256(changed) != digest
