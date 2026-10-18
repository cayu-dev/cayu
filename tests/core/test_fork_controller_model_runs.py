"""The host's model-run gate for fork controllers, on every built-in session store."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_session_fork_overlays import (
    LOOKUP,
    SUBJECT,
    ScriptedProvider,
    _calls,
    _completed,
    _controller_server,
    _controller_toolset,
    _drain,
    _fork_request,
    _host_app,
    _http,
    _run_trunk,
)

from cayu import EventType, Message, ResumeRequest
from cayu.fork_controllers import ForkControllerDenied
from cayu.sessions.event_queries import EventQuery

__all__ = ["conformance_postgres_dsn", "session_store_case"]

DENIED = "The host has not allowed this controller to run its forks."
CHILD = "compass-child"
RUN_CHILD = "compass-run-child"


def _in_store(case, scenario):
    async def run():
        store = await _open_store(case)
        try:
            return await scenario(store)
        finally:
            await _close_store(store)

    return asyncio.run(run())


async def _refusal(stream) -> str | None:
    try:
        await _drain(stream)
    except ForkControllerDenied as exc:
        return str(exc)
    return None


def _invocation(session_id: str) -> ResumeRequest:
    return ResumeRequest(session_id=session_id, messages=[Message.text("user", "Investigate.")])


def _invoking_fork(source, session_id: str = RUN_CHILD):
    return _fork_request(
        session_id,
        expected_source=source,
        initial_invocation=_invocation(session_id),
        initial_dispatch_id=f"{session_id}-dispatch",
    )


def test_default_enrollment_creates_forks_but_never_runs_them(session_store_case):
    async def scenario(store):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed")])
        app, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        server = _controller_server(app)
        source = await controller.snapshot_source("trunk")
        refused = {
            "sdk_fork": await _refusal(controller.fork(_invoking_fork(source))),
            "http_fork": await _http(
                server,
                "POST",
                "/fork-controller/fork",
                json=_invoking_fork(source).model_dump(mode="json"),
            ),
        }
        invoked_child = await store.load(RUN_CHILD)
        # A fork without an invocation still works; running it does not.
        forked = await _drain(controller.fork(_fork_request(CHILD)))
        created = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        refused["sdk_resume"] = await _refusal(controller.resume(_invocation(CHILD)))
        refused["http_resume"] = await _http(
            server,
            "POST",
            "/fork-controller/resume",
            json=_invocation(CHILD).model_dump(mode="json"),
        )
        after = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        return refused, invoked_child, forked, (created, after), provider, controller_server

    refused, invoked_child, forked, (created, after), provider, controller_server = _in_store(
        session_store_case, scenario
    )
    assert refused["sdk_fork"] == DENIED and refused["sdk_resume"] == DENIED
    for name in ("http_fork", "http_resume"):
        assert refused[name].status_code == 403
        assert refused[name].json() == {"detail": "Fork controller access denied."}
    # Nothing was created, dispatched or run for the refused requests.
    assert invoked_child is None
    assert [event.type for event in forked] == [EventType.SESSION_FORKED]
    assert [record.sequence for record in after] == [record.sequence for record in created]
    assert EventType.MODEL_STARTED not in {record.event.type for record in after}
    assert len(provider.requests) == 1  # the host's own trunk run
    assert controller_server.calls == []


def test_an_allowed_controller_runs_its_forks(session_store_case):
    async def scenario(store):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls(LOOKUP, {"q": "accruals"}), _completed("finding")]
        )
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        server = _controller_server(app)
        source = await controller.snapshot_source("trunk")
        invoked = await _http(
            server,
            "POST",
            "/fork-controller/fork",
            json=_invoking_fork(source).model_dump(mode="json"),
        )
        await _drain(controller.fork(_fork_request(CHILD)))
        resumed = await _drain(controller.resume(_invocation(CHILD)))
        return invoked, await store.load(RUN_CHILD), resumed, provider, controller_server

    invoked, invoked_child, resumed, provider, controller_server = _in_store(
        session_store_case, scenario
    )
    assert invoked.status_code == 200 and f"{RUN_CHILD}-dispatch" in invoked.text
    assert invoked_child is not None
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 3
    assert controller_server.calls == [("lookup", {"q": "accruals"})]


def test_narrowing_the_gate_on_restart_stops_existing_forks(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store, ScriptedProvider([_completed("closed"), _completed("ran")]), toolset=toolset
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        await _drain(controller.resume(_invocation(CHILD)))
        before = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        # The host restarts with the same enrollment but without model runs.
        toolset, restarted_server = _controller_toolset()
        provider = ScriptedProvider([])
        restarted, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        narrowed = restarted.fork_controller_access(SUBJECT)
        sdk = await _refusal(narrowed.resume(_invocation(CHILD)))
        http = await _http(
            _controller_server(restarted),
            "POST",
            "/fork-controller/resume",
            json=_invocation(CHILD).model_dump(mode="json"),
        )
        after = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        return before, after, sdk, http, provider, restarted_server

    before, after, sdk, http, provider, restarted_server = _in_store(session_store_case, scenario)
    assert EventType.MODEL_STARTED in {record.event.type for record in before}
    assert sdk == DENIED
    assert http.status_code == 403
    assert [record.sequence for record in after] == [record.sequence for record in before]
    assert provider.requests == []
    assert restarted_server.calls == []


def test_allow_model_runs_must_be_a_bool() -> None:
    from cayu.fork_controllers import ForkController

    for value in (1, "true", None):
        with pytest.raises(TypeError, match="allow_model_runs must be a bool"):
            ForkController(
                name="compass",
                subjects=(SUBJECT,),
                agents=("finance-monthend",),
                allow_model_runs=value,  # ty: ignore[invalid-argument-type]
            )
