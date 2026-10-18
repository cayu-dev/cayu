"""Keyed correlation values on fork-controller reads, on every built-in session store."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

from tests.core.session_store_conformance.support import (
    _close_store,
    _new_postgres_store,
    _open_store,
    _public_authority_alias_codec,
    _reset_postgres_data,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_fork_controller_reads import _controller_server
from tests.core.test_session_fork_overlays import _controller_toolset, _http
from tests.runtime.runtime_supervision_support import (
    WINDOW_END,
    WINDOW_START,
    approve_pending,
    supervision_window,
)
from tests.runtime.runtime_tool_support import SideEffectApprovalPolicy, SideEffectTool

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.evals import ScriptedModelProvider
from cayu.events import EventType
from cayu.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.fork_controllers import ForkController, ForkControllerDenied
from cayu.messages import TextPart
from cayu.providers import ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions import ForkOverlay, ForkOverlayToolGrant, ForkSessionRequest, ResumeRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.forks import ForkExecutionProfileSelection
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import CompactSessionRequest
from cayu.storage import SQLiteSessionStore
from cayu.tools import ToolCapabilityCeiling
from cayu.tools.base import Tool, ToolResult, ToolSpec
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits
from cayu.tools.user_input import UserInputTool

__all__ = ["conformance_postgres_dsn", "session_store_case"]

AGENT = "assistant"
COMPASS = "compass"
SUBJECT = "svc:compass"
HOST = "host-session"
FORK = "compass-fork"
ASKER = "asker-session"
PAUSED = "paused-session"
LOOKUP = "mcp__compass__lookup"
NESTED = [Message.text("user", "nested")]
LIMITS = InferenceLimits(
    max_input_tokens=100, max_output_tokens=100, timeout_seconds=10, max_response_bytes=4096
)
NO_RETRY_DELAY = RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter_s=0.0)
WINDOW = {"start_at": WINDOW_START.isoformat(), "end_at": WINDOW_END.isoformat()}
CORRELATION_DOMAINS = (
    "tool_call_correlation",
    "tool_round_correlation",
    "model_step_correlation",
    "model_attempt_correlation",
    "auxiliary_operation_correlation",
)
CORRELATION_FIELDS = tuple(f"{domain}_id" for domain in CORRELATION_DOMAINS)
TOOL_TERMINALS = {
    "tool.call.completed",
    "tool.call.failed",
    "tool.call.blocked",
    "tool.call.approval_denied",
    "tool.call.approval_expired",
}
MODEL_TERMINALS = {"model.completed", "model.error"}


class Summarize(Tool):
    """A tool whose one auxiliary inference attempt the ledger must see."""

    spec = ToolSpec(
        name="summarize",
        description="Summarize with one auxiliary inference.",
        input_schema={"type": "object"},
        auxiliary_inference=AuxiliaryInferencePolicy(limits=LIMITS, purposes=("tool.summary",)),
    )

    async def run(self, ctx, args):
        await ctx.inference.invoke(
            ModelRequest(model="fake-model", messages=NESTED), purpose="tool.summary", limits=LIMITS
        )
        return ToolResult(content="summarized")


def _usage(tokens: int = 1) -> dict[str, Any]:
    return {"usage": {"input_tokens": tokens, "output_tokens": tokens}}


# The tool calls each user text asks for.
_TURNS: dict[str, list[tuple[str, str]]] = {
    "go": [("side_effect", "call_side"), ("summarize", "call_summary")],
    "ask": [("ask_user", "call_ask")],
    "pause": [("side_effect", "call_paused")],
}


class _Script:
    """The first model attempt fails retryably; later turns follow the last user text."""

    def __init__(self) -> None:
        self.failed_once = False

    def __call__(self, request: ModelRequest) -> list[ModelStreamEvent]:
        if request.messages == NESTED:
            return [ModelStreamEvent.text_delta("short"), ModelStreamEvent.completed(_usage(2))]
        if not self.failed_once:
            self.failed_once = True
            error = ModelProviderError(
                "overloaded", provider="scripted", retryable=True, status_code=503
            )
            return [ModelStreamEvent.error("overloaded", cause=error), ModelStreamEvent.completed()]
        last = request.messages[-1]
        text = next((part.text for part in last.content if isinstance(part, TextPart)), "")
        calls = _TURNS.get(text if last.role == "user" else "", [])
        if calls:
            return [
                *(
                    ModelStreamEvent.tool_call(
                        id=call_id,
                        name=name,
                        arguments={"question": "Which?"} if name == "ask_user" else {},
                    )
                    for name, call_id in calls
                ),
                ModelStreamEvent.completed(_usage()),
            ]
        return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed(_usage())]


def _app(store) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    provider = ScriptedModelProvider(response_factory=_Script())
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name=AGENT, model="fake-model"),
        tools=[SideEffectTool(), Summarize(), UserInputTool()],
        tool_policy=SideEffectApprovalPolicy(),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="fake-model"), max_user_turns=1
        ),
    )
    toolset, _server = _controller_toolset()
    app.enroll_fork_controller(
        ForkController(
            name=COMPASS,
            subjects=(SUBJECT,),
            agents=(AGENT,),
            mcp_toolsets=(toolset,),
            allow_model_runs=True,
        )
    )
    return app


async def _drain(stream) -> None:
    async for _event in stream:
        pass


async def _start(app: CayuApp, session_id: str, text: str) -> None:
    await _drain(
        app.run(
            RunRequest(
                agent_name=AGENT,
                session_id=session_id,
                messages=[Message.text("user", text)],
                retry_policy=NO_RETRY_DELAY,
            )
        )
    )


async def _world(store) -> CayuApp:
    """A session with a retried model call, an approved tool call and an auxiliary
    inference, a compaction, and a controller fork whose profile adoption was decided."""

    app = _app(store)
    await _start(app, HOST, "go")
    await approve_pending(app, store, HOST)
    await _drain(
        app.resume(ResumeRequest(session_id=HOST, messages=[Message.text("user", "next")]))
    )
    session = await store.load(HOST)
    await _drain(
        app.compact_session(
            CompactSessionRequest(
                session_id=HOST,
                idempotency_key="compact-host",
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=len(await store.load_transcript(HOST)),
            )
        )
    )
    compass = app.fork_controller_access(SUBJECT)
    await _drain(
        compass.fork(
            ForkSessionRequest(
                source_session_id=HOST,
                session_id=FORK,
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="compass-fork",
                    reason="Compass investigates the approval.",
                    requested_by=ResolutionActor(
                        subject=SUBJECT, source=ResolutionActorSource.REQUEST
                    ),
                ),
                overlay=ForkOverlay(
                    controller=COMPASS, tools=(ForkOverlayToolGrant(tool_name=LOOKUP),)
                ),
            )
        )
    )
    await _drain(
        compass.resume(ResumeRequest(session_id=FORK, messages=[Message.text("user", "look")]))
    )
    return app


async def _controller_reads(app: CayuApp) -> dict[str, Any]:
    """Read everything once through every controller read route."""

    server = _controller_server(app, SUBJECT)
    queried = await _http(server, "POST", "/fork-controller/events/query", json={"limit": 1000})
    supervision = await _http(
        server, "POST", "/fork-controller/supervision-events/query", json=WINDOW
    )
    listed = await _http(server, "GET", "/fork-controller/sessions")
    per_session = {
        session_id: (
            await _http(server, "GET", f"/fork-controller/sessions/{session_id}"),
            await _http(
                server,
                "GET",
                f"/fork-controller/sessions/{session_id}/events",
                params={"limit": 1000},
            ),
        )
        for session_id in (HOST, FORK)
    }
    for response in (
        queried,
        supervision,
        listed,
        *(r for pair in per_session.values() for r in pair),
    ):
        assert response.status_code == 200, response.text
    return {
        "events": queried.json()["events"],
        "supervision": supervision.json()["events"],
        "sessions": listed.json()["sessions"],
        "details": {sid: detail.json() for sid, (detail, _events) in per_session.items()},
        "session_events": {sid: events.json() for sid, (_detail, events) in per_session.items()},
        "text": json.dumps(
            [
                queried.json(),
                supervision.json(),
                listed.json(),
                *(r.json() for pair in per_session.values() for r in pair),
            ]
        ),
    }


def _private_ids(records) -> set[str]:
    found: set[str] = set()
    for record in records:
        payload = record.event.payload
        for key in ("tool_call_id", "tool_round_id", "model_step_id", "model_attempt_id"):
            if type(payload.get(key)) is str:
                found.add(payload[key])
        attribution = payload.get("auxiliary_inference")
        if type(attribution) is dict and type(attribution.get("operation_id")) is str:
            found.add(attribution["operation_id"])
    return found


def _correlation_values(reads: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {field: set() for field in CORRELATION_FIELDS}
    for event in reads["events"]:
        for field in CORRELATION_FIELDS:
            if field in event["payload"]:
                values[field].add(event["payload"][field])
    return values


def _of(events: list[dict[str, Any]], *types: str, session_id: str = HOST) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] in types and e["session_id"] == session_id]


async def _scenario_in(case, scenario):
    store = await _open_store(case)
    try:
        return await scenario(store)
    finally:
        await _close_store(store)


def test_controller_reads_pair_starts_with_terminals_and_join_supervision(session_store_case):
    async def scenario(store):
        app = await _world(store)
        reads = await _controller_reads(app)
        again = await _controller_reads(app)
        records = await store.query_events(EventQuery(session_ids=(HOST, FORK), limit=5000))
        return reads, again, records

    reads, again, records = asyncio.run(_scenario_in(session_store_case, scenario))
    events = reads["events"]
    # Every tool start pairs with exactly one terminal by its call correlation.
    starts = _of(events, "tool.call.started")
    terminals = _of(events, *TOOL_TERMINALS)
    assert len(starts) == 2
    terminal_calls = Counter(e["payload"]["tool_call_correlation_id"] for e in terminals)
    for start in starts:
        assert terminal_calls[start["payload"]["tool_call_correlation_id"]] == 1
    assert len(terminal_calls) == len(starts)
    # Starts carry the proposing attempt; one round holds both calls.
    proposing = [
        e for e in _of(events, "model.completed") if "tool_round_correlation_id" in e["payload"]
    ]
    assert len(proposing) == 1
    for event in (*starts, *terminals):
        assert (
            event["payload"]["tool_round_correlation_id"]
            == (proposing[0]["payload"]["tool_round_correlation_id"])
        )
    for start in starts:
        for field in ("model_step_correlation_id", "model_attempt_correlation_id"):
            assert start["payload"][field] == proposing[0]["payload"][field]
        assert start["tool_name"] in {"side_effect", "summarize"}
        assert start["timestamp"] and type(start["sequence"]) is int
    # Every model start pairs with exactly one completion or error by attempt;
    # the retried step keeps its step correlation across both attempts.
    for session_id in (HOST, FORK):
        model_starts = _of(events, "model.started", session_id=session_id)
        model_ends = Counter(
            e["payload"]["model_attempt_correlation_id"]
            for e in _of(events, *MODEL_TERMINALS, session_id=session_id)
        )
        assert model_starts
        for start in model_starts:
            assert model_ends[start["payload"]["model_attempt_correlation_id"]] == 1
        assert len(model_ends) == len(model_starts)
    errors = _of(events, "model.error")
    assert len(errors) == 1
    retried = [
        e
        for e in _of(events, "model.started")
        if e["payload"]["model_step_correlation_id"]
        == errors[0]["payload"]["model_step_correlation_id"]
    ]
    assert len({e["payload"]["model_attempt_correlation_id"] for e in retried}) == 2
    # Compaction attempts keep their purpose; completions keep usage.
    compactions = _of(events, "model.started", "model.completed")
    assert {e["payload"].get("purpose") for e in compactions} >= {"context_compaction"}
    assert all("usage_metrics" in e["payload"] for e in _of(events, "model.completed"))
    # Auxiliary attempts pair by operation correlation plus attempt.
    auxiliary_starts = _of(events, "model.auxiliary.attempt_started")
    auxiliary_settled = _of(events, "model.auxiliary.attempt_settled")
    assert len(auxiliary_starts) == len(auxiliary_settled) == 1
    key = ("auxiliary_operation_correlation_id", "attempt")
    assert [tuple(e["payload"][k] for k in key) for e in auxiliary_starts] == [
        tuple(e["payload"][k] for k in key) for e in auxiliary_settled
    ]
    settled = auxiliary_settled[0]["payload"]
    assert settled["auxiliary_inference"]["purpose"] == "tool.summary"
    assert settled["auxiliary_outcome"] == "completed"
    assert settled["usage_metrics"]["total_tokens"] == 4
    # Profile decisions keep their kind and candidate fingerprint, which resolve
    # the fingerprints on the fork's model calls and on supervision touches.
    decided = {
        sid: [
            e["payload"] for e in _of(events, "session.execution_profile.decided", session_id=sid)
        ]
        for sid in (HOST, FORK)
    }
    assert "adopted" in {payload["decision"] for payload in decided[FORK]}
    candidates = {
        sid: {payload["candidate_profile"]["fingerprint"] for payload in decided[sid]}
        for sid in (HOST, FORK)
    }
    assert {
        e["payload"]["execution_profile_fingerprint"]
        for e in _of(events, "model.started", session_id=FORK)
    } <= candidates[FORK]
    # Supervision touches join their tool calls and model attempts.
    [touch] = reads["supervision"]
    assert touch["kind"] == "approve" and touch["outcome"] == "approved"
    assert touch["approval_id"] and touch["tool_name"] == "side_effect"
    assert touch["execution_profile_fingerprint"] in candidates[HOST]
    assert touch["requested_at"] and touch["resolved_at"]
    side_effect = next(e for e in starts if e["tool_name"] == "side_effect")
    assert touch["tool_call_correlation_ids"] == [
        side_effect["payload"]["tool_call_correlation_id"]
    ]
    assert touch["tool_round_correlation_id"] == side_effect["payload"]["tool_round_correlation_id"]
    for field in ("model_step_correlation_id", "model_attempt_correlation_id"):
        assert touch[field] == proposing[0]["payload"][field]
    # One public session id on every route.
    assert {s["id"] for s in reads["sessions"]} == {HOST, FORK}
    for sid in (HOST, FORK):
        assert reads["details"][sid]["id"] == sid
        assert reads["session_events"][sid]["session_id"] == sid
        assert {e["session_id"] for e in reads["session_events"][sid]["events"]} == {sid}
        assert reads["session_events"][sid]["events"] == _of(
            events, *{e["type"] for e in events}, session_id=sid
        )
    assert touch["session_id"] == HOST
    # Values are identical across reads, and values of one kind never equal another's.
    assert again["events"] == reads["events"] and again["supervision"] == reads["supervision"]
    values = _correlation_values(reads)
    for field, domain in zip(CORRELATION_FIELDS, CORRELATION_DOMAINS, strict=True):
        assert values[field] and all(f".{domain}." in value for value in values[field])
    # No private identity reaches controller-read output.
    private = _private_ids(records)
    assert private and all(value not in reads["text"] for value in private)
    assert all(value["timestamp"] for value in events)
    by_sequence = {(r.event.session_id, r.sequence): r.event.timestamp for r in records}
    assert all(
        e["timestamp"] == by_sequence[(e["session_id"], e["sequence"])].isoformat() for e in events
    )


def test_sdk_correlation_ids_equal_the_http_payload(session_store_case):
    async def scenario(store):
        app = await _world(store)
        reads = await _controller_reads(app)
        compass = app.fork_controller_access(SUBJECT)
        events = {
            (record.event.session_id, record.sequence): compass.event_correlation_ids(record)
            for record in await compass.query_events(EventQuery(limit=5000))
        }
        page = await compass.supervision_events(supervision_window())
        touches = [compass.supervision_correlation_ids(touch) for touch in page.events]
        return reads, events, touches

    reads, events, touches = asyncio.run(_scenario_in(session_store_case, scenario))
    assert len(events) == len(reads["events"])
    for event in reads["events"]:
        expected = {k: v for k, v in event["payload"].items() if k in CORRELATION_FIELDS}
        assert events[(event["session_id"], event["sequence"])] == expected
    assert touches == [
        {
            field: touch[field]
            for field in (
                "tool_call_correlation_ids",
                "tool_round_correlation_id",
                "model_step_correlation_id",
                "model_attempt_correlation_id",
            )
        }
        for touch in reads["supervision"]
    ]


def test_correlation_values_are_stable_across_a_store_reopen(session_store_case):
    store_kind, tmp_path, postgres_dsn = session_store_case

    async def scenario():
        store = await _open_store(session_store_case)
        try:
            before = await _controller_reads(await _world(store))
        finally:
            if store_kind != "memory":
                await _close_store(store)
        # The host restarts with the same key material in a new keyring object.
        codec = _public_authority_alias_codec()
        if store_kind == "sqlite":
            store = SQLiteSessionStore(
                tmp_path / "sessions.sqlite", public_authority_alias_codec=codec
            )
        elif store_kind == "postgres":
            store = _new_postgres_store(postgres_dsn, public_authority_alias_codec=codec)
        try:
            return before, await _controller_reads(_app(store))
        finally:
            await _close_store(store)

    before, after = asyncio.run(scenario())
    values = _correlation_values(before)
    assert all(values[field] for field in CORRELATION_FIELDS)
    assert all(touch["tool_call_correlation_ids"] for touch in before["supervision"])
    assert after["events"] == before["events"]
    assert after["supervision"] == before["supervision"]


def test_keyless_controller_reads_omit_correlation_fields(session_store_case):
    from cayu.server.contracts import ApiSupervisionEvent

    store_kind, tmp_path, postgres_dsn = session_store_case

    async def scenario():
        if store_kind == "memory":
            # An in-memory store always holds a random key for as long as its data lives.
            store = InMemorySessionStore()
        elif store_kind == "sqlite":
            store = SQLiteSessionStore(tmp_path / "keyless.sqlite")
        else:
            await _reset_postgres_data(postgres_dsn)
            store = _new_postgres_store(postgres_dsn)
        try:
            app = await _world(store)
            compass = app.fork_controller_access(SUBJECT)
            records = await compass.query_events(EventQuery(limit=5000))
            page = await compass.supervision_events(supervision_window())
            sdk = (
                [compass.event_correlation_ids(record) for record in records],
                [compass.supervision_correlation_ids(touch) for touch in page.events],
            )
            return await _controller_reads(app), sdk
        finally:
            await _close_store(store)

    reads, (sdk_events, sdk_touches) = asyncio.run(scenario())
    supervision_fields = {
        "tool_call_correlation_ids",
        "tool_round_correlation_id",
        "model_step_correlation_id",
        "model_attempt_correlation_id",
    }
    assert reads["events"] and reads["supervision"]
    if store_kind == "memory":
        assert all(_correlation_values(reads)[field] for field in CORRELATION_FIELDS)
        assert all(supervision_fields <= set(touch) for touch in reads["supervision"])
        assert any(sdk_events) and all(sdk_touches)
        return
    # Without a keyring every field is absent, including the control plane's
    # process-local tool_call_correlation_id.
    for event in reads["events"]:
        assert not set(CORRELATION_FIELDS) & set(event["payload"]), event["type"]
    for listed in reads["session_events"].values():
        for event in listed["events"]:
            assert not set(CORRELATION_FIELDS) & set(event["payload"])
    for touch in reads["supervision"]:
        assert set(touch) == set(ApiSupervisionEvent.model_fields)
    assert not any(field in reads["text"] for field in CORRELATION_FIELDS)
    assert sdk_events == [{}] * len(sdk_events)
    assert sdk_touches == [{}] * len(sdk_touches)


def test_correlation_values_carry_no_authority(session_store_case):
    async def scenario(store):
        app = await _world(store)
        reads = await _controller_reads(app)
        values = _correlation_values(reads)
        for touch in reads["supervision"]:
            values["tool_call_correlation_id"].update(touch["tool_call_correlation_ids"])
        await _start(app, PAUSED, "pause")
        await _start(app, ASKER, "ask")
        compass = app.fork_controller_access(SUBJECT)
        server = _controller_server(app, SUBJECT)
        pending = await _pending_approval(store)
        before = await _pending_kinds(store)
        outcomes: dict[str, list[object]] = {}
        for field in CORRELATION_FIELDS:
            value = sorted(values[field])[0]
            results: list[object] = []
            # The store's alias registry resolves none of them, for any field.
            for field_name in (
                "session_id",
                "causal_budget_id",
                "interaction_id",
                "approval_id",
                "input_id",
                "tool_call_id",
                "tool_round_id",
                *CORRELATION_DOMAINS,
            ):
                for scope in (None, HOST):
                    try:
                        resolved = await store.resolve_public_authority_alias(
                            value, field_name=field_name, scope_session_id=scope
                        )
                    except ValueError:
                        resolved = None
                    results.append(resolved)
            # Session resolvers on the SDK and HTTP entrances.
            results.append(await _raises(app._resolve_public_session_id(value), ValueError))
            results.append(await _raises(compass.session(value), ForkControllerDenied))
            results.append(await _raises(compass.snapshot_source(value), ValueError))
            for path in (
                f"/fork-controller/sessions/{value}",
                f"/fork-controller/sessions/{value}/events",
                f"/fork-controller/sources/{value}/snapshot",
            ):
                results.append((await _http(server, "GET", path)).status_code)
            results.append(
                await _raises(
                    app._resolve_public_interaction_id(session_id=HOST, value=value), ValueError
                )
            )
            # Action linkage resolvers never map them to private linkage.
            for field_name in ("approval_id", "input_id", "tool_call_id", "tool_round_id"):
                results.append(
                    await app._resolve_public_action_linkage(
                        session_id=HOST, value=value, field_name=field_name
                    )
                )
            # Approval and user-input resolution refuse them and stay pending.
            for update in (
                {"approval_id": value},
                {"tool_round_id": value},
                {"tool_call_id": value},
            ):
                request = ToolApprovalRequest(
                    session_id=PAUSED,
                    approval_id=pending.approval_id,
                    tool_round_id=pending.tool_round_id,
                    tool_call_id=pending.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                ).model_copy(update=update)
                results.append(
                    await _raises(_drain(app.resolve_tool_approval(request)), ValueError)
                )
            results.append(
                await _raises(
                    _drain(
                        app.resolve_user_input(
                            UserInputResponse(session_id=ASKER, input_id=value, answer="prod")
                        )
                    ),
                    ValueError,
                )
            )
            results.append(await _pending_kinds(store))
            outcomes[value] = results
        return outcomes, before

    outcomes, before = asyncio.run(_scenario_in(session_store_case, scenario))
    assert before == {PAUSED: ["tool_approval"], ASKER: ["user_input"]}
    assert len(outcomes) == len(CORRELATION_FIELDS)
    unresolved = [None] * 2 * (7 + len(CORRELATION_DOMAINS))
    for value, results in outcomes.items():
        assert results == [
            *unresolved,
            True,
            True,
            True,
            403,
            403,
            403,
            True,
            value,
            value,
            value,
            value,
            True,
            True,
            True,
            True,
            before,
        ]


async def _raises(operation, error: type[BaseException]) -> bool:
    try:
        await operation
    except error:
        return True
    return False


async def _pending_approval(store):
    from cayu.sessions import _pending_approval_reader as reader

    pending = reader.pending_approval_from_checkpoint(await store.load_checkpoint(PAUSED))
    assert pending is not None
    return pending


async def _pending_kinds(store) -> dict[str, list[str]]:
    from cayu.sessions.pending_action_contracts import PendingActionQuery

    kinds = {}
    for session_id in (PAUSED, ASKER):
        page = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
        kinds[session_id] = [action.kind.value for action in page.actions]
    return kinds


def test_controller_events_publish_only_derived_correlation(session_store_case):
    from cayu.server.fork_controller import _controller_event
    from cayu.sessions.records import EventRecord

    async def scenario(store):
        app = _app(store)
        await _start(app, HOST, "done")
        compass = app.fork_controller_access(SUBJECT)
        records = await compass.query_events(EventQuery(session_id=HOST, limit=1000))
        forged = {field: "forged" for field in CORRELATION_FIELDS}
        published = []
        for record in records:
            event = record.event.model_copy(update={"payload": {**record.event.payload, **forged}})
            published.append(
                _controller_event(app, compass, EventRecord(sequence=record.sequence, event=event))
            )
        return published

    published = asyncio.run(_scenario_in(session_store_case, scenario))
    assert published
    for event in published:
        assert "forged" not in event["payload"].values()
    started = [e for e in published if e["type"] == EventType.MODEL_STARTED.value]
    assert started and all("model_attempt_correlation_id" in e["payload"] for e in started)


def test_correlation_ids_follow_the_active_key_and_cover_denials() -> None:
    import base64

    from pydantic import SecretStr
    from tests.core.test_fork_controller_reads import _settled_approval
    from tests.runtime.runtime_supervision_support import approval_app

    from cayu.runtime.public_authority import PublicAuthorityAliasCodec
    from cayu.sessions import _correlation as correlation

    async def scenario():
        codec = _public_authority_alias_codec()
        store = InMemorySessionStore(public_authority_alias_codec=codec)
        await _settled_approval(approval_app(store, sessions=1), store, HOST, AGENT)
        return codec, await store.load_events(HOST)

    codec, events = asyncio.run(scenario())
    [denied] = [event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_DENIED]
    ids = correlation.event_correlation_ids(denied, codec)
    assert set(ids) == {"tool_call_correlation_id", "tool_round_correlation_id"}
    rotated = PublicAuthorityAliasCodec(
        codec.keyring.rotated(
            active_key_id="next",
            key=SecretStr(base64.urlsafe_b64encode(bytes([7]) * 32).decode().rstrip("=")),
        )
    )
    after = correlation.event_correlation_ids(denied, rotated)
    assert set(after) == set(ids)
    assert all(after[field] != ids[field] for field in ids)
    # A touch without linkage has empty or null correlation.
    assert correlation.supervision_correlation_ids(
        codec,
        session_id=HOST,
        tool_round_id=None,
        tool_call_ids=("call_1",),
        model_step_id=None,
        model_attempt_id=None,
    ) == {
        "tool_call_correlation_ids": [],
        "tool_round_correlation_id": None,
        "model_step_correlation_id": None,
        "model_attempt_correlation_id": None,
    }


def test_supervision_pages_stay_within_max_bytes_with_correlation() -> None:
    from tests.runtime.runtime_supervision_support import approval_app, pause_for_approval

    async def scenario():
        store = InMemorySessionStore(public_authority_alias_codec=_public_authority_alias_codec())
        app = approval_app(store, sessions=1)
        app.enroll_fork_controller(
            ForkController(name=COMPASS, subjects=(SUBJECT,), agents=(AGENT,))
        )
        await pause_for_approval(app, store, HOST)
        await approve_pending(app, store, HOST)
        page = await app.fork_controller_access(SUBJECT).supervision_events(supervision_window())
        private_bytes = len(
            json.dumps(page.model_dump(mode="json"), separators=(",", ":")).encode("utf-8")
        )
        server = _controller_server(app, SUBJECT)
        fits = await _http(server, "POST", "/fork-controller/supervision-events/query", json=WINDOW)
        bounded = await _http(
            server,
            "POST",
            "/fork-controller/supervision-events/query",
            json={**WINDOW, "max_bytes": private_bytes},
        )
        # The whole response counts, not just its events.
        exact, short = [
            await _http(
                server,
                "POST",
                "/fork-controller/supervision-events/query",
                json={**WINDOW, "max_bytes": max_bytes},
            )
            for max_bytes in (len(fits.content), len(fits.content) - 1)
        ]
        return private_bytes, fits, bounded, exact, short

    private_bytes, fits, bounded, exact, short = asyncio.run(scenario())
    published = len(json.dumps(fits.json()["events"], separators=(",", ":")).encode("utf-8"))
    # The store admits the private page, but the published page is larger.
    assert published > private_bytes
    assert bounded.status_code == 413
    assert bounded.json() == {
        "detail": "The result exceeds the byte limit; request a smaller page."
    }
    assert len(fits.content) > published
    assert exact.status_code == 200
    assert exact.content == fits.content
    assert short.status_code == 413
