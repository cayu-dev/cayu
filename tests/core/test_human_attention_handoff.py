"""Human-attention handoff against a durable SQLite Runtime and a reference receiver."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from examples.human_attention.app import ApprovalPolicy, ApprovalTool

from cayu import (
    AgentSpec,
    CayuApp,
    InterruptSessionRequest,
    Message,
    ModelStreamEvent,
    PendingActionKind,
    PendingActionQuery,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.human_attention_handoff import (
    PROTOCOL,
    HumanAttentionHandoff,
    HumanAttentionHandoffError,
    HumanAttentionReceiver,
)
from cayu.runtime.human_attention import HumanAttentionReference
from cayu.tools.user_input import UserInputTool

TERMINAL = {"resolved", "cancelled", "superseded", "expired"}
TOKEN = "receiver-token"


class ReferenceReceiver:
    """The receiver obligations of cayu.human-attention-handoff/v1, in memory."""

    def __init__(self) -> None:
        self.requests: dict[str, dict[str, Any]] = {}
        self.scan_cursor: str | None = None
        self.reports: list[dict[str, Any]] = []
        self.bodies: list[str] = []
        self.down = False
        self.lose_acknowledgements = 0
        self.decline: int | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("receiver down", request=request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"detail": {"code": "credential_invalid"}})
        if self.decline is not None:
            return httpx.Response(self.decline, json={"detail": {"code": "declined"}})
        path = request.url.path.rsplit("/", 1)[-1]
        if request.method == "GET" and path == "state":
            return httpx.Response(
                200, json={"reconcile_interval_seconds": 300, "scan_cursor": self.scan_cursor}
            )
        if request.method == "GET" and path == "references":
            after = request.url.params.get("after")
            limit = int(request.url.params.get("limit", "100"))
            session_id = request.url.params.get("session_id")
            open_items = sorted(
                (key, item["reference"])
                for key, item in self.requests.items()
                if item["state"] not in TERMINAL
                and (after is None or key > after)
                and (session_id is None or item["reference"]["session_id"] == session_id)
            )[:limit]
            return httpx.Response(
                200,
                json={
                    "items": [reference for _, reference in open_items],
                    "next_after": open_items[-1][0] if len(open_items) == limit else None,
                },
            )
        body = json.loads(request.content)
        self.bodies.append(request.content.decode())
        assert body["protocol"] == PROTOCOL
        if path == "reconciliations":
            self.reports.append(body)
            return httpx.Response(200, json={"recorded": True})
        assert path == "observations"
        staged = {key: dict(value) for key, value in self.requests.items()}
        for item in body["observations"]:
            reference = HumanAttentionReference.model_validate(item["reference"])
            key = reference.attention_id
            existing = staged.get(key)
            if existing is not None and existing["reference"] != {
                **reference.model_dump(mode="json"),
                "source_sequence": existing["reference"]["source_sequence"],
            }:
                return httpx.Response(409, json={"detail": {"code": "identity_conflict"}})
            if existing is None:
                if item["state"] == "unavailable":
                    continue
                staged[key] = {
                    "reference": reference.model_dump(mode="json"),
                    "requester_ref": item.get("requester_ref"),
                    "state": item["state"],
                }
            elif existing["state"] not in TERMINAL and item["state"] != "unavailable":
                existing["state"] = item["state"]
        self.requests = staged
        if "scan" in body:
            self.scan_cursor = body["scan"]["cursor"]
        verify = []
        scope = body.get("session_scope")
        if scope is not None and scope["complete"]:
            active = {
                item["reference"]["attention_id"]
                for item in body["observations"]
                if item["state"] == "active"
            }
            verify = [
                value["reference"]
                for key, value in sorted(self.requests.items())
                if value["state"] not in TERMINAL
                and value["reference"]["session_id"] == scope["session_id"]
                and key not in active
            ]
        response = httpx.Response(200, json={"verify": verify})
        if self.lose_acknowledgements:
            self.lose_acknowledgements -= 1
            raise httpx.ReadError("acknowledgement lost", request=request)
        return response


def _handoff(receiver: ReferenceReceiver, **options: Any) -> HumanAttentionHandoff:
    return HumanAttentionHandoff(
        HumanAttentionReceiver(
            url="http://receiver.test/attention",
            token=TOKEN,
            transport=httpx.MockTransport(receiver.handle),
        ),
        **options,
    )


def _pause(kind: str = "user_input") -> list[list[ModelStreamEvent]]:
    return [
        [
            ModelStreamEvent.tool_call(
                id=f"call-{kind}",
                name="ask_user" if kind == "user_input" else "record",
                arguments={"question": "Which environment?"} if kind == "user_input" else {},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("Finished."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]


class Agent:
    """One application process over a durable SQLite Runtime store."""

    def __init__(
        self, store: SQLiteSessionStore, handoff: HumanAttentionHandoff | None, batches=None
    ) -> None:
        self.store = store
        self.app = CayuApp(
            session_store=self.store,
            event_sinks=[] if handoff is None else handoff.event_sinks(),
            enable_logging=False,
        )
        self.app.register_provider(ScriptedModelProvider(batches or _pause()), default=True)
        self.app.register_agent(
            AgentSpec(name="attention-demo", model="fixture"),
            tools=[UserInputTool(), ApprovalTool()],
            tool_policy=ApprovalPolicy(),
        )
        if handoff is not None:
            handoff.bind(self.app)

    async def start(self, session_id: str, **labels: str) -> list[Any]:
        return [
            event
            async for event in self.app.run(
                RunRequest(
                    agent_name="attention-demo",
                    session_id=session_id,
                    messages=[Message.text("user", "go")],
                    labels=labels,
                )
            )
        ]


def _states(receiver: ReferenceReceiver) -> list[str]:
    return [item["state"] for _, item in sorted(receiver.requests.items())]


def test_a_committed_question_is_handed_off_and_its_answer_settles_it(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            handoff = _handoff(receiver)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            await agent.start("question", requester="customer-42")
            ((key, request),) = receiver.requests.items()
            assert request["state"] == "active"
            assert request["requester_ref"] == "customer-42"
            action = (
                await agent.store.query_pending_actions(PendingActionQuery(session_id="question"))
            ).actions[0]
            assert key == action.attention_id
            # Identities and Runtime's fixed summary only; no question text on the wire.
            assert all("Which environment?" not in body for body in receiver.bodies)
            assert any("User input required." in body for body in receiver.bodies)

            reference = HumanAttentionReference.model_validate(request["reference"])
            _ = [
                event
                async for event in agent.app.resolve_user_input(
                    UserInputResponse(
                        session_id=reference.session_id,
                        input_id=reference.action_id,
                        answer="staging",
                    )
                )
            ]
            assert _states(receiver) == ["resolved"]

            # A late opening observation cannot reopen the settled request.
            await handoff._receiver.observations(
                [
                    {
                        "reason": "current_action",
                        "reference": request["reference"],
                        "state": "active",
                        "summary": "User input required.",
                    }
                ]
            )
            assert _states(receiver) == ["resolved"]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("decision", "expected"),
    [(ToolApprovalDecision.APPROVE, "resolved"), (ToolApprovalDecision.DENY, "cancelled")],
)
def test_approvals_settle_with_runtime_meaning(sqlite_resources, decision, expected):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            agent = Agent(
                resources.own(SQLiteSessionStore(resources.path())),
                _handoff(receiver),
                _pause("tool_approval"),
            )
            await agent.start("approval")
            ((_, request),) = receiver.requests.items()
            reference = request["reference"]
            assert reference["kind"] == "tool_approval"
            _ = [
                event
                async for event in agent.app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="approval",
                        approval_id=reference["action_id"],
                        tool_round_id=reference["round_id"],
                        tool_call_id=reference["tool_call_id"],
                        decision=decision,
                    )
                )
            ]
            assert _states(receiver) == [expected]

    asyncio.run(run())


def test_supersession_is_reported_by_the_repair_pass(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            handoff = _handoff(receiver)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            await agent.start("superseded")
            _ = [
                event
                async for event in agent.app.interrupt_session(
                    InterruptSessionRequest(session_id="superseded", reason="operator")
                )
            ]
            await handoff.reconcile_once()
            assert _states(receiver) == ["superseded"]
            assert receiver.reports[-1]["verify_complete"] is True

    asyncio.run(run())


def test_outage_and_lost_acknowledgement_leave_the_pause_and_one_request(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            handoff = _handoff(receiver, failure_backoff_seconds=0)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            receiver.down = True
            events = await agent.start("outage")
            assert str(events[-1].type) == "session.interrupted"
            page = await agent.store.query_pending_actions(PendingActionQuery(session_id="outage"))
            assert len(page.actions) == 1
            assert receiver.requests == {}

            receiver.down = False
            receiver.lose_acknowledgements = 1
            with pytest.raises(HumanAttentionHandoffError):
                await handoff.sync_session("outage")
            assert len(receiver.requests) == 1  # committed before the ack was lost
            await handoff.reconcile_once()
            assert list(receiver.requests) == [page.actions[0].attention_id]
            assert _states(receiver) == ["active"]

    asyncio.run(run())


def test_fail_fast_window_after_a_failure(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            now = [100.0]
            handoff = _handoff(receiver, failure_backoff_seconds=30, clock=lambda: now[0])
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), None)
            handoff.bind(agent.app)
            await agent.start("window")
            receiver.down = True
            with pytest.raises(HumanAttentionHandoffError):
                await handoff.sync_session("window")
            receiver.down = False
            calls_before = len(receiver.bodies)
            with pytest.raises(HumanAttentionHandoffError, match="failed recently"):
                await handoff.sync_session("window")
            assert len(receiver.bodies) == calls_before
            now[0] += 31
            await handoff.sync_session("window")
            assert _states(receiver) == ["active"]

    asyncio.run(run())


def test_restart_backfills_pre_enrollment_pauses_across_pages(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            path = resources.path()
            for index in range(3):
                plain = Agent(resources.own(SQLiteSessionStore(path)), None)
                await plain.start(f"backfill-{index}")
            receiver = ReferenceReceiver()

            first = _handoff(receiver, page_size=1, max_pages=2)
            Agent(resources.own(SQLiteSessionStore(path)), first, [])
            await first.reconcile_once()
            assert len(receiver.requests) == 2
            assert receiver.scan_cursor is not None
            assert receiver.reports[-1]["scan_complete"] is False

            # A restarted process resumes from the cursor the receiver persisted.
            restarted = _handoff(receiver, page_size=1, max_pages=5)
            Agent(resources.own(SQLiteSessionStore(path)), restarted, [])
            await restarted.reconcile_once()
            assert len(receiver.requests) == 3
            assert receiver.scan_cursor is None
            assert receiver.reports[-1]["scan_complete"] is True
            assert receiver.reports[-1]["issues"] == []

    asyncio.run(run())


def test_declines_are_not_retried_but_outages_are(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            handoff = _handoff(receiver, failure_backoff_seconds=0)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            await agent.start("declined")
            (sink,) = handoff.event_sinks()
            event = SimpleNamespace(type="session.interrupted", session_id="declined")
            for status in (401, 403, 409):
                receiver.decline = status
                await sink.emit(event)  # acknowledged: retrying cannot succeed
            receiver.decline = None
            receiver.down = True
            with pytest.raises(HumanAttentionHandoffError):
                await sink.emit(event)  # retried by Runtime's persisted delivery

    asyncio.run(run())


def test_a_delegated_parent_is_navigation_only(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            handoff = _handoff(ReferenceReceiver())
            Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            parent = SimpleNamespace(
                attention_id=None,
                kind=PendingActionKind.DELEGATED_ACTION,
                session=SimpleNamespace(instance_id="instance", labels={}, parent_session_id=None),
            )
            assert await handoff._observe_action(parent) is None

    asyncio.run(run())


def test_an_unconfigured_handoff_is_inert():
    async def run():
        handoff = HumanAttentionHandoff.from_environment({})
        assert not handoff.enabled
        assert handoff.event_sinks() == []
        await handoff.run()

    asyncio.run(run())
    with pytest.raises(ValueError, match="set together"):
        HumanAttentionHandoff.from_environment({"CAYU_ATTENTION_HANDOFF_URL": "https://r.test"})
    configured = HumanAttentionHandoff.from_environment(
        {
            "CAYU_ATTENTION_HANDOFF_TOKEN": "secret-token",
            "CAYU_ATTENTION_HANDOFF_URL": "https://r.test/a",
        }
    )
    assert configured.enabled
    assert "secret-token" not in repr(configured._receiver)


def test_bounded_verification_progresses_and_retries_unacknowledged_pages():
    import hashlib

    async def run():
        receiver = ReferenceReceiver()
        for index in range(101):
            identity = [f"session-{index}", "instance", "user_input", f"input-{index}", None]
            reference = HumanAttentionReference(
                attention_id="attention_"
                + hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
                session_id=identity[0],
                session_instance_id="instance",
                kind="user_input",
                action_id=identity[3],
                source_sequence=1,
            )
            receiver.requests[reference.attention_id] = {
                "reference": reference.model_dump(mode="json"),
                "state": "active",
            }
        last = sorted(receiver.requests)[-1]
        checked = []

        async def pending(query):
            return SimpleNamespace(actions=[], issues=[], has_more=False, next_cursor=None)

        async def observe(reference):
            checked.append(reference.attention_id)
            return SimpleNamespace(
                state="resolved" if reference.attention_id == last else "active",
                reason="durable_closure" if reference.attention_id == last else "current_action",
            )

        handoff = _handoff(receiver, max_pages=1, failure_backoff_seconds=0)
        handoff.bind(
            SimpleNamespace(
                session_store=SimpleNamespace(query_pending_actions=pending),
                get_human_attention_state=observe,
            )
        )
        await handoff.reconcile_once()
        assert len(set(checked)) == 100
        assert receiver.reports[-1]["verify_complete"] is False
        cursors = []
        original_references = handoff._receiver.references

        async def references(**kwargs):
            cursors.append(kwargs.get("after"))
            return await original_references(**kwargs)

        handoff._receiver.references = references

        # Lose the acknowledgement of the verification batch, not the scan batch.
        original = handoff._receiver.observations

        async def lose_ack(observations, **kwargs):
            if observations:
                receiver.lose_acknowledgements = 1
            return await original(observations, **kwargs)

        handoff._receiver.observations = lose_ack
        with pytest.raises(HumanAttentionHandoffError):
            await handoff.reconcile_once()
        handoff._receiver.observations = original
        await handoff.reconcile_once()
        assert len(set(checked)) == 101
        assert receiver.requests[last]["state"] == "resolved"
        assert receiver.reports[-1]["verify_complete"] is True
        assert cursors[0] is not None
        assert cursors[0] == cursors[1]
        checked.clear()
        await handoff.reconcile_once()
        assert cursors[-1] is None
        assert len(set(checked)) == 100  # wrapped to revisit the earlier open requests

    asyncio.run(run())


def test_unavailable_reads_make_scan_and_verification_incomplete(sqlite_resources, monkeypatch):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            handoff = _handoff(receiver)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), handoff)
            await agent.start("unavailable")

            async def unavailable(reference):
                return SimpleNamespace(state="unavailable", reason="store_unavailable")

            with monkeypatch.context() as patch:
                patch.setattr(agent.app, "get_human_attention_state", unavailable)
                await handoff.reconcile_once()
            report = receiver.reports[-1]
            assert report["scan_complete"] is False
            assert report["verify_complete"] is False
            assert report["issues"] == ["attention_unavailable"]
            scan = [json.loads(body)["scan"] for body in receiver.bodies if '"scan"' in body]
            assert scan[-1]["complete"] is False
            assert _states(receiver) == ["active"]
            await handoff.reconcile_once()
            assert receiver.reports[-1]["scan_complete"] is True
            assert receiver.reports[-1]["verify_complete"] is True
            assert receiver.reports[-1]["issues"] == []

    asyncio.run(run())


def test_rejected_scan_cursor_restarts_with_one_page_budget(sqlite_resources):
    async def run():
        async with sqlite_resources as resources:
            receiver = ReferenceReceiver()
            receiver.scan_cursor = "obsolete-cursor"
            handoff = _handoff(receiver, max_pages=1)
            agent = Agent(resources.own(SQLiteSessionStore(resources.path())), None)
            handoff.bind(agent.app)
            await agent.start("missed-pause")

            await handoff.reconcile_once()

            assert _states(receiver) == ["active"]
            assert receiver.scan_cursor is None
            assert receiver.reports[-1]["pages"] == 1
            assert receiver.reports[-1]["scan_complete"] is True
            assert receiver.reports[-1]["issues"] == ["scan_cursor_reset"]

            await handoff.reconcile_once()
            assert _states(receiver) == ["active"]
            assert receiver.reports[-1]["issues"] == []

    asyncio.run(run())
