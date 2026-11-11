"""An operator interruption requested from another process, for runtime tests.

Two ``CayuApp`` instances share one store. The one that owns no active run
interrupts a session the other is running, so its request is only written
durably: status ``interrupting`` with the ``pending_session_interrupt`` and
``pending_interruption_cascade`` markers. The owner must finalize it.
"""

from __future__ import annotations

import asyncio
import contextvars

from tests.runtime.runtime_session_support import _approval_pause_app, collect_events
from tests.runtime.runtime_tool_support import PureRecordingTool, RequireApprovalPolicy

from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest

OPERATOR_REASON = "operator stop"


async def session_paused_for_approval(store, *, session_id: str):
    """Return a worker, an API app and the approval the session is paused on."""

    worker, _ = _approval_pause_app(
        store=store, tool=PureRecordingTool(), policy=RequireApprovalPolicy()
    )
    # A second app on the same store stands in for another process: it owns no
    # active run, so its interrupt only writes the durable request.
    api, _ = _approval_pause_app(
        store=store, tool=PureRecordingTool(), policy=RequireApprovalPolicy()
    )
    initial = await collect_events(
        worker,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "Use the tool")],
        ),
    )
    approval = next(e for e in initial if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    resolution = ToolApprovalRequest(
        session_id=session_id,
        approval_id=approval.payload["approval_id"],
        tool_round_id=approval.payload["tool_round_id"],
        tool_call_id=approval.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )
    return worker, api, resolution


class OperatorInterrupt:
    """Interrupt through the API app, as another process would, and wait for the store."""

    def __init__(self, api, store, *, session_id: str) -> None:
        self._api = api
        self._store = store
        self._session_id = session_id
        self.tasks: list[asyncio.Task[None]] = []
        self.outcome: dict[str, object] = {}

    async def _interrupt(self) -> None:
        try:
            self.outcome["events"] = [
                event
                async for event in self._api.interrupt_session(
                    InterruptSessionRequest(session_id=self._session_id, reason=OPERATOR_REASON)
                )
            ]
        except TimeoutError as exc:
            # Documented answer when another process owns the run: the request
            # is durable and the caller waits for ``session.interrupted``.
            self.outcome["error"] = exc

    async def commit(self) -> None:
        # A fresh context keeps the worker's run fence out of the API call.
        task = asyncio.get_running_loop().create_task(
            self._interrupt(), context=contextvars.Context()
        )
        self.tasks.append(task)
        for _ in range(5000):
            current = await self._store.load(self._session_id)
            assert current is not None
            if current.status is SessionStatus.INTERRUPTING or task.done():
                break
            await asyncio.sleep(0.001)
        current = await self._store.load(self._session_id)
        assert current is not None
        # The store accepted the operator's request while recovery was running.
        assert current.status is SessionStatus.INTERRUPTING, self.outcome
        checkpoint = await self._store.load_checkpoint(self._session_id)
        assert checkpoint is not None and "pending_session_interrupt" in checkpoint
        assert "pending_interruption_cascade" in checkpoint

    async def settle(self) -> None:
        for task in self.tasks:
            await task


async def assert_operator_interruption_finalized(
    store, worker, *, session_id: str, events_before: int
) -> None:
    session = await store.load(session_id)
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED
    interrupted = [
        event
        for event in (await store.load_events(session_id))[events_before:]
        if event.type is EventType.SESSION_INTERRUPTED
    ]
    # The only interruption since `events_before` is the operator's own.
    assert [event.payload.get("interruption_type") for event in interrupted] == [
        "operator_requested"
    ]
    assert interrupted[0].payload["reason"] == OPERATOR_REASON
    assert await worker.drain_background_interruptions(timeout_s=5) is True
    checkpoint = await store.load_checkpoint(session_id)
    assert checkpoint is not None
    assert "pending_session_interrupt" not in checkpoint
    assert "pending_interruption_cascade" not in checkpoint
    # The approval stays pending for an explicit retry.
    assert "pending_tool_approval" in checkpoint
