"""Route interruption through the owner of each session runtime."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Literal

from cayu.environments.snapshots import ExecutionSnapshotError
from cayu.events import Event
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, copy_interrupt_session_request

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.sessions.records import Session


def require_native_session(
    session: Session | None, operation: Literal["capture", "resume", "fork"]
) -> None:
    """Keep native lifecycle operations out of external harness state."""
    external = session is not None and session.runtime_name == "external_agent"
    session = None
    if not external:
        return
    if operation == "capture":
        raise ExecutionSnapshotError(
            "External agent state cannot be captured in execution snapshots."
        )
    raise ValueError(f"External agent sessions cannot {operation}; schedule a fresh run.")


async def interrupt(
    app: CayuApp,
    request: InterruptSessionRequest,
    *,
    store_resolved_session_id: str | None = None,
) -> AsyncGenerator[Event, None]:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
    request = copy_interrupt_session_request(request)
    stored = await app.session_store.load(request.session_id)
    if stored is not None and stored.runtime_name == "external_agent":
        if stored.status not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
            return
        await app.session_store.transition_status(
            stored.id,
            from_statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTING},
            to_status=SessionStatus.INTERRUPTING,
        )
        from cayu.external_agents._authority import cancel_run_once

        cancel_run_once(
            stored,
            app._session_control._interrupt_targets(stored.id),
            message="External session interruption requested.",
        )
        yield await app.emit_event(
            Event(
                type="custom.external_agent.interruption_requested",
                session_id=stored.id,
                agent_name=stored.agent_name,
                payload={"external_agent": True},
            )
        )
        return
    stream = app._session_interruption.interrupt_session(
        request=request,
        store_resolved_session_id=store_resolved_session_id,
    )
    async with _close_delegated_event_stream(stream) as owned_stream:
        async for item in owned_stream:
            yield item
