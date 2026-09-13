"""One tool-list subscription multiplexed by the existing stdio reader."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from cayu.mcp._jsonrpc import McpProtocolError, jsonrpc_request_payload
from cayu.mcp._subscriptions import ModernToolSubscription, SubscriptionEvent

if TYPE_CHECKING:
    from cayu.mcp.stdio import StdioMcpSession

_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"


class StdioToolSubscription:
    """Own establishment, cancellation, and re-listen; never read stdout."""

    def __init__(self, session: StdioMcpSession) -> None:
        self.session = session
        self.task: asyncio.Task[None] | None = None
        self.state: ModernToolSubscription | None = None
        self.changed = asyncio.Event()
        self.complete = False
        self.unsupported = False
        self.deadline = 0.0
        self.started_at = 0.0
        self.stopping = False
        self.failure: str | None = None
        # Cancelled streams can have already-buffered frames. Keep only bounded
        # identities, never payloads; older unknown identities fail closed.
        self.retired: deque[int] = deque(maxlen=64)

    def start(self) -> None:
        if self.task is not None or self.session._closed:
            return
        if self.unsupported:
            self._continuity(True)
            return
        self.stopping = False
        self.task = asyncio.create_task(self._run())
        self.task.add_done_callback(self._finished)

    def stop(self) -> None:
        task = self.task
        if task is not None and not task.done() and not self.stopping:
            self.stopping = True
            task.cancel()

    async def close(self) -> None:
        task = self.task
        if task is not None:
            await asyncio.gather(asyncio.shield(task), return_exceptions=True)

    def _finished(self, task: asyncio.Task[None]) -> None:
        if self.task is task:
            self.task = None
        if not task.cancelled():
            task.exception()
        if self.session._tools_list_changed_handler is not None:
            self.start()

    def _continuity(self, ready: bool) -> None:
        handler = self.session._tools_list_changed_continuity_handler
        if handler is not None:
            handler(ready)

    def _check_idle_gap(self) -> None:
        gap = self.session._last_expired_idle_gap
        if gap is not None and (
            max(gap[0], self.started_at) + self.session.transport_limits.idle_timeout_s <= gap[1]
        ):
            raise McpProtocolError("MCP stdio subscription crossed its channel idle deadline.")

    def consume(self, message: dict[str, Any]) -> bool:
        """Route subscription frames without stealing ordinary RPC responses."""

        notification = "method" in message
        payload = message.get("params" if notification else "result")
        meta = payload.get("_meta") if type(payload) is dict else None
        correlated = type(meta) is dict and _SUBSCRIPTION_ID in meta
        identity = meta.get(_SUBSCRIPTION_ID) if correlated else None
        state = self.state
        if state is not None and not state.acknowledged and not self.stopping:
            self._check_idle_gap()
        response_matches = (
            not notification
            and state is not None
            and type(message.get("id")) is int
            and message["id"] == state.request_id
        )
        if (
            not correlated
            and not response_matches
            and message.get("method") != "notifications/subscriptions/acknowledged"
        ):
            return False
        if type(identity) is int and identity in self.retired:
            return True
        if self.stopping:
            return True
        if state is None:
            raise McpProtocolError("MCP stdio message has no active subscription.")
        if self.complete or (self.unsupported and notification):
            raise McpProtocolError(
                "MCP stdio subscription sent a message after its terminal filter or result."
            )
        if not state.acknowledged and asyncio.get_running_loop().time() >= self.deadline:
            raise McpProtocolError("MCP stdio subscription acknowledgement timed out.")
        event = state.consume(message)
        if event is SubscriptionEvent.ACKNOWLEDGED:
            self._continuity(True)
        elif event is SubscriptionEvent.TOOLS_CHANGED:
            handler = self.session._tools_list_changed_handler
            if handler is not None:
                handler()
        else:
            self.complete = event is SubscriptionEvent.COMPLETE
            self.unsupported = self.unsupported or event is SubscriptionEvent.UNSUPPORTED
            self._continuity(False)
        self.changed.set()
        return True

    async def _cancel(self) -> None:
        state = self.state
        if state is None or self.complete:
            return
        self.retired.append(state.request_id)
        # This owner never waits for session close: interrupted writes can
        # schedule that close, which in turn joins this owner before closing stdin.
        async with asyncio.timeout(self.session.cancellation_notification_timeout_s):
            await self.session._notify(
                "notifications/cancelled",
                {"requestId": state.request_id, "reason": "Cayu subscription owner stopped."},
            )

    async def _run(self) -> None:
        payload: dict[str, Any] = {}
        try:
            while not self.session._closed:
                request_id = self.session._next_id
                self.session._next_id += 1
                self.state = ModernToolSubscription(request_id)
                self.complete = False
                self.changed.clear()
                self.started_at = asyncio.get_running_loop().time()
                self.deadline = self.started_at + self.session.transport_limits.total_call_timeout_s
                params = self.session._wire_protocol.prepare_request_params(
                    "subscriptions/listen", {"notifications": {"toolsListChanged": True}}
                )
                payload = jsonrpc_request_payload(request_id, "subscriptions/listen", params)
                params.clear()
                await self.session._write_with_timeout(
                    payload,
                    timeout_message="MCP stdio subscription write timed out.",
                    call_deadline=self.deadline,
                )
                payload.clear()
                while not self.complete and not self.unsupported:
                    self.changed.clear()
                    if self.state.acknowledged:
                        # Stdio subscriptions have no heartbeat contract. Silence
                        # is normal until the catalogue changes; ordinary RPCs
                        # retain their own independent idle and total budgets.
                        await self.changed.wait()
                        continue
                    self._check_idle_gap()
                    idle_deadline = (
                        max(self.session._last_read_activity, self.started_at)
                        + self.session.transport_limits.idle_timeout_s
                    )
                    deadline = min(self.deadline, idle_deadline)
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise McpProtocolError("MCP stdio subscription exceeded its deadline.")
                    try:
                        async with asyncio.timeout(remaining):
                            await self.changed.wait()
                    except TimeoutError:
                        # Ordinary replies also prove shared-channel activity.
                        # Recompute idle time, but never extend acknowledgement.
                        continue
                if self.unsupported:
                    break
                self.state = None
                # Completion has already fenced dispatch in the reader. There
                # is no process restart and no replay of ordinary requests.
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            pass
        except BaseException:
            self.failure = "MCP stdio subscription failed; session closed."
            self.session._schedule_close()
        finally:
            payload.clear()
            try:
                await self._cancel()
            except (Exception, asyncio.CancelledError):
                self.failure = "MCP stdio subscription cancellation failed; session closed."
                self.session._schedule_close()
            self.state = None
            if self.unsupported and not self.session._closed:
                self._continuity(True)
