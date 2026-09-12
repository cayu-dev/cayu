"""Validated tool-list subscription state for MCP 2026-07-28."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, cast

from cayu.mcp._jsonrpc import McpProtocolError

_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
_ACKNOWLEDGED = "notifications/subscriptions/acknowledged"
_TOOLS_CHANGED = "notifications/tools/list_changed"


class SubscriptionEvent(Enum):
    ACKNOWLEDGED = auto()
    UNSUPPORTED = auto()
    TOOLS_CHANGED = auto()
    COMPLETE = auto()


@dataclass(slots=True)
class ModernToolSubscription:
    """One request identity and the filter acknowledged on its response stream."""

    request_id: int
    acknowledged: bool = False

    def consume(self, message: dict[str, Any]) -> SubscriptionEvent:
        error_message: str | None = None
        try:
            return self._consume(message)
        except McpProtocolError as error:
            error_message = str(error)
        finally:
            message.clear()
        # Raw wire values and inner validation frames must not escape in a
        # public traceback, even when the peer sends malformed control fields.
        raise McpProtocolError(error_message) from None

    def _consume(self, message: dict[str, Any]) -> SubscriptionEvent:
        if message.get("jsonrpc") != "2.0":
            raise McpProtocolError("MCP subscription message has invalid JSON-RPC version.")
        if "method" in message:
            if "id" in message or "result" in message or "error" in message:
                raise McpProtocolError("MCP subscription expected a notification.")
            params = self._validate_correlation(message.get("params"))
            method = message["method"]
            if not self.acknowledged:
                if method != _ACKNOWLEDGED:
                    raise McpProtocolError(
                        "MCP subscription notification preceded acknowledgement."
                    )
                notifications = params.get("notifications")
                if type(notifications) is not dict or set(notifications) - {
                    "toolsListChanged",
                    "promptsListChanged",
                    "resourcesListChanged",
                    "resourceSubscriptions",
                }:
                    raise McpProtocolError("MCP subscription acknowledged an unrequested filter.")
                for field in ("promptsListChanged", "resourcesListChanged"):
                    if field in notifications and notifications[field] is not False:
                        raise McpProtocolError(
                            "MCP subscription acknowledged an unrequested filter."
                        )
                if "resourceSubscriptions" in notifications:
                    uris = notifications["resourceSubscriptions"]
                    if type(uris) is not list or uris:
                        raise McpProtocolError(
                            "MCP subscription acknowledged an unrequested filter."
                        )
                supported = notifications.get("toolsListChanged", False)
                if type(supported) is not bool:
                    raise McpProtocolError("MCP subscription filter must contain a boolean.")
                self.acknowledged = True
                return (
                    SubscriptionEvent.ACKNOWLEDGED if supported else SubscriptionEvent.UNSUPPORTED
                )
            if method != _TOOLS_CHANGED:
                raise McpProtocolError("MCP subscription delivered an unrequested notification.")
            return SubscriptionEvent.TOOLS_CHANGED
        if not self.acknowledged:
            raise McpProtocolError("MCP subscription ended before acknowledgement.")
        if type(message.get("id")) is not int or message["id"] != self.request_id:
            raise McpProtocolError("MCP subscription completion has an invalid response ID.")
        if "error" in message:
            raise McpProtocolError("MCP subscription returned a JSON-RPC error.")
        result = self._validate_correlation(message.get("result"))
        if result.get("resultType", "complete") != "complete":
            raise McpProtocolError("MCP subscription returned an unsupported resultType.")
        return SubscriptionEvent.COMPLETE

    def _validate_correlation(self, payload: object) -> dict[str, Any]:
        if type(payload) is not dict:
            raise McpProtocolError("MCP subscription payload must be an object.")
        payload = cast("dict[str, Any]", payload)
        meta = payload.get("_meta")
        subscription_id = meta.get(_SUBSCRIPTION_ID) if type(meta) is dict else None
        if type(subscription_id) is not int or subscription_id != self.request_id:
            raise McpProtocolError("MCP subscription message has an invalid subscription ID.")
        return payload
