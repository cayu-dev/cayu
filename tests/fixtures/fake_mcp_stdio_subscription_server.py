"""Deterministic shared-channel subscription peer, including hostile frames."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fake_mcp_2026_stdio_server import _append_request, _respond, _write

KEY = "io.modelcontextprotocol/subscriptionId"


def notification(method, identity, **params):
    return {"jsonrpc": "2.0", "method": method, "params": {"_meta": {KEY: identity}, **params}}


def main():
    identity = None
    previous = None
    added = False
    pending_call = None
    mode = os.environ.get("CAYU_SUBSCRIPTION_MODE", "normal")
    log = Path(os.environ["CAYU_FAKE_MCP_REQUEST_LOG"])
    for line in sys.stdin:
        request = json.loads(line)
        _append_request(log, request)
        method = request["method"]
        if method == "notifications/cancelled":
            previous = request["params"]["requestId"]
            # Deliberately emit already-buffered events after cancellation.
            _write(notification("notifications/tools/list_changed", previous))
            continue
        request_id = request["id"]
        if method == "subscriptions/listen":
            identity = request_id
            if mode == "no-ack":
                continue
            if mode == "pre-ack":
                _write(notification("notifications/tools/list_changed", identity))
                continue
            honored = {} if mode == "unsupported" else {"toolsListChanged": True}
            _write(
                notification(
                    "notifications/subscriptions/acknowledged", identity, notifications=honored
                )
            )
            continue
        if method == "server/discover":
            result = {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {"listChanged": True}},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "search", "description": "Search.", "inputSchema": {}}]}
            if added:
                result["tools"].append(
                    {"name": "added", "description": "Added.", "inputSchema": {}}
                )
        elif method == "tools/call":
            if request["params"].get("arguments", {}).get("defer"):
                pending_call = request_id
                continue
            result = {"content": [{"type": "text", "text": "ok"}]}
        elif method == "resources/read":
            action = request["params"]["uri"]
            if action == "control://change":
                added = True
                for _ in range(10):
                    _write(notification("notifications/tools/list_changed", identity))
            elif action == "control://release":
                _write(notification("notifications/tools/list_changed", identity))
                _respond(pending_call, {"content": [{"type": "text", "text": "settled"}]})
                pending_call = None
            elif action == "control://complete":
                _respond(identity, {"resultType": "complete", "_meta": {KEY: identity}})
            elif action == "control://exit":
                return
            elif action == "control://old":
                _write(notification("notifications/tools/list_changed", previous))
            elif action == "control://wrong-id":
                _write(notification("notifications/tools/list_changed", identity + 100))
            elif action == "control://boolean-id":
                _write(notification("notifications/tools/list_changed", True))
            elif action == "control://duplicate-ack":
                _write(
                    notification(
                        "notifications/subscriptions/acknowledged",
                        identity,
                        notifications={"toolsListChanged": True},
                    )
                )
            elif action == "control://unrequested":
                _write(notification("notifications/prompts/list_changed", identity))
            elif action == "control://bad-complete":
                _respond(identity, {"resultType": "input_required", "_meta": {KEY: identity}})
            elif action == "control://uncorrelated":
                _write(
                    {"jsonrpc": "2.0", "method": "notifications/tools/list_changed", "params": {}}
                )
            elif action == "control://oversize":
                sys.stdout.write("x" * 8192 + "\n")
                sys.stdout.flush()
            elif action == "control://malformed":
                sys.stdout.write("{malformed\n")
                sys.stdout.flush()
            result = {"contents": [{"uri": action, "text": "ok"}]}
        else:
            raise AssertionError(method)
        _respond(
            request_id, {"resultType": "complete", "ttlMs": 0, "cacheScope": "private", **result}
        )


if __name__ == "__main__":
    main()
