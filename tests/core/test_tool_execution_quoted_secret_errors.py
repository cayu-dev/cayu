"""Native exception summaries must redact Python-quoted secret values too."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_tool_execution import _ScriptedProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolEffect, ToolSpec
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("error_kind", ["key_error", "file_error"])
def test_quoted_secret_in_native_exception_is_not_published(error_kind):
    secret = "credential-first-line\ncredential-private-second-line"

    class LookupTool(Tool):
        spec = ToolSpec(name="lookup", effect=ToolEffect.NONE, max_terminal_payload_bytes=4096)

        async def run(self, ctx, args):
            # Both exceptions use repr() internally, without application code
            # deliberately transforming or printing the credential.
            if error_kind == "key_error":
                return {}[secret]
            raise FileNotFoundError(2, "No such file or directory", secret)

    async def scenario():
        provider = _ScriptedProvider([("lookup-call", "lookup", {})])
        app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[LookupTool()])
        live = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="quoted-secret-error",
                    agent_name="agent",
                    messages=[Message.text("user", "Look up the credential.")],
                )
            )
        ]
        durable = await app.session_store.load_events("quoted-secret-error")
        transcript = await app.session_store.load_transcript("quoted-secret-error")
        return live, durable, transcript, provider.requests

    live, durable, transcript, requests = asyncio.run(scenario())
    publications = {}
    for name, events in (("live terminal", live), ("durable terminal", durable)):
        terminal = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
        publications[name] = terminal.payload["result"]["content"]
    for name, messages in (
        ("transcript", transcript),
        ("next model request", requests[-1].messages),
    ):
        publications[name] = json.dumps([message.model_dump(mode="json") for message in messages])
    leaked_to = [
        name for name, text in publications.items() if "credential-private-second-line" in text
    ]
    assert leaked_to == [], f"Python-quoted credential was published in: {leaked_to}"
    assert REDACTED_SECRET in publications["live terminal"]
