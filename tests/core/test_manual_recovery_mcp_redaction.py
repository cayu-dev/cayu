"""Manual results retain the registered MCP transport's secret-redaction boundary."""

from __future__ import annotations

import asyncio
import json

from pydantic import SecretStr
from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions
from tests.runtime.runtime_provider_support import FakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.mcp.base import McpToolResult
from cayu.mcp.tools import McpToolAdapter, McpToolset
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _LostTerminalStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__(
            public_authority_alias_codec=PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="test", keys={"test": SecretStr("A" * 43)}
                )
            )
        )
        self.failed_terminal = False

    async def append_events(self, session_id, events):
        if not self.failed_terminal and any(
            e.type is EventType.TOOL_CALL_COMPLETED for e in events
        ):
            self.failed_terminal = True
            raise ConnectionError("terminal acknowledgement unavailable")
        await super().append_events(session_id, events)


def test_manual_recovery_redacts_mcp_transport_credentials_before_publication():
    asyncio.run(_scenario())


async def _scenario():
    secret = "mcp-manual-result-transport-credential-canary"
    store = _LostTerminalStore()
    definition = _fake_tool_definitions("echo")[0].model_copy(
        update={"annotations": {"readOnlyHint": True}}
    )

    class Session(FakeMcpSession):
        @property
        def secret_redactor(self):
            return SecretRedactor(secret)

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return McpToolResult(content=[{"type": "text", "text": secret}])

    transport = Session(definitions=(definition,))
    toolset = McpToolset(
        server=_fake_server_spec().model_copy(update={"connection_id": "manual-mcp"}),
        session=transport,
        definitions=(definition,),
    )
    adapter = McpToolAdapter(toolset=toolset, definition=definition)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="echo", name=adapter.name, arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
        ]
    )
    app = CayuApp(session_store=store, secret_redactor=SecretRedactor(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="test"), tools=[adapter])
    try:
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="manual-mcp",
                    agent_name="assistant",
                    messages=[Message.text("user", "Read the status")],
                )
            )
        ]
        assert store.failed_terminal
        assert initial[-1].type is EventType.SESSION_FAILED
        assert len(transport.calls) == 1
        checkpoint = await store.load_checkpoint("manual-mcp")
        pending = checkpoint["pending_tool_round"]
        assert pending["assistant_publication"]["secret_resolution_scope"] == "static"
        # The transport credential is intentionally not in the application registry.
        assert app._secret_redactor.redact_text(secret) == secret
        recovered = [
            event
            async for event in app.recover_tool_round(
                ToolRoundRecoveryRequest(
                    session_id="manual-mcp",
                    round_id=pending["tool_round_id"],
                    tool_call_id="echo",
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message=f"Externally verified status: {secret}",
                    structured={"raw_status": secret},
                )
            )
        ]
        assert (await store.load("manual-mcp")).status is SessionStatus.COMPLETED
        terminal = next(
            event
            for event in recovered
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("manual_recovery") is True
        )
        persisted = next(
            event
            for event in await store.load_events("manual-mcp")
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("manual_recovery") is True
        )
        assert secret not in persisted.model_dump_json(), (
            "Manual recovery persisted the MCP transport credential without composing "
            "the adapter redactor"
        )
        assert secret not in terminal.model_dump_json()
        assert REDACTED_SECRET in terminal.payload["result"]["content"]
        assert secret not in json.dumps(
            [e.model_dump(mode="json") for e in await store.load_events("manual-mcp")]
        )
        assert len(transport.calls) == 1
    finally:
        await app.aclose()
        await toolset.close()
