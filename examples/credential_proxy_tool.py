from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    CredentialProxy,
    Environment,
    EnvironmentSpec,
    Message,
    PassthroughProxy,
    ProxyAuthorizationResult,
    RunRequest,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_send_email",
                    name="send_demo_email",
                    arguments={
                        "to": "ops@example.com",
                        "template": "invoice_reminder",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("email queued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


class SendgridAllowlistProxy(CredentialProxy):
    """Trusted demo proxy that only authorizes one SendGrid action."""

    def __init__(self, vault: StaticVault) -> None:
        self._passthrough = PassthroughProxy(vault)

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        return await self._passthrough.resolve(ref, scope=scope)

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        allowed = (
            destination == "https://api.sendgrid.com/v3/mail/send"
            and credential is not None
            and credential.name == "sendgrid_api_key"
            and action == "send_email"
        )
        if not allowed:
            return ProxyAuthorizationResult(
                allowed=False,
                reason="Destination, credential, or action is not allowed.",
                metadata={"policy": "sendgrid_allowlist"},
            )
        return ProxyAuthorizationResult(
            allowed=True,
            metadata={
                "policy": "sendgrid_allowlist",
                "template": (metadata or {}).get("template"),
            },
        )


class SendDemoEmailTool(Tool):
    spec = ToolSpec(
        name="send_demo_email",
        description="Queue a demo email through a trusted proxy-aware tool.",
        input_schema={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "template": {"type": "string"},
            },
            "required": ["to", "template"],
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.proxy is None:
            return ToolResult(content="No credential proxy configured.", is_error=True)

        credential = SecretRef(name="sendgrid_api_key")
        authorization = await ctx.proxy.authorize_request(
            destination="https://api.sendgrid.com/v3/mail/send",
            credential=credential,
            action="send_email",
            metadata={
                "to": args["to"],
                "template": args["template"],
            },
        )
        if not authorization.allowed:
            return ToolResult(
                content=f"Email blocked: {authorization.reason}",
                structured=authorization.model_dump(mode="json"),
                is_error=True,
            )

        resolved = await ctx.proxy.resolve(
            credential,
            scope={
                "session_id": ctx.session_id,
                "tool": "send_demo_email",
            },
        )
        raw_secret = resolved.value.get_secret_value()

        return ToolResult(
            # This intentionally includes the resolved value to demonstrate that
            # Cayu redacts proxy-resolved secrets before persistence/logging.
            content=f"Queued {args['template']} to {args['to']} using {raw_secret}.",
            structured={
                "to": args["to"],
                "template": args["template"],
                "authorized": True,
                "resolved_secret": raw_secret,
            },
        )


async def main() -> None:
    vault = StaticVault({"sendgrid_api_key": "SG.demo-secret-value"})
    app = CayuApp(enable_logging=False)
    app.register_provider(FakeProvider(), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="trusted-tools"),
            vault=vault,
            proxy=SendgridAllowlistProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SendDemoEmailTool()],
    )

    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="demo_credential_proxy_tool",
            messages=[Message.text("user", "send the demo email")],
        )
    ):
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )


if __name__ == "__main__":
    asyncio.run(main())
