"""Redaction contracts."""

from __future__ import annotations

import asyncio
from typing import Any

from tests.runtime.runtime_environment_support import ProvisionedTestRunner
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _assert_proxy_tool_effect_remains_unknown,
    _private_record_for_public_event,
    collect_events,
)
from tests.runtime.runtime_tool_support import EchoTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message, ToolResultPart
from cayu.observability.hooks import RuntimeHook, ToolCallHookContext
from cayu.providers.base import ModelStreamEvent
from cayu.proxies.base import CredentialProxy, ProxyAuthorizationResult
from cayu.proxies.passthrough import PassthroughProxy
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tools.base import (
    _POLICY_DENIAL_TEXT_MAX_BYTES,
    _POLICY_DENIAL_TRUNCATION_MARKER,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    _bound_policy_denial_text,
)
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    ExecCommandTool,
)
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult
from cayu.vaults.base import ResolvedSecret, SecretRef
from cayu.vaults.static import StaticVault


def test_cayu_app_redacts_tool_results_before_events_transcript_and_context() -> None:
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "sk-runtime-secret-value"

    class LeakyTool(Tool):
        spec = ToolSpec(
            name="leak",
            description="Return a result containing a known secret.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(
                content=f"connected with {secret_value}",
                structured={
                    "token": secret_value,
                    "nested": ["safe", f"prefix-{secret_value}-suffix"],
                },
                artifacts=[
                    {
                        "type": "debug",
                        "metadata": {"authorization": f"Bearer {secret_value}"},
                    }
                ],
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_leak", name="leak", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[LeakyTool()])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_result_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert secret_value not in str(result_payload)
    assert REDACTED_SECRET in result_payload["content"]
    assert result_payload["structured"]["token"] == REDACTED_SECRET
    assert result_payload["structured"]["nested"][1] == f"prefix-{REDACTED_SECRET}-suffix"
    assert result_payload["artifacts"][0]["metadata"]["authorization"] == (
        f"Bearer {REDACTED_SECRET}"
    )

    transcript = asyncio.run(store.load_transcript("sess_tool_result_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"connected with {REDACTED_SECRET}"
    assert tool_part.structured is not None
    assert tool_part.structured["token"] == REDACTED_SECRET
    assert tool_part.artifacts[0]["metadata"]["authorization"] == f"Bearer {REDACTED_SECRET}"

    second_request = provider.requests[1]
    assert secret_value not in str(
        [message.model_dump(mode="json") for message in second_request.messages]
    )


def test_cayu_app_redacts_proxy_resolved_secrets_from_tool_results() -> None:
    from cayu.vaults.redaction import REDACTED_SECRET

    secret_value = "sk-proxy-resolved-secret"

    class ProxyLeakingTool(Tool):
        spec = ToolSpec(
            name="proxy_leak",
            description="Resolve a proxy secret and accidentally return it.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            resolved = await ctx.proxy.resolve(
                SecretRef(name="api_key"),
                scope={"session_id": ctx.session_id},
            )
            raw_secret = resolved.value.get_secret_value()
            return ToolResult(
                content=f"connected with {raw_secret}",
                structured={
                    "token": raw_secret,
                    "nested": ["safe", f"prefix-{raw_secret}-suffix"],
                },
                artifacts=[
                    {
                        "type": "debug",
                        "metadata": {"authorization": f"Bearer {raw_secret}"},
                    }
                ],
            )

    store = InMemorySessionStore()
    vault = StaticVault({"api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_leak", name="proxy_leak", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyLeakingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_secret_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_event = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    result_payload = tool_event.payload["result"]
    assert secret_value not in str(result_payload)
    assert result_payload["content"] == f"connected with {REDACTED_SECRET}"
    assert result_payload["structured"]["token"] == REDACTED_SECRET
    assert result_payload["structured"]["nested"][1] == f"prefix-{REDACTED_SECRET}-suffix"
    assert result_payload["artifacts"][0]["metadata"]["authorization"] == (
        f"Bearer {REDACTED_SECRET}"
    )

    transcript = asyncio.run(store.load_transcript("sess_proxy_secret_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    tool_part = tool_message.content[0]
    assert isinstance(tool_part, ToolResultPart)
    assert secret_value not in str(tool_part.model_dump(mode="json"))
    assert tool_part.content == f"connected with {REDACTED_SECRET}"
    assert tool_part.structured is not None
    assert tool_part.structured["token"] == REDACTED_SECRET
    assert tool_part.artifacts[0]["metadata"]["authorization"] == f"Bearer {REDACTED_SECRET}"

    second_request = provider.requests[1]
    assert secret_value not in str(
        [message.model_dump(mode="json") for message in second_request.messages]
    )


def test_cayu_app_emits_redacted_proxy_authorization_events() -> None:
    from cayu.vaults.redaction import REDACTED_SECRET

    secret_value = "sk-authorized-proxy-secret"

    class AuditingProxy(CredentialProxy):
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
            if action == "deny_email":
                return ProxyAuthorizationResult(
                    allowed=False,
                    reason="destination denied",
                    metadata={"policy": "blocked", "debug": secret_value},
                )
            return ProxyAuthorizationResult(
                allowed=True,
                metadata={"policy": "allowlist", "debug": secret_value},
            )

    class ProxyAuthorizingTool(Tool):
        spec = ToolSpec(
            name="proxy_auth",
            description="Resolve and authorize proxy use.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.resolve(SecretRef(name="sendgrid_api_key"))
            allowed = await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="send_email",
                metadata={"template": "invoice_reminder"},
            )
            denied = await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(name="sendgrid_api_key"),
                action="deny_email",
                metadata={"template": "invoice_reminder"},
            )
            return ToolResult(
                content="checked",
                structured={
                    "allowed": allowed.allowed,
                    "denied": denied.allowed,
                },
            )

    store = InMemorySessionStore()
    vault = StaticVault({"sendgrid_api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_auth", name="proxy_auth", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=AuditingProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyAuthorizingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_authorization_events",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    proxy_events = [event for event in events if event.type == EventType.CREDENTIAL_PROXY_CHECKED]
    assert len(proxy_events) == 2
    assert [event.type for event in events].index(EventType.CREDENTIAL_PROXY_CHECKED) < [
        event.type for event in events
    ].index(EventType.TOOL_CALL_COMPLETED)

    allowed_event = proxy_events[0]
    assert allowed_event.tool_name == "proxy_auth"
    assert [event.payload["tool_call_id"] for event in proxy_events] == [
        PRIVATE_EVENT_AUTHORITY,
        PRIVATE_EVENT_AUTHORITY,
    ]
    durable_records = asyncio.run(
        store.query_events(EventQuery(session_id="sess_proxy_authorization_events"))
    )
    assert [
        _private_record_for_public_event(durable_records, event).event.payload["tool_call_id"]
        for event in proxy_events
    ] == ["call_proxy_auth", "call_proxy_auth"]
    assert allowed_event.payload["destination"] == "https://api.sendgrid.com/v3/mail/send"
    assert allowed_event.payload["credential"] == "sendgrid_api_key"
    assert allowed_event.payload["action"] == "send_email"
    assert allowed_event.payload["metadata"] == {"template": "invoice_reminder"}
    assert allowed_event.payload["allowed"] is True
    assert allowed_event.payload["reason"] is None
    assert allowed_event.payload["result_metadata"] == {
        "policy": "allowlist",
        "debug": REDACTED_SECRET,
    }

    denied_event = proxy_events[1]
    assert denied_event.payload["allowed"] is False
    assert denied_event.payload["reason"] == "destination denied"
    assert denied_event.payload["result_metadata"] == {
        "policy": "blocked",
        "debug": REDACTED_SECRET,
    }
    assert secret_value not in str([event.model_dump(mode="json") for event in proxy_events])


def test_cayu_app_records_original_proxy_authorization_metadata() -> None:
    class MutatingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("resolve should not be called")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            if metadata is None:
                raise AssertionError("metadata should be copied before proxy delegation")
            metadata["template"] = "mutated"
            metadata["new_field"] = "proxy-added"
            if credential is not None:
                credential.metadata["scope"] = "mutated"
            return ProxyAuthorizationResult(allowed=True)

    class ProxyAuthorizingTool(Tool):
        spec = ToolSpec(
            name="proxy_auth",
            description="Authorize proxy use.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            await ctx.proxy.authorize_request(
                destination="https://api.sendgrid.com/v3/mail/send",
                credential=SecretRef(
                    name="sendgrid_api_key",
                    metadata={"scope": "email"},
                ),
                action="send_email",
                metadata={"template": "invoice_reminder"},
            )
            return ToolResult(content="checked")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_proxy_auth", name="proxy_auth", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=MutatingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyAuthorizingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_authorization_metadata_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    proxy_events = [event for event in events if event.type == EventType.CREDENTIAL_PROXY_CHECKED]
    assert len(proxy_events) == 1
    assert proxy_events[0].payload["metadata"] == {"template": "invoice_reminder"}
    assert proxy_events[0].payload["credential"] == "sendgrid_api_key"


def test_cayu_app_redacts_proxy_secret_after_tool_mutates_resolved_secret() -> None:
    from pydantic import SecretStr

    from cayu.vaults.redaction import REDACTED_SECRET

    secret_value = "sk-mutated-proxy-secret"

    class MutatingSecretTool(Tool):
        spec = ToolSpec(
            name="mutate_secret",
            description="Mutate returned proxy secret and leak original value.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            secret = await ctx.proxy.resolve(SecretRef(name="api_key"))
            raw_value = secret.value.get_secret_value()
            secret.value = SecretStr("changed-after-resolve")
            secret.metadata["leak"] = "mutated"
            return ToolResult(
                content=f"using {raw_value}",
                structured={"token": raw_value},
                artifacts=[
                    {
                        "type": "test.secret",
                        "value": raw_value,
                    }
                ],
            )

    vault = StaticVault({"api_key": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_mutate_secret",
                    name="mutate_secret",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[MutatingSecretTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_resolved_secret_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(tool_events) == 1
    result = tool_events[0].payload["result"]
    assert result["content"] == f"using {REDACTED_SECRET}"
    assert result["structured"] == {"token": REDACTED_SECRET}
    assert result["artifacts"] == [{"type": "test.secret", "value": REDACTED_SECRET}]
    assert secret_value not in str([event.model_dump(mode="json") for event in events])


def test_cayu_app_copies_proxy_resolve_inputs_before_delegation() -> None:
    from pydantic import SecretStr

    class MutatingResolveProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            ref.metadata["scope"] = "proxy-mutated"
            if scope is not None:
                scope["tenant"] = "proxy-mutated"
            return ResolvedSecret(
                name=ref.name,
                value=SecretStr("resolved-secret"),
                metadata={"source": "mutating-proxy"},
            )

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            raise AssertionError("authorize_request should not be called")

    class ProxyResolvingTool(Tool):
        spec = ToolSpec(
            name="resolve_proxy",
            description="Resolve proxy secret with mutable inputs.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            ref = SecretRef(name="api_key", metadata={"scope": "email"})
            scope = {"tenant": "acme"}
            await ctx.proxy.resolve(ref, scope=scope)
            return ToolResult(
                content="checked",
                structured={
                    "ref_metadata": ref.metadata,
                    "scope": scope,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_resolve_proxy",
                    name="resolve_proxy",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=MutatingResolveProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ProxyResolvingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_resolve_input_copy",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(tool_events) == 1
    assert tool_events[0].payload["result"]["structured"] == {
        "ref_metadata": {"scope": "email"},
        "scope": {"tenant": "acme"},
    }


def test_cayu_app_rejects_invalid_proxy_resolve_scope_before_delegation() -> None:
    delegated: list[str] = []
    validation_errors: list[str] = []

    class FailingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            delegated.append("resolve")
            raise AssertionError("invalid scope should fail before proxy delegation")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            raise AssertionError("authorize_request should not be called")

    class InvalidScopeTool(Tool):
        spec = ToolSpec(
            name="invalid_scope",
            description="Resolve proxy secret with invalid scope.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            try:
                await ctx.proxy.resolve(
                    SecretRef(name="api_key"),
                    scope=["not", "an", "object"],  # type: ignore[arg-type]
                )
            except ValueError as exc:
                validation_errors.append(str(exc))
                raise
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_invalid_scope",
                    name="invalid_scope",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=FailingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidScopeTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_invalid_scope",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert delegated == []
    assert validation_errors == ["`scope` must be a JSON object."]
    assert not any(
        event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
        for event in events
    )
    asyncio.run(
        _assert_proxy_tool_effect_remains_unknown(app.session_store, "sess_proxy_invalid_scope")
    )


def test_cayu_app_rejects_invalid_proxy_authorization_metadata_before_delegation() -> None:
    from cayu._validation import DurableValueError

    delegated: list[str] = []
    validation_codes: list[str] = []

    class FailingProxy(CredentialProxy):
        async def resolve(
            self,
            ref: SecretRef,
            *,
            scope: dict[str, Any] | None = None,
        ) -> ResolvedSecret:
            raise AssertionError("resolve should not be called")

        async def authorize_request(
            self,
            *,
            destination: str,
            credential: SecretRef | None = None,
            action: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> ProxyAuthorizationResult:
            delegated.append("authorize_request")
            raise AssertionError("invalid metadata should fail before proxy delegation")

    class InvalidMetadataTool(Tool):
        spec = ToolSpec(
            name="invalid_metadata",
            description="Authorize proxy use with invalid metadata.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.proxy is not None
            try:
                await ctx.proxy.authorize_request(
                    destination="https://api.sendgrid.com/v3/mail/send",
                    metadata=["not", "an", "object"],  # type: ignore[arg-type]
                )
            except DurableValueError as exc:
                validation_codes.append(exc.code)
                raise
            return ToolResult(content="unexpected")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_invalid_metadata",
                    name="invalid_metadata",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            proxy=FailingProxy(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidMetadataTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_invalid_metadata",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert delegated == []
    assert validation_codes == ["invalid_json_object"]
    assert not any(
        event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
        for event in events
    )
    asyncio.run(
        _assert_proxy_tool_effect_remains_unknown(app.session_store, "sess_proxy_invalid_metadata")
    )
    assert EventType.CREDENTIAL_PROXY_CHECKED not in [event.type for event in events]


def test_cayu_app_redacts_blocked_tool_result_event_payload() -> None:
    from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor

    secret_value = "BOUNDARY_SECRET_policy_value"
    raw_reason = "a" * 4050 + secret_value
    observed: dict[str, str] = {}

    class ObserveBlockedResult(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            observed["reason"] = context.tool_event.payload["reason"]
            observed["content"] = context.result.content

    class SecretPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=raw_reason,
                metadata={"token": secret_value},
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        tool_policy=SecretPolicy(),
        runtime_hooks=[ObserveBlockedResult()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blocked_tool_result_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    blocked_event = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    expected_reason = _bound_policy_denial_text(raw_reason.replace(secret_value, REDACTED_SECRET))
    assert "BOUNDARY" not in str(blocked_event.payload)
    assert blocked_event.payload["denied_by"] == "tool_policy"
    assert blocked_event.payload["decision"] == "deny"
    assert blocked_event.payload["tool_name"] == "echo"
    assert blocked_event.payload["reason"] == expected_reason
    assert blocked_event.payload["metadata"]["token"] == REDACTED_SECRET
    assert blocked_event.payload["result"]["content"] == expected_reason
    assert blocked_event.payload["result"]["structured"]["metadata"]["token"] == REDACTED_SECRET
    assert observed == {"reason": expected_reason, "content": expected_reason}

    transcript = asyncio.run(store.load_transcript("sess_blocked_tool_result_redaction"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert isinstance(tool_message.content[0], ToolResultPart)
    assert "BOUNDARY" not in str(tool_message.model_dump(mode="json"))


def test_cayu_app_bounds_policy_denial_after_secret_redaction_expands_it() -> None:
    from cayu.vaults.redaction import SecretRedactor

    secret_value = "qz"
    raw_reason = secret_value * 3000

    class ExpandingSecretPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason=raw_reason)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_echo", name="echo", arguments={"text": "x"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret_value),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        tool_policy=ExpandingSecretPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_expanding_policy_denial_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    for value in (
        blocked.payload["reason"],
        blocked.payload["result"]["content"],
        blocked.payload["result"]["structured"]["reason"],
    ):
        assert len(value.encode("utf-8")) <= _POLICY_DENIAL_TEXT_MAX_BYTES
        assert value.endswith(_POLICY_DENIAL_TRUNCATION_MARKER)
    transcript = asyncio.run(store.load_transcript(blocked.session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert len(tool_result.content.encode("utf-8")) <= _POLICY_DENIAL_TEXT_MAX_BYTES


def test_policy_denial_redaction_preserves_protocol_fields_that_match_secrets() -> None:
    from cayu.vaults.redaction import SecretRedactor

    secret_values = [
        "reason",
        "denied",
        "deny",
        "tool",
        "decision",
        "result",
        "quarantined",
        "unavailable",
        "-",
    ]
    redactor = SecretRedactor(secret_values)
    raw_reason = "reason denied deny tool echo decision result " * 300
    raw_metadata = {"detail": "tool", "safe": "deny this value"}
    observed: dict[str, Any] = {}

    class CollisionPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=raw_reason,
                metadata=raw_metadata,
            )

    class ObserveCanonicalDenial(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            observed["payload"] = context.tool_event.payload
            observed["structured"] = context.result.structured

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_echo", name="echo", arguments={"text": "x"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(session_store=store, secret_redactor=redactor, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model"),
        tools=[EchoTool()],
        tool_policy=CollisionPolicy(),
        runtime_hooks=[ObserveCanonicalDenial()],
    )
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_protocol_collision",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    expected_reason = _bound_policy_denial_text(redactor.redact_text(raw_reason))
    expected_metadata = redactor.redact_json(raw_metadata)
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert blocked.payload["tool_name"] == "echo"
    assert blocked.payload["denied_by"] == "tool_policy"
    assert blocked.payload["decision"] == "deny"
    assert blocked.payload["reason"] == expected_reason
    assert blocked.payload["metadata"] == expected_metadata
    assert blocked.payload["result"]["content"] == expected_reason
    assert blocked.payload["result"]["structured"] == {
        "decision": "deny",
        "reason": expected_reason,
        "metadata": expected_metadata,
    }
    durable_records = asyncio.run(
        store.query_events(EventQuery(session_id="sess_protocol_collision"))
    )
    durable_blocked = _private_record_for_public_event(durable_records, blocked).event
    observed_payload = dict(observed["payload"])
    observed_payload["tool_name"] = durable_blocked.payload["tool_name"]
    assert observed_payload == durable_blocked.payload
    assert blocked.payload["tool_call_id"] == f"{blocked.id}:tool_call_id"
    assert blocked.payload["tool_round_id"] == f"{blocked.id}:tool_round_id"
    assert blocked.payload["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert blocked.payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert blocked.payload["idempotency_key"] == PRIVATE_EVENT_AUTHORITY
    assert observed["structured"] == blocked.payload["result"]["structured"]
    assert len(expected_reason.encode("utf-8")) <= _POLICY_DENIAL_TEXT_MAX_BYTES

    transcript = asyncio.run(store.load_transcript(blocked.session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_result.content == expected_reason
    assert tool_result.structured == blocked.payload["result"]["structured"]


def test_command_policy_redacts_proxy_resolved_secret_before_bounding_denial() -> None:
    from cayu.vaults.redaction import REDACTED_SECRET

    secret_value = "BOUNDARY_DYNAMIC_PROXY_SECRET"
    raw_prefix = "a" * 4050

    class ProxyResolvingDenyPolicy(CommandPolicy):
        async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
            del request
            assert ctx.proxy is not None
            resolved = await ctx.proxy.resolve(
                SecretRef(name="policy_secret"),
                scope={"session_id": ctx.session_id},
            )
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason=raw_prefix + resolved.value.get_secret_value(),
            )

    store = InMemorySessionStore()
    vault = StaticVault({"policy_secret": secret_value})
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_proxy_policy",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="proxy-policy"),
            runner=ProvisionedTestRunner(),
            vault=vault,
            proxy=PassthroughProxy(vault),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=ProxyResolvingDenyPolicy())],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_proxy_policy_denial_redaction",
                messages=[Message.text("user", "push")],
            ),
        )
    )

    expected_reason = _bound_policy_denial_text(raw_prefix + REDACTED_SECRET)
    expected_content = _bound_policy_denial_text(
        "Command denied by policy. " + raw_prefix + REDACTED_SECRET
    )
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert blocked.payload["denied_by"] == "command_policy"
    assert blocked.payload["metadata"] == {}
    assert blocked.payload["reason"] == expected_reason
    assert blocked.payload["result"]["content"] == expected_content
    assert blocked.payload["result"]["structured"]["reason"] == expected_reason
    assert "BOUNDARY" not in str(blocked.payload)

    transcript = asyncio.run(store.load_transcript(blocked.session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_result.content == expected_content
    assert "BOUNDARY" not in str(tool_result.model_dump(mode="json"))
