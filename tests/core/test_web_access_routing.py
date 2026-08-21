from __future__ import annotations

import asyncio
import json
import multiprocessing
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SecretRef,
    SQLiteSessionStore,
    WebAccessCircuitPolicy,
    WebAccessEvidenceSource,
    WebAccessOutcome,
    WebAccessRouteAction,
    WebAccessRoutePolicy,
    WebAccessRouteRule,
    WebAccessSignal,
    WebBridge,
    WebBridgeCredentialAuthority,
    WebBridgeRoute,
    WebFetchTool,
)
from cayu.core.events import EventType
from cayu.core.tools import (
    ToolContext,
    ToolResult,
    _bind_runtime_tool_invocation_authority,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.proxies import CredentialProxy, ProxyAuthorizationResult
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentityStrength,
    execution_profile_from_session_metadata,
)
from cayu.runtime.hooks import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.runtime.sessions import EventQuery
from cayu.tools import WebFetchAdapterRequest
from cayu.tools.web import HttpxWebFetchTransport, SystemWebFetchResolver, WebFetchHttpResponse
from cayu.tools.web_access import classify_http_access, web_destination_fingerprint
from cayu.vaults import ResolvedSecret, SecretRedactor

_ROUTE_ENVIRONMENT_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="routed-web-access-test-environment",
    behavior_version="1",
    implementation_version="1",
)


class _FetchAdapter:
    def __init__(self, *results: ToolResult, secret_name: str = "fixture_api_key") -> None:
        self.results = list(results)
        self.calls = 0
        self.secret_name = secret_name

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        return WebBridgeCredentialAuthority(
            provider="fixture",
            origin="https://provider.example",
            secret_refs=(SecretRef(name=self.secret_name),),
        )

    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        del ctx, request
        self.calls += 1
        if not self.results:
            raise AssertionError("The route was invoked more often than configured.")
        return self.results.pop(0)


class _SharedLedgerFetchAdapter(_FetchAdapter):
    def __init__(
        self,
        ledger: dict[str, ToolResult],
        result: ToolResult,
        *,
        secret_name: str,
    ) -> None:
        super().__init__(secret_name=secret_name)
        self.ledger = ledger
        self.result = result
        self.keys: list[str] = []

    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        del request
        self.calls += 1
        key = ctx.idempotency_key
        assert type(key) is str
        self.keys.append(key)
        return self.ledger.setdefault(key, self.result)


class _CredentialResolvingFetchAdapter(_FetchAdapter):
    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        if ctx.proxy is None:
            raise AssertionError("The credentialed route requires an invocation proxy.")
        resolved = await ctx.proxy.resolve(
            SecretRef(name=self.secret_name),
            scope={"provider": "fixture"},
        )
        assert type(resolved) is ResolvedSecret
        return await super().fetch(ctx, request)


class _RaisingFetchAdapter(_FetchAdapter):
    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        del ctx, request
        self.calls += 1
        raise RuntimeError("private route failure diagnostic")


class _CrashAfterStagedWebTerminalStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        checkpoint = await super().load_checkpoint(session_id)
        pending_round = None if checkpoint is None else checkpoint.get("pending_tool_round")
        staged_terminals = (
            None if type(pending_round) is not dict else pending_round.get("staged_terminals")
        )
        if (
            not self.crashed
            and type(staged_terminals) is list
            and len(staged_terminals) >= 2
            and all(
                type(item) is dict and item.get("hooks_state") == "completed"
                for item in staged_terminals
            )
        ):
            self.crashed = True
            raise RuntimeError("simulated process loss after web terminal staging")
        return checkpoint


class _ProcessFetchAdapter:
    def __init__(self, calls_path: str) -> None:
        self.calls_path = calls_path

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        return WebBridgeCredentialAuthority(
            provider="process-fixture",
            origin="https://provider.example",
            secret_refs=(SecretRef(name="process_fixture_key"),),
        )

    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        del ctx, request
        path = Path(self.calls_path)
        count = 0 if not path.exists() else int(path.read_text(encoding="utf-8"))
        path.write_text(str(count + 1), encoding="utf-8")
        return _status(401)


class _RouteCredentialProxy(CredentialProxy):
    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        return type(authority) is WebBridgeCredentialAuthority

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del ref, scope
        raise AssertionError("Routing fixtures must not resolve credentials.")

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        del destination, credential, action, metadata
        raise AssertionError("Routing fixtures must not authorize requests.")


class _ResolvingRouteCredentialProxy(CredentialProxy):
    def __init__(self) -> None:
        self.resolutions = 0

    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        return type(authority) is WebBridgeCredentialAuthority

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del scope
        self.resolutions += 1
        return ResolvedSecret(
            name=ref.name,
            value=SecretStr("credential-resolved-after-primary-denial"),
        )

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        del destination, credential, action, metadata
        return ProxyAuthorizationResult(allowed=True)


def _run_routed_sqlite_process(
    database_path: str,
    calls_path: str,
    *,
    resume: bool,
) -> None:
    adapter = _ProcessFetchAdapter(calls_path)
    hosted = WebBridge.hosted(
        adapter=adapter,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="process-fixture-adapter",
            behavior_version="1",
            implementation_version="1",
        ),
    )
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", hosted),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="fetch",
                    name="web_fetch",
                    arguments={"url": "https://blocked.example/article"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = SQLiteSessionStore(database_path)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="routed",
                execution_profile_identity=_ROUTE_ENVIRONMENT_IDENTITY,
            ),
            proxy=_RouteCredentialProxy(),
        ),
        default=True,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="fixture"),
        tools=bridge.tools,
        execution_requirements=bridge.execution_requirements,
    )

    async def execute() -> None:
        stream = (
            app.resume(
                ResumeRequest(
                    session_id="routed-process-restart",
                    messages=[Message.text("user", "Try again after restart")],
                )
            )
            if resume
            else app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id="routed-process-restart",
                    messages=[Message.text("user", "Fetch once")],
                )
            )
        )
        async for _event in stream:
            pass

    asyncio.run(execute())


def _bridge(
    adapter: _FetchAdapter,
    *,
    identity_name: str = "fixture-web-adapter",
) -> WebBridge:
    return WebBridge.hosted(
        adapter=adapter,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name=identity_name,
            behavior_version="1",
            implementation_version="1",
        ),
    )


def _success(url: str = "https://replacement.example/article") -> ToolResult:
    return ToolResult(
        content="<untrusted_web_content>replacement</untrusted_web_content>",
        structured={
            "requested_url": url,
            "final_url": url,
            "title": "Replacement",
            "representation": "text",
            "content": "replacement",
            "redirects": [],
            "truncated": False,
            "truncation_reasons": [],
        },
    )


def _status(status_code: int, *, retry_after: int | None = None) -> ToolResult:
    structured: dict[str, Any] = {"error": "http_status", "status_code": status_code}
    if retry_after is not None:
        structured = {
            "error": "rate_limited",
            "status_code": status_code,
            "provider_metadata": {"fixture": {"retry_after_seconds": retry_after}},
        }
    return ToolResult(
        content="blocked page text is discarded", structured=structured, is_error=True
    )


def _access_denial(
    *,
    outcome: str,
    status_code: int,
    effective_source_url: str,
) -> ToolResult:
    return ToolResult(
        content="protected response content is discarded",
        structured={
            "error": "access_blocked",
            "access": {
                "schema_version": 1,
                "outcome": outcome,
                "source": "hosted_provider",
                "signal": "provider_status",
                "destination_fingerprint": web_destination_fingerprint(effective_source_url),
                "status_code": status_code,
                "retry_after_seconds": None,
                "retry_after_unrepresentable": False,
            },
            "effective_source_url": effective_source_url,
        },
        is_error=True,
    )


def _context(
    records: dict[str, dict[str, Any]],
    *,
    args: dict[str, Any],
    idempotency_key: str,
    load_error: BaseException | None = None,
    compare_error: BaseException | None = None,
    seal: Any | None = None,
    compare_failure_mode: str | None = None,
) -> ToolContext:
    ctx = ToolContext(
        session_id="route-parent",
        idempotency_key=idempotency_key,
        proxy=_RouteCredentialProxy(),
    )

    compare_calls = 0

    async def load(key: str) -> dict[str, Any] | None:
        if load_error is not None:
            raise load_error
        if compare_failure_mode == "commit_then_read_failure" and compare_calls >= 2:
            raise ConnectionError("private readback diagnostic")
        value = records.get(key)
        return None if value is None else json.loads(json.dumps(value))

    async def compare_and_set(
        key: str,
        expected: dict[str, Any] | None,
        desired: dict[str, Any],
        secondary: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any]:
        nonlocal compare_calls
        compare_calls += 1
        if compare_error is not None:
            raise compare_error
        assert secondary == {}
        assert records.get(key) == expected
        records[key] = json.loads(json.dumps(desired))
        if compare_failure_mode == "fail_after_first" and compare_calls >= 2:
            if expected is None:
                records.pop(key, None)
            else:
                records[key] = json.loads(json.dumps(expected))
            raise ConnectionError("private publication diagnostic")
        if compare_failure_mode == "commit_then_read_failure" and compare_calls >= 2:
            raise ConnectionError("private acknowledgement diagnostic")
        return json.loads(json.dumps(desired))

    _bind_runtime_tool_invocation_authority(
        ctx,
        parent_task_id="task-1",
        parent_run_epoch=1,
        model_step_id="step-1",
        model_attempt_id="attempt-1",
        tool_round_id="round-1",
        tool_call_id=idempotency_key,
        tool_name="web_fetch",
        idempotency_key=idempotency_key,
        effective_arguments=args,
        execution_profile_fingerprint="e" * 64,
        environment_allocation_fingerprint=None,
        load_durable_operation=load,
        compare_and_set_durable_operation=compare_and_set,
        seal_durable_output=(
            (lambda value: json.loads(json.dumps(value))) if seal is None else seal
        ),
        secret_publication_sealer=lambda: None,
    )
    return ctx


def _routed_runtime_app(
    *,
    store: InMemorySessionStore,
    provider: ScriptedModelProvider,
    bridge: WebBridge,
    secret_redactor: SecretRedactor | None = None,
    app_hooks: tuple[RuntimeHook, ...] = (),
    agent_hooks: tuple[RuntimeHook, ...] = (),
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        secret_redactor=secret_redactor,
        runtime_hooks=app_hooks,
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="routed",
                execution_profile_identity=_ROUTE_ENVIRONMENT_IDENTITY,
            ),
            proxy=_RouteCredentialProxy(),
        ),
        default=True,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="fixture"),
        tools=bridge.tools,
        runtime_hooks=agent_hooks,
        execution_requirements=bridge.execution_requirements,
    )
    return app


async def _run_routed_runtime_result(
    bridge: WebBridge,
    *,
    session_id: str,
    secret_redactor: SecretRedactor | None = None,
    app_hooks: tuple[RuntimeHook, ...] = (),
    agent_hooks: tuple[RuntimeHook, ...] = (),
) -> tuple[list[Any], list[Any], ToolResult]:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="fetch",
                    name="web_fetch",
                    arguments={"url": "https://blocked.example/article"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = _routed_runtime_app(
        store=store,
        provider=provider,
        bridge=bridge,
        secret_redactor=secret_redactor,
        app_hooks=app_hooks,
        agent_hooks=agent_hooks,
    )
    public_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="researcher",
                session_id=session_id,
                messages=[Message.text("user", "Fetch the article")],
            )
        )
    ]
    durable_events = await store.query_events(EventQuery(session_id=session_id))
    transcript = await store.load_transcript(session_id)
    result = next(message for message in transcript if message.role == "tool").content[0]
    return public_events, durable_events, result


@pytest.mark.parametrize(
    ("status", "headers", "outcome", "signal"),
    [
        (
            401,
            {"www-authenticate": 'Basic realm="fixture"'},
            "authentication_required",
            "www_authenticate",
        ),
        (401, {"cache-control": "no-store"}, "bot_challenge", "status_code"),
        (403, {"cf-mitigated": "challenge"}, "bot_challenge", "challenge_header"),
        (428, {"x-cayu-access-requirement": "consent"}, "consent_required", "consent_header"),
        (429, {"retry-after": "120"}, "rate_limited", "retry_after"),
        (403, {}, "destination_denied", "status_code"),
        (404, {}, "content_unavailable", "status_code"),
        (503, {}, "transient_transport_failure", "status_code"),
    ],
)
def test_http_access_classification_uses_bounded_transport_evidence(
    status: int,
    headers: dict[str, str],
    outcome: str,
    signal: str,
) -> None:
    evidence = classify_http_access(
        "https://blocked.example/protected/path?token=not-published",
        status_code=status,
        headers=headers,
        source=WebAccessEvidenceSource.HTTP_RESPONSE,
    )

    assert evidence is not None
    assert evidence.outcome.value == outcome
    assert evidence.signal.value == signal
    assert len(evidence.destination_fingerprint) == 64
    assert "protected" not in evidence.model_dump_json()
    assert "token" not in evidence.model_dump_json()


def test_oversized_retry_after_delta_is_authoritative_but_unrepresentable() -> None:
    evidence = classify_http_access(
        "https://limited.example/",
        status_code=429,
        headers={"retry-after": "9" * 129},
        source=WebAccessEvidenceSource.HTTP_RESPONSE,
    )

    assert evidence is not None
    assert evidence.signal is WebAccessSignal.RETRY_AFTER
    assert evidence.retry_after_seconds is None
    assert evidence.retry_after_unrepresentable is True


@pytest.mark.parametrize(
    ("value", "seconds", "unrepresentable"),
    [
        ("000001", 1, False),
        ("086400", 86_400, False),
        ("086401", None, True),
        ("0" * 129, None, True),
    ],
)
def test_retry_after_delta_accepts_leading_zeroes_without_weakening_bounds(
    value: str,
    seconds: int | None,
    unrepresentable: bool,
) -> None:
    evidence = classify_http_access(
        "https://limited.example/",
        status_code=429,
        headers={"retry-after": value},
        source=WebAccessEvidenceSource.HTTP_RESPONSE,
    )

    assert evidence is not None
    assert evidence.retry_after_seconds == seconds
    assert evidence.retry_after_unrepresentable is unrepresentable


@pytest.mark.parametrize(
    ("result", "outcome", "signal", "unrepresentable"),
    [
        (_status(407), "authentication_required", "provider_status", False),
        (
            ToolResult(
                content="provider timeout",
                structured={"error": "timeout"},
                is_error=True,
            ),
            "transient_transport_failure",
            "provider_status",
            False,
        ),
        (
            ToolResult(
                content="rate limited",
                structured={
                    "error": "rate_limited",
                    "status_code": 429,
                    "provider_metadata": {"fixture": {"retry_after_seconds": 86_401.25}},
                },
                is_error=True,
            ),
            "rate_limited",
            "retry_after",
            True,
        ),
        (
            ToolResult(
                content="conflicting retry authority",
                structured={
                    "error": "rate_limited",
                    "status_code": 429,
                    "provider_metadata": {
                        "bounded": {"retry_after_seconds": 60},
                        "unbounded": {"retry_after_unrepresentable": True},
                    },
                },
                is_error=True,
            ),
            "rate_limited",
            "retry_after",
            True,
        ),
        (
            ToolResult(
                content="rate limited",
                structured={
                    "error": "rate_limited",
                    "status_code": 429,
                    "provider_metadata": {"fixture": {"retry_after_seconds": 2**63 - 1}},
                },
                is_error=True,
            ),
            "rate_limited",
            "retry_after",
            True,
        ),
    ],
)
def test_hosted_fetch_normalizes_provider_access_evidence(
    result: ToolResult,
    outcome: str,
    signal: str,
    unrepresentable: bool,
) -> None:
    projected = asyncio.run(
        WebFetchTool(adapter=_FetchAdapter(result)).run(
            ToolContext(session_id="hosted-evidence", idempotency_key="hosted-evidence"),
            {"url": "https://entry.example/article"},
        )
    )

    assert projected.structured["access"]["outcome"] == outcome
    assert projected.structured["access"]["signal"] == signal
    assert projected.structured["access"]["retry_after_unrepresentable"] is unrepresentable


def test_hostile_success_page_text_cannot_forge_an_access_block() -> None:
    assert (
        classify_http_access(
            "https://hostile.example/",
            status_code=200,
            headers={"content-type": "text/html"},
            source=WebAccessEvidenceSource.BROWSER_RESPONSE,
        )
        is None
    )


@pytest.mark.parametrize(
    "structured",
    [
        {
            "error": "access_blocked",
            "access": {
                "schema_version": 1,
                "outcome": "bot_challenge",
                "source": "browser_response",
                "signal": "status_code",
                "destination_fingerprint": "a" * 64,
                "status_code": 401,
                "retry_after_seconds": None,
                "retry_after_unrepresentable": False,
            },
            "protected_url": "https://private.example/path?token=SECRET_CANARY",
        },
        {
            "error": "rate_limited",
            "status_code": 500,
            "provider_metadata": {
                "fixture": {
                    "retry_after_seconds": 60,
                    "challenge": "SECRET_CANARY",
                }
            },
        },
    ],
)
def test_invalid_adapter_access_claim_is_replaced_with_a_fixed_diagnostic(
    structured: dict[str, Any],
) -> None:
    adapter = _FetchAdapter(
        ToolResult(content="CAPTCHA_SECRET_CANARY", structured=structured, is_error=True)
    )
    tool = WebFetchTool(adapter=adapter)

    result = asyncio.run(
        tool.run(
            ToolContext(session_id="malformed-access", idempotency_key="malformed-access"),
            {"url": "https://actual.example/article"},
        )
    )

    rendered = result.model_dump_json()
    assert result.structured == {"error": "malformed_access_evidence"}
    assert "SECRET_CANARY" not in rendered
    assert "private.example" not in rendered


def test_hosted_raw_destination_denial_cannot_claim_egress_policy_authority() -> None:
    adapter = _FetchAdapter(
        ToolResult(
            content="hosted denial",
            structured={"error": "destination_denied"},
            is_error=True,
        )
    )

    result = asyncio.run(
        WebFetchTool(adapter=adapter).run(
            ToolContext(session_id="hosted-denial", idempotency_key="hosted-denial"),
            {"url": "https://entry.example/article"},
        )
    )

    assert result.structured == {"error": "malformed_access_evidence"}


def test_typed_access_evidence_must_match_the_declared_effective_origin() -> None:
    result = asyncio.run(
        WebFetchTool(
            adapter=_FetchAdapter(
                ToolResult(
                    content="conflicting access evidence",
                    structured={
                        "error": "access_blocked",
                        "access": {
                            "schema_version": 1,
                            "outcome": "bot_challenge",
                            "source": "hosted_provider",
                            "signal": "provider_status",
                            "destination_fingerprint": web_destination_fingerprint(
                                "https://entry.example/article"
                            ),
                            "status_code": 401,
                            "retry_after_seconds": None,
                            "retry_after_unrepresentable": False,
                        },
                        "effective_source_url": "https://challenge.example/protected",
                    },
                    is_error=True,
                )
            )
        ).run(
            ToolContext(session_id="origin-conflict", idempotency_key="origin-conflict"),
            {"url": "https://entry.example/article"},
        )
    )

    assert result.structured == {"error": "malformed_access_evidence"}


def test_anubis_style_bare_401_uses_one_explicit_fallback_with_attribution() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/web_access/anubis_challenge.json").read_text()
    )
    access = classify_http_access(
        "https://blocked.example/protected",
        status_code=fixture["status_code"],
        headers=fixture["headers"],
        source=WebAccessEvidenceSource.HTTP_RESPONSE,
    )
    assert access is not None
    assert access.outcome is WebAccessOutcome.BOT_CHALLENGE
    assert access.signal is WebAccessSignal.STATUS_CODE
    primary = _FetchAdapter(_status(fixture["status_code"]))
    replacement = _FetchAdapter(_success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("browser", _bridge(primary)),
            WebBridgeRoute("hosted", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="browser",
            rules=(
                WebAccessRouteRule(
                    "browser",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("hosted"),
                ),
            ),
        ),
    )
    args = {"url": "https://blocked.example/protected"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="fallback-1"),
            args,
        )
    )

    assert result.is_error is False
    route = result.structured["webbridge_route"]
    assert route["terminal_disposition"] == "fallback_succeeded"
    assert route["original_access"]["outcome"] == "bot_challenge"
    assert route["selected_route"]["route_id"] == "hosted"
    assert route["effective_source_url"] == "https://replacement.example/article"
    assert route["execution_profile_fingerprint"] == "e" * 64
    assert primary.calls == replacement.calls == 1


def test_routed_bridge_uses_normal_runtime_durable_tool_authority() -> None:
    primary = _FetchAdapter(_status(401))
    replacement = _FetchAdapter(_success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("browser", _bridge(primary)),
            WebBridgeRoute("hosted", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="browser",
            rules=(
                WebAccessRouteRule(
                    "browser",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("hosted"),
                ),
            ),
        ),
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="fetch",
                    name="web_fetch",
                    arguments={"url": "https://blocked.example/article"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="routed",
                execution_profile_identity=_ROUTE_ENVIRONMENT_IDENTITY,
            ),
            proxy=_RouteCredentialProxy(),
        ),
        default=True,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="fixture"),
        tools=bridge.tools,
        execution_requirements=bridge.execution_requirements,
    )

    async def scenario() -> ToolResult:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id="routed-runtime",
                    messages=[Message.text("user", "Fetch the article")],
                )
            )
        ]
        assert events[-1].type.value == "session.completed"
        session = await store.load("routed-runtime")
        assert session is not None
        component = execution_profile_from_session_metadata(session.metadata).component(
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS
        )
        assert component.strength is not ExecutionProfileIdentityStrength.PROCESS_LOCAL
        transcript = await store.load_transcript("routed-runtime")
        return transcript[2].content[0]

    result = asyncio.run(scenario())

    assert result.structured["webbridge_route"]["terminal_disposition"] == ("fallback_succeeded")
    assert primary.calls == replacement.calls == 1


@pytest.mark.parametrize(
    "secret",
    ["primary", "hosted_provider", "bot_challenge", "fallback_succeeded"],
)
def test_routed_runtime_preserves_attested_controls_under_short_secret_collision(
    secret: str,
) -> None:
    primary = _FetchAdapter(_status(401))
    replacement = _FetchAdapter(_success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="primary-adapter")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="replacement-adapter"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
        ),
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="fetch",
                    name="web_fetch",
                    arguments={"url": "https://blocked.example/article"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="routed",
                execution_profile_identity=_ROUTE_ENVIRONMENT_IDENTITY,
            ),
            proxy=_RouteCredentialProxy(),
        ),
        default=True,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="fixture"),
        tools=bridge.tools,
        execution_requirements=bridge.execution_requirements,
    )

    async def scenario() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        session_id = "routed-control-collision"
        public_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id=session_id,
                    messages=[Message.text("user", "Fetch the article")],
                )
            )
        ]
        records = await store.query_events(EventQuery(session_id=session_id))
        durable = next(
            record.event.payload["result"]["structured"]["webbridge_route"]
            for record in records
            if record.event.type.value == "tool.call.completed"
        )
        public = next(
            event.payload["result"]["structured"]["webbridge_route"]
            for event in public_events
            if event.type.value == "tool.call.completed"
        )
        transcript = await store.load_transcript(session_id)
        tool_part = next(message for message in transcript if message.role == "tool").content[0]
        return durable, public, tool_part.structured["webbridge_route"]

    durable, public, transcript = asyncio.run(scenario())

    for route in (durable, public):
        assert route["selected_route"]["route_id"] == "replacement"
        assert route["selected_route"]["kind"] == "hosted_provider"
        assert route["terminal_disposition"] == "fallback_succeeded"
        assert route["original_access"]["outcome"] == "bot_challenge"
    assert transcript["selected_route"]["kind"] == "hosted_provider"
    assert transcript["terminal_disposition"] == "fallback_succeeded"
    assert transcript["original_access"]["outcome"] == "bot_challenge"
    assert transcript["history"][0]["route"]["route_id"] == "primary"
    assert primary.calls == replacement.calls == 1


def test_hook_short_circuit_cannot_attest_synthetic_web_access_evidence(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "hook-authored-web-access-secret-canary"

    class SyntheticAccessHook(RuntimeHook):
        async def before_tool_call(
            self,
            context: BeforeToolCallHookContext,
        ) -> BeforeToolCallDecision:
            del context
            return BeforeToolCallDecision(
                action="short_circuit",
                synthetic_result=ToolResult(
                    content="synthetic",
                    structured={
                        "access": {
                            "schema_version": 1,
                            "outcome": "bot_challenge",
                            "source": "hosted_provider",
                            "signal": "provider_status",
                            "destination_fingerprint": secret,
                        }
                    },
                ),
            )

    adapter = _FetchAdapter(_success())
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        public_events, durable_events, result = asyncio.run(
            _run_routed_runtime_result(
                WebBridge.routed(
                    routes=(WebBridgeRoute("hosted", _bridge(adapter)),),
                    policy=WebAccessRoutePolicy(entry_route_id="hosted"),
                ),
                session_id="synthetic-web-access-authority",
                secret_redactor=SecretRedactor(secret),
                app_hooks=(SyntheticAccessHook(),),
            )
        )

    captured_io = capsys.readouterr()
    rendered = repr(
        (
            public_events,
            durable_events,
            result,
            captured_warnings,
            caplog.records,
            captured_io.out,
            captured_io.err,
        )
    )
    assert secret not in rendered
    assert adapter.calls == 0


def test_restart_recovery_restores_exact_staged_web_access_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session_id = "staged-web-access-restart"
        store = _CrashAfterStagedWebTerminalStore()
        transport_calls: dict[str, int] = {}

        async def resolve(
            _resolver: SystemWebFetchResolver,
            _hostname: str,
            _port: int,
        ) -> tuple[str, ...]:
            return ("93.184.216.34",)

        async def fetch(
            _transport: HttpxWebFetchTransport,
            _request: Any,
        ) -> WebFetchHttpResponse:
            call_count = transport_calls.get(_request.url, 0) + 1
            transport_calls[_request.url] = call_count
            if call_count == 1:
                return WebFetchHttpResponse(status_code=401, headers={}, body=b"")
            return WebFetchHttpResponse(
                status_code=200,
                headers={"content-type": "text/html; charset=utf-8"},
                body=b"<html><body>replacement</body></html>",
            )

        monkeypatch.setattr(SystemWebFetchResolver, "resolve", resolve)
        monkeypatch.setattr(HttpxWebFetchTransport, "fetch", fetch)

        def bridge() -> WebBridge:
            return WebBridge.routed(
                routes=(
                    WebBridgeRoute("primary", WebBridge.trusted_local()),
                    WebBridgeRoute("replacement", WebBridge.trusted_local()),
                ),
                policy=WebAccessRoutePolicy(
                    entry_route_id="primary",
                    rules=(
                        WebAccessRouteRule(
                            "primary",
                            WebAccessOutcome.BOT_CHALLENGE,
                            WebAccessRouteAction.fallback_to("replacement"),
                        ),
                    ),
                ),
            )

        initial_app = _routed_runtime_app(
            store=store,
            provider=ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="fetch-one",
                            name="web_fetch",
                            arguments={"url": "https://blocked.example/one"},
                        ),
                        ModelStreamEvent.tool_call(
                            id="fetch-two",
                            name="web_fetch",
                            arguments={"url": "https://blocked.example/two"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            ),
            bridge=bridge(),
            secret_redactor=SecretRedactor(),
        )
        initial_events = [
            event
            async for event in initial_app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id=session_id,
                    messages=[Message.text("user", "Fetch the article")],
                )
            )
        ]

        checkpoint = await store.load_checkpoint(session_id)
        assert store.crashed is True
        assert checkpoint is not None
        assert checkpoint.get("pending_tool_round") is not None
        staged = checkpoint["pending_tool_round"]["staged_terminals"]
        assert staged, checkpoint["pending_tool_round"]
        assert all(item["hooks_state"] == "completed" for item in staged)
        assert all(
            item["event"]["payload"]["result"]["structured"]["webbridge_route"][
                "terminal_disposition"
            ]
            == "fallback_succeeded"
            for item in staged
        )
        unattested_checkpoint = json.loads(json.dumps(checkpoint))
        unattested_checkpoint["pending_tool_round"]["staged_terminals"][0]["event"]["payload"].pop(
            "web_access_result_authority"
        )
        from cayu.runtime import _tool_round_recovery as tool_round_recovery

        with pytest.raises(ValueError, match="workload secret"):
            tool_round_recovery.pending_tool_round_from_checkpoint(
                unattested_checkpoint,
                redactor=SecretRedactor("fallback_succeeded"),
            )
        malformed_checkpoint = json.loads(json.dumps(checkpoint))
        malformed_checkpoint["pending_tool_round"]["staged_terminals"][0]["event"]["payload"][
            "result"
        ]["structured"]["webbridge_route"]["selected_route"]["route_id"] = "checkpoint-route-secret"
        with pytest.raises(ValueError, match="workload secret"):
            tool_round_recovery.pending_tool_round_from_checkpoint(
                malformed_checkpoint,
                redactor=SecretRedactor("checkpoint-route-secret"),
            )
        calls_after_crash = sum(transport_calls.values())
        assert calls_after_crash >= 2
        assert initial_events[-1].type is EventType.SESSION_FAILED

        recovery_app = _routed_runtime_app(
            store=store,
            provider=ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            bridge=bridge(),
            secret_redactor=SecretRedactor("fallback_succeeded"),
        )
        resumed_events = [
            event
            async for event in recovery_app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]

        assert sum(transport_calls.values()) == calls_after_crash
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        records = await store.query_events(EventQuery(session_id=session_id))
        routed_terminals = [
            record.event
            for record in records
            if record.event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
            }
            and "webbridge_route" in (record.event.payload["result"].get("structured") or {})
        ]
        assert len(routed_terminals) == len(staged), ",".join(
            record.event.type.value for record in records
        )
        for terminal in routed_terminals:
            route = terminal.payload["result"]["structured"]["webbridge_route"]
            assert route["terminal_disposition"] == "fallback_succeeded"
            assert route["effective_source_url"] in {
                "https://blocked.example/one",
                "https://blocked.example/two",
            }

    asyncio.run(scenario())


def test_after_hooks_preserve_only_genuine_runtime_routing_controls() -> None:
    class AppRewriteHook(RuntimeHook):
        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision:
            assert context.result.structured["webbridge_route"]["terminal_disposition"] == (
                "fallback_succeeded"
            )
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="app-rewritten"),
            )

    class AgentConflictHook(RuntimeHook):
        def __init__(self) -> None:
            self.observed_route = False
            self.observed_source = False

        async def after_tool_call(
            self,
            context: ToolCallHookContext,
        ) -> AfterToolCallDecision:
            self.observed_route = "webbridge_route" in (context.result.structured or {})
            self.observed_source = (
                context.result.structured["webbridge_route"]["effective_source_url"]
                == "https://replacement.example/article"
            )
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(
                    content="agent-rewritten",
                    structured={
                        "webbridge_route": {
                            "terminal_disposition": "stopped",
                            "selected_route": {"route_id": "forged"},
                            "effective_source_url": "https://forged.example/article",
                        }
                    },
                ),
            )

    primary = _FetchAdapter(_status(401))
    replacement = _FetchAdapter(_success())
    agent_hook = AgentConflictHook()
    public_events, durable_events, result = asyncio.run(
        _run_routed_runtime_result(
            WebBridge.routed(
                routes=(
                    WebBridgeRoute("primary", _bridge(primary, identity_name="hook-primary")),
                    WebBridgeRoute(
                        "replacement",
                        _bridge(replacement, identity_name="hook-replacement"),
                    ),
                ),
                policy=WebAccessRoutePolicy(
                    entry_route_id="primary",
                    rules=(
                        WebAccessRouteRule(
                            "primary",
                            WebAccessOutcome.BOT_CHALLENGE,
                            WebAccessRouteAction.fallback_to("replacement"),
                        ),
                    ),
                ),
            ),
            session_id="after-hook-routing-authority",
            secret_redactor=SecretRedactor("fallback_succeeded"),
            app_hooks=(AppRewriteHook(),),
            agent_hooks=(agent_hook,),
        )
    )

    route = result.structured["webbridge_route"]
    assert result.content == "agent-rewritten"
    assert route["terminal_disposition"] == "fallback_succeeded"
    assert route["selected_route"]["route_id"] == "replacement"
    assert route["original_access"]["outcome"] == "bot_challenge"
    assert route["effective_source_url"] == "https://replacement.example/article"
    assert agent_hook.observed_route is True
    assert agent_hook.observed_source is True
    assert "forged" not in repr((public_events, durable_events, result))


def test_fallback_adapter_exception_is_owned_by_the_runtime_routing_terminal() -> None:
    primary = _FetchAdapter(_status(401))
    replacement = _RaisingFetchAdapter()
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="raising-primary")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="raising-replacement"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
        ),
    )

    public_events, durable_events, result = asyncio.run(
        _run_routed_runtime_result(
            bridge,
            session_id="fallback-adapter-exception",
        )
    )

    route = result.structured["webbridge_route"]
    assert result.structured["error"] == "route_failed"
    assert route["terminal_disposition"] == "route_failed"
    assert route["selected_route"]["route_id"] == "replacement"
    assert route["original_access"]["outcome"] == "bot_challenge"
    assert route["history"][0]["access"]["outcome"] == "bot_challenge"
    assert route["history"][1]["disposition"] == "route_failed"
    assert route["effective_source_url"] == "https://blocked.example/"
    assert "private route failure diagnostic" not in repr((public_events, durable_events, result))
    assert primary.calls == replacement.calls == 1


def test_intermediate_circuit_publication_does_not_seal_credentialed_fallback() -> None:
    primary = _FetchAdapter(_status(401), secret_name="primary_key")
    replacement = _CredentialResolvingFetchAdapter(
        _success(),
        secret_name="replacement_key",
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="primary-adapter")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="replacement-adapter"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="fetch",
                    name="web_fetch",
                    arguments={"url": "https://blocked.example/article"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    proxy = _ResolvingRouteCredentialProxy()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="routed",
                execution_profile_identity=_ROUTE_ENVIRONMENT_IDENTITY,
            ),
            proxy=proxy,
        ),
        default=True,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="researcher", model="fixture"),
        tools=bridge.tools,
        execution_requirements=bridge.execution_requirements,
    )

    async def scenario() -> ToolResult:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id="credentialed-routed-runtime",
                    messages=[Message.text("user", "Fetch the article")],
                )
            )
        ]
        assert events[-1].type.value == "session.completed"
        transcript = await store.load_transcript("credentialed-routed-runtime")
        return transcript[2].content[0]

    result = asyncio.run(scenario())

    assert result.is_error is False
    assert result.structured["webbridge_route"]["terminal_disposition"] == ("fallback_succeeded")
    assert primary.calls == replacement.calls == 1
    assert proxy.resolutions == 1


def test_intermediate_circuit_preserves_other_destination_origins() -> None:
    first_url = "https://first.example/article"
    second_url = "https://second.example/article"
    first_challenge_origin = "https://first-challenge.example/protected"
    first_login_origin = "https://first-login.example/private"
    primary = _FetchAdapter(
        _access_denial(
            outcome="bot_challenge",
            status_code=403,
            effective_source_url=first_challenge_origin,
        ),
        _access_denial(
            outcome="bot_challenge",
            status_code=403,
            effective_source_url="https://second-challenge.example/protected",
        ),
    )
    replacement = _FetchAdapter(
        _access_denial(
            outcome="authentication_required",
            status_code=401,
            effective_source_url=first_login_origin,
        ),
        _success(second_url),
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="primary-adapter")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="replacement-adapter"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(
            _context(records, args={"url": first_url}, idempotency_key="first-denial"),
            {"url": first_url},
        )
    )
    second = asyncio.run(
        bridge.tools[0].run(
            _context(records, args={"url": second_url}, idempotency_key="second-success"),
            {"url": second_url},
        )
    )
    replay = asyncio.run(
        bridge.tools[0].run(
            _context(records, args={"url": first_url}, idempotency_key="first-replay"),
            {"url": first_url},
        )
    )

    assert first.structured["webbridge_route"]["effective_source_url"] == (
        "https://first-login.example/"
    )
    assert second.structured["webbridge_route"]["terminal_disposition"] == ("fallback_succeeded")
    assert replay.structured["webbridge_route"]["effective_source_url"] == (
        "https://first-login.example/"
    )
    assert [item["invoked"] for item in replay.structured["webbridge_route"]["history"]] == [
        False,
        False,
    ]
    assert primary.calls == replacement.calls == 2


def test_failed_fallback_and_circuit_replay_retain_only_the_effective_origin() -> None:
    primary = _FetchAdapter(
        _access_denial(
            outcome="bot_challenge",
            status_code=403,
            effective_source_url="https://challenge.example/protected?token=primary-secret",
        )
    )
    replacement = _FetchAdapter(
        _access_denial(
            outcome="authentication_required",
            status_code=401,
            effective_source_url="https://login.example/private?token=fallback-secret",
        )
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="primary-adapter")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="replacement-adapter"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    args = {"url": "https://entry.example/article"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="denied-1"), args)
    )
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="denied-2"), args)
    )

    assert first.is_error is second.is_error is True
    assert primary.calls == replacement.calls == 1
    for result in (first, second):
        route = result.structured["webbridge_route"]
        assert route["terminal_disposition"] == "stopped"
        assert route["selected_route"]["route_id"] == "replacement"
        assert route["original_access"]["outcome"] == "bot_challenge"
        assert route["effective_source_url"] == "https://login.example/"
        assert "token=" not in repr(route)
    assert [item["invoked"] for item in second.structured["webbridge_route"]["history"]] == [
        False,
        False,
    ]


def test_open_circuit_survives_a_fresh_process_sqlite_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "sessions.db"
    calls_path = tmp_path / "provider-calls.txt"
    context = multiprocessing.get_context("spawn")

    first = context.Process(
        target=_run_routed_sqlite_process,
        args=(str(database_path), str(calls_path)),
        kwargs={"resume": False},
    )
    first.start()
    first.join(timeout=20)
    assert first.exitcode == 0
    assert calls_path.read_text(encoding="utf-8") == "1"

    recovered = context.Process(
        target=_run_routed_sqlite_process,
        args=(str(database_path), str(calls_path)),
        kwargs={"resume": True},
    )
    recovered.start()
    recovered.join(timeout=20)
    assert recovered.exitcode == 0
    assert calls_path.read_text(encoding="utf-8") == "1"


def test_no_configured_fallback_stops_without_another_route() -> None:
    primary = _FetchAdapter(_status(401))
    unused = _FetchAdapter(_success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("browser", _bridge(primary)),
            WebBridgeRoute("hosted", _bridge(unused)),
        ),
        policy=WebAccessRoutePolicy(entry_route_id="browser"),
    )
    args = {"url": "https://blocked.example/private/path"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="stop-1"),
            args,
        )
    )

    assert result.is_error is True
    route = result.structured["webbridge_route"]
    assert route["terminal_disposition"] == "stopped"
    assert "private/path" not in result.model_dump_json()
    assert primary.calls == 1
    assert unused.calls == 0


def test_trusted_local_route_uses_http_response_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve(
        _resolver: SystemWebFetchResolver,
        _hostname: str,
        _port: int,
    ) -> tuple[str, ...]:
        return ("93.184.216.34",)

    async def fetch(
        _transport: HttpxWebFetchTransport,
        _request: Any,
    ) -> WebFetchHttpResponse:
        return WebFetchHttpResponse(status_code=401, headers={}, body=b"")

    monkeypatch.setattr(SystemWebFetchResolver, "resolve", resolve)
    monkeypatch.setattr(HttpxWebFetchTransport, "fetch", fetch)
    local = WebBridge.trusted_local()
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("local", local),),
        policy=WebAccessRoutePolicy(entry_route_id="local"),
    )
    args = {"url": "https://blocked.example/"}

    result = asyncio.run(
        bridge.tools[0].run(_context({}, args=args, idempotency_key="local-source"), args)
    )

    access = result.structured["webbridge_route"]["original_access"]
    assert access["source"] == "http_response"
    assert access["outcome"] == "bot_challenge"


def test_trusted_local_redirect_attributes_access_to_the_effective_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve(
        _resolver: SystemWebFetchResolver,
        _hostname: str,
        _port: int,
    ) -> tuple[str, ...]:
        return ("93.184.216.34",)

    async def fetch(
        _transport: HttpxWebFetchTransport,
        request: Any,
    ) -> WebFetchHttpResponse:
        if request.url == "https://entry.example/":
            return WebFetchHttpResponse(
                status_code=302,
                headers={"location": "https://challenge.example/"},
                body=b"",
            )
        assert request.url == "https://challenge.example/"
        return WebFetchHttpResponse(status_code=401, headers={}, body=b"")

    monkeypatch.setattr(SystemWebFetchResolver, "resolve", resolve)
    monkeypatch.setattr(HttpxWebFetchTransport, "fetch", fetch)
    replacement = _FetchAdapter(_success("https://entry.example/"))
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("local", WebBridge.trusted_local()),
            WebBridgeRoute("hosted", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="local",
            rules=(
                WebAccessRouteRule(
                    "local",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("hosted"),
                ),
            ),
        ),
    )
    args = {"url": "https://entry.example/"}

    result = asyncio.run(
        bridge.tools[0].run(_context({}, args=args, idempotency_key="redirect-hop"), args)
    )

    route = result.structured["webbridge_route"]
    assert route["terminal_disposition"] == "fallback_succeeded"
    assert route["original_access"]["destination_fingerprint"] == (
        web_destination_fingerprint("https://challenge.example/")
    )
    assert route["history"][0]["invoked"] is True
    assert replacement.calls == 1


def test_blocked_fallback_authority_does_not_expand_or_retry() -> None:
    primary = _FetchAdapter(_status(403))
    blocked = _FetchAdapter(
        ToolResult(
            content="The active runner did not match the configured route.",
            structured={"error": "capability_refused"},
            is_error=True,
        )
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("direct", _bridge(primary)),
            WebBridgeRoute("sandbox", _bridge(blocked)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="direct",
            rules=(
                WebAccessRouteRule(
                    "direct",
                    WebAccessOutcome.DESTINATION_DENIED,
                    WebAccessRouteAction.fallback_to("sandbox"),
                ),
            ),
        ),
    )
    args = {"url": "https://blocked.example/"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="blocked-authority"),
            args,
        )
    )

    assert result.is_error is True
    assert result.structured["error"] == "capability_refused"
    assert result.structured["webbridge_route"]["terminal_disposition"] == "route_failed"
    assert result.structured["webbridge_route"]["effective_source_url"] == (
        "https://blocked.example/"
    )
    assert primary.calls == blocked.calls == 1


def test_durable_circuit_skips_denied_route_until_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter((1_000, 1_000, 1_000, 1_001, 1_001, 1_301, 1_301, 1_301))
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: next(times))
    primary = _FetchAdapter(_status(401), _status(401))
    replacement = _FetchAdapter(_success(), _success(), _success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("browser", _bridge(primary)),
            WebBridgeRoute("hosted", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="browser",
            rules=(
                WebAccessRouteRule(
                    "browser",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("hosted"),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    args = {"url": "https://blocked.example/article"}
    records: dict[str, dict[str, Any]] = {}

    results = [
        asyncio.run(
            bridge.tools[0].run(
                _context(records, args=args, idempotency_key=f"circuit-{index}"),
                args,
            )
        )
        for index in range(3)
    ]

    assert [item.is_error for item in results] == [False, False, False]
    assert primary.calls == 2
    assert replacement.calls == 3
    assert results[1].structured["webbridge_route"]["original_access"]["source"] == (
        "hosted_provider"
    )
    assert results[1].structured["webbridge_route"]["history"][0]["invoked"] is False
    assert any(key.startswith("web_access_circuit:") for key in records)


def test_redirected_denial_circuit_is_keyed_by_the_requested_route_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 1_000)
    blocked_origin = "https://challenge.example/"
    primary = _FetchAdapter(
        ToolResult(
            content="challenge content must not survive",
            structured={
                "error": "access_blocked",
                "access": {
                    "schema_version": 1,
                    "outcome": "bot_challenge",
                    "source": "hosted_provider",
                    "signal": "provider_status",
                    "destination_fingerprint": web_destination_fingerprint(blocked_origin),
                    "status_code": 401,
                    "retry_after_seconds": None,
                    "retry_after_unrepresentable": False,
                },
                "effective_source_url": blocked_origin,
            },
            is_error=True,
        )
    )
    replacement = _FetchAdapter(_success(), _success())
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary)),
            WebBridgeRoute("replacement", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    args = {"url": "https://entry.example/protected"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="redirect-1"), args)
    )
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="redirect-2"), args)
    )

    assert first.is_error is second.is_error is False
    assert primary.calls == 1
    assert replacement.calls == 2
    assert second.structured["webbridge_route"]["history"][0]["invoked"] is False
    assert "challenge content" not in first.model_dump_json()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry.__setitem__("fingerprint", "f" * 64),
        lambda entry: entry.__setitem__("route_id", "unknown"),
        lambda entry: entry.__setitem__("source", "transport"),
        lambda entry: entry.__setitem__("updated_at", 10**100),
        lambda entry: entry.__setitem__("denial_count", True),
        lambda entry: entry.__setitem__("next_eligible_at", entry["updated_at"]),
    ],
)
def test_durable_circuit_rejects_conflicting_authority_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 1_000)
    primary = _FetchAdapter(_status(401), _success())
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("primary", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    args = {"url": "https://entry.example/"}
    records: dict[str, dict[str, Any]] = {}
    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="valid"), args)
    )
    assert first.is_error is True
    record = next(iter(records.values()))
    mutation(record["entries"][0])

    result = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="tampered"), args)
    )

    assert result.structured["error"] == "durable_authority_unavailable"
    assert primary.calls == 1


@pytest.mark.parametrize(
    "secret",
    [
        "cayu.web-access-circuit",
        "primary",
        "bot_challenge",
        "hosted_provider",
        "entry.example",
    ],
)
def test_circuit_sealing_preserves_authenticated_controls_on_short_secret_collision(
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 1_000)
    redactor = SecretRedactor(secret)
    primary = _FetchAdapter(_status(401))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("primary", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=300),
        ),
    )
    args = {"url": "https://entry.example/"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(
            _context(
                records,
                args=args,
                idempotency_key="sealed-first",
                seal=redactor.redact_json_values,
            ),
            args,
        )
    )
    second = asyncio.run(
        bridge.tools[0].run(
            _context(
                records,
                args=args,
                idempotency_key="sealed-second",
                seal=redactor.redact_json_values,
            ),
            args,
        )
    )

    assert first.is_error is second.is_error is True
    assert first.structured["error"] != "durable_authority_unavailable"
    assert second.structured["error"] != "durable_authority_unavailable"
    entry = next(iter(records.values()))["entries"][0]
    assert entry["route_id"] == "primary"
    assert entry["outcome"] == "bot_challenge"
    assert entry["source"] == "hosted_provider"
    assert primary.calls == 1


def test_durable_circuit_store_failure_stops_before_route_dispatch() -> None:
    primary = _FetchAdapter(_success())
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("direct", _bridge(primary)),),
        policy=WebAccessRoutePolicy(entry_route_id="direct"),
    )
    args = {"url": "https://example.com/article"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context(
                {},
                args=args,
                idempotency_key="store-failure",
                load_error=ConnectionError("private store diagnostic"),
            ),
            args,
        )
    )

    assert result.is_error is True
    assert result.structured["error"] == "durable_authority_unavailable"
    assert result.structured["webbridge_route"]["terminal_disposition"] == (
        "durable_authority_unavailable"
    )
    assert result.structured["webbridge_route"]["history"] == []
    assert "private store diagnostic" not in result.model_dump_json()
    assert primary.calls == 0


def test_durable_circuit_publication_failure_retains_the_classified_attempt() -> None:
    primary = _FetchAdapter(_status(403))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("direct", _bridge(primary)),),
        policy=WebAccessRoutePolicy(entry_route_id="direct"),
    )
    args = {"url": "https://blocked.example/article"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context(
                {},
                args=args,
                idempotency_key="publication-failure",
                compare_error=ConnectionError("private publication diagnostic"),
            ),
            args,
        )
    )

    route = result.structured["webbridge_route"]
    assert result.structured["error"] == "durable_authority_unavailable"
    assert route["terminal_disposition"] == "durable_authority_unavailable"
    assert route["original_access"]["outcome"] == "destination_denied"
    assert route["history"][0]["access"]["outcome"] == "destination_denied"
    assert "private publication diagnostic" not in result.model_dump_json()
    assert primary.calls == 1


@pytest.mark.parametrize(
    "failure_mode",
    ["fail_after_first", "commit_then_read_failure"],
)
def test_success_settlement_failure_retains_the_replacement_source(
    failure_mode: str,
) -> None:
    primary = _FetchAdapter(_status(401))
    successful = _success("https://replacement.example/article")
    replacement = _FetchAdapter(
        ToolResult(
            content=successful.content,
            structured={
                **successful.structured,
                "effective_source_url": "https://conflicting.example/article",
            },
        )
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="settlement-primary")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="settlement-replacement"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
        ),
    )
    args = {"url": "https://blocked.example/article"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context(
                {},
                args=args,
                idempotency_key=f"success-settlement-{failure_mode}",
                compare_failure_mode=failure_mode,
            ),
            args,
        )
    )

    route = result.structured["webbridge_route"]
    assert result.structured["error"] == "durable_authority_unavailable"
    assert route["terminal_disposition"] == "durable_authority_unavailable"
    assert route["history"][-1]["disposition"] == "success_unrecorded"
    assert route["effective_source_url"] == "https://replacement.example/article"
    assert primary.calls == replacement.calls == 1


def test_rate_limit_wait_uses_authoritative_retry_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 2_000)
    primary = _FetchAdapter(_status(429, retry_after=120))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(5),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=1, open_seconds=30),
        ),
    )
    args = {"url": "https://limited.example/"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="rate-limit"),
            args,
        )
    )

    route = result.structured["webbridge_route"]
    assert route["terminal_disposition"] == "wait"
    assert route["next_eligible_at"] == "1970-01-01T00:35:20Z"
    assert result.structured["access"]["retry_after_seconds"] == 120
    assert primary.calls == 1


def test_retry_after_opens_immediately_and_does_not_slide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 2_000)
    primary = _FetchAdapter(_status(429, retry_after=120))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(5),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=3, open_seconds=30),
        ),
    )
    args = {"url": "https://limited.example/"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="retry-1"), args)
    )
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="retry-2"), args)
    )

    assert primary.calls == 1
    assert first.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:35:20Z")
    assert second.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:35:20Z")
    assert second.structured["webbridge_route"]["history"][0]["invoked"] is False


def test_configured_wait_is_an_absolute_durable_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 2_000)
    primary = _FetchAdapter(_status(429))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(60),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=3, open_seconds=30),
        ),
    )
    args = {"url": "https://limited.example/"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="wait-1"), args)
    )
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="wait-2"), args)
    )

    assert primary.calls == 1
    assert first.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:34:20Z")
    assert second.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:34:20Z")


def test_configured_wait_preserves_denial_series_at_the_next_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 2_000
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: now)
    primary = _FetchAdapter(_status(429), _status(429))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(5),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=2, open_seconds=60),
        ),
    )
    args = {"url": "https://limited.example/"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="series-1"), args)
    )
    now = 2_005
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="series-2"), args)
    )
    third = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="series-3"), args)
    )

    assert primary.calls == 2
    assert first.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:33:25Z")
    assert second.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:34:25Z")
    assert third.structured["webbridge_route"]["next_eligible_at"] == ("1970-01-01T00:34:25Z")
    assert third.structured["webbridge_route"]["history"][0]["invoked"] is False


def test_denial_count_saturates_when_a_threshold_32_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 2_000
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: now)
    primary = _FetchAdapter(*(_status(429) for _ in range(33)))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(1),
                ),
            ),
            circuit=WebAccessCircuitPolicy(threshold=32, open_seconds=1),
        ),
    )
    args = {"url": "https://limited.example/"}
    records: dict[str, dict[str, Any]] = {}

    for attempt in range(32):
        result = asyncio.run(
            bridge.tools[0].run(
                _context(records, args=args, idempotency_key=f"saturate-{attempt}"),
                args,
            )
        )
        assert result.structured["webbridge_route"]["terminal_disposition"] == "wait"
        now += 1

    failed_probe = asyncio.run(
        bridge.tools[0].run(
            _context(records, args=args, idempotency_key="saturate-probe"),
            args,
        )
    )
    suppressed = asyncio.run(
        bridge.tools[0].run(
            _context(records, args=args, idempotency_key="saturate-suppressed"),
            args,
        )
    )

    assert primary.calls == 33
    assert failed_probe.structured["webbridge_route"]["terminal_disposition"] == "wait"
    assert suppressed.structured["webbridge_route"]["history"][0]["invoked"] is False
    stored = next(iter(records.values()))
    assert stored["entries"][0]["denial_count"] == 32


def test_unrepresentable_retry_after_fails_closed_without_shortening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 2_000)
    primary = _FetchAdapter(_status(429, retry_after=172_800))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("hosted", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="hosted",
            rules=(
                WebAccessRouteRule(
                    "hosted",
                    WebAccessOutcome.RATE_LIMITED,
                    WebAccessRouteAction.wait(5),
                ),
            ),
        ),
    )
    args = {"url": "https://limited.example/"}
    records: dict[str, dict[str, Any]] = {}

    first = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="long-1"), args)
    )
    second = asyncio.run(
        bridge.tools[0].run(_context(records, args=args, idempotency_key="long-2"), args)
    )

    assert primary.calls == 1
    assert first.structured["access"]["retry_after_unrepresentable"] is True
    assert first.structured["webbridge_route"]["terminal_disposition"] == "operator_action"
    assert "next_eligible_at" not in first.structured["webbridge_route"]
    assert second.structured["webbridge_route"]["history"][0]["invoked"] is False


def test_success_resets_the_prior_denial_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cayu.tools.web_access._utc_now_seconds", lambda: 3_000)
    primary = _FetchAdapter(_status(403), _success(), _status(403), _status(403))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("direct", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="direct",
            circuit=WebAccessCircuitPolicy(threshold=3, open_seconds=300),
        ),
    )
    args = {"url": "https://sometimes.example/"}
    records: dict[str, dict[str, Any]] = {}

    results = [
        asyncio.run(
            bridge.tools[0].run(
                _context(records, args=args, idempotency_key=f"series-{index}"), args
            )
        )
        for index in range(4)
    ]

    assert primary.calls == 4
    assert results[1].is_error is False
    assert all(
        result.structured["webbridge_route"]["history"][0]["invoked"] is True
        for result in (results[2], results[3])
    )


def test_fallback_without_provider_final_url_uses_the_requested_url() -> None:
    primary = _FetchAdapter(_status(401))
    replacement = _FetchAdapter(ToolResult(content="bounded replacement"))
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary)),
            WebBridgeRoute("replacement", _bridge(replacement)),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
        ),
    )
    args = {"url": "https://requested.example/article"}

    result = asyncio.run(
        bridge.tools[0].run(_context({}, args=args, idempotency_key="source-url"), args)
    )

    assert result.structured["webbridge_route"]["effective_source_url"] == args["url"]


def test_route_and_policy_identity_bind_adapter_and_credential_authority() -> None:
    first = WebBridge.routed(
        routes=(
            WebBridgeRoute(
                "hosted",
                _bridge(
                    _FetchAdapter(_success(), secret_name="first_key"),
                    identity_name="adapter-one",
                ),
            ),
        ),
        policy=WebAccessRoutePolicy(entry_route_id="hosted"),
    )
    changed_adapter = WebBridge.routed(
        routes=(
            WebBridgeRoute(
                "hosted",
                _bridge(
                    _FetchAdapter(_success(), secret_name="first_key"),
                    identity_name="adapter-two",
                ),
            ),
        ),
        policy=WebAccessRoutePolicy(entry_route_id="hosted"),
    )
    changed_credential = WebBridge.routed(
        routes=(
            WebBridgeRoute(
                "hosted",
                _bridge(
                    _FetchAdapter(_success(), secret_name="second_key"),
                    identity_name="adapter-one",
                ),
            ),
        ),
        policy=WebAccessRoutePolicy(entry_route_id="hosted"),
    )

    assert first.tools[0].policy_fingerprint != changed_adapter.tools[0].policy_fingerprint
    assert first.tools[0].policy_fingerprint != changed_credential.tools[0].policy_fingerprint
    assert first.tools[0]._execution_profile_material() is not None
    assert first.tools[0].spec.parallel_safe is False


@pytest.mark.parametrize(
    ("option", "first_value", "second_value"),
    [
        ("max_response_bytes", 32_000, 64_000),
        ("max_content_bytes", 8_000, 16_000),
        ("timeout_seconds", 5.0, 10.0),
        ("max_redirects", 0, 1),
    ],
)
def test_hosted_route_identity_binds_cayu_fetch_configuration(
    option: str,
    first_value: int | float,
    second_value: int | float,
) -> None:
    def routed(value: int | float) -> WebBridge:
        hosted = WebBridge.hosted(
            adapter=_FetchAdapter(_success()),
            fetch_options={option: value},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="configured-hosted-adapter",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        return WebBridge.routed(
            routes=(WebBridgeRoute("hosted", hosted),),
            policy=WebAccessRoutePolicy(entry_route_id="hosted"),
        )

    first = routed(first_value)
    second = routed(second_value)

    assert first.tools[0].policy_fingerprint != second.tools[0].policy_fingerprint
    assert (
        first.tools[0]._execution_profile_material()
        != second.tools[0]._execution_profile_material()
    )


def test_fallback_uses_a_stable_distinct_downstream_idempotency_identity_per_route() -> None:
    ledger: dict[str, ToolResult] = {}
    primary = _SharedLedgerFetchAdapter(
        ledger,
        _status(401),
        secret_name="primary_key",
    )
    replacement = _SharedLedgerFetchAdapter(
        ledger,
        _success("https://entry.example/"),
        secret_name="replacement_key",
    )
    bridge = WebBridge.routed(
        routes=(
            WebBridgeRoute("primary", _bridge(primary, identity_name="primary-ledger")),
            WebBridgeRoute(
                "replacement",
                _bridge(replacement, identity_name="replacement-ledger"),
            ),
        ),
        policy=WebAccessRoutePolicy(
            entry_route_id="primary",
            rules=(
                WebAccessRouteRule(
                    "primary",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.fallback_to("replacement"),
                ),
            ),
        ),
    )
    args = {"url": "https://entry.example/"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="outer-operation"),
            args,
        )
    )

    assert result.is_error is False
    assert primary.calls == replacement.calls == 1
    assert primary.keys[0] != replacement.keys[0]
    assert primary.keys[0].startswith("cayu-web-route:v1:")
    assert replacement.keys[0].startswith("cayu-web-route:v1:")
    assert "outer-operation" not in primary.keys[0]
    assert "outer-operation" not in replacement.keys[0]


def test_auth_operator_action_is_bounded_and_omits_protected_url() -> None:
    primary = _FetchAdapter(_status(401))
    bridge = WebBridge.routed(
        routes=(WebBridgeRoute("provider", _bridge(primary)),),
        policy=WebAccessRoutePolicy(
            entry_route_id="provider",
            rules=(
                WebAccessRouteRule(
                    "provider",
                    WebAccessOutcome.BOT_CHALLENGE,
                    WebAccessRouteAction.operator_action(
                        "Ask the site owner for approved evidence."
                    ),
                ),
            ),
        ),
    )
    args = {"url": "https://blocked.example/protected?secret=canary"}

    result = asyncio.run(
        bridge.tools[0].run(
            _context({}, args=args, idempotency_key="operator"),
            args,
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert result.structured["webbridge_route"]["terminal_disposition"] == "operator_action"
    assert "Ask the site owner" in result.content
    assert "protected" not in rendered
    assert "canary" not in rendered


def test_access_evidence_model_rejects_non_rate_limit_retry_timing() -> None:
    with pytest.raises(ValueError, match="Only rate-limit"):
        from cayu import WebAccessEvidence

        WebAccessEvidence(
            outcome=WebAccessOutcome.CONSENT_REQUIRED,
            source=WebAccessEvidenceSource.HTTP_RESPONSE,
            signal=WebAccessSignal.CONSENT_HEADER,
            destination_fingerprint="a" * 64,
            retry_after_seconds=1,
        )
