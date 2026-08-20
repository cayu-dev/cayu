from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    BROWSER_FETCH_PLAYWRIGHT_VERSION,
    BROWSER_FETCH_PROTOCOL_VERSION,
    BROWSER_FETCH_WORKER_VERSION,
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    CayuApp,
    CredentialProxy,
    ExaWebAdapter,
    ExecutionProfileBehaviorIdentity,
    LocalArtifactStore,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SecretRef,
    VirtualEgressEnvironmentFactory,
    WebBridge,
    WebBridgeCredentialAuthority,
    WebBridgeProfileKind,
    WebSearchRestrictions,
)
from cayu.core import AgentSpec, ToolContext, ToolResult
from cayu.environments import (
    Environment,
    EnvironmentSpec,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionEnvironmentAuthority,
)
from cayu.proxies import ProxyAuthorizationResult
from cayu.runners import (
    BROWSER_FETCH_WORKLOAD_NAME,
    PINNED_BROWSER_FETCH_WORKLOAD,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkloadAuthority,
)
from cayu.tools import WebFetchAdapterRequest, WebSearchAdapterRequest
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import ResolvedSecret

_BROWSER_ENVIRONMENT_AUTHORITY = ExecutionEnvironmentAuthority(
    identity="browser-test-environment",
    profile_identity="browser-test-profile",
)


def _browser_candidate(
    *,
    omit: str | None = None,
    identity: str = "browser-test-runner",
) -> ExecutionAdmissionCandidate:
    capabilities = (
        "deny_by_default_network",
        "brokered_egress",
        "confirmed_cancellation",
        "confirmed_cleanup",
    )
    return ExecutionAdmissionCandidate(
        candidate=identity,
        evidence=ExecutionCapabilityEvidence(
            subject=identity,
            claims=tuple(
                ExecutionCapabilityClaim.available(capability)
                for capability in capabilities
                if capability != omit
            ),
        ),
    )


class _BrowserRunner(Runner):
    def __init__(
        self,
        candidate: ExecutionAdmissionCandidate | None = None,
        *,
        environment_authority: ExecutionEnvironmentAuthority = _BROWSER_ENVIRONMENT_AUTHORITY,
    ) -> None:
        super().__init__()
        self._candidate = candidate or _browser_candidate()
        self._environment_authority = environment_authority

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        del command, kwargs
        raise AssertionError("Construction must not dispatch the browser worker.")

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        return self._candidate

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
        return self._environment_authority

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        if name != BROWSER_FETCH_WORKLOAD_NAME:
            return None
        return PINNED_BROWSER_FETCH_WORKLOAD


class _ImplicitAuthorityBrowserRunner(_BrowserRunner):
    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
        return Runner.execution_environment_authority(self)


class _HostedAdapter:
    def __init__(self) -> None:
        self.authority_calls = 0
        self.search_calls = 0
        self.fetch_calls = 0
        self.authority = WebBridgeCredentialAuthority(
            provider="test-provider",
            origin="https://provider.example",
            secret_refs=(SecretRef(name="provider_api_key"),),
        )

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        self.authority_calls += 1
        return self.authority

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        self.search_calls += 1
        return ToolResult(content="search")

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        self.fetch_calls += 1
        return ToolResult(content="fetch")


class _FetchOnlyHostedAdapter:
    def __init__(self) -> None:
        self.authority_calls = 0

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        self.authority_calls += 1
        return WebBridgeCredentialAuthority(
            provider="fetch-provider",
            origin="https://fetch.example",
            secret_refs=(SecretRef(name="fetch_api_key"),),
        )

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        return ToolResult(content="fetch")


class _SearchOnlyHostedAdapter:
    def __init__(self) -> None:
        self.authority_calls = 0

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        self.authority_calls += 1
        return WebBridgeCredentialAuthority(
            provider="search-provider",
            origin="https://search.example",
            secret_refs=(SecretRef(name="search_api_key"),),
        )

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        return ToolResult(content="search")


class _FailingAuthorityHostedAdapter(_HostedAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.fail_authority = False

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        if self.fail_authority:
            raise RuntimeError("private authority diagnostic")
        return super().webbridge_credential_authority()


class _ConfiguredCredentialProxy(CredentialProxy):
    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        return authority == WebBridgeCredentialAuthority(
            provider="test-provider",
            origin="https://provider.example",
            secret_refs=(SecretRef(name="provider_api_key"),),
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del ref, scope
        raise AssertionError("Registration must not resolve credentials.")

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        del destination, credential, action, metadata
        raise AssertionError("Registration must not authorize requests.")


class _IncompatibleCredentialProxy(_ConfiguredCredentialProxy):
    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        del authority
        return False


class _FailingCompatibilityCredentialProxy(_ConfiguredCredentialProxy):
    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        del authority
        raise RuntimeError("private proxy diagnostic")


def test_trusted_local_profile_exposes_only_stable_web_fetch() -> None:
    bridge = WebBridge.trusted_local()

    assert bridge.kind is WebBridgeProfileKind.TRUSTED_LOCAL
    assert [tool.spec.name for tool in bridge.tools] == ["web_fetch"]
    assert bridge.execution_location == "host_process"
    assert bridge.credential_path == "none"
    assert bridge.execution_requirements.network_access == "unrestricted"


def test_profile_constructor_cannot_bypass_validated_factories() -> None:
    with pytest.raises(TypeError, match="Construct WebBridge through"):
        WebBridge(
            kind=WebBridgeProfileKind.SANDBOXED_BROWSER,
            tools=(WebBridge.trusted_local().tools[0],),
            execution_requirements=WebBridge.trusted_local().execution_requirements,
            execution_location="admitted_runner",
            credential_path="none",
            workspace_requirement="none",
        )


@pytest.mark.parametrize("option", ["adapter", "resolver", "transport", "spec"])
def test_profiles_reject_adapter_and_schema_authority_overrides(option: str) -> None:
    with pytest.raises(ValueError, match="cannot override"):
        WebBridge.trusted_local(fetch_options={option: object()})

    with pytest.raises(ValueError, match="cannot override"):
        WebBridge.hosted(adapter=_HostedAdapter(), fetch_options={option: object()})


def test_hosted_profile_discovers_supported_tools_and_credential_authority() -> None:
    adapter = _HostedAdapter()

    bridge = WebBridge.hosted(adapter=adapter)

    assert bridge.kind is WebBridgeProfileKind.HOSTED_PROVIDER
    assert [tool.spec.name for tool in bridge.tools] == ["web_search", "web_fetch"]
    assert bridge.credential_authority == WebBridgeCredentialAuthority(
        provider="test-provider",
        origin="https://provider.example",
        secret_refs=(SecretRef(name="provider_api_key"),),
    )
    assert adapter.authority_calls == 1


def test_hosted_profile_preserves_application_owned_search_restrictions() -> None:
    restrictions = WebSearchRestrictions(include_domains=("provider.example",))

    bridge = WebBridge.hosted(
        adapter=_HostedAdapter(),
        search_options={"restrictions": restrictions},
    )

    assert bridge.tools[0].restrictions == restrictions


def test_hosted_profile_binds_explicit_restart_safe_tool_identity() -> None:
    identity = ExecutionProfileBehaviorIdentity(
        name="hosted-web-adapter",
        behavior_version="1",
        implementation_version="2026-08-20",
    )

    bridge = WebBridge.hosted(
        adapter=_HostedAdapter(),
        execution_profile_identity=identity,
    )

    assert [tool.execution_profile_identity for tool in bridge.tools] == [identity, identity]


def test_hosted_profile_registration_requires_configured_proxy() -> None:
    bridge = WebBridge.hosted(adapter=_HostedAdapter())
    missing_proxy = CayuApp(enable_logging=False)
    missing_proxy.register_environment(
        Environment(EnvironmentSpec(name="hosted")),
        default=True,
    )

    with pytest.raises(ValueError, match="credential proxy"):
        bridge.register_agent(
            missing_proxy,
            AgentSpec(name="hosted-agent", model="test-model"),
        )

    configured = CayuApp(enable_logging=False)
    configured.register_environment(
        Environment(
            EnvironmentSpec(name="hosted"),
            proxy=_ConfiguredCredentialProxy(),
        ),
        default=True,
    )
    registered = bridge.register_agent(
        configured,
        AgentSpec(name="hosted-agent", model="test-model"),
    )

    assert registered.name == "hosted-agent"
    assert "provider_api_key" not in repr(bridge.tools)


def test_hosted_profile_registration_rejects_incompatible_proxy_authority() -> None:
    bridge = WebBridge.hosted(adapter=_HostedAdapter())
    incompatible = CayuApp(enable_logging=False)
    incompatible.register_environment(
        Environment(
            EnvironmentSpec(name="hosted"),
            proxy=_IncompatibleCredentialProxy(),
        ),
        default=True,
    )

    with pytest.raises(ValueError, match="does not support"):
        bridge.register_agent(
            incompatible,
            AgentSpec(name="hosted-agent", model="test-model"),
        )


def test_hosted_profile_registration_fails_closed_on_proxy_compatibility_error() -> None:
    bridge = WebBridge.hosted(adapter=_HostedAdapter())
    app = CayuApp(enable_logging=False)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="hosted"),
            proxy=_FailingCompatibilityCredentialProxy(),
        ),
        default=True,
    )

    with pytest.raises(ValueError, match="does not support") as caught:
        bridge.register_agent(
            app,
            AgentSpec(name="hosted-agent", model="test-model"),
        )

    assert "private proxy diagnostic" not in str(caught.value)


def test_hosted_profile_revalidates_active_proxy_before_adapter_dispatch() -> None:
    adapter = _HostedAdapter()
    bridge = WebBridge.hosted(adapter=adapter)

    refused = asyncio.run(
        bridge.tools[1].run(
            ToolContext(
                session_id="wrong-hosted-environment",
                proxy=_IncompatibleCredentialProxy(),
            ),
            {"url": "https://provider.example/reference"},
        )
    )
    accepted = asyncio.run(
        bridge.tools[1].run(
            ToolContext(
                session_id="right-hosted-environment",
                proxy=_ConfiguredCredentialProxy(),
            ),
            {"url": "https://provider.example/reference"},
        )
    )

    assert refused.structured == {"error": "capability_refused"}
    assert refused.is_error is True
    assert accepted.content == "fetch"
    assert adapter.fetch_calls == 1


def test_hosted_profile_rejects_adapter_authority_drift_before_dispatch() -> None:
    adapter = _HostedAdapter()
    bridge = WebBridge.hosted(adapter=adapter)
    adapter.authority = WebBridgeCredentialAuthority(
        provider="other-provider",
        origin="https://other.example",
        secret_refs=(SecretRef(name="other_api_key"),),
    )

    refused = asyncio.run(
        bridge.tools[1].run(
            ToolContext(
                session_id="hosted-authority-drift",
                proxy=_ConfiguredCredentialProxy(),
            ),
            {"url": "https://provider.example/reference"},
        )
    )

    assert refused.structured == {"error": "capability_refused"}
    assert refused.is_error is True
    assert adapter.fetch_calls == 0


def test_hosted_profile_fails_closed_when_authority_revalidation_raises() -> None:
    adapter = _FailingAuthorityHostedAdapter()
    bridge = WebBridge.hosted(adapter=adapter)
    adapter.fail_authority = True

    refused = asyncio.run(
        bridge.tools[1].run(
            ToolContext(
                session_id="hosted-authority-failure",
                proxy=_ConfiguredCredentialProxy(),
            ),
            {"url": "https://provider.example/reference"},
        )
    )

    assert refused.structured == {"error": "capability_refused"}
    assert "private authority diagnostic" not in str(refused)
    assert adapter.fetch_calls == 0


def test_hosted_profile_exposes_only_adapter_supported_capabilities() -> None:
    bridge = WebBridge.hosted(adapter=_FetchOnlyHostedAdapter())

    assert [tool.spec.name for tool in bridge.tools] == ["web_fetch"]


@pytest.mark.parametrize(
    ("adapter", "options_name", "message"),
    [
        (_FetchOnlyHostedAdapter(), "search_options", "search support"),
        (_SearchOnlyHostedAdapter(), "fetch_options", "fetch support"),
    ],
)
def test_hosted_profile_rejects_options_for_unsupported_capability(
    adapter: _FetchOnlyHostedAdapter | _SearchOnlyHostedAdapter,
    options_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WebBridge.hosted(
            adapter=adapter,
            **{options_name: {"timeout_seconds": 1.0}},
        )
    assert adapter.authority_calls == 0


def test_exa_profile_declares_reference_only_credential_authority() -> None:
    bridge = WebBridge.hosted(adapter=ExaWebAdapter(api_key_ref=SecretRef(name="exa_api_key")))

    assert bridge.credential_authority == WebBridgeCredentialAuthority(
        provider="exa",
        origin="https://api.exa.ai",
        secret_refs=(SecretRef(name="exa_api_key"),),
    )


@pytest.mark.parametrize(
    "adapter, message",
    [
        (object(), "search or fetch"),
        (_HostedAdapter, "adapter must be an instance"),
    ],
)
def test_hosted_profile_rejects_missing_capabilities_or_adapter_instances(
    adapter: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        WebBridge.hosted(adapter=adapter)


def test_hosted_profile_requires_declared_credential_authority() -> None:
    class _UndeclaredAdapter:
        async def fetch(
            self,
            ctx: ToolContext,
            request: WebFetchAdapterRequest,
        ) -> ToolResult:
            del ctx, request
            return ToolResult(content="fetch")

    with pytest.raises(ValueError, match="credential authority"):
        WebBridge.hosted(adapter=_UndeclaredAdapter())


def test_sandboxed_profile_validates_and_binds_setup_authorities(tmp_path: Path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    environment = Environment(
        EnvironmentSpec(name="browser"),
        runner=_BrowserRunner(),
        artifact_store=artifact_store,
    )

    bridge = WebBridge.sandboxed_browser(
        environment=environment,
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )

    assert bridge.kind is WebBridgeProfileKind.SANDBOXED_BROWSER
    assert [tool.spec.name for tool in bridge.tools] == ["web_fetch", "screenshot_page"]
    assert bridge.execution_location == "admitted_runner"
    assert bridge.workspace_requirement == "none"
    assert bridge.artifact_store_id == artifact_store.id
    assert bridge.browser_protocol == BROWSER_FETCH_PROTOCOL_VERSION
    assert bridge.browser_worker_version == BROWSER_FETCH_WORKER_VERSION
    assert bridge.playwright_version == BROWSER_FETCH_PLAYWRIGHT_VERSION
    assert bridge.execution_requirements.network_access == "brokered_egress"
    assert bridge.execution_requirements.cancellation == "confirmed"
    assert bridge.execution_requirements.cleanup == "confirmed"


def test_sandboxed_static_profile_binds_one_exact_runner_instance(tmp_path: Path) -> None:
    configured_runner = _ImplicitAuthorityBrowserRunner()
    bridge = WebBridge.sandboxed_browser(
        environment=Environment(
            EnvironmentSpec(name="browser"),
            runner=configured_runner,
            artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        ),
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )

    assert bridge.environment_authority == configured_runner.execution_environment_authority()
    assert (
        bridge.environment_authority
        != _ImplicitAuthorityBrowserRunner().execution_environment_authority()
    )


def test_sandboxed_profile_registers_with_factory_backed_virtual_egress(
    tmp_path: Path,
) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "factory-artifacts")
    policy = BrowserEgressPolicy(
        name="public-docs",
        allowed_hosts=["docs.example.com"],
        allowed_path_prefixes=["/"],
    )
    factory = VirtualEgressEnvironmentFactory(
        policies={"public-docs": policy},
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="public-docs",
            )
        ],
        runner_kind="docker",
        image="cayu-browser-fetch:3-playwright-1.62.0",
        artifact_store=artifact_store,
    )

    bridge = WebBridge.sandboxed_browser(
        environment=factory,
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(
        EnvironmentSpec(name="browser"),
        factory,
        artifact_store=artifact_store,
        default=True,
    )
    bridge.register_agent(
        app,
        AgentSpec(name="browser-agent", model="test-model"),
    )

    assert [tool.spec.name for tool in bridge.tools] == ["web_fetch", "screenshot_page"]
    assert bridge.artifact_store_id == artifact_store.id


def test_sandboxed_profile_rejects_same_candidate_from_different_factory_authority(
    tmp_path: Path,
) -> None:
    configured_store = LocalArtifactStore(tmp_path / "configured-artifacts")
    shared_profile_identity = ExecutionProfileBehaviorIdentity(
        name="browser-egress",
        behavior_version="1",
        implementation_version="shared-but-not-live-authority",
    )
    restrictive = VirtualEgressEnvironmentFactory(
        policies={
            "restricted": BrowserEgressPolicy(
                name="restricted",
                allowed_hosts=["docs.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="restricted",
            )
        ],
        runner_kind="docker",
        image="cayu-browser-fetch:3-playwright-1.62.0",
        artifact_store=configured_store,
        execution_profile_identity=shared_profile_identity,
    )
    broader = VirtualEgressEnvironmentFactory(
        policies={
            "broader": BrowserEgressPolicy(
                name="broader",
                allowed_hosts=["docs.example.com", "public.example.com"],
            )
        },
        approved_destinations=[
            ApprovedEgressDestination(
                destination="docs.example.com",
                policy_name="broader",
            ),
            ApprovedEgressDestination(
                destination="public.example.com",
                policy_name="broader",
            ),
        ],
        runner_kind="docker",
        image="cayu-browser-fetch:3-playwright-1.62.0",
        artifact_store=configured_store,
        execution_profile_identity=shared_profile_identity,
    )
    bridge = WebBridge.sandboxed_browser(
        environment=restrictive,
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )
    active_runner = _BrowserRunner(environment_authority=broader.execution_environment_authority())

    result = asyncio.run(
        bridge.tools[0].run(
            ToolContext(session_id="mismatch", runner=active_runner),
            {"url": "https://docs.example.com/"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "capability_refused"}


def test_sandboxed_profile_reconstructs_fetch_and_screenshot_identity_after_restart(
    tmp_path: Path,
) -> None:
    profile_identity = ExecutionProfileBehaviorIdentity(
        name="browser-egress",
        behavior_version="1",
        implementation_version="2026-08-20",
    )

    def build_app() -> CayuApp:
        artifact_store = LocalArtifactStore(tmp_path / "restart-artifacts")
        policy = BrowserEgressPolicy(
            name="public-docs",
            allowed_hosts=["docs.example.com"],
        )
        factory = VirtualEgressEnvironmentFactory(
            policies={policy.name: policy},
            approved_destinations=[
                ApprovedEgressDestination(
                    destination="docs.example.com",
                    policy_name=policy.name,
                )
            ],
            runner_kind="docker",
            image="cayu-browser-fetch:3-playwright-1.62.0",
            artifact_store=artifact_store,
            execution_profile_identity=profile_identity,
        )
        bridge = WebBridge.sandboxed_browser(
            environment=factory,
            browser_image="cayu-browser-fetch:3-playwright-1.62.0",
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_environment_factory(
            EnvironmentSpec(
                name="browser",
                execution_profile_identity=profile_identity,
            ),
            factory,
            artifact_store=artifact_store,
            default=True,
        )
        bridge.register_agent(
            app,
            AgentSpec(name="browser-agent", model="scripted-model"),
        )
        return app

    async def prepare(app: CayuApp):
        return await app._session_engine._prepare_initial_run(
            RunRequest(
                agent_name="browser-agent",
                messages=[Message.text("user", "inspect")],
            )
        )

    first = asyncio.run(prepare(build_app()))
    restarted = asyncio.run(prepare(build_app()))

    assert list(first.registered_agent.tools) == [
        "web_fetch",
        "screenshot_page",
    ]
    assert first.execution_profile == restarted.execution_profile


def test_sandboxed_profile_never_dispatches_a_different_runner_candidate(
    tmp_path: Path,
) -> None:
    configured_store = LocalArtifactStore(tmp_path / "configured-artifacts")
    bridge = WebBridge.sandboxed_browser(
        environment=Environment(
            EnvironmentSpec(name="browser"),
            runner=_BrowserRunner(),
            artifact_store=configured_store,
        ),
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )
    active_runner = _BrowserRunner(_browser_candidate(identity="other-browser-runner"))

    result = asyncio.run(
        bridge.tools[0].run(
            ToolContext(session_id="mismatch", runner=active_runner),
            {"url": "https://example.com/"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "capability_refused"}


def test_sandboxed_profile_revalidates_materialized_worker_authority(
    tmp_path: Path,
) -> None:
    bridge = WebBridge.sandboxed_browser(
        environment=Environment(
            EnvironmentSpec(name="browser"),
            runner=_BrowserRunner(),
            artifact_store=LocalArtifactStore(tmp_path / "configured-artifacts"),
        ),
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )

    result = asyncio.run(
        bridge.tools[0].run(
            ToolContext(
                session_id="mismatch",
                runner=_RunnerWithoutBrowserWorkerAuthority(),
            ),
            {"url": "https://example.com/"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "capability_refused"}


def test_invocation_runner_handle_preserves_detached_workload_authority() -> None:
    handle = InvocationRunnerHandle(
        _BrowserRunner(),
        redactor_snapshot_provider=SecretRedactor,
    )

    assert handle.workload_authority(BROWSER_FETCH_WORKLOAD_NAME) == PINNED_BROWSER_FETCH_WORKLOAD


def test_invocation_runner_handle_preserves_detached_environment_authority() -> None:
    handle = InvocationRunnerHandle(
        _BrowserRunner(),
        redactor_snapshot_provider=SecretRedactor,
    )

    assert handle.execution_environment_authority() == _BROWSER_ENVIRONMENT_AUTHORITY


def test_sandboxed_profile_never_publishes_to_a_different_artifact_store(
    tmp_path: Path,
) -> None:
    configured_store = LocalArtifactStore(tmp_path / "configured-artifacts")
    bridge = WebBridge.sandboxed_browser(
        environment=Environment(
            EnvironmentSpec(name="browser"),
            runner=_BrowserRunner(),
            artifact_store=configured_store,
        ),
        browser_image="cayu-browser-fetch:3-playwright-1.62.0",
    )

    result = asyncio.run(
        bridge.tools[1].run(
            ToolContext(
                session_id="mismatch",
                runner=_BrowserRunner(),
                artifact_store=LocalArtifactStore(tmp_path / "other-artifacts"),
            ),
            {"url": "https://example.com/"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "capability_refused"}


@pytest.mark.parametrize(
    "environment_factory, image, message",
    [
        (
            lambda tmp_path: Environment(
                EnvironmentSpec(name="browser"),
                artifact_store=LocalArtifactStore(tmp_path / "missing-runner"),
            ),
            "cayu-browser-fetch:3-playwright-1.62.0",
            "runner",
        ),
        (
            lambda tmp_path: Environment(
                EnvironmentSpec(name="browser"),
                runner=_BrowserRunner(_browser_candidate(omit="brokered_egress")),
                artifact_store=LocalArtifactStore(tmp_path / "bad-runner"),
            ),
            "cayu-browser-fetch:3-playwright-1.62.0",
            "brokered browser execution",
        ),
        (
            lambda tmp_path: Environment(
                EnvironmentSpec(name="browser"),
                runner=_BrowserRunner(),
            ),
            "cayu-browser-fetch:3-playwright-1.62.0",
            "artifact store",
        ),
        (
            lambda tmp_path: Environment(
                EnvironmentSpec(name="browser"),
                runner=_BrowserRunner(),
                artifact_store=LocalArtifactStore(tmp_path / "bad-image"),
            ),
            "browser:latest",
            "pinned browser image",
        ),
        (
            lambda tmp_path: Environment(
                EnvironmentSpec(name="browser"),
                runner=_RunnerWithoutBrowserWorkerAuthority(),
                artifact_store=LocalArtifactStore(tmp_path / "missing-worker"),
            ),
            "cayu-browser-fetch:3-playwright-1.62.0",
            "image and worker",
        ),
    ],
)
def test_sandboxed_profile_rejects_unsupported_setup(
    tmp_path: Path,
    environment_factory: Any,
    image: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        WebBridge.sandboxed_browser(
            environment=environment_factory(tmp_path),
            browser_image=image,
        )


class _RunnerWithoutBrowserWorkerAuthority(_BrowserRunner):
    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        del name
        return None
