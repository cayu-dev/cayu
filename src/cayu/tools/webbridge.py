"""Explicit construction profiles for Cayu's provider-neutral web tools."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from cayu._validation import require_durable_clean_nonblank
from cayu.core.agents import AgentSpec
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.tools import ToolContext, ToolResult
from cayu.environments import Environment, EnvironmentFactory
from cayu.environments.admission import (
    ExecutionEnvironmentAuthority,
    ExecutionRequirements,
)
from cayu.runners import BROWSER_FETCH_WORKLOAD_NAME, PINNED_BROWSER_FETCH_WORKLOAD
from cayu.tools.browser import (
    BROWSER_FETCH_PLAYWRIGHT_VERSION,
    BROWSER_FETCH_PROTOCOL_VERSION,
    BROWSER_FETCH_WORKER_VERSION,
    BrowserWebFetchAdapter,
    ScreenshotPageTool,
    _browser_runner_is_admitted,
)
from cayu.tools.web import (
    WebFetchAdapter,
    WebFetchAdapterRequest,
    WebFetchTool,
    WebSearchAdapter,
    WebSearchAdapterRequest,
    WebSearchTool,
)
from cayu.vaults import SecretRef, copy_secret_ref

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp

DEFAULT_WEBBRIDGE_BROWSER_IMAGE = PINNED_BROWSER_FETCH_WORKLOAD.image
_WEBBRIDGE_CONSTRUCTION_TOKEN = object()
_FETCH_OPTION_NAMES = frozenset(
    {"max_response_bytes", "max_content_bytes", "timeout_seconds", "max_redirects"}
)
_SEARCH_OPTION_NAMES = frozenset(
    {
        "default_results",
        "max_results",
        "max_snippet_bytes",
        "max_total_snippet_bytes",
        "timeout_seconds",
        "restrictions",
    }
)
_SCREENSHOT_OPTION_NAMES = frozenset(
    {
        "max_response_bytes",
        "timeout_seconds",
        "max_redirects",
        "max_requests",
        "max_screenshot_bytes",
        "viewport_width",
        "viewport_height",
        "max_page_width",
        "max_page_height",
        "max_page_pixels",
    }
)


class WebBridgeProfileKind(StrEnum):
    """The application-selected execution and credential boundary."""

    TRUSTED_LOCAL = "trusted_local"
    HOSTED_PROVIDER = "hosted_provider"
    SANDBOXED_BROWSER = "sandboxed_browser"


@dataclass(frozen=True, slots=True)
class WebBridgeCredentialAuthority:
    """Secret-reference authority declared by one hosted web adapter.

    This object contains references only. Constructing a bridge never resolves
    credential values or moves them into a tool schema or model context.
    """

    provider: str
    origin: str
    secret_refs: tuple[SecretRef, ...]

    def __post_init__(self) -> None:
        provider = require_durable_clean_nonblank(self.provider, "provider")
        if len(provider.encode("utf-8")) > 128:
            raise ValueError("provider must not exceed 128 bytes.")
        origin = _credential_origin(self.origin)
        if type(self.secret_refs) is not tuple or not self.secret_refs:
            raise ValueError("secret_refs must contain at least one SecretRef.")
        if len(self.secret_refs) > 16:
            raise ValueError("secret_refs must not contain more than 16 entries.")
        refs = tuple(copy_secret_ref(ref) for ref in self.secret_refs)
        for index, ref in enumerate(refs):
            if any(ref == existing for existing in refs[:index]):
                raise ValueError("secret_refs must not contain duplicates.")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "secret_refs", refs)


@runtime_checkable
class WebBridgeCredentialAuthorityProvider(Protocol):
    """Hosted adapter contract used for side-effect-free credential binding."""

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority: ...


class _HostedBoundWebSearchAdapter:
    def __init__(
        self,
        adapter: WebSearchAdapter,
        authority: WebBridgeCredentialAuthority,
    ) -> None:
        self._adapter = adapter
        self._authority = _copy_webbridge_credential_authority(authority)

    async def search(self, ctx: ToolContext, request: WebSearchAdapterRequest) -> ToolResult:
        refused = _hosted_credential_authority_refusal(
            ctx,
            adapter=self._adapter,
            authority=self._authority,
        )
        if refused is not None:
            return refused
        return await self._adapter.search(ctx, request)


class _HostedBoundWebFetchAdapter:
    def __init__(
        self,
        adapter: WebFetchAdapter,
        authority: WebBridgeCredentialAuthority,
    ) -> None:
        self._adapter = adapter
        self._authority = _copy_webbridge_credential_authority(authority)

    async def fetch(self, ctx: ToolContext, request: WebFetchAdapterRequest) -> ToolResult:
        refused = _hosted_credential_authority_refusal(
            ctx,
            adapter=self._adapter,
            authority=self._authority,
        )
        if refused is not None:
            return refused
        return await self._adapter.fetch(ctx, request)


@dataclass(frozen=True, slots=True)
class WebBridge:
    """A validated set of ordinary Cayu web tools for one explicit profile."""

    kind: WebBridgeProfileKind
    tools: tuple[Any, ...]
    execution_requirements: ExecutionRequirements
    execution_location: str
    credential_path: str
    workspace_requirement: str
    credential_authority: WebBridgeCredentialAuthority | None = None
    environment_authority: ExecutionEnvironmentAuthority | None = None
    artifact_store_id: str | None = None
    browser_protocol: str | None = None
    browser_worker_version: str | None = None
    playwright_version: str | None = None
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _WEBBRIDGE_CONSTRUCTION_TOKEN:
            raise TypeError(
                "Construct WebBridge through trusted_local(), hosted(), or sandboxed_browser()."
            )

    @classmethod
    def trusted_local(
        cls,
        *,
        fetch_options: dict[str, Any] | None = None,
    ) -> WebBridge:
        """Build credentialless host-process ``web_fetch``.

        The host process owns DNS and network access. This profile deliberately
        makes no sandbox or brokered-egress claim.
        """

        return cls(
            kind=WebBridgeProfileKind.TRUSTED_LOCAL,
            tools=(WebFetchTool(**_profile_options(fetch_options, _FETCH_OPTION_NAMES, "fetch")),),
            execution_requirements=ExecutionRequirements.trusted(),
            execution_location="host_process",
            credential_path="none",
            workspace_requirement="none",
            _construction_token=_WEBBRIDGE_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def hosted(
        cls,
        *,
        adapter: object,
        search_options: dict[str, Any] | None = None,
        fetch_options: dict[str, Any] | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> WebBridge:
        """Build the capabilities implemented by one explicit hosted adapter."""

        if isinstance(adapter, type):
            raise TypeError("adapter must be an instance, not a class.")
        search_configuration = _profile_options(
            search_options,
            _SEARCH_OPTION_NAMES,
            "search",
        )
        fetch_configuration = _profile_options(
            fetch_options,
            _FETCH_OPTION_NAMES,
            "fetch",
        )
        supports_search = isinstance(adapter, WebSearchAdapter)
        supports_fetch = isinstance(adapter, WebFetchAdapter)
        if not supports_search and not supports_fetch:
            raise ValueError("Hosted adapter must implement web search or fetch.")
        if search_options is not None and not supports_search:
            raise ValueError("search_options require a hosted adapter with web search support.")
        if fetch_options is not None and not supports_fetch:
            raise ValueError("fetch_options require a hosted adapter with web fetch support.")
        if not isinstance(adapter, WebBridgeCredentialAuthorityProvider):
            raise ValueError("Hosted adapter must declare its WebBridge credential authority.")
        authority = adapter.webbridge_credential_authority()
        if type(authority) is not WebBridgeCredentialAuthority:
            raise TypeError(
                "webbridge_credential_authority() must return WebBridgeCredentialAuthority."
            )
        owned_authority = _copy_webbridge_credential_authority(authority)
        owned_profile_identity = copy_execution_profile_behavior_identity(
            execution_profile_identity
        )
        tools: list[Any] = []
        if supports_search:
            tools.append(
                WebSearchTool(
                    adapter=_HostedBoundWebSearchAdapter(adapter, owned_authority),
                    execution_profile_identity=owned_profile_identity,
                    **search_configuration,
                )
            )
        if supports_fetch:
            tools.append(
                WebFetchTool(
                    adapter=_HostedBoundWebFetchAdapter(adapter, owned_authority),
                    execution_profile_identity=owned_profile_identity,
                    **fetch_configuration,
                )
            )
        return cls(
            kind=WebBridgeProfileKind.HOSTED_PROVIDER,
            tools=tuple(tools),
            execution_requirements=ExecutionRequirements.trusted(),
            execution_location="trusted_provider_adapter",
            credential_path="invocation_credential_proxy",
            workspace_requirement="none",
            credential_authority=owned_authority,
            _construction_token=_WEBBRIDGE_CONSTRUCTION_TOKEN,
        )

    @classmethod
    def sandboxed_browser(
        cls,
        *,
        environment: Environment | EnvironmentFactory,
        browser_image: str,
        fetch_options: dict[str, Any] | None = None,
        screenshot_options: dict[str, Any] | None = None,
    ) -> WebBridge:
        """Build browser tools for a static or factory-backed environment."""

        if not isinstance(environment, Environment | EnvironmentFactory):
            raise TypeError("environment must be an Environment or EnvironmentFactory.")
        if type(browser_image) is not str or browser_image != DEFAULT_WEBBRIDGE_BROWSER_IMAGE:
            raise ValueError(
                "sandboxed WebBridge requires the pinned browser image "
                f"{DEFAULT_WEBBRIDGE_BROWSER_IMAGE!r}."
            )
        if isinstance(environment, Environment):
            runner = environment.runner
            if runner is None:
                raise ValueError("sandboxed WebBridge requires a configured runner.")
            configured_worker = runner.workload_authority(BROWSER_FETCH_WORKLOAD_NAME)
            artifact_store = environment.artifact_store
            environment_authority = runner.execution_environment_authority()
            stage = "pre_exposure"
            try:
                candidate = runner.execution_admission_candidate()
            except Exception as exc:
                raise ValueError(
                    "sandboxed WebBridge could not inspect runner admission evidence."
                ) from exc
        else:
            configured_worker = environment.workload_authority(BROWSER_FETCH_WORKLOAD_NAME)
            artifact_store = environment.configured_artifact_store
            environment_authority = environment.execution_environment_authority()
            stage = "pre_create"
            try:
                candidate = environment.construction_admission_candidate()
            except Exception as exc:
                raise ValueError(
                    "sandboxed WebBridge could not inspect factory admission evidence."
                ) from exc
        if configured_worker != PINNED_BROWSER_FETCH_WORKLOAD:
            raise ValueError(
                "sandboxed WebBridge environment does not prove the selected pinned browser "
                "image and worker."
            )
        if not _browser_runner_is_admitted(candidate, stage=stage):
            raise ValueError(
                "sandboxed WebBridge requires admitted brokered browser execution "
                "with confirmed cancellation and cleanup."
            )
        if candidate is None:  # kept explicit for static narrowing after admission
            raise ValueError("sandboxed WebBridge requires runner admission evidence.")
        if type(environment_authority) is not ExecutionEnvironmentAuthority:
            raise ValueError("sandboxed WebBridge requires exact environment authority evidence.")
        owned_environment_authority = ExecutionEnvironmentAuthority(
            identity=environment_authority.identity,
            profile_identity=environment_authority.profile_identity,
        )
        if artifact_store is None:
            raise ValueError("sandboxed WebBridge requires a configured artifact store.")
        artifact_store_id = artifact_store.id
        if type(artifact_store_id) is not str or not artifact_store_id:
            raise ValueError("sandboxed WebBridge artifact store must have a stable id.")
        expected_candidate = candidate.candidate
        browser_adapter = BrowserWebFetchAdapter(
            expected_runner_candidate=expected_candidate,
            expected_environment_authority=owned_environment_authority,
            expected_workload_authority=PINNED_BROWSER_FETCH_WORKLOAD,
        )
        web_fetch = WebFetchTool(
            adapter=browser_adapter,
            **_profile_options(fetch_options, _FETCH_OPTION_NAMES, "fetch"),
        )
        screenshot = ScreenshotPageTool(
            expected_runner_candidate=expected_candidate,
            expected_environment_authority=owned_environment_authority,
            expected_workload_authority=PINNED_BROWSER_FETCH_WORKLOAD,
            expected_artifact_store_id=artifact_store_id,
            **_profile_options(
                screenshot_options,
                _SCREENSHOT_OPTION_NAMES,
                "screenshot",
            ),
        )
        return cls(
            kind=WebBridgeProfileKind.SANDBOXED_BROWSER,
            tools=(web_fetch, screenshot),
            execution_requirements=ExecutionRequirements.trusted(
                network_access="brokered_egress",
                cancellation="confirmed",
                cleanup="confirmed",
                minimum_evidence="available",
            ),
            execution_location="admitted_runner",
            credential_path="none",
            workspace_requirement="none",
            environment_authority=owned_environment_authority,
            artifact_store_id=artifact_store_id,
            browser_protocol=BROWSER_FETCH_PROTOCOL_VERSION,
            browser_worker_version=BROWSER_FETCH_WORKER_VERSION,
            playwright_version=BROWSER_FETCH_PLAYWRIGHT_VERSION,
            _construction_token=_WEBBRIDGE_CONSTRUCTION_TOKEN,
        )

    def register_agent(
        self,
        app: CayuApp,
        spec: AgentSpec,
        *,
        environment_name: str | None = None,
    ) -> AgentSpec:
        """Validate this bridge against an app environment, then register its agent."""

        from cayu.runtime.app import CayuApp

        if not isinstance(app, CayuApp):
            raise TypeError("app must be a CayuApp.")
        if type(spec) is not AgentSpec:
            raise TypeError("spec must be an AgentSpec.")
        if self.kind is WebBridgeProfileKind.HOSTED_PROVIDER:
            try:
                registered_environment = app.get_environment(environment_name)
            except RuntimeError as exc:
                raise ValueError(
                    "Hosted WebBridge registration requires a concrete environment with "
                    "a credential proxy."
                ) from exc
            proxy = registered_environment.environment.proxy
            if proxy is None:
                raise ValueError(
                    "Hosted WebBridge registration requires the selected environment to "
                    "expose a credential proxy."
                )
            authority = self.credential_authority
            if authority is None:
                raise ValueError("Hosted WebBridge has no credential authority declaration.")
            try:
                supported = proxy.supports_webbridge_credential_authority(
                    _copy_webbridge_credential_authority(authority)
                )
            except Exception:
                supported = False
            if type(supported) is not bool or not supported:
                raise ValueError(
                    "The selected credential proxy does not support the hosted WebBridge authority."
                )
        elif self.kind is WebBridgeProfileKind.SANDBOXED_BROWSER:
            self._validate_sandbox_registration(app, environment_name=environment_name)
        return app.register_agent(
            spec,
            tools=self.tools,
            execution_requirements=self.execution_requirements,
        )

    def _validate_sandbox_registration(
        self,
        app: CayuApp,
        *,
        environment_name: str | None,
    ) -> None:
        try:
            registered_environment = app.get_environment(environment_name)
        except RuntimeError:
            factory = app.get_environment_factory(environment_name)
            authority = factory.execution_environment_authority()
            artifact_store = factory.configured_artifact_store
        else:
            runner = registered_environment.environment.runner
            authority = None if runner is None else runner.execution_environment_authority()
            artifact_store = registered_environment.environment.artifact_store
        if authority != self.environment_authority:
            raise ValueError(
                "The selected environment does not match the sandboxed WebBridge authority."
            )
        if artifact_store is None or artifact_store.id != self.artifact_store_id:
            raise ValueError(
                "The selected environment does not match the sandboxed WebBridge artifact store."
            )


def _copy_webbridge_credential_authority(
    authority: WebBridgeCredentialAuthority,
) -> WebBridgeCredentialAuthority:
    if type(authority) is not WebBridgeCredentialAuthority:
        raise TypeError("authority must be a WebBridgeCredentialAuthority.")
    return WebBridgeCredentialAuthority(
        provider=authority.provider,
        origin=authority.origin,
        secret_refs=authority.secret_refs,
    )


def _hosted_credential_authority_refusal(
    ctx: ToolContext,
    *,
    adapter: object,
    authority: WebBridgeCredentialAuthority,
) -> ToolResult | None:
    try:
        if not isinstance(adapter, WebBridgeCredentialAuthorityProvider):
            raise TypeError("Hosted adapter no longer declares credential authority.")
        current_authority = adapter.webbridge_credential_authority()
        if type(current_authority) is not WebBridgeCredentialAuthority:
            raise TypeError("Hosted adapter returned invalid credential authority.")
        authority_matches = _copy_webbridge_credential_authority(current_authority) == authority
    except Exception:
        authority_matches = False
    if not authority_matches:
        return ToolResult(
            content="The hosted adapter no longer matches the registered WebBridge authority.",
            structured={"error": "capability_refused"},
            is_error=True,
        )
    proxy = ctx.proxy
    if proxy is None:
        return ToolResult(
            content="The active environment has no compatible hosted credential authority.",
            structured={"error": "credential_authority_unavailable"},
            is_error=True,
        )
    supports = getattr(proxy, "supports_webbridge_credential_authority", None)
    if not callable(supports):
        return ToolResult(
            content="The active environment has no compatible hosted credential authority.",
            structured={"error": "credential_authority_unavailable"},
            is_error=True,
        )
    try:
        supported = supports(_copy_webbridge_credential_authority(authority))
    except Exception:
        supported = False
    if type(supported) is not bool or not supported:
        return ToolResult(
            content="The active environment does not match the hosted WebBridge authority.",
            structured={"error": "capability_refused"},
            is_error=True,
        )
    return None


def _credential_origin(value: str) -> str:
    if type(value) is not str or len(value) > 2048:
        raise ValueError("origin must be a bounded HTTPS origin.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("origin must be a bounded HTTPS origin.") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port not in {None, 443}
    ):
        raise ValueError("origin must be a credential-free HTTPS origin.")
    return f"https://{parsed.hostname.lower()}"


def _profile_options(
    value: dict[str, Any] | None,
    allowed: frozenset[str],
    capability: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if type(value) is not dict:
        raise TypeError(f"{capability}_options must be a dict or None.")
    unsupported = sorted(set(value) - allowed)
    if unsupported:
        raise ValueError(
            f"{capability}_options cannot override WebBridge authority or schema: "
            + ", ".join(unsupported)
        )
    return dict(value)
