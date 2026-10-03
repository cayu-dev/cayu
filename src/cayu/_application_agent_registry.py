"""Agent registration, lookup and atomic MCP catalogue publication."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from itertools import islice
from types import MappingProxyType

from cayu._application_registration import (
    _copy_registered_tool,
    _registered_tool_descriptor,
    _snapshot_context_behavior_execution_profile_identities,
    _validate_agent_spec,
    _validate_registered_tool,
    _validate_runtime_hooks,
)
from cayu._validation import require_clean_nonblank
from cayu.agents import AgentSpec
from cayu.configuration import CayuConfigSource
from cayu.context.base import ContextPolicy, DefaultContextPolicy
from cayu.context.thinking import ThinkingConfig
from cayu.environments.admission import ExecutionRequirements
from cayu.mcp.tools import (
    McpToolAdapter,
    McpToolset,
    McpToolsetRefreshBlocked,
    McpToolsetRefreshResult,
    McpToolsetUnavailable,
    mcp_toolset_manifest_diff,
)
from cayu.observability.hooks import RuntimeHook
from cayu.providers.hosted import OpenAIWebSearch, copy_openai_web_search
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._tool_effect_reconciliation import register_tool_effect_reconciler
from cayu.runtime.application_lifecycle import (
    ApplicationAdmission,
    ApplicationAdmissionsSealed,
    _admitted_entrance,
)
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.mcp_manifest_policy import McpManifestPolicy, McpManifestPolicyAction
from cayu.runtime.tool_effects import ToolEffectReconciliationRegistration
from cayu.sessions.base import SessionStore
from cayu.sessions.child_context import ChildSessionContextContributor
from cayu.tools.base import Tool, ToolEffect
from cayu.tools.catalogue import build_tool_catalog_snapshot
from cayu.tools.discovery import (
    TOOL_DISCOVERY_ONLY_PROFILE_ID,
    SearchToolsTool,
    ToolDiscoveryMode,
    copy_tool_discovery_mode,
)
from cayu.tools.exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    AllRegisteredToolsExposurePolicy,
    RegisteredToolCapability,
    ResolvedToolExposure,
    StaticToolExposurePolicy,
    ToolExposurePolicy,
)
from cayu.tools.policy import AllowAllToolPolicy, ToolPolicy
from cayu.tools.targeted_projection import TargetedToolMode, copy_targeted_tool_mode
from cayu.vaults.redaction import SecretRedactor


class ApplicationAgentRegistry:
    """Own admitted agent declarations and atomic publication of MCP catalogues.

    The caller admits toolset use and explicit refreshes through the supplied
    admission gate, then seals it before releasing MCP ownership. Notification
    refreshes use that gate by default; ``notification_refresh`` can route them
    through the caller's admitted entrance instead.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        secret_redactor: SecretRedactor,
        tool_timeout_seconds: float | None,
        mcp_manifest_policy: McpManifestPolicy | None,
        admission: ApplicationAdmission,
        notification_refresh: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._admission = admission
        self._static_mcp_toolsets: dict[int, McpToolset] = {}
        self._released_mcp_refreshes: set[asyncio.Task[None]] = set()
        self._session_store = session_store
        self._secret_redactor = secret_redactor
        self._tool_timeout_seconds = tool_timeout_seconds
        self._mcp_manifest_policy = mcp_manifest_policy
        self._agents: dict[str, runtime_records.RegisteredAgentState] = {}
        self._agent_thinking_sources: dict[str, CayuConfigSource] = {}
        self._mcp_refresh_owner = object()
        self._mcp_publication_lock = asyncio.Lock()
        self._refreshable_mcp_toolsets: dict[int, McpToolset] = {}
        self._notification_refresh = (
            self._refresh_after_notification
            if notification_refresh is None
            else notification_refresh
        )

    async def release_mcp_toolsets(self, timeout_s: float) -> bool:
        """Hand caller-owned MCP toolsets back so another app can own them.

        Releasing refresh ownership removes this app's notification handler and
        cancels its pending notification refresh; a refresh already running is
        an admitted operation that shutdown waited for.
        """

        # A running operation still relies on this app's list-changed fence;
        # releasing it then would let that operation dispatch stale tools.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while self._admission.in_flight:
            remaining = deadline - loop.time()
            if remaining <= 0 or not await self._admission.wait_idle(remaining):
                return False
        # No await from the check above to the release below.
        for toolset in self._refreshable_mcp_toolsets.values():
            task = toolset._refresh_source.release_refresh_owner(self._mcp_refresh_owner)
            if task is not None:
                self._released_mcp_refreshes.add(task)
                task.add_done_callback(self._released_mcp_refreshes.discard)
        for toolset in self._static_mcp_toolsets.values():
            toolset._refresh_source.release_static_owner(self._mcp_refresh_owner)
        if not self._released_mcp_refreshes:
            return True
        _done, pending = await asyncio.wait(
            set(self._released_mcp_refreshes), timeout=max(0.0, deadline - loop.time())
        )
        return not pending

    @property
    def mcp_refreshes_pending(self) -> bool:
        return bool(self._released_mcp_refreshes)

    @property
    def registrations(self) -> Mapping[str, runtime_records.RegisteredAgentState]:
        """Read the current publication; refresh atomically replaces its map."""
        return MappingProxyType(self._agents)

    def thinking_source(self, agent_name: str) -> CayuConfigSource:
        return self._agent_thinking_sources[agent_name]

    def current_mcp_toolset(self, source_key: int) -> McpToolset | None:
        return self._refreshable_mcp_toolsets.get(source_key)

    def replace_for_replay(
        self, registrations: dict[str, runtime_records.RegisteredAgentState]
    ) -> None:
        """Install isolated replay declarations without claiming live MCP sources."""
        self._agents = registrations

    @_admitted_entrance
    async def _refresh_after_notification(self, source_key: int) -> None:
        if self._admission.sealed:
            raise ApplicationAdmissionsSealed("The application is shutting down.")
        current = self.current_mcp_toolset(source_key)
        if current is not None:
            await self.refresh_mcp_toolset(current)

    def register_agent(
        self,
        spec: AgentSpec,
        *,
        default_thinking: ThinkingConfig | None = None,
        default_thinking_source: CayuConfigSource = "framework",
        registration_site: tuple[str | None, str | None] = (None, None),
        tools: Iterable[Tool] | None = None,
        tool_effect_reconcilers: Mapping[str, ToolEffectReconciliationRegistration] | None = None,
        mcp_toolsets: Iterable[McpToolset] | None = None,
        hosted_tools: Iterable[OpenAIWebSearch] | None = None,
        context_policy: ContextPolicy | None = None,
        context_overflow_policy: ContextPolicy | None = None,
        child_session_context: ChildSessionContextContributor | None = None,
        tool_exposure_policy: ToolExposurePolicy | None = None,
        targeted_tool_mode: TargetedToolMode | str | None = None,
        tool_discovery_mode: ToolDiscoveryMode | str | None = None,
        tool_policy: ToolPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        loop_policies: Iterable[LoopPolicy] | None = None,
        execution_requirements: ExecutionRequirements | None = None,
    ) -> AgentSpec:
        if type(spec) is not AgentSpec:
            raise TypeError("Agent registration requires an AgentSpec.")
        stored_spec = _validate_agent_spec(spec)
        thinking_source: CayuConfigSource = (
            "explicit" if stored_spec.thinking is not None else default_thinking_source
        )
        if stored_spec.thinking is None and default_thinking is not None:
            stored_spec = stored_spec.model_copy(
                update={
                    "thinking": ThinkingConfig.model_validate(
                        default_thinking.model_dump(
                            mode="python",
                            warnings=False,
                        )
                    )
                },
                deep=True,
            )
        if stored_spec.name in self._agents:
            raise ValueError(f"Agent already registered: {stored_spec.name}")
        if context_policy is None:
            stored_context_policy = DefaultContextPolicy()
        elif isinstance(context_policy, ContextPolicy):
            stored_context_policy = context_policy
        else:
            raise TypeError("context_policy must be a ContextPolicy.")
        if context_overflow_policy is None:
            stored_context_overflow_policy = None
        elif isinstance(context_overflow_policy, ContextPolicy):
            stored_context_overflow_policy = context_overflow_policy
        else:
            raise TypeError("context_overflow_policy must be a ContextPolicy.")
        if child_session_context is None:
            stored_child_session_context = None
        elif type(child_session_context) is ChildSessionContextContributor:
            if (
                self._session_store.child_session_notification_version != 1
                or not self._session_store.supports_public_authority_aliases
                or self._session_store.public_authority_alias_codec is None
            ):
                raise RuntimeError(
                    "child_session_context requires a v1 child-notification SessionStore "
                    "with configured public authority aliases."
                )
            stored_child_session_context = child_session_context
        else:
            raise TypeError(
                "child_session_context must be a ChildSessionContextContributor or None."
            )
        stored_tool_discovery_mode = (
            None if tool_discovery_mode is None else copy_tool_discovery_mode(tool_discovery_mode)
        )
        if tool_exposure_policy is None:
            stored_tool_exposure_policy = (
                AllRegisteredToolsExposurePolicy()
                if stored_tool_discovery_mode is None
                else StaticToolExposurePolicy(
                    profile_id=TOOL_DISCOVERY_ONLY_PROFILE_ID,
                    tools=(),
                )
            )
        elif isinstance(tool_exposure_policy, ToolExposurePolicy):
            stored_tool_exposure_policy = tool_exposure_policy
        else:
            raise TypeError("tool_exposure_policy must be a ToolExposurePolicy.")
        stored_targeted_tool_mode = (
            None if targeted_tool_mode is None else copy_targeted_tool_mode(targeted_tool_mode)
        )
        if stored_targeted_tool_mode is not None and (
            not self._session_store.supports_targeted_tool_grants
            or not self._session_store.supports_public_authority_aliases
            or self._session_store.public_authority_alias_codec is None
        ):
            raise RuntimeError(
                "targeted_tool_mode requires a SessionStore with durable targeted-grant "
                "state and configured public authority aliases."
            )
        if tool_policy is None:
            stored_tool_policy = AllowAllToolPolicy()
        elif isinstance(tool_policy, ToolPolicy):
            stored_tool_policy = tool_policy
        else:
            raise TypeError("tool_policy must be a ToolPolicy.")
        stored_runtime_hooks = _validate_runtime_hooks(
            runtime_hooks,
            field_name="runtime_hooks",
            redactor=self._secret_redactor,
        )
        stored_loop_policies = validate_loop_policies(
            loop_policies,
            field_name="loop_policies",
        )
        if execution_requirements is None:
            stored_execution_requirements = ExecutionRequirements.trusted()
        elif isinstance(execution_requirements, ExecutionRequirements):
            stored_execution_requirements = ExecutionRequirements.model_validate(
                execution_requirements.model_dump(mode="python", warnings=False)
            )
        else:
            raise TypeError("execution_requirements must be ExecutionRequirements or None.")

        requested_mcp_toolsets = _copy_refreshable_mcp_toolsets(mcp_toolsets)
        resolved_mcp_toolsets: list[McpToolset] = []
        seen_mcp_sources: set[int] = set()
        seen_mcp_manifest_identities: set[str] = set()
        for index, toolset in enumerate(requested_mcp_toolsets):
            source_key = _mcp_refresh_source_key(toolset)
            if source_key in seen_mcp_sources:
                raise ValueError("mcp_toolsets must contain unique MCP sources.")
            seen_mcp_sources.add(source_key)
            current = self._refreshable_mcp_toolsets.get(source_key)
            resolved = toolset if current is None else current
            if not resolved._refresh_source.registration_authority_is_current(resolved.generation):
                raise ValueError("A refreshable MCP source must be ready during registration.")
            if not resolved.manifest_identity_is_explicit:
                raise ValueError(
                    f"mcp_toolsets[{index}] requires an explicit McpServerSpec.connection_id."
                )
            if resolved.manifest_identity in seen_mcp_manifest_identities:
                raise ValueError("mcp_toolsets must contain unique MCP connection identities.")
            seen_mcp_manifest_identities.add(resolved.manifest_identity)
            if current is None and any(
                registered.manifest_identity == resolved.manifest_identity
                for registered in self._refreshable_mcp_toolsets.values()
            ):
                raise ValueError(
                    "Refreshable MCP sources require unique connection identities "
                    "within one CayuApp."
                )
            if current is None and _agents_contain_mcp_source(self._agents, resolved):
                raise ValueError(
                    "A refreshable MCP source cannot also be registered through static tools."
                )
            resolved_mcp_toolsets.append(resolved)
        stored_mcp_toolsets = tuple(resolved_mcp_toolsets)

        if tools is None:
            agent_tools = []
        else:
            if isinstance(tools, str | bytes):
                raise TypeError("Agent tools must be an iterable of Tool instances.")
            try:
                agent_tools = list(tools)
            except TypeError as exc:
                raise TypeError("Agent tools must be an iterable of Tool instances.") from exc

        refreshable_source_keys = {
            _mcp_refresh_source_key(toolset) for toolset in stored_mcp_toolsets
        }
        static_mcp_toolsets: dict[int, McpToolset] = {}
        for tool in agent_tools:
            if isinstance(tool, McpToolAdapter):
                source_key = _mcp_refresh_source_key(tool.toolset)
                if (
                    source_key in refreshable_source_keys
                    or source_key in self._refreshable_mcp_toolsets
                ):
                    raise ValueError(
                        "A refreshable MCP source cannot also be registered through static tools."
                    )
                static_mcp_toolsets.setdefault(source_key, tool.toolset)
        for toolset in stored_mcp_toolsets:
            agent_tools.extend(sorted(toolset.tools, key=lambda tool: tool.name))

        tools_by_name: dict[str, runtime_records.RegisteredTool] = {}
        for tool in agent_tools:
            if not isinstance(tool, Tool):
                raise TypeError("Agent tools must be Tool instances.")
            registered_tool = _validate_registered_tool(
                tool,
                redactor=self._secret_redactor,
                timeout_seconds=self._tool_timeout_seconds,
            )
            if registered_tool.name in tools_by_name:
                raise ValueError(f"Duplicate tool registered for agent: {registered_tool.name}")
            tools_by_name[registered_tool.name] = registered_tool

        if tool_effect_reconcilers is not None:
            if not isinstance(tool_effect_reconcilers, Mapping):
                raise TypeError(
                    "tool_effect_reconcilers must map exact tool names to registrations."
                )
            for tool_name, registration in tool_effect_reconcilers.items():
                if type(tool_name) is not str or tool_name not in tools_by_name:
                    raise ValueError("Effect reconciler targets an unregistered tool.")
                registered_tool = tools_by_name[tool_name]
                tools_by_name[tool_name] = replace(
                    registered_tool,
                    effect_reconciler=register_tool_effect_reconciler(
                        registration,
                        effect=registered_tool.effect,
                        redactor=self._secret_redactor,
                    ),
                )

        runtime_tools_by_name: dict[str, runtime_records.RegisteredTool] = {}
        if stored_tool_discovery_mode is not None:
            search_tool = _validate_registered_tool(
                SearchToolsTool(),
                redactor=self._secret_redactor,
                framework_owned=True,
                timeout_seconds=self._tool_timeout_seconds,
            )
            runtime_tools_by_name[search_tool.name] = search_tool

        if hosted_tools is None:
            stored_hosted_tools: tuple[OpenAIWebSearch, ...] = ()
        else:
            if isinstance(hosted_tools, str | bytes):
                raise TypeError("Agent hosted_tools must be an iterable of hosted tool instances.")
            try:
                copied_hosted_tools = tuple(
                    copy_openai_web_search(hosted_tool) for hosted_tool in hosted_tools
                )
            except TypeError as exc:
                raise TypeError(
                    "Agent hosted_tools must be an iterable of hosted tool instances."
                ) from exc
            if len(copied_hosted_tools) > 1:
                raise ValueError("Duplicate OpenAI web search hosted tool registered for agent.")
            stored_hosted_tools = copied_hosted_tools

        registration_source, registration_symbol = registration_site
        descriptors_by_name = {
            tool.name: _registered_tool_descriptor(tool) for tool in tools_by_name.values()
        }
        tool_catalogue = build_tool_catalog_snapshot(descriptors_by_name.values())
        tool_capabilities = tuple(
            RegisteredToolCapability(**descriptors_by_name[name].exposure_capability_material())
            for name in tools_by_name
        )
        all_registered_tool_exposure = ResolvedToolExposure(
            profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
            catalogue_revision=tool_catalogue.revision,
            tools=tool_capabilities,
            registered_count=len(tool_capabilities),
            ceiling_count=len(tool_capabilities),
        )
        # Keep one frozen exposure graph for registration/profile admission and
        # the expose-all snapshot; the catalogue remains its canonical source.
        tool_capabilities = all_registered_tool_exposure.tools
        registered_agent = runtime_records.RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            runtime_tools=MappingProxyType(runtime_tools_by_name),
            tool_catalogue=tool_catalogue,
            tool_capabilities=tool_capabilities,
            all_registered_tool_exposure=all_registered_tool_exposure,
            tool_exposure_policy=stored_tool_exposure_policy,
            tool_exposure_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_tool_exposure_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name=("tool_exposure_policy.execution_profile_identity"),
                )
            ),
            targeted_tool_mode=stored_targeted_tool_mode,
            tool_discovery_mode=stored_tool_discovery_mode,
            hosted_tools=stored_hosted_tools,
            context_policy=stored_context_policy,
            context_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_context_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="context_policy.execution_profile_identity",
                )
            ),
            context_overflow_policy=stored_context_overflow_policy,
            context_overflow_policy_execution_profile_identity=(
                None
                if stored_context_overflow_policy is None
                else copy_secret_free_execution_profile_behavior_identity(
                    stored_context_overflow_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="context_overflow_policy.execution_profile_identity",
                )
            ),
            tool_policy=stored_tool_policy,
            tool_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_tool_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="tool_policy.execution_profile_identity",
                )
            ),
            runtime_hooks=stored_runtime_hooks,
            loop_policies=stored_loop_policies,
            loop_policy_execution_profile_identities=tuple(
                copy_secret_free_execution_profile_behavior_identity(
                    policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name=f"loop_policies[{index}].execution_profile_identity",
                )
                for index, policy in enumerate(stored_loop_policies)
            ),
            execution_requirements=stored_execution_requirements,
            mcp_toolsets=stored_mcp_toolsets,
            context_behavior_execution_profile_identities=(
                _snapshot_context_behavior_execution_profile_identities(
                    stored_context_policy,
                    stored_context_overflow_policy,
                    redactor=self._secret_redactor,
                )
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
            child_session_context_contributor=stored_child_session_context,
        )
        if (static_mcp_toolsets or stored_mcp_toolsets) and self._admission.sealed:
            # Shutdown hands MCP toolsets back to their caller; a closing app
            # must not take them again.
            raise ApplicationAdmissionsSealed(
                "A closing application cannot take ownership of MCP toolsets."
            )
        newly_claimed_static: list[McpToolset] = []
        newly_claimed_refreshable: list[McpToolset] = []
        try:
            for toolset in static_mcp_toolsets.values():
                if toolset._refresh_source.claim_static_owner(self._mcp_refresh_owner):
                    newly_claimed_static.append(toolset)
            for toolset in stored_mcp_toolsets:
                source_key = _mcp_refresh_source_key(toolset)
                if source_key in self._refreshable_mcp_toolsets:
                    continue
                toolset._refresh_source.claim_refresh_owner(
                    self._mcp_refresh_owner,
                    notification_refresh=(
                        lambda source_key=source_key: self._notification_refresh(source_key)
                    ),
                )
                newly_claimed_refreshable.append(toolset)
        except BaseException:
            for toolset in newly_claimed_refreshable:
                toolset._refresh_source.release_refresh_owner(self._mcp_refresh_owner)
            for toolset in newly_claimed_static:
                toolset._refresh_source.release_static_owner(self._mcp_refresh_owner)
            raise
        self._agents[stored_spec.name] = registered_agent
        self._agent_thinking_sources[stored_spec.name] = thinking_source
        for toolset in stored_mcp_toolsets:
            self._refreshable_mcp_toolsets[_mcp_refresh_source_key(toolset)] = toolset
        for toolset in newly_claimed_static:
            self._static_mcp_toolsets[_mcp_refresh_source_key(toolset)] = toolset
        return spec

    async def refresh_mcp_toolset(
        self,
        toolset: McpToolset,
    ) -> McpToolsetRefreshResult:
        """Re-list and atomically publish one explicitly registered MCP source."""

        if not isinstance(toolset, McpToolset):
            raise TypeError("toolset must be a McpToolset.")
        source_key = _mcp_refresh_source_key(toolset)
        current = self._refreshable_mcp_toolsets.get(source_key)
        if current is None:
            raise ValueError("MCP toolset refresh requires explicit mcp_toolsets= registration.")
        source = current._refresh_source
        previous_generation = current.generation
        refresh_started = False
        refresh_dirty_epoch = 0
        discovery = None
        try:
            refresh_dirty_epoch = await source.begin_refresh(
                owner=self._mcp_refresh_owner,
                expected_generation=previous_generation,
            )
            refresh_started = True
            candidate, discovery = await current._prepare_refresh()
            staged_discovery = discovery
            diff = mcp_toolset_manifest_diff(current, candidate)
            decision = None
            if self._mcp_manifest_policy is not None:
                decision = self._mcp_manifest_policy.decide(
                    status="changed" if diff.changed else "unchanged",
                    diff=diff.policy_input(),
                )
                if decision.action is McpManifestPolicyAction.BLOCK:
                    source.quarantine_refresh(
                        owner=self._mcp_refresh_owner,
                        expected_generation=previous_generation,
                        expected_dirty_epoch=refresh_dirty_epoch,
                    )
                    raise McpToolsetRefreshBlocked(decision.reason)
            if not diff.changed:

                def require_unchanged_refresh_current() -> None:
                    source.require_refresh_current(
                        owner=self._mcp_refresh_owner,
                        expected_generation=previous_generation,
                        expected_dirty_epoch=refresh_dirty_epoch,
                    )

                await source.finish_unchanged(
                    owner=self._mcp_refresh_owner,
                    expected_generation=previous_generation,
                    expected_dirty_epoch=refresh_dirty_epoch,
                    publish=lambda: staged_discovery.commit(
                        validate=require_unchanged_refresh_current
                    ),
                )
                discovery = None
                return McpToolsetRefreshResult(
                    toolset=current,
                    status="unchanged",
                    previous_generation=previous_generation,
                    generation=previous_generation,
                    previous_manifest_hash=current.manifest_hash,
                    manifest_hash=current.manifest_hash,
                    diff=diff,
                    policy_action=(None if decision is None else decision.action.value),
                )

            async def publish() -> None:
                async with self._mcp_publication_lock:
                    published_current = self._refreshable_mcp_toolsets.get(source_key)
                    if (
                        published_current is None
                        or published_current._refresh_source is not source
                        or published_current.generation != previous_generation
                    ):
                        raise McpToolsetUnavailable(
                            "MCP application publication authority changed during refresh."
                        )
                    replacements = {
                        name: _registered_agent_after_mcp_refresh(
                            registered_agent,
                            source=source,
                            candidate=candidate,
                            redactor=self._secret_redactor,
                            timeout_seconds=self._tool_timeout_seconds,
                        )
                        for name, registered_agent in self._agents.items()
                        if _registered_agent_contains_mcp_source(registered_agent, source)
                    }
                    if not replacements:
                        raise RuntimeError("Refreshable MCP source lost its agent registrations.")
                    await staged_discovery.commit(
                        validate=lambda: source.require_refresh_current(
                            owner=self._mcp_refresh_owner,
                            expected_generation=previous_generation,
                            expected_dirty_epoch=refresh_dirty_epoch,
                        )
                    )
                    # Synchronous registration can complete while discovery commits.
                    # Merge the current maps without yielding again before publication.
                    next_agents = {**self._agents, **replacements}
                    next_toolsets = dict(self._refreshable_mcp_toolsets)
                    next_toolsets[source_key] = candidate
                    self._agents = next_agents
                    self._refreshable_mcp_toolsets = next_toolsets

            await source.publish_refresh(
                owner=self._mcp_refresh_owner,
                expected_generation=previous_generation,
                generation=candidate.generation,
                expected_dirty_epoch=refresh_dirty_epoch,
                publish=publish,
            )
            discovery = None
            return McpToolsetRefreshResult(
                toolset=candidate,
                status="accepted",
                previous_generation=previous_generation,
                generation=candidate.generation,
                previous_manifest_hash=current.manifest_hash,
                manifest_hash=candidate.manifest_hash,
                diff=diff,
                policy_action=(None if decision is None else decision.action.value),
            )
        except BaseException:
            if discovery is not None:
                discovery.discard()
            if refresh_started:
                source.quarantine_refresh(
                    owner=self._mcp_refresh_owner,
                    expected_generation=previous_generation,
                    expected_dirty_epoch=refresh_dirty_epoch,
                )
            raise

    def get_agent(self, name: str) -> runtime_records.RegisteredAgent:
        agent_name = require_clean_nonblank(name, "agent.name")
        registered_agent = self._get_registered_agent(agent_name)
        return runtime_records.RegisteredAgent(
            spec=registered_agent.spec.model_copy(deep=True),
            tools={
                name: _copy_registered_tool(tool) for name, tool in registered_agent.tools.items()
            },
            hosted_tools=tuple(
                copy_openai_web_search(tool) for tool in registered_agent.hosted_tools
            ),
        )

    def list_agents(self) -> tuple[str, ...]:
        """Return the names of all registered agents, sorted."""
        return tuple(sorted(self._agents))

    def _get_registered_agent(self, name: str) -> runtime_records.RegisteredAgentState:
        agent_name = require_clean_nonblank(name, "agent.name")
        try:
            return self._agents[agent_name]
        except KeyError as exc:
            raise KeyError(f"Agent not registered: {agent_name}") from exc


def _copy_refreshable_mcp_toolsets(
    value: Iterable[McpToolset] | None,
) -> tuple[McpToolset, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes | bytearray | Mapping):
        raise TypeError("mcp_toolsets must be an iterable of McpToolset instances.")
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise TypeError("mcp_toolsets must be an iterable of McpToolset instances.") from exc
    copied = tuple(islice(iterator, 10_001))
    if len(copied) > 10_000:
        raise ValueError("mcp_toolsets cannot contain more than 10,000 sources.")
    for index, toolset in enumerate(copied):
        if not isinstance(toolset, McpToolset):
            raise TypeError(f"mcp_toolsets[{index}] must be a McpToolset.")
    return copied


def _mcp_refresh_source_key(toolset: McpToolset) -> int:
    if not isinstance(toolset, McpToolset):
        raise TypeError("toolset must be a McpToolset.")
    return id(toolset._refresh_source)


def _registered_agent_contains_mcp_source(
    registered_agent: runtime_records.RegisteredAgentState,
    source: object,
) -> bool:
    return any(toolset._refresh_source is source for toolset in registered_agent.mcp_toolsets)


def _agents_contain_mcp_source(
    agents: Mapping[str, runtime_records.RegisteredAgentState],
    toolset: McpToolset,
) -> bool:
    source = toolset._refresh_source
    return any(
        isinstance(registered.tool, McpToolAdapter)
        and registered.tool.toolset._refresh_source is source
        for agent in agents.values()
        for registered in agent.tools.values()
    )


def _registered_agent_after_mcp_refresh(
    registered_agent: runtime_records.RegisteredAgentState,
    *,
    source: object,
    candidate: McpToolset,
    redactor: SecretRedactor,
    timeout_seconds: float | None,
) -> runtime_records.RegisteredAgentState:
    if not _registered_agent_contains_mcp_source(registered_agent, source):
        raise ValueError("Registered agent does not contain the refreshed MCP source.")
    refreshed_toolsets = tuple(
        candidate if toolset._refresh_source is source else toolset
        for toolset in registered_agent.mcp_toolsets
    )
    dynamic_sources = {_mcp_refresh_source_key(toolset) for toolset in refreshed_toolsets}
    tools_by_name: dict[str, runtime_records.RegisteredTool] = {}
    for name, registered in registered_agent.tools.items():
        tool = registered.tool
        if (
            isinstance(tool, McpToolAdapter)
            and _mcp_refresh_source_key(tool.toolset) in dynamic_sources
        ):
            continue
        tools_by_name[name] = registered
    for toolset in refreshed_toolsets:
        for adapter in sorted(toolset.tools, key=lambda tool: tool.name):
            registered = _validate_registered_tool(
                adapter,
                redactor=redactor,
                timeout_seconds=timeout_seconds,
            )
            if registered.name in tools_by_name:
                raise ValueError(
                    f"Refreshed MCP tool collides with registered tool: {registered.name}"
                )
            previous = registered_agent.tools.get(registered.name)
            if previous is not None and previous.effect_reconciler is not None:
                if (
                    not isinstance(previous.tool, McpToolAdapter)
                    or previous.tool.toolset._refresh_source is not toolset._refresh_source
                    or registered.effect is not ToolEffect.EXTERNAL
                ):
                    raise ValueError("MCP refresh conflicts with a registered effect reconciler.")
                registered = replace(registered, effect_reconciler=previous.effect_reconciler)
            tools_by_name[registered.name] = registered

    if any(
        previous.effect_reconciler is not None and name not in tools_by_name
        for name, previous in registered_agent.tools.items()
    ):
        raise ValueError("MCP refresh removed a tool with a registered effect reconciler.")

    executable_names = frozenset((*tools_by_name, *registered_agent.runtime_tools))
    missing_workflow_tools = tuple(
        name for name in registered_agent.spec.workflow_tool_names if name not in executable_names
    )
    if missing_workflow_tools:
        raise ValueError("MCP refresh removed a configured workflow tool.")
    exposure_policy = registered_agent.tool_exposure_policy
    if isinstance(exposure_policy, StaticToolExposurePolicy) and any(
        name not in tools_by_name for name in exposure_policy.tools
    ):
        raise ValueError("MCP refresh removed a statically exposed tool.")

    descriptors_by_name = {
        tool.name: _registered_tool_descriptor(tool) for tool in tools_by_name.values()
    }
    tool_catalogue = build_tool_catalog_snapshot(descriptors_by_name.values())
    tool_capabilities = tuple(
        RegisteredToolCapability(**descriptors_by_name[name].exposure_capability_material())
        for name in tools_by_name
    )
    all_registered_tool_exposure = ResolvedToolExposure(
        profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
        catalogue_revision=tool_catalogue.revision,
        tools=tool_capabilities,
        registered_count=len(tool_capabilities),
        ceiling_count=len(tool_capabilities),
    )
    return replace(
        registered_agent,
        tools=MappingProxyType(tools_by_name),
        tool_catalogue=tool_catalogue,
        tool_capabilities=all_registered_tool_exposure.tools,
        all_registered_tool_exposure=all_registered_tool_exposure,
        mcp_toolsets=refreshed_toolsets,
    )
