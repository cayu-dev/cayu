"""CayuApp's registration entrances: agents, providers and outside fork controllers.

``CayuApp`` inherits these; they live here so the application module stays within its size
limit.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from cayu._application_registration import _registration_site
from cayu.fork_controllers import fork_controller_access
from cayu.tasks import session_wakes as _session_wakes

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.context.base import ContextPolicy
    from cayu.environments.admission import ExecutionRequirements
    from cayu.environments.base import NoEnvironment
    from cayu.fork_controllers import ForkController, ForkControllerAccess
    from cayu.managed_settings import ManagedSetting
    from cayu.mcp.tools import McpToolset
    from cayu.observability.hooks import RuntimeHook
    from cayu.providers.base import ModelProvider
    from cayu.providers.hosted import OpenAIWebSearch
    from cayu.runtime.loop_policies import LoopPolicy
    from cayu.runtime.tool_effects import ToolEffectReconciliationRegistration
    from cayu.sessions.child_context import ChildSessionContextContributor
    from cayu.tools.base import Tool
    from cayu.tools.discovery import ToolDiscoveryMode
    from cayu.tools.exposure import ToolExposureMode, ToolExposurePolicy
    from cayu.tools.policy import ToolPolicy
    from cayu.tools.targeted_projection import TargetedToolMode


class RegistrationEntrances:
    """Registering an application's agents and providers."""

    def register_agent(
        self: CayuApp,
        spec: AgentSpec,
        *,
        tools: Iterable[Tool] | None = None,
        proposal_tools: Iterable[Tool] | None = None,
        tool_effect_reconcilers: Mapping[str, ToolEffectReconciliationRegistration] | None = None,
        mcp_toolsets: Iterable[McpToolset] | None = None,
        hosted_tools: Iterable[OpenAIWebSearch] | None = None,
        context_policy: ContextPolicy | None = None,
        context_overflow_policy: ContextPolicy | None = None,
        child_session_context: ChildSessionContextContributor | None = None,
        tool_exposure_policy: ToolExposurePolicy | None = None,
        tool_exposure_mode: ToolExposureMode | str | None = None,
        targeted_tool_mode: TargetedToolMode | str | None = None,
        tool_discovery_mode: ToolDiscoveryMode | str | None = None,
        tool_policy: ToolPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        loop_policies: Iterable[LoopPolicy] | None = None,
        execution_requirements: ExecutionRequirements | None = None,
        environment_name: str | NoEnvironment | None = None,
        settings: Iterable[ManagedSetting] | None = None,
        model_slot: str | None = None,
    ) -> AgentSpec:
        spec, declared = self._agent_managed_settings(
            spec, settings, default_thinking=self._config.run.thinking
        )
        tools, proposal_tools, kits = self._toolkit_slots(tools, proposal_tools)
        connections = {*self._connection_slot_declarations, *kits}
        slot, setting_slot = self._agent_model_slot(spec, model_slot, declared, connections)
        registered = self._agent_registry.register_agent(
            spec,
            default_thinking=self._config.run.thinking,
            default_thinking_source=self._config_sources["run.thinking"],
            registration_site=_registration_site(),
            tools=tools,
            proposal_tools=proposal_tools,
            pause_check=partial(_session_wakes._require_servable_pause, self),
            tool_effect_reconcilers=tool_effect_reconcilers,
            mcp_toolsets=mcp_toolsets,
            hosted_tools=hosted_tools,
            context_policy=context_policy,
            context_overflow_policy=context_overflow_policy,
            child_session_context=child_session_context,
            tool_exposure_policy=tool_exposure_policy,
            tool_exposure_mode=tool_exposure_mode,
            targeted_tool_mode=targeted_tool_mode,
            tool_discovery_mode=tool_discovery_mode,
            tool_policy=tool_policy,
            runtime_hooks=runtime_hooks,
            loop_policies=loop_policies,
            execution_requirements=execution_requirements,
            environment_name=environment_name,
            model_slot=slot,
        )
        self._add_managed_settings(declared)
        self._add_agent_slots(registered.name, setting_slot, kits)
        return registered

    def enroll_fork_controller(self: CayuApp, controller: ForkController) -> None:
        """Let one outside controller fork the named agents with its own overlay."""
        self._agent_registry.enroll_fork_controller(controller, budget_ledger=self.budget_ledger)

    def fork_controller_access(self: CayuApp, subject: str) -> ForkControllerAccess:
        """Return the overlay entrance for a subject the host already authenticated."""
        return fork_controller_access(self, subject, self._agent_registry.fork_controllers)

    def register_provider(
        self: CayuApp,
        provider: ModelProvider,
        *,
        default: bool = False,
        model_patterns: Iterable[str] | None = None,
        settings: Iterable[ManagedSetting] | None = None,
    ) -> ModelProvider:
        declared = self._part_managed_settings(settings, part="provider", name=provider.name)
        registered = self._provider_registry.register(
            provider,
            default=default,
            model_patterns=model_patterns,
            registration_site=_registration_site(),
        )
        self._add_managed_settings(declared)
        return registered
