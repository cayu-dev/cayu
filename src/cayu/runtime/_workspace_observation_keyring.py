"""Check that workspace-mutating tools can record their observation identity.

A workspace-mutating tool writes its workspace observation intent before it
runs. When the environment resolves secrets dynamically, the configured
workspace, observer and artifact-store identities cannot be proven secret-free
at that point, so the intent stores them only as keyed aliases. That needs the
session store's public authority alias codec. Persistent stores have no codec
unless the deployment configures a keyring. A concrete environment's workspace
is known at registration, so registration fails there. A factory's workspace is
known only once it creates the environment, so that check runs at
materialization, before the first model call, instead of at the first
``write_file`` or ``exec_command``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

from cayu.environments.base import Environment
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._invocation_secrets import registered_environment_secret_resolution_scope

WORKSPACE_OBSERVATION_KEYRING_UNAVAILABLE = "workspace_observation_keyring_unavailable"

_KEYRING_FIX = (
    "give the session store a persistent public authority alias keyring "
    "(public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)) and pass the same "
    "keyring to CayuApp(public_authority_alias_keyring=keyring); "
    "public_authority_alias_codec_from_environment() reads one from "
    "CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID and CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS"
)
_STATIC_FACTORY_FIX = (
    'declare secret_resolution_scope = "static" on the environment factory if it never '
    "returns a vault or credential proxy"
)


def environment_needs_workspace_observation_alias_codec(
    registered_environment: runtime_records.RegisteredEnvironment,
) -> bool:
    """Whether a concrete registration's workspace tools need a keyed alias codec.

    Factory registrations return ``False``: whether a factory result has a
    workspace is unknown until it is created, and one without a workspace never
    records an observation. ``require_materialized_workspace_observation_keyring``
    checks the created environment instead.
    """

    if registered_environment.factory is not None or registered_environment.factory_backed:
        return False
    if registered_environment_secret_resolution_scope(registered_environment) != "dynamic":
        return False
    return registered_environment.environment.workspace is not None


def workspace_mutating_tool_names(
    registered_agent: runtime_records.RegisteredAgentState,
) -> tuple[str, ...]:
    """Names of the agent's registered tools that mutate the workspace."""

    return tuple(
        sorted(name for name, tool in registered_agent.tools.items() if tool.workspace_mutation)
    )


def workspace_observation_keyring_message(
    *,
    tool_names: Iterable[str],
    environment_name: str | None,
    factory_backed: bool,
    agent_name: str | None = None,
) -> str:
    """Explain the missing keyring and both ways to fix it."""

    names = tuple(tool_names)
    subject = ("Tool " if len(names) == 1 else "Tools ") + ", ".join(repr(name) for name in names)
    if agent_name is not None:
        subject += f" of agent {agent_name!r}"
    environment = "the environment" if environment_name is None else repr(environment_name)
    scope_reason = (
        "comes from an environment factory that does not declare a static secret scope"
        if factory_backed
        else "has a vault or credential proxy"
    )
    fixes = (
        f"Either {_STATIC_FACTORY_FIX}, or {_KEYRING_FIX}."
        if factory_backed
        else f"To fix it, {_KEYRING_FIX}."
    )
    return (
        f"{subject} cannot record workspace observations in environment {environment}: "
        f"the environment {scope_reason}, so its workspace identity must be stored as a "
        "keyed alias, and the session store has no public authority alias keyring. "
        f"{fixes} CayuApp() without a session_store does not show this, because its "
        "in-memory store makes its own key."
    )


def require_workspace_observation_keyring(
    *,
    agents: Mapping[str, runtime_records.RegisteredAgentState],
    environments: Mapping[str, runtime_records.RegisteredEnvironment],
    alias_codec_configured: bool,
) -> None:
    """Fail registration when a workspace-mutating tool could never record its intent.

    Agents are not bound to environments at registration, so any registered
    agent can run in any registered concrete environment. Check every pair.
    """

    if alias_codec_configured:
        return
    registration = next(
        (
            registration
            for registration in environments.values()
            if environment_needs_workspace_observation_alias_codec(registration)
        ),
        None,
    )
    if registration is None:
        return
    for agent_name, registered_agent in agents.items():
        tool_names = workspace_mutating_tool_names(registered_agent)
        if tool_names:
            raise RuntimeError(
                workspace_observation_keyring_message(
                    tool_names=tool_names,
                    environment_name=registration.spec.name,
                    factory_backed=False,
                    agent_name=agent_name,
                )
            )


def require_materialized_workspace_observation_keyring(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str,
    environment: Environment,
    factory_secret_resolution_scope: str,
    alias_codec_configured: bool,
    permitted_tool_names: Collection[str] | None = None,
) -> None:
    """Fail a session whose factory-created workspace its tools could never observe.

    Runs when a dynamic factory creates a fresh environment, before it is bound
    and before the first model call. ``permitted_tool_names`` is the session's
    tool ceiling; tools outside it can't run, so they don't count.
    """

    if (
        alias_codec_configured
        or factory_secret_resolution_scope == "static"
        or environment.workspace is None
    ):
        return
    tool_names = tuple(
        name
        for name in workspace_mutating_tool_names(registered_agent)
        if permitted_tool_names is None or name in permitted_tool_names
    )
    if tool_names:
        raise RuntimeError(
            workspace_observation_keyring_message(
                tool_names=tool_names,
                environment_name=environment_name,
                factory_backed=True,
                agent_name=registered_agent.spec.name,
            )
        )
