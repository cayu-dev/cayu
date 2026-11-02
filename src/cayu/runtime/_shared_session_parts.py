"""What a factory must satisfy to share an allocation, and what each session gets as its own.

Sessions that share one allocation (an ``allocation_scope`` other than
``session``) run on the first session's runner and workspace, and later
sessions never reach the factory's ``create()``. The rule these checks keep: no
session of a scope gets authority (secrets, credentials, egress grants, MCP
tools, knowledge or artifact scope) its own ``create()`` wouldn't have given it.

- Secrets stay refused. Whatever a session resolves from a vault or a
  credential proxy can reach the shared sandbox (a command's environment, a
  file it writes), and every other session's commands run as the same user in
  the same sandbox, so they could read it. A runner gives no per-command or
  per-session process boundary to scope it to.
- Egress authority shares only when the factory says the authority belongs to
  the allocation (``EnvironmentFactory.egress_authority_follows_allocation``).
  One sandbox has one network path to one broker, which can't tell which
  session sent a request, so per-session grants can't be kept apart.
- Artifact and knowledge stores, the knowledge scope and MCP servers live in
  this process, outside the sandbox, so they can be per session: a factory that
  sets ``attaches_session_parts`` gives every attaching session its own through
  ``attach_session_parts``. Without it, the shared result's parts must be the
  factory's fixed configuration.
"""

from __future__ import annotations

from typing import Any

from cayu.environments.base import Environment, copy_environment_spec
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.environments.shared_allocations import SharedSessionParts

_INSTEAD = "Give the environment allocation_scope='session' so each session gets its own."


def require_shareable_factory(factory: EnvironmentFactory) -> None:
    """Refuse a factory whose allocations can't be shared, before asking it for one."""

    if (
        factory.egress_authority_identity is not None
        and factory.egress_authority_follows_allocation is not True
    ):
        raise RuntimeError(
            "This environment factory's egress authority belongs to each session, so it "
            "can't share an allocation: one sandbox has one network path to one broker, "
            "which can't tell which session sent a request, so per-session grants can't "
            "be kept apart. A factory whose grants come only from its fixed "
            "configuration can declare egress_authority_follows_allocation. " + _INSTEAD
        )


def require_shareable_result(
    factory: EnvironmentFactory, environment: Environment, environment_name: str
) -> None:
    """Refuse a shared allocation whose authority might depend on the session that created it."""

    reasons: list[str] = []
    secrets: list[str] = []
    if factory.secret_resolution_scope != "static" and not _secrets_follow_allocation(factory):
        secrets.append("its secret resolution scope is dynamic")
    if environment.vault is not None or environment.proxy is not None:
        secrets.append("it exposes a vault or credential proxy")
    if secrets:
        reasons.append(
            " and ".join(secrets)
            + ", and a secret one session resolves can reach the shared sandbox (a "
            "command's environment, a file), where every other session's commands "
            "could read it"
        )
    if factory.attaches_session_parts is not True:
        fixed: list[str] = []
        if environment.knowledge_store is not getattr(
            factory, "configured_knowledge_store", None
        ) or environment.knowledge_access_scope != getattr(
            factory, "configured_knowledge_access_scope", None
        ):
            fixed.append("its knowledge store or scope isn't the factory's configured one")
        if environment.artifact_store is not factory.configured_artifact_store:
            fixed.append("its artifact store isn't the factory's configured one")
        if environment.mcp_servers:
            fixed.append("it exposes MCP servers")
        if fixed:
            reasons.append(
                "; ".join(fixed)
                + ", so every session of the scope would get the first session's (a "
                "factory can give each session its own by setting attaches_session_parts "
                "and implementing attach_session_parts)"
            )
    if reasons:
        raise RuntimeError(
            f"Environment {environment_name!r} can't share its allocation: "
            + "; ".join(reasons)
            + ". "
            + _INSTEAD
        )


def _secrets_follow_allocation(factory: EnvironmentFactory) -> bool:
    # A virtual-egress factory declares a dynamic scope because its runner
    # holds the grants' virtual credentials. When the egress authority belongs
    # to the allocation, those are the same for every session of the scope and
    # no session resolves anything into the sandbox; a vault or proxy on the
    # result is still refused below.
    return (
        factory.egress_authority_identity is not None
        and factory.egress_authority_follows_allocation is True
    )


async def session_parts(
    factory: EnvironmentFactory,
    request: EnvironmentFactoryRequest,
    shared: EnvironmentFactoryResult,
) -> SharedSessionParts:
    """The parts ``request``'s session gets on the shared allocation ``shared``.

    A factory that attaches session parts is asked with the session's own
    request; otherwise every session gets the shared result's parts, which
    ``require_shareable_result`` checked are the factory's fixed configuration.
    """

    if factory.attaches_session_parts is True:
        parts = await factory.attach_session_parts(request, shared)
        if type(parts) is not SharedSessionParts:
            raise TypeError("attach_session_parts must return SharedSessionParts.")
        return parts
    environment = shared.environment
    return SharedSessionParts(
        artifact_store=environment.artifact_store,
        knowledge_store=environment.knowledge_store,
        knowledge_access_scope=environment.knowledge_access_scope,
        mcp_servers=environment.mcp_servers,
    )


def session_environment(
    shared: Environment,
    parts: SharedSessionParts,
    *,
    bound: BoundWorkspace | None,
    runner: Any,
    binding: WorkspaceBinding,
) -> Environment:
    """One session's view of a shared allocation: the shared runner and workspace, its own parts."""

    return Environment(
        copy_environment_spec(shared.spec),
        workspace=bound.workspace if bound is not None else shared.workspace,
        artifact_store=parts.artifact_store,
        runner=runner,
        vault=shared.vault,
        proxy=shared.proxy,
        knowledge_store=parts.knowledge_store,
        knowledge_access_scope=parts.knowledge_access_scope,
        binding=binding,
        mcp_servers=parts.mcp_servers,
        workspace_instructions=shared.workspace_instructions,
    )
