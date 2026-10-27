"""Whether an agent's knowledge tools will find a knowledge store when they run.

A tool call gets its knowledge store from the environment it runs in, never from
the application. Registration and ``cayu check`` use this to warn about an agent
whose knowledge tools would fail every call with ``missing_knowledge_store``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from cayu.environments.base import NO_ENVIRONMENT, NoEnvironment

if TYPE_CHECKING:
    from cayu.runtime._runtime_records import RegisteredEnvironment

KnowledgeSupply = Literal["supplied", "missing", "unknown"]


def is_knowledge_tool(tool: object) -> bool:
    """Whether ``tool`` is one of Cayu's tools that read or write a knowledge store."""

    from cayu.tools.knowledge import (
        ListKnowledgeTool,
        ReadKnowledgeTool,
        RememberKnowledgeTool,
        SearchKnowledgeTool,
    )

    return isinstance(
        tool, ListKnowledgeTool | ReadKnowledgeTool | RememberKnowledgeTool | SearchKnowledgeTool
    )


def environment_knowledge_supply(registered: RegisteredEnvironment) -> KnowledgeSupply:
    """Whether tool calls in this environment get a knowledge store.

    A concrete environment answers directly. A factory answers only when it
    declares ``configured_knowledge_store``, as the Docker and brokered coding
    factories do; any other factory may attach a store per session, so the
    answer is ``unknown``.
    """

    if registered.environment.knowledge_store is not None:
        return "supplied"
    factory = registered.factory
    if factory is None:
        return "missing"
    if getattr(type(factory), "configured_knowledge_store", None) is None:
        return "unknown"
    try:
        store = getattr(factory, "configured_knowledge_store", None)
    except Exception:
        return "unknown"
    return "missing" if store is None else "supplied"


def agent_knowledge_supply(
    default_environment_name: str | NoEnvironment | None,
    *,
    environments: Mapping[str, RegisteredEnvironment],
    app_default_environment: str | None,
    complete: bool,
) -> KnowledgeSupply:
    """The supply in the environment a new run of the agent gets by default.

    ``complete`` says every environment is registered (``cayu check``). During
    registration an environment named but not registered yet is ``unknown``.
    """

    if default_environment_name is NO_ENVIRONMENT:
        return "missing"
    name = default_environment_name or app_default_environment
    if name is None:
        return "missing" if complete else "unknown"
    registered = environments.get(name)
    if registered is None:
        return "missing" if complete else "unknown"
    return environment_knowledge_supply(registered)


def knowledge_store_unavailable_message(agent: str, tools: tuple[str, ...]) -> str:
    return (
        f"Agent {agent} has {', '.join(tools)}, but no knowledge store reaches them, so "
        "every call fails with missing_knowledge_store. Tools get their knowledge store "
        "from the environment they run in, and this agent runs by default in one without "
        "a store (or in none). Pass knowledge_store= to the Environment or factory, or "
        "inherit_app_knowledge=True to register_coding_environment."
    )
