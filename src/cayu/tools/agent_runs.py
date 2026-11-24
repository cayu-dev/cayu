"""Let a tool run another registered agent, with file inputs, from its ToolContext.

A tool opts in by naming the agents it may run in ``Tool.runs_agents``. During
its call the runtime binds ``ToolContext.agents`` to an :class:`AgentRuns`
handle for exactly those agents. Each run is a session of its own with the
request limits and budget the call passes, inside the parent's causal budget.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cayu._validation import copy_durable_metadata, require_clean_nonblank
from cayu.environments.base import NO_ENVIRONMENT
from cayu.messages import FilePart, Message, MessageRole

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.budgets.base import BudgetLimit
    from cayu.budgets.run_limits import RunLimits
    from cayu.context.structured_output import StructuredOutputSpec
    from cayu.sessions.outcomes import RunOutcome
    from cayu.tools.base import ToolContext

AGENT_RUN_SESSION_PREFIX = "tool-agent-run-"


@dataclass(frozen=True)
class AgentFile:
    """One file to send to the agent: an image (jpeg/png/gif/webp) or a PDF document."""

    content: bytes = field(repr=False)
    filename: str
    kind: str = "image"
    content_type: str | None = None


def agent_run_session_id(
    *, parent_session_id: str, idempotency_key: str, agent_name: str, key: str
) -> str:
    """The session id for one agent run of one tool call.

    It is the same for a retried call, so a call that runs again after its
    agent already finished gets that finished run back instead of paying twice.
    """

    digest = hashlib.sha256(
        "\x00".join((parent_session_id, idempotency_key, agent_name, key)).encode()
    ).hexdigest()
    return AGENT_RUN_SESSION_PREFIX + digest[:40]


class AgentRuns:
    """Runs the agents a tool declared in ``Tool.runs_agents``, during one call."""

    def __init__(self, app: CayuApp, ctx: ToolContext, agent_names: frozenset[str]) -> None:
        self._app = app
        self._ctx = ctx
        self._agent_names = agent_names

    @property
    def agent_names(self) -> frozenset[str]:
        return self._agent_names

    async def run(
        self,
        agent_name: str,
        messages: str | Sequence[Message],
        *,
        files: Sequence[AgentFile] = (),
        key: str = "run",
        structured_output: StructuredOutputSpec | None = None,
        max_steps: int | None = None,
        limits: RunLimits | None = None,
        budget_limits: Sequence[BudgetLimit] = (),
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunOutcome:
        """Run ``agent_name`` in its own session and return its outcome.

        ``files`` are stored as attachments of that session and added to the
        last user message. ``key`` tells apart several runs in one tool call.
        The session id comes from the tool call, so a retried call returns the
        earlier run's outcome when it finished (see ``cayu.sessions.run_once``);
        one that is still running raises ``RunOnceSessionNotFinished``.
        """

        from cayu.budgets.base import copy_request_budget_limits
        from cayu.sessions.idempotent_runs import run_once
        from cayu.sessions.requests import RunRequest, run_request_with_runtime_generated_authority

        agent_name = require_clean_nonblank(agent_name, "agent_name")
        if agent_name not in self._agent_names:
            raise PermissionError(
                f"This tool may not run agent {agent_name!r}; add it to the tool's runs_agents."
            )
        key = require_clean_nonblank(key, "key")
        ctx = self._ctx
        if ctx.idempotency_key is None:
            raise RuntimeError("Agent runs need the runtime's tool call identity.")
        session_id = agent_run_session_id(
            parent_session_id=ctx.session_id,
            idempotency_key=ctx.idempotency_key,
            agent_name=agent_name,
            key=key,
        )
        request_messages = (
            [Message.text("user", messages)] if isinstance(messages, str) else list(messages)
        )
        environment_name = self._environment_for(
            agent_name, environment_name, needs_files=bool(files)
        )
        if files and await self._app.session_store.load(session_id) is None:
            parts = [
                await self._app.attach_file(
                    file.content,
                    filename=file.filename,
                    kind=file.kind,
                    content_type=file.content_type,
                    environment_name=environment_name,
                    session_id=session_id,
                    agent_name=agent_name,
                )
                for file in files
            ]
            request_messages = _with_files(request_messages, parts)
        run_metadata = {
            **copy_durable_metadata(metadata or {}, "metadata"),
            "tool_agent_run": {
                "parent_session_id": ctx.session_id,
                "tool_call_id": ctx.metadata.get("tool_call_id"),
                "key": key,
            },
        }
        fields: dict[str, Any] = {
            "agent_name": agent_name,
            "session_id": session_id,
            "causal_budget_id": ctx.causal_budget_id or ctx.session_id,
            "messages": request_messages,
            "environment_name": environment_name,
            "metadata": run_metadata,
            # The parent's causal limits come first, so the run stays inside them.
            "budget_limits": (
                *copy_request_budget_limits(ctx._causal_budget_limits_for_builtin()),
                *budget_limits,
            ),
            "structured_output": structured_output,
        }
        if max_steps is not None:
            fields["max_steps"] = max_steps
        if limits is not None:
            fields["limits"] = limits
        request = run_request_with_runtime_generated_authority(
            RunRequest(**fields), "causal_budget_id"
        )
        return await run_once(self._app, request)

    def _environment_for(
        self, agent_name: str, environment_name: str | None, *, needs_files: bool
    ) -> str | None:
        """The environment the run will use, so its files are stored where it reads them.

        The same order a new run resolves: the request, then the agent's own
        default, then the application default (``None`` here).
        """

        if environment_name is not None:
            return environment_name
        registered = self._app._agents.get(agent_name)
        default = None if registered is None else registered.default_environment_name
        if default is NO_ENVIRONMENT:
            if needs_files:
                raise ValueError(
                    f"Agent {agent_name!r} runs without an environment, so it has no "
                    "artifact store to hold files."
                )
            return None
        return default


def _with_files(messages: list[Message], parts: list[FilePart]) -> list[Message]:
    if messages and messages[-1].role == MessageRole.USER:
        last = messages[-1]
        return [*messages[:-1], Message(role=last.role, content=(*last.content, *parts))]
    return [*messages, Message(role=MessageRole.USER, content=tuple(parts))]


def declared_agent_runs(tool: object) -> frozenset[str]:
    """The agent names a tool declared in ``runs_agents``, validated."""

    declared = getattr(tool, "runs_agents", ())
    if isinstance(declared, str) or not isinstance(declared, tuple | list | frozenset | set):
        raise TypeError("Tool.runs_agents must be a tuple of agent names.")
    return frozenset(require_clean_nonblank(name, "runs_agents") for name in declared)
