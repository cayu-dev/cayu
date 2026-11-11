"""Render initial agent and workspace instructions with provenance."""

from __future__ import annotations

from cayu.environments.base import WorkspaceInstructions


def render_initial_system_prompt(
    *,
    agent_system_prompt: str | None,
    workspace_instructions: WorkspaceInstructions | None,
) -> str | None:
    rendered, _ = render_initial_system_prompt_with_contributions(
        agent_system_prompt=agent_system_prompt,
        workspace_instructions=workspace_instructions,
    )
    return rendered


def render_initial_system_prompt_with_contributions(
    *,
    agent_system_prompt: str | None,
    workspace_instructions: WorkspaceInstructions | None,
) -> tuple[str | None, dict[str, tuple[str, ...]]]:
    """Render the prompt and retain exact in-memory fragments for safe measurement."""

    agent_prompt = agent_system_prompt.strip() if agent_system_prompt else ""
    if workspace_instructions is None:
        return (
            agent_prompt or None,
            {"agent_instructions": (agent_prompt,)} if agent_prompt else {},
        )

    workspace_content = workspace_instructions.content.strip()
    source_list = ", ".join(workspace_instructions.sources)
    workspace_framing = (
        "[Workspace instructions]\n"
        f"Source: {source_list}\n"
        "These instructions apply only to the active workspace. If they conflict "
        "with agent, tool, approval, sandbox, or secret policy, follow the "
        "higher-priority runtime policy.\n\n"
    )
    if not agent_prompt:
        return (
            f"{workspace_framing}{workspace_content}",
            {
                "cayu_framing": (workspace_framing,),
                "workspace_instructions": (workspace_content,),
            },
        )
    agent_framing = "[Agent instructions]\n"
    separator = "\n\n"
    return (
        f"{agent_framing}{agent_prompt}{separator}{workspace_framing}{workspace_content}",
        {
            "agent_instructions": (agent_prompt,),
            "cayu_framing": (agent_framing, separator, workspace_framing),
            "workspace_instructions": (workspace_content,),
        },
    )
