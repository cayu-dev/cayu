"""The one-sentence description of Cayu shown where people and coding agents look first.

`cayu guide` (the index) and generated `AGENTS.md` files render these values. The
repository README repeats them verbatim; a documentation test keeps the copies equal.
"""

from __future__ import annotations

CAYU_IDENTITY = (
    "Cayu is a Python runtime for long-running agents: durable sessions that survive "
    "restarts and redeploys, human approval before side effects, spend caps enforced "
    "before each model call, and multi-agent collaboration."
)

# (capability, the Cayu parts that provide it)
CAYU_CAPABILITIES = (
    (
        "Durable sessions and recovery",
        "`SQLiteSessionStore` or `PostgresSessionStore` (`cayu.storage`) with "
        "`ResumeRequest`; `ExecutionProfilePolicy` (`cayu.runtime`) governs resumes on a "
        "new build",
    ),
    (
        "Approval before side effects",
        "`ToolPolicy` returning `ToolPolicyDecision.REQUIRE_APPROVAL` (`cayu.tools`), "
        "resolved through `cayu.approvals`",
    ),
    (
        "Spend caps",
        "`BudgetLimit` per request and `BudgetPolicy` per app (`cayu.budgets`), checked "
        "before each model call",
    ),
    (
        "Timers and background work",
        "`TaskStore` workers and `TaskSchedulePolicy` for scheduled follow-ups (`cayu.tasks`)",
    ),
    (
        "Multi-agent collaboration",
        "`SubagentTool` (`cayu.tools`), forks, and participants with durable requests "
        "(`cayu.collaboration`)",
    ),
    (
        "Claude Code or Codex runs",
        "`run_external_agent` with `ClaudeAgentAdapter` or `CodexAgentAdapter` "
        "(`cayu.external_agents`, extras `cayu[claude-agent-sdk]` and `cayu[codex]`) "
        "and a launcher such as `local_agent_launch`, recorded as sessions with usage "
        "and deadlines; see `cayu guide external-agents`",
    ),
)

CAYU_IS_NOT = (
    "Cayu is Python only and is not a graph DSL: agents, tools and workflows are "
    "ordinary Python. It is not only a hosted service: run it on your own "
    "infrastructure or deploy to the optional Cayu Cloud."
)


def capability_table() -> str:
    """Return the capability list as the Markdown table the README also uses."""

    rows = "".join(f"| {capability} | {parts} |\n" for capability, parts in CAYU_CAPABILITIES)
    return "| Capability | Cayu parts |\n| --- | --- |\n" + rows


def identity_markdown() -> str:
    """Return the identity sentence, capability table and boundary line as Markdown."""

    return f"{CAYU_IDENTITY}\n\n{capability_table()}\n{CAYU_IS_NOT}\n"
