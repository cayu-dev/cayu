from __future__ import annotations

from cayu import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    ToolContext,
)

_ALLOWED_QA_COMMANDS = {
    "pytest",
    "python",
    "python3",
    "uv",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "node",
    "make",
    "go",
    "cargo",
    "tox",
    "ruff",
    "mypy",
    "jest",
}
_DENYLISTED_TOKENS = {"rm", "curl", "wget", "sudo", "ssh", "git", "scp"}


class QaCommandPolicy(CommandPolicy):
    """Allow only recognized test/build invocations; deny raw shell strings."""

    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        command = request.command
        if command.kind == "shell":
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason="Raw shell strings are not allowed for QA; use kind='process' with argv.",
            )
        argv = command.argv or []
        if not argv or argv[0] not in _ALLOWED_QA_COMMANDS:
            got = argv[0] if argv else "<empty>"
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY, reason=f"'{got}' is not an allowed QA command."
            )
        if any(token in _DENYLISTED_TOKENS for token in argv):
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY, reason="Command contains a disallowed token."
            )
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)
