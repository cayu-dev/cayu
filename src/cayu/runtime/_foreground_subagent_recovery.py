"""A child-linked recovery pause, never a terminal result or dispatch grant."""

from typing import TYPE_CHECKING

from cayu.core.tools import ToolResult
from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired
from cayu.runtime.sessions import MAX_SESSION_ID_BYTES
from cayu.runtime.tool_effects import _bounded_text

if TYPE_CHECKING:
    from cayu.runtime._child_session_identity import ChildSessionRecoveryMatcher
    from cayu.runtime.sessions import Session


async def project_authenticated_child_result(
    matcher: "ChildSessionRecoveryMatcher",
    child: "Session",
    *,
    tool_call_id: str,
    tool_name: str,
    tool_round_id: str,
) -> ToolResult:
    """Shared live/recovery projection after the caller authenticates the child."""
    projected = await matcher.project_recoverable_child(child.model_copy(deep=True))
    if projected is not None:
        if type(projected) is not ToolResult:
            raise TypeError("Child result projection must return a ToolResult or None.")
        return projected.model_copy(deep=True)
    from cayu.runtime._tool_round_recovery import recovered_subagent_tool_result

    return recovered_subagent_tool_result(
        tool_call_id=tool_call_id, tool_name=tool_name, tool_round_id=tool_round_id, child=child
    )


class ForegroundSubagentRecoveryRequired(ToolEffectReconciliationRequired):
    """An authenticated foreground child must terminalize before parent recovery."""

    def __init__(self, *, child_session_id: str, tool_round_id: str, tool_call_id: str):
        RuntimeError.__init__(
            self,
            "A foreground child requires interruption or recovery before parent continuation.",
        )
        self.child_session_id = _bounded_text(
            child_session_id, "child_session_id", maximum=MAX_SESSION_ID_BYTES, identifier=True
        )
        self.tool_round_id = _bounded_text(
            tool_round_id, "tool_round_id", maximum=256, identifier=True
        )
        self.tool_call_id = _bounded_text(
            tool_call_id, "tool_call_id", maximum=256, identifier=True
        )

    def interruption_evidence(self) -> dict[str, object]:
        return {
            "recovery_required": "foreground_subagent",
            "pending_subagent_session_ids": [self.child_session_id],
            "tool_round_id": self.tool_round_id,
            "tool_call_id": self.tool_call_id,
        }
