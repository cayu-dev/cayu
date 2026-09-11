"""A child-linked recovery pause, never a terminal result or dispatch grant."""

from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired
from cayu.runtime.sessions import MAX_SESSION_ID_BYTES
from cayu.runtime.tool_effects import _bounded_text


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
