"""Runtime hooks shared by tests."""

from __future__ import annotations

from cayu.observability.hooks import RuntimeHook, ToolCallHookContext
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity


class NoopAfterToolHook(RuntimeHook):
    """Any after-tool hook keeps a round's results publishing one at a time."""

    @property
    def name(self) -> str:
        return "noop-after-tool"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:noop-after-tool", behavior_version="1", implementation_version="1"
        )

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        del context
