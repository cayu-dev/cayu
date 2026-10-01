from __future__ import annotations

from typing import Any

from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

ASK_USER_TOOL_NAME = "ask_user"


class UserInputTool(Tool):
    """Built-in tool that pauses the session to ask the user a question.

    Opt-in: register it explicitly on the agents that should be able to ask. When the
    model calls it, the runtime intercepts the call before execution, checkpoints the
    question, emits ``session.awaiting_user_input``, and interrupts the session; the
    caller resumes with ``CayuApp.resolve_user_input(...)`` and the answer becomes this
    tool's result. Because the runtime handles the pause, ``run`` is never invoked on the
    normal path; it returns an error only if the tool is somehow executed directly.
    """

    # Decoupled marker the runtime checks (via getattr) to recognize a pausing tool
    # without importing this class — cayu.tools depends on cayu.runtime, not the reverse.
    pauses_session = True

    spec = ToolSpec(
        name=ASK_USER_TOOL_NAME,
        description=(
            "Ask the user a question and wait for their answer before continuing. Use "
            "this only when you genuinely need input you cannot obtain otherwise. Provide "
            "a clear question; optionally provide a list of choices."
        ),
        # A pause must not run alongside other tools in a concurrent batch.
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r"\S",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
                    "description": "Optional list of choices to present to the user.",
                },
            },
            "required": ["question"],
        },
    )

    def _execution_profile_material(self) -> dict[str, object]:
        return {}

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return ToolResult(
                content="Invalid ask_user arguments: 'question' must be a non-blank string.",
                is_error=True,
            )
        # A valid question can reach direct execution after another tool's approval
        # took precedence over the user-input pause in this round.
        return ToolResult(
            content=(
                "ask_user did not pause this round: another tool in the round required "
                "approval first (approval takes precedence). If you "
                "still need the user's input, call ask_user again in a later round."
            ),
            is_error=True,
        )
