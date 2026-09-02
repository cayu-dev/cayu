from __future__ import annotations

from typing import Any

from cayu._validation import require_clean_nonblank
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runtime.child_session_context import (
    CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS,
    CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS,
    CHILD_SESSION_RESULT_REFERENCE_VERSION,
    ChildSessionResultReference,
)
from cayu.runtime.child_session_results import (
    DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS,
    MAX_CHILD_SESSION_RESULT_MAX_CHARS,
    ChildSessionResultUnavailable,
    project_terminal_child_session_result,
)
from cayu.runtime.sessions import SessionStore
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation


class ChildSessionResultTool(Tool):
    """Resolve a notification reference under the calling parent's authority."""

    def __init__(
        self,
        session_store: SessionStore,
        *,
        name: str = "child_session_result",
        description: str | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> None:
        if not isinstance(session_store, SessionStore):
            raise TypeError("ChildSessionResultTool requires a SessionStore.")
        self._session_store = session_store
        super().__init__(
            ToolSpec(
                name=require_clean_nonblank(name, "name"),
                effect=ToolEffect.NONE,
                description=description
                or (
                    "Retrieve the bounded final result for one terminal child-session "
                    "notification. The reference does not authorize another parent or "
                    "sibling to read the child."
                ),
                execution_profile_identity=execution_profile_identity,
                input_schema={
                    "type": "object",
                    "properties": {
                        "reference": {
                            "type": "object",
                            "properties": {
                                "schema_version": {
                                    "type": "string",
                                    "const": CHILD_SESSION_RESULT_REFERENCE_VERSION,
                                },
                                "resolver": {
                                    "type": "string",
                                    "const": "child_session_result",
                                },
                                "child_session_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": CHILD_SESSION_PUBLIC_ALIAS_MAX_CHARS,
                                    "pattern": (
                                        r"^cayu_authority_v1\.[a-z][a-z0-9_-]{0,31}"
                                        r"\.session_id\.[A-Za-z0-9_-]{43}$"
                                    ),
                                },
                                "terminal_occurrence_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": CHILD_SESSION_PUBLIC_OCCURRENCE_ID_MAX_CHARS,
                                    "pattern": r"^cayu_child_occurrence_v1_[0-9a-f]{64}$",
                                },
                            },
                            "required": [
                                "schema_version",
                                "resolver",
                                "child_session_id",
                                "terminal_occurrence_id",
                            ],
                            "additionalProperties": False,
                        },
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_CHILD_SESSION_RESULT_MAX_CHARS,
                            "default": DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS,
                        },
                    },
                    "required": ["reference"],
                    "additionalProperties": False,
                },
            )
        )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        with tool_argument_validation():
            reference = ChildSessionResultReference.model_validate(args.get("reference"))
            max_chars = args.get(
                "max_chars",
                DEFAULT_CHILD_SESSION_RESULT_MAX_CHARS,
            )
            if type(max_chars) is not int:
                raise ValueError("max_chars must be an integer.")
            if not 1 <= max_chars <= MAX_CHILD_SESSION_RESULT_MAX_CHARS:
                raise ValueError(
                    f"max_chars must be between 1 and {MAX_CHILD_SESSION_RESULT_MAX_CHARS}."
                )
        try:
            result = await project_terminal_child_session_result(
                self._session_store,
                parent_session_id=ctx.session_id,
                reference=reference,
                max_chars=max_chars,
            )
        except ChildSessionResultUnavailable:
            return ToolResult(
                content="Child-session result is unavailable to this session.",
                structured={
                    "child_session_id": reference.child_session_id,
                    "terminal_occurrence_id": reference.terminal_occurrence_id,
                    "retrieval_status": "unavailable",
                },
                is_error=True,
            )
        material = result.model_dump(mode="json")
        return ToolResult(
            content=result.result_text or f"Child session ended with status {result.state.value}.",
            structured={**material, "retrieval_status": "ready"},
            is_error=result.state.value != "completed",
        )


__all__ = ["ChildSessionResultTool"]
