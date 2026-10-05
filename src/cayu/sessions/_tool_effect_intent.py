"""Exact saved identity of an effective tool call, shared with checkpoint evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from cayu.runtime.tool_effects import _bounded_text
from cayu.sessions.base import MAX_SESSION_ID_BYTES


class ToolEffectIntent(BaseModel):
    """Immutable identity of the actual effective call, not its model proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr
    session_instance_id: StrictStr
    source_run_epoch: StrictInt = Field(ge=0)
    interaction_id: StrictStr
    model_step_id: StrictStr
    model_attempt_id: StrictStr
    tool_round_id: StrictStr
    tool_call_id: StrictStr
    agent_name: StrictStr
    tool_name: StrictStr
    idempotency_key: StrictStr
    effect: Literal["external"] = "external"
    execution_profile_fingerprint: StrictStr
    schema_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    arguments_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: StrictStr | None = None
    pause_id: StrictStr | None = None
    environment_name: StrictStr | None = None
    allocation_fingerprint: StrictStr | None = None
    reconciler_fingerprint: StrictStr | None = None
    targeted_invocation_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("*", mode="after")
    @classmethod
    def bound_identity(cls, value, info):
        if isinstance(value, str):
            return _bounded_text(
                value,
                info.field_name,
                maximum=MAX_SESSION_ID_BYTES if info.field_name == "session_id" else 256,
                identifier=True,
            )
        return value
