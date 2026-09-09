"""Exact invocation identity for cooperative, complete-tool-round stopping.

This control does not enqueue input or change the meaning of ``next_turn``.
Acceptance is not terminal settlement: the execution owner must first finish
its current model/tool operation and publish the ordinary terminal boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank


class SessionSteeringConflict(ValueError):
    """The named interaction or its previously accepted request conflicts."""

    def __init__(self) -> None:
        super().__init__("Session steering authority or accepted request conflicts.")


class StopAfterCurrentToolRoundRequest(BaseModel):
    """Request a cooperative stop of exactly one current interaction.

    The caller observes these identities before requesting control. A changed
    run epoch before acceptance is a conflict, not permission to stop its
    replacement. An accepted request follows recovery of the same interaction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr = Field(min_length=1, max_length=512)
    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    interaction_id: StrictStr = Field(min_length=1, max_length=512)
    expected_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    idempotency_key: StrictStr = Field(min_length=1, max_length=512)

    @field_validator("session_id", "session_instance_id", "interaction_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class SessionSteeringReceipt(BaseModel):
    """Durable acceptance evidence; does not claim the execution has stopped."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    request: StopAfterCurrentToolRoundRequest
    execution_profile_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported session steering receipt schema.")
        return value

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> StopAfterCurrentToolRoundRequest:
        if isinstance(value, StopAfterCurrentToolRoundRequest):
            return copy_stop_after_current_tool_round_request(value)
        return StopAfterCurrentToolRoundRequest.model_validate(value)


def copy_stop_after_current_tool_round_request(
    request: StopAfterCurrentToolRoundRequest,
) -> StopAfterCurrentToolRoundRequest:
    """Validate caller-owned fields without invoking a model serializer."""

    if type(request) is not StopAfterCurrentToolRoundRequest:
        raise TypeError("Session steering requires a StopAfterCurrentToolRoundRequest.")
    return StopAfterCurrentToolRoundRequest(
        session_id=request.session_id,
        session_instance_id=request.session_instance_id,
        interaction_id=request.interaction_id,
        expected_run_epoch=request.expected_run_epoch,
        idempotency_key=request.idempotency_key,
    )
