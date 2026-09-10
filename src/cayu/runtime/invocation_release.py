"""Portable evidence of a release validated by the invocation lifecycle owner."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank


class InvocationReleaseEvidence(BaseModel):
    """A bounded receipt projection, not caller authority to release an invocation.

    The runtime obtains this value from the lifecycle owner's exact readback.
    A downstream settlement must bind every field to its expected invocation;
    accepting an arbitrary caller-created instance does not establish cleanup.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.invocation-release-evidence"] = "cayu.invocation-release-evidence"
    schema_version: Literal[1] = 1
    session_id: str = Field(max_length=1024)
    session_instance_id: str = Field(max_length=1024)
    interaction_id: str = Field(max_length=1024)
    command_identity: str = Field(max_length=4096)
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_epoch: StrictInt = Field(ge=1, lt=MAX_DURABLE_JSON_INTEGER)
    released_run_epoch: StrictInt = Field(ge=2, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("session_id", "session_instance_id", "interaction_id", "command_identity")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_release_epoch(self) -> InvocationReleaseEvidence:
        if self.released_run_epoch != self.run_epoch + 1:
            raise ValueError("Invocation release evidence requires the exact successor epoch.")
        return self
