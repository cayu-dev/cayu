"""Non-secret durable references to application authorization.

Possession of this record is not admission authority. Only trusted application
policy configuration can reconstruct an execution from a stored record.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import require_durable_clean_nonblank


class ResourceExecutionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    authority: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=512)
    admitted_json: str = Field(min_length=2, max_length=65536)
    policy_revision: int = Field(
        default=0, ge=0, le=9007199254740991, strict=True, exclude_if=lambda v: v == 0
    )

    @field_validator("authority", "subject")
    @classmethod
    def validate_identity(cls, value, info):
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("admitted_json")
    @classmethod
    def validate_admitted(cls, value):
        from cayu.resource_access import decode_grant, encode_grant

        # Validate the complete bounded predicate, not just syntactic JSON.
        validated = encode_grant(decode_grant(json.loads(value)))
        if len(validated) > 65536:
            raise ValueError("Durable access grant exceeds 65536 characters.")
        return validated
