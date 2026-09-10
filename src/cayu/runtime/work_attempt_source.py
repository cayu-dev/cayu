"""Bounded original request data for exact pre-dispatch reconstruction."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.runtime.work_contracts import (
    WORK_CONTRACT_TASK_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_ITEMS,
    require_bounded_work_completion_document,
)

WORK_ATTEMPT_SOURCE_MAX_BYTES = WORK_CONTRACT_TASK_MAX_BYTES
WORK_ATTEMPT_SOURCE_MAX_ITEMS = WORK_CONTRACT_TASK_MAX_ITEMS
WORK_ATTEMPT_SOURCE_MAX_FIELDS = 256


def work_attempt_source_digest(
    *,
    kind: Literal["initial", "continuation"],
    request: dict[str, Any],
    fields_set: tuple[str, ...],
    loop_policy_authority: list[dict[str, Any]],
) -> str:
    """Bind source data and explicit controls, including excluded policy objects."""
    return sha256(
        canonical_durable_json_bytes(
            {
                "kind": kind,
                "request": request,
                "fields_set": list(fields_set),
                "loop_policy_authority": loop_policy_authority,
            },
            "work_attempt_source_request",
        )
    ).hexdigest()


class WorkAttemptSourceRequest(BaseModel):
    """Portable data, never authentication of a caller or invocation.

    Explicit field names matter even when values equal their defaults. Private
    provenance is deliberately not serialized; recovery must resolve it from
    the admission owner. Request-local Python policies cannot be reconstructed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["initial", "continuation"]
    request: dict[str, Any]
    fields_set: tuple[str, ...]
    source_request_sha256: str
    content_sha256: str

    @classmethod
    def capture(
        cls,
        *,
        kind: Literal["initial", "continuation"],
        request: dict[str, Any],
        fields_set: tuple[str, ...],
        source_request_sha256: str,
    ) -> WorkAttemptSourceRequest:
        return cls(
            kind=kind,
            request=request,
            fields_set=fields_set,
            source_request_sha256=source_request_sha256,
            content_sha256=work_attempt_source_digest(
                kind=kind, request=request, fields_set=fields_set, loop_policy_authority=[]
            ),
        )

    @field_validator("source_request_sha256", "content_sha256")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Work-attempt source requires lowercase SHA-256 digests.")
        return value

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> dict[str, Any]:
        require_bounded_work_completion_document(
            value,
            "work_attempt_source_request",
            max_bytes=WORK_ATTEMPT_SOURCE_MAX_BYTES,
            max_items=WORK_ATTEMPT_SOURCE_MAX_ITEMS,
        )
        return copy_durable_json_object(value, "work_attempt_source_request")

    @field_validator("fields_set", mode="before")
    @classmethod
    def require_field_names(cls, value: object) -> tuple[str, ...]:
        if type(value) is not list and type(value) is not tuple:
            raise ValueError("Work-attempt source requires bounded explicit field names.")
        if len(value) > WORK_ATTEMPT_SOURCE_MAX_FIELDS:
            raise ValueError("Work-attempt source requires bounded explicit field names.")
        if any(type(item) is not str or not item.isidentifier() for item in value):
            raise ValueError("Work-attempt source contains invalid field names.")
        if tuple(value) != tuple(sorted(set(value))):
            raise ValueError("Work-attempt source field names must be unique and sorted.")
        return cast("tuple[str, ...]", tuple(value))

    @model_validator(mode="after")
    def require_bounded_source(self) -> WorkAttemptSourceRequest:
        if set(self.fields_set) - self.request.keys() - {"loop_policies"}:
            raise ValueError("Work-attempt source field names conflict with its request.")
        if self.content_sha256 != work_attempt_source_digest(
            kind=self.kind,
            request=self.request,
            fields_set=self.fields_set,
            loop_policy_authority=[],
        ):
            raise ValueError("Work-attempt source conflicts with its content digest.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "work_attempt_source",
            max_bytes=WORK_ATTEMPT_SOURCE_MAX_BYTES,
            max_items=WORK_ATTEMPT_SOURCE_MAX_ITEMS,
        )
        return self

    def require_binding(
        self,
        *,
        kind: str,
        source_request_sha256: str,
        session_id: str,
        task_id: str,
    ) -> None:
        if (
            self.kind != kind
            or self.source_request_sha256 != source_request_sha256
            or self.request.get("session_id") != session_id
            or (kind == "initial" and self.request.get("task_id") != task_id)
        ):
            raise ValueError("Work-attempt source conflicts with its admission authority.")
