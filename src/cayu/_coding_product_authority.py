"""Private runtime authority shared by the maintained coding product and Docker binding."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.workspaces.revisions import WorkspaceRevisionObservationLimits

CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY = "cayu.coding_product_source_authority.v1"
CODING_PRODUCT_FINAL_GIT_MAX_CHANGES = 200
CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA = "cayu.final_git_receipt.v1"


class CodingProductSourceCopyAuthority(BaseModel):
    """Exact admitted source snapshot that one Docker copy-in must reproduce."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_fingerprint: str
    source_workspace_id: str
    baseline_revision: str
    observation_limits: WorkspaceRevisionObservationLimits

    @field_validator("request_fingerprint", "baseline_revision")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        raw = value.removeprefix("sha256:")
        if len(value) not in {64, 71} or len(raw) != 64:
            raise ValueError(f"{info.field_name} must be a SHA-256 identity.")
        if any(character not in "0123456789abcdef" for character in raw):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 identity.")
        return value

    @field_validator("source_workspace_id")
    @classmethod
    def validate_source_workspace_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "source_workspace_id")
        if len(value.encode("utf-8")) > 512:
            raise ValueError("source_workspace_id must not exceed 512 UTF-8 bytes.")
        return value


def source_copy_authority_from_metadata(
    metadata: dict[str, Any],
) -> CodingProductSourceCopyAuthority | None:
    """Read and strictly validate optional runtime-injected copy authority."""

    if type(metadata) is not dict:
        raise TypeError("Coding-product source authority metadata must be a dict.")
    value = metadata.get(CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY)
    if value is None:
        return None
    return CodingProductSourceCopyAuthority.model_validate(value)


def is_final_git_result_envelope(value: object, *, mode: str) -> bool:
    """Return whether one trusted final Git result has the exact v1 envelope."""

    base_keys = {
        "mode",
        "scope",
        "changes",
        "returned",
        "offset",
        "limit",
        "truncated",
        "truncation_reasons",
        "next_offset",
    }
    expected_keys = (
        base_keys | {"diff_offset", "next_diff_offset", "binary_omitted"}
        if mode == "diff"
        else base_keys
    )
    if type(value) is not dict or set(value) != expected_keys:
        return False
    value = cast("dict[str, Any]", value)
    changes = value.get("changes")
    returned = value.get("returned")
    limit = value.get("limit")
    truncated = value.get("truncated")
    reasons = value.get("truncation_reasons")
    next_offset = value.get("next_offset")
    if (
        value.get("mode") != mode
        or value.get("scope") != "all"
        or type(changes) is not list
        or type(returned) is not int
        or returned != len(changes)
        or type(limit) is not int
        or limit != CODING_PRODUCT_FINAL_GIT_MAX_CHANGES
        or not 0 <= returned <= limit
        or type(value.get("offset")) is not int
        or value.get("offset") != 0
        or type(truncated) is not bool
        or type(reasons) is not list
        or any(type(reason) is not str or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
        or truncated is not bool(reasons)
        or (next_offset is not None and (type(next_offset) is not int or next_offset < returned))
        or (not truncated and next_offset is not None)
    ):
        return False
    if mode != "diff":
        return True
    next_diff_offset = value.get("next_diff_offset")
    return (
        type(value.get("diff_offset")) is int
        and value.get("diff_offset") == 0
        and (next_diff_offset is None or (type(next_diff_offset) is int and next_diff_offset > 0))
        and type(value.get("binary_omitted")) is bool
    )


__all__ = [
    "CODING_PRODUCT_FINAL_GIT_MAX_CHANGES",
    "CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA",
    "CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY",
    "CodingProductSourceCopyAuthority",
    "is_final_git_result_envelope",
    "source_copy_authority_from_metadata",
]
