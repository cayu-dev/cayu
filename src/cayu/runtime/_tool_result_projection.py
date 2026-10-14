"""Validated tool-result projection and preparation for runtime publication.

Event adapters establish provenance before retaining these values. Transcript
adapters independently validate references and share the content helper below.
Parsing a matching application-authored shape never grants runtime authority.
This owner has no artifact-store or tool-execution collaborator: preparing an
existing projection cannot persist another artifact or repeat a tool call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cayu._validation import copy_durable_json_value, freeze_json_value, thaw_json_value
from cayu.tools.base import ToolResult
from cayu.tools.result_projection import (
    ToolResultProjection,
    ToolResultProjectionRecord,
    ToolResultProjectionStatus,
    _validate_externalized_projection_reference,
    redact_tool_result_projection_content,
    reestimate_tool_result_projection_tokens,
)
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True, slots=True)
class ValidatedToolResultProjection:
    """Detached projection evidence; its caller still owns provenance admission."""

    result: ToolResult = field(repr=False)
    record: ToolResultProjectionRecord = field(repr=False)
    _references: tuple[tuple[int, Mapping[str, Any]], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        projection = ToolResultProjection(result=self.result, record=self.record)
        references: tuple[tuple[int, Mapping[str, Any]], ...] = ()
        if projection.record.status is ToolResultProjectionStatus.EXTERNALIZED:
            reference = _validate_externalized_projection_reference(projection)
            references = tuple(
                (index, freeze_json_value(reference))
                for index, artifact in enumerate(projection.result.artifacts)
                if artifact == reference
            )
        object.__setattr__(self, "result", projection.result)
        object.__setattr__(self, "record", projection.record)
        object.__setattr__(self, "_references", references)

    def references(self) -> dict[int, dict[str, Any]]:
        """Copy strict references for an adapter's schema-aware redaction pass."""

        return {index: thaw_json_value(reference) for index, reference in self._references}

    def record_for_boundary(self, redactor: SecretRedactor | None) -> dict[str, Any]:
        """Omit secret-bearing optional metadata before restoring the record."""

        record = self.record.model_dump(mode="json", exclude_none=True)
        if redactor is None:
            return record
        settlement = record.get("artifact_write_settlement")
        if type(settlement) is not dict:
            return record
        settlement = dict(settlement)
        for name in ("backend_locator", "backend_version"):
            value = settlement.get(name)
            if type(value) is str and redactor.redact_text(value) != value:
                settlement.pop(name, None)
        record["artifact_write_settlement"] = settlement
        return record


@dataclass(frozen=True, slots=True)
class PreparedToolResultProjection:
    """Immutable projection fields; other result fields belong to the adapter."""

    content: str = field(repr=False)
    artifacts: tuple[Mapping[str, Any], ...] = field(repr=False)
    _record: Mapping[str, Any] = field(repr=False)

    def __init__(
        self,
        source: ValidatedToolResultProjection,
        *,
        redacted_result: dict[str, Any],
        redactor: SecretRedactor,
    ) -> None:
        """Prepare validated evidence after the adapter's schema redaction.

        The adapter establishes provenance and supplies owned durable JSON from
        its schema redactor. The source type establishes shape validity, not
        authority. Only externalized results rebuild an artifact-reading
        instruction; other decisions retain the adapter's content. Final byte
        and token accounting waits until the adapter finishes its other policies.
        """

        if type(source) is not ValidatedToolResultProjection:
            raise TypeError("Projection preparation requires a validated projection value.")
        if type(redacted_result) is not dict:
            raise TypeError("Projection preparation requires an object result.")
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        content = redacted_result.get("content")
        artifacts = redacted_result.get("artifacts", [])
        references = source.references()
        if references:
            # ToolResult defaults absent content to ""; retain the boundary's
            # missing-field rejection before reconstructing from that value.
            if type(content) is not str:
                raise AssertionError("Validated projection lost its content or artifacts.")
            content, artifacts = prepare_runtime_reference_content(
                original_content=source.result.content,
                redacted_artifacts=artifacts,
                references=references,
                redactor=redactor,
            )
        if type(content) is not str:
            raise AssertionError("Validated projection lost its content.")
        if type(artifacts) is not list:
            raise AssertionError("Validated projection lost its content or artifacts.")
        if any(type(artifact) is not dict for artifact in artifacts):
            raise TypeError("Prepared projection artifacts must be objects.")
        record = source.record_for_boundary(redactor)
        object.__setattr__(self, "content", content)
        # The schema redactor returns valid, owned JSON. Freeze that snapshot
        # directly; repeating durable admission here adds a full artifact walk.
        object.__setattr__(self, "artifacts", tuple(freeze_json_value(a) for a in artifacts))
        object.__setattr__(self, "_record", freeze_json_value(record))

    def artifact_payload(self) -> list[dict[str, Any]]:
        return [thaw_json_value(artifact) for artifact in self.artifacts]

    def record_for_content(self, content: object) -> dict[str, Any]:
        """Account after an adapter finishes its other terminal-result policies.

        Those policies retain their existing order. Accounting must describe the
        final content, including when another attested result owner restores it.
        """

        if type(content) is not str:
            raise AssertionError("Validated projection lost its content.")
        copied = thaw_json_value(self._record)
        _account_content(copied, content)
        return copied


def read_runtime_tool_result_projection(
    payload: dict[str, Any],
) -> ValidatedToolResultProjection | None:
    """Parse already-owned runtime data without conferring provenance on it.

    Preserve the event boundary's distinction between malformed containers
    (errors) and invalid evidence inside object containers (untrusted fallback).
    """

    record_payload = payload.get("tool_result_projection")
    if record_payload is None:
        return None
    result_payload = payload.get("result")
    if type(record_payload) is not dict or type(result_payload) is not dict:
        raise TypeError("Projected tool-result events require object result and record fields.")
    try:
        return ValidatedToolResultProjection(
            result=ToolResult(**copy_durable_json_value(result_payload, "projected_tool_result")),
            record=ToolResultProjectionRecord.model_validate(
                copy_durable_json_value(record_payload, "tool_result_projection")
            ),
        )
    except (TypeError, ValueError):
        return None


def prepare_runtime_reference_content(
    *,
    original_content: object,
    redacted_artifacts: object,
    references: dict[int, dict[str, Any]],
    redactor: SecretRedactor,
) -> tuple[str, list[object]]:
    """Prepare an adapter's already-validated runtime artifact reference.

    Transcript messages retain the reference without a projection record. Their
    runtime-owned adapter uses this same content preparation operation.
    """

    if type(original_content) is not str or type(redacted_artifacts) is not list:
        raise AssertionError("Validated projection lost its content or artifacts.")
    if len(references) != 1:
        raise AssertionError("Validated projection must own exactly one artifact reference.")
    artifact_index, reference = next(iter(references.items()))
    if artifact_index >= len(redacted_artifacts):
        raise AssertionError("Validated projection artifact position changed during redaction.")
    artifact_id = reference.get("artifact_id")
    if type(artifact_id) is not str:
        raise AssertionError("Validated projection reference lost its artifact identity.")
    readback_max_bytes = reference.get("readback_max_bytes")
    if type(readback_max_bytes) is not int:
        raise AssertionError("Validated projection reference lost its readback bound.")
    content = redact_tool_result_projection_content(
        original_content,
        artifact_id=artifact_id,
        readback_max_bytes=readback_max_bytes,
        redact_text=redactor.redact_text,
    )
    artifacts = list(redacted_artifacts)
    artifacts[artifact_index] = reference
    return content, artifacts


def _account_content(record: dict[str, Any], content: object) -> None:
    if type(content) is not str:
        raise AssertionError("Validated projection lost its content.")
    record["projected_bytes"] = len(content.encode("utf-8"))
    record["projected_token_estimate"], record["token_estimation_method"] = (
        reestimate_tool_result_projection_tokens(
            content, token_estimation_method=record.get("token_estimation_method")
        )
    )
