"""Validated tool-result projection and preparation for runtime publication.

Event and transcript adapters establish provenance before retaining these values.
Parsing a matching application-authored shape never grants runtime authority.
This owner has no artifact-store or tool-execution collaborator: preparing an
existing projection cannot persist another artifact or repeat a tool call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from cayu._validation import (
    copy_durable_json_value,
    freeze_json_value,
    require_durable_text,
    thaw_json_value,
)
from cayu.tools.base import ToolResult
from cayu.tools.result_projection import (
    _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS,
    MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES,
    TOOL_RESULT_ARTIFACT_TYPE,
    ToolResultProjection,
    ToolResultProjectionRecord,
    ToolResultProjectionStatus,
    _validate_builtin_tool_result_artifact_reference,
    _validate_externalized_projection_reference,
    redact_tool_result_projection_content,
    reestimate_tool_result_projection_tokens,
)
from cayu.vaults.redaction import SecretRedactor

_INLINE_TRUNCATION_MARKER = "\n[tool result truncated after redaction]"


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

    @property
    def original_content(self) -> str:
        return self.result.content

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
class ValidatedToolResultMessage:
    """Validated references from an admitted runtime transcript result.

    The durable message shape has no projection record. Its runtime adapter
    establishes provenance; matching references in untrusted ingress do not
    enter this path. Reference validation belongs to this owner in both shapes.
    """

    original_content: str = field(repr=False)
    _references: Mapping[int, Mapping[str, Any]] = field(repr=False)

    def __init__(self, content: str, artifacts: list[dict[str, Any]]) -> None:
        references: dict[int, Mapping[str, Any]] = {}
        for index, artifact in enumerate(artifacts):
            if type(artifact) is not dict or artifact.get("type") != TOOL_RESULT_ARTIFACT_TYPE:
                continue
            try:
                reference = _validate_builtin_tool_result_artifact_reference(artifact)
            except (TypeError, ValueError):
                continue
            references[index] = freeze_json_value(reference)
        object.__setattr__(self, "original_content", require_durable_text(content, "content"))
        object.__setattr__(self, "_references", MappingProxyType(references))

    def references(self) -> Mapping[int, Mapping[str, Any]]:
        """Share immutable, already-validated references across boundary checks."""

        return self._references

    def record_for_boundary(self, redactor: SecretRedactor | None) -> None:
        """The durable transcript shape does not carry projection evidence."""

        return None

    def require_secret_free_authority(self, redactor: SecretRedactor, field_name: str) -> None:
        """Apply the stricter admission rule for unverified stored transcripts."""

        for index, reference in self.references().items():
            if redactor.redact_json_values(reference) != reference:
                raise ValueError(
                    f"{field_name}[{index}] contains a workload secret and cannot be "
                    "used as runtime projection authority."
                )


@dataclass(frozen=True, slots=True)
class ToolResultPublication:
    """Final content and durable evidence for the event adapter."""

    content: str = field(repr=False)
    _record: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_record", freeze_json_value(self._record))

    def record_payload(self) -> dict[str, Any]:
        return thaw_json_value(self._record)


@dataclass(frozen=True, slots=True)
class PreparedToolResultProjection:
    """Immutable projection fields; other result fields belong to the adapter."""

    content: str = field(repr=False)
    artifacts: tuple[Mapping[str, Any], ...] = field(repr=False)
    _record: Mapping[str, Any] | None = field(repr=False)
    _redactor: SecretRedactor = field(repr=False)
    _reference: Mapping[str, Any] | None = field(repr=False)

    def __init__(
        self,
        source: ValidatedToolResultProjection | ValidatedToolResultMessage,
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

        if type(source) not in (ValidatedToolResultProjection, ValidatedToolResultMessage):
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
            content, artifacts = _prepare_reference_content(
                original_content=source.original_content,
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
        object.__setattr__(self, "_redactor", redactor)
        object.__setattr__(
            self, "_reference", freeze_json_value(next(iter(references.values()), None))
        )

    def artifact_payload(self) -> list[dict[str, Any]]:
        return [thaw_json_value(artifact) for artifact in self.artifacts]

    def finish(self, content: object) -> ToolResultPublication:
        """Finish content and accounting after other terminal-result policies.

        Those policies retain their existing order. Accounting must describe the
        final content, including when another attested result owner restores it.
        """

        if type(content) is not str:
            raise AssertionError("Validated projection lost its content.")
        if self._record is None:
            raise AssertionError("Only event projections carry publication evidence.")
        if len(content.encode("utf-8")) > MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES:
            # Keep redaction markers atomic so the same registry cannot change
            # the bounded content again on replay. The shared helper omits the
            # notice if appending it would reconstruct a registered secret.
            if self._reference is not None:
                content = redact_tool_result_projection_content(
                    content,
                    artifact_id=self._reference["artifact_id"],
                    readback_max_bytes=self._reference["readback_max_bytes"],
                    redactor=self._redactor,
                )
            else:
                content, _ = self._redactor.redact_text_bounded_with_marker(
                    content,
                    max_bytes=MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES,
                    truncation_marker=self._redactor.redact_text(_INLINE_TRUNCATION_MARKER),
                )
        # Accounting replaces only top-level scalars. Retain immutable
        # nested evidence until the adapter requests its mutable payload.
        record = dict(self._record)
        _account_content(record, content)
        return ToolResultPublication(content, record)


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


def read_runtime_tool_result_message(part: dict[str, Any]) -> ValidatedToolResultMessage | None:
    """Read an already-admitted runtime message, without accepting ingress authority."""

    if part.get("type") != "tool_result":
        return None
    artifacts = part.get("artifacts")
    if type(artifacts) is not list:
        return None
    content = part.get("content")
    if type(content) is not str:
        raise AssertionError("Validated projection lost its content or artifacts.")
    if not artifacts:
        return None
    source = ValidatedToolResultMessage(content, cast("list[dict[str, Any]]", artifacts))
    return source if source._references else None


def require_secret_free_tool_result_artifact_keys(
    artifacts: object,
    *,
    source: ValidatedToolResultMessage | None,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Exempt only a positively validated runtime reference's fixed schema."""

    if type(artifacts) is not list:
        raise AssertionError("Tool result artifacts serialized as a non-list.")
    references = {} if source is None else source.references()
    for index, artifact in enumerate(artifacts):
        if type(artifact) is not dict:
            raise AssertionError("Tool result artifact serialized as a non-object.")
        reference = references.get(index)
        redactor.require_no_secret_keys(
            artifact if reference is None else reference,
            field_name=f"{field_name}[{index}]",
            preserve_keys=()
            if reference is None
            else _BUILTIN_TOOL_RESULT_ARTIFACT_REFERENCE_FIELDS,
            match_short_substrings=True,
        )


def _prepare_reference_content(
    *,
    original_content: object,
    redacted_artifacts: object,
    references: Mapping[int, Mapping[str, Any]],
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
        redactor=redactor,
    )
    artifacts = list(redacted_artifacts)
    artifacts[artifact_index] = thaw_json_value(reference)
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
