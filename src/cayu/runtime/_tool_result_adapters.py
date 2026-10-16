"""Translate admitted event and transcript containers for projection preparation.

The projector owns validation, redaction, reference reconstruction and accounting.
These adapters retain container omissions and the event boundary's legacy fallback.
They do not issue authority or persist artifacts.
"""

from collections.abc import Mapping
from typing import Any

from cayu.runtime._tool_result_projection import (
    PreparedToolResultProjection,
    ValidatedToolResultMessage,
    ValidatedToolResultProjection,
)
from cayu.vaults.redaction import SecretRedactor


def prepare_tool_result_event(
    *,
    redacted_payload: dict[str, Any],
    projection: ValidatedToolResultProjection | None,
    controls: Mapping[str, Any],
    redactor: SecretRedactor,
) -> PreparedToolResultProjection | None:
    """Reinsert prepared fields at the event's existing publication stage."""

    if projection is None:
        return None
    if "tool_result_projection" not in controls and not projection.references():
        # A malformed sibling control can discard inline evidence on exposure.
        # Retain that fallback; externalized references still need preparation.
        return None
    result = redacted_payload.get("result")
    if type(result) is not dict:
        raise AssertionError("Validated projection lost its result object.")
    prepared = PreparedToolResultProjection(projection, redacted_result=result, redactor=redactor)
    result["content"] = prepared.content
    if "artifacts" in result:
        result["artifacts"] = prepared.artifact_payload()
    return prepared


def finish_tool_result_event(
    payload: dict[str, Any],
    *,
    controls: Mapping[str, Any],
    prepared: PreparedToolResultProjection | None,
) -> None:
    """Reinsert evidence after the event's other terminal-result policies."""

    if type(controls.get("tool_result_projection")) is not dict:
        return
    result = payload.get("result")
    record = payload.get("tool_result_projection")
    if type(result) is not dict or type(record) is not dict or prepared is None:
        raise AssertionError("Validated projection lost its result or evidence record.")
    publication = prepared.finish(result.get("content"))
    evidence = publication.record_payload()
    result["content"] = publication.content
    record.clear()
    record.update(evidence)


def prepare_tool_result_message(
    part: dict[str, Any],
    *,
    projection: ValidatedToolResultMessage | None,
    redactor: SecretRedactor,
) -> None:
    """Reinsert prepared fields in an admitted runtime transcript result."""

    if projection is None:
        return
    prepared = PreparedToolResultProjection(projection, redacted_result=part, redactor=redactor)
    part["content"] = prepared.content
    part["artifacts"] = prepared.artifact_payload()
