"""Publication-safe projections of model-authored tool arguments.

Raw arguments are executable private state.  A start event is emitted before an
invocation can discover every workload secret, so it may carry only a
quarantine marker.  Arguments become publishable only after the invocation's
secret tracker has been sealed; abnormal recovery without that tracker remains
explicitly unavailable instead of guessing from the application redactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.vaults import SecretRedactor

ARGUMENTS_STATE_FIELD = "arguments_state"
ARGUMENTS_FIELD = "arguments"
ArgumentPublicationState = Literal["quarantined", "finalized", "unavailable"]
TERMINAL_ARGUMENT_STATES = frozenset({"finalized", "unavailable"})


@dataclass(frozen=True, slots=True)
class ToolArgumentProjection:
    """One immutable public projection of a private argument object."""

    state: Literal["finalized", "unavailable"]
    arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state == "finalized":
            if type(self.arguments) is not dict:
                raise TypeError("Finalized tool arguments must be an object.")
            object.__setattr__(
                self,
                "arguments",
                copy_durable_json_object(self.arguments, "arguments"),
            )
            return
        if self.arguments is not None:
            raise ValueError("Unavailable tool arguments cannot carry an argument object.")

    def payload_fields(self) -> dict[str, Any]:
        payload: dict[str, Any] = {ARGUMENTS_STATE_FIELD: self.state}
        if self.arguments is not None:
            payload[ARGUMENTS_FIELD] = copy_durable_json_object(
                self.arguments,
                "arguments",
            )
        return payload

    def transcript_arguments(self) -> dict[str, Any]:
        """Return the only argument object safe for model-visible history."""

        return (
            {} if self.arguments is None else copy_durable_json_object(self.arguments, "arguments")
        )


def quarantined_argument_fields() -> dict[str, str]:
    """Return the only argument evidence safe before invocation dispatch."""

    return {ARGUMENTS_STATE_FIELD: "quarantined"}


def finalized_argument_projection(
    arguments: dict[str, Any],
    *,
    redactor: SecretRedactor,
    scope_finalized: bool = True,
) -> ToolArgumentProjection:
    """Project arguments after positive evidence that secret discovery ended."""

    if type(arguments) is not dict:
        raise TypeError("Tool arguments must be an object.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if not scope_finalized:
        return unavailable_argument_projection()
    projected = redactor.redact_json(arguments)
    if type(projected) is not dict:
        raise AssertionError("Tool argument projection returned a non-object.")
    return ToolArgumentProjection(state="finalized", arguments=projected)


def unavailable_argument_projection() -> ToolArgumentProjection:
    """Return a fail-closed projection when no sealed invocation scope exists."""

    return ToolArgumentProjection(state="unavailable")


def started_arguments_match_private_call(
    payload: dict[str, Any],
    *,
    private_arguments: dict[str, Any],
) -> bool:
    """Accept new quarantined starts and exact legacy argument-bearing starts."""

    state = payload.get(ARGUMENTS_STATE_FIELD)
    if state is None:
        arguments = payload.get(ARGUMENTS_FIELD)
        return type(arguments) is dict and canonical_durable_json_bytes(
            arguments,
            "started_arguments",
        ) == canonical_durable_json_bytes(private_arguments, "private_arguments")
    return state == "quarantined" and ARGUMENTS_FIELD not in payload


def pause_checkpoint_validation_view(
    value: Any,
) -> tuple[dict[str, Any], bool]:
    """Return a model-compatible copy of one durable pause descriptor.

    Current pause events intentionally omit every executable argument object.
    Their private checkpoint remains authoritative; this view supplies empty
    placeholders solely so the existing typed model can validate the durable
    identity and configuration fields. Legacy descriptors with arguments are
    returned unchanged. Mixed or malformed quarantine states fail closed.
    """

    checkpoint = copy_durable_json_object(value, "pause_checkpoint")
    tool_calls = checkpoint.get("tool_calls")
    if type(tool_calls) is not list:
        raise ValueError("Pause checkpoint tool_calls must be a list.")

    descriptors = [checkpoint, *tool_calls]
    quarantined: list[bool] = []
    for descriptor in descriptors:
        if type(descriptor) is not dict:
            raise ValueError("Pause checkpoint tool calls must be objects.")
        state = descriptor.pop(ARGUMENTS_STATE_FIELD, None)
        if state is None:
            quarantined.append(False)
            continue
        if state != "quarantined" or ARGUMENTS_FIELD in descriptor:
            raise ValueError("Pause checkpoint has an invalid argument quarantine state.")
        descriptor[ARGUMENTS_FIELD] = {}
        quarantined.append(True)

    if any(quarantined) and not all(quarantined):
        raise ValueError("Pause checkpoint mixes private and quarantined arguments.")
    if "input_id" in checkpoint and "question" not in checkpoint:
        # Dynamic-secret user-input events withhold their prompt.  Recovery
        # validates only durable pause identity here; the private checkpoint,
        # not this fixed model-compatible placeholder, remains authoritative
        # for the actual question and options.
        checkpoint["question"] = "Input required"
        checkpoint["options"] = []
    return checkpoint, bool(quarantined and quarantined[0])


def terminal_argument_projection(
    payload: dict[str, Any],
    *,
    legacy_arguments: dict[str, Any],
) -> ToolArgumentProjection:
    """Parse one terminal projection, retaining compatibility with old events."""

    state = payload.get(ARGUMENTS_STATE_FIELD)
    if state is None:
        return ToolArgumentProjection(
            state="finalized",
            arguments=copy_durable_json_object(legacy_arguments, "legacy_arguments"),
        )
    if state == "finalized":
        arguments = payload.get(ARGUMENTS_FIELD)
        if type(arguments) is not dict:
            raise ValueError("Finalized terminal arguments must be an object.")
        return ToolArgumentProjection(state="finalized", arguments=arguments)
    if state == "unavailable" and ARGUMENTS_FIELD not in payload:
        return unavailable_argument_projection()
    raise ValueError("Terminal event has an invalid tool-argument publication state.")
