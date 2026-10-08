"""Fresh, secret-safe decoding of saved pending tool rounds."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cayu._validation import copy_durable_json_value
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._checkpoint_secret_validation import durable_value_contains_secret
from cayu.sessions.base import SessionStore
from cayu.sessions.checkpoints import _DecodedRuntimeCheckpoint
from cayu.sessions.records import Session
from cayu.vaults.redaction import SecretRedactor, contains_redacted_secret


async def load_pending_tool_round(
    session_store: SessionStore,
    session_id: str,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
    runtime_session: Session | None = None,
) -> tuple[dict[str, Any] | None, pending_rounds.PendingToolRound | None]:
    """Read a fresh round and retain its exact source snapshot for publication.

    Each call loads and validates once with the caller's current context. The
    round is detached from the returned checkpoint; neither is cached. Callers
    already inside a checkpoint transform use the synchronous parser instead.
    """

    checkpoint = None
    copied_checkpoint = None
    decoded = None
    try:
        load_decoded = getattr(session_store, "_load_decoded_runtime_checkpoint", None)
        if callable(load_decoded) and getattr(load_decoded, "__self__", None) is session_store:
            decoded = await load_decoded(session_id)
            if type(decoded) is not _DecodedRuntimeCheckpoint:
                raise TypeError("Runtime checkpoint reader must return a decoded snapshot.")
            checkpoint = decoded.take(session_id=session_id)
        else:
            checkpoint = await session_store.load_checkpoint(session_id)
        if type(consume_on_rejection) is not bool:
            raise TypeError("consume_on_rejection must be a bool.")
        if checkpoint is None:
            return None, None
        if decoded is None:
            copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
        else:
            # The decoder admitted the whole document and transferred its
            # private snapshot without exposing it to a callback. Only the
            # round needs another copy to keep the returned pair detached.
            # Secret/provenance checks below still use this call's context.
            key = pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY
            copied_checkpoint = {key: deepcopy(checkpoint.get(key))}
        return checkpoint, _pending_tool_round_from_owned_checkpoint(
            checkpoint,
            copied_checkpoint,
            redactor=redactor,
            consume_on_rejection=consume_on_rejection,
            runtime_session=runtime_session,
        )
    finally:
        # The parser clears rejected private data. Its loader must not retain
        # the original snapshot in an exception traceback either.
        checkpoint = None
        copied_checkpoint = None
        decoded = None


def pending_tool_round_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
    runtime_session: Session | None = None,
) -> pending_rounds.PendingToolRound | None:
    if type(consume_on_rejection) is not bool:
        raise TypeError("consume_on_rejection must be a bool.")
    if checkpoint is None:
        return None
    copied_checkpoint = copy_durable_json_value(checkpoint, "checkpoint")
    try:
        return _pending_tool_round_from_owned_checkpoint(
            checkpoint,
            copied_checkpoint,
            redactor=redactor,
            consume_on_rejection=consume_on_rejection,
            runtime_session=runtime_session,
        )
    finally:
        # The inner parser clears rejected private data. Do not retain the
        # caller-owned source in this wrapper's exception traceback.
        checkpoint = None
        copied_checkpoint = None


def _pending_tool_round_from_owned_checkpoint(
    checkpoint: dict[str, Any] | None,
    copied_checkpoint: dict[str, Any],
    *,
    redactor: SecretRedactor | None = None,
    consume_on_rejection: bool = False,
    runtime_session: Session | None = None,
) -> pending_rounds.PendingToolRound | None:
    """Parse an immediately owned, validated snapshot; never retain or cache it."""
    value = copied_checkpoint.get(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY)
    if value is None:
        return None
    if redactor is not None and durable_value_contains_secret(
        value,
        redactor=redactor,
        path=(pending_rounds.PENDING_TOOL_ROUND_CHECKPOINT_KEY,),
        runtime_session=runtime_session,
    ):
        # Public callers retain their input by default. Runtime callers opt in
        # to consuming their private checkpoint copy so no outer traceback
        # frame keeps executable secret-bearing state.
        if type(value) is dict:
            value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection and checkpoint is not None:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool-round checkpoint contains a workload secret and cannot be executed."
        ) from None
    if type(value) is not dict:
        raise ValueError("Pending tool round checkpoint must be an object.")
    validation_rejected = False
    try:
        pending_round = pending_rounds.PendingToolRound.model_validate(
            value, context=pending_rounds._OWNED_ROUND_JSON_CONTEXT
        )
    except Exception:
        if redactor is None:
            raise
        validation_rejected = True
    if validation_rejected:
        value.clear()
        value = None
        copied_checkpoint.clear()
        if consume_on_rejection and checkpoint is not None:
            checkpoint.clear()
        checkpoint = None
        raise ValueError(
            "Pending tool-round checkpoint is invalid and cannot be executed."
        ) from None
    _require_executable_pending_tool_round(pending_round)
    return pending_round


def _require_executable_pending_tool_round(
    pending_round: pending_rounds.PendingToolRound,
) -> None:
    has_redacted_arguments = any(
        call.targeted_tool_rejection is None and contains_redacted_secret(call.arguments)
        for call in pending_round.tool_calls
    )
    is_internal_structured_output_round = pending_round.structured_output is not None and any(
        call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls
    )
    if has_redacted_arguments and not is_internal_structured_output_round:
        raise ValueError(
            "Pending tool-round arguments contain a redaction marker and cannot be executed."
        )
