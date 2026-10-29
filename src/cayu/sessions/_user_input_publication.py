"""Closed user-input publication and checkpoint authority validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType
from cayu.messages import MessageRole, ToolCallPart, ToolResultPart
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
)
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)


def _validate_user_input_checkpoint_mutation(
    request: RuntimePublicationRequest,
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    session_instance_id: str,
    current_run_epoch: int,
    durable_events_by_id: Mapping[str, Event] | None = None,
) -> None:
    """Validate the closed atomic publication contract for user-input pauses."""

    if request.kind in {"wake-open", "wake-close"}:
        _validate_wake_checkpoint_mutation(
            request,
            checkpoint,
            session_id=session_id,
            session_instance_id=session_instance_id,
            current_run_epoch=current_run_epoch,
        )
        return
    if request.kind not in {"user-input-open", "user-input-close"}:
        return
    from cayu.approvals.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
        USER_INPUT_SUPERSESSION_INTENT_KEY,
        PendingUserInput,
        UserInputResolutionIntent,
        pending_user_input_identity,
        require_resolution_intent_matches_pending,
    )

    if request.operation_record_mutations:
        raise ValueError("A user-input publication cannot mutate session-operation records.")
    operations = {operation.key: operation for operation in request.mutation.operations}
    schema_operation = operations.get(CHECKPOINT_SCHEMA_VERSION_KEY)
    if schema_operation is not None:
        supported_schema_digests = {
            runtime_publication_checkpoint_value_digest(version)
            for version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION + 1)
        }
        if (
            schema_operation.action != "set"
            or schema_operation.value != CURRENT_CHECKPOINT_SCHEMA_VERSION
            or schema_operation.expected_value_digest not in supported_schema_digests | {None}
        ):
            raise ValueError("User-input checkpoint schema publication is invalid.")
    identity_fields = {
        "schema_version",
        "session_id",
        "session_instance_id",
        "source_interaction_id",
        "source_run_epoch",
        "input_id",
        "tool_call_id",
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
        "execution_profile_fingerprint",
        "pause_digest",
    }
    extra_fields = (
        {"source_round_digest", "event_ids"}
        if request.kind == "user-input-open"
        else {
            "claim_run_epoch",
            "answer_request_digest",
            "execution_state",
            "resolution_request_digest",
            "tool_call_ids",
            "event_ids",
            "referenced_event_ids",
        }
    )
    intent = request.intent
    if request.kind == "user-input-close" and "post_action_continuation_digest" in intent:
        extra_fields.add("post_action_continuation_digest")
    if request.kind == "user-input-close" and "foreground_parent_continuation" in intent:
        extra_fields.add("foreground_parent_continuation")
    if set(intent) != identity_fields | extra_fields or intent.get("schema_version") != 1:
        raise ValueError("User-input publication intent is malformed.")
    if (
        intent.get("session_id") != session_id
        or intent.get("session_instance_id") != session_instance_id
        or request.interaction_id != intent.get("source_interaction_id")
        or request.publication_id != f"{request.kind}:{intent.get('input_id')}"
    ):
        raise ValueError("User-input publication identity conflicts with its session.")
    for field_name in (
        "session_instance_id",
        "source_interaction_id",
        "input_id",
        "tool_call_id",
        "tool_round_id",
        "model_step_id",
        "model_attempt_id",
    ):
        require_clean_nonblank(cast("str", intent.get(field_name)), f"intent.{field_name}")
    if type(intent.get("source_run_epoch")) is not int or intent["source_run_epoch"] < 0:
        raise ValueError("User-input publication source_run_epoch is invalid.")
    for field_name in ("execution_profile_fingerprint", "pause_digest"):
        session_authority_rules._require_raw_sha256_digest(intent.get(field_name))
    event_ids = intent.get("event_ids")
    if (
        type(event_ids) is not list
        or len(event_ids) != len(set(event_ids))
        or any(type(event_id) is not str or not event_id for event_id in event_ids)
        or event_ids != [event.id for event in request.events]
    ):
        raise ValueError("User-input publication event identities are malformed.")

    current = {} if checkpoint is None else checkpoint
    active_profile = active_invocation_execution_profile_from_checkpoint(current)
    if (
        active_profile is None
        or not active_invocation_execution_profile_matches_session_epoch(
            active_profile,
            session_id=session_id,
            run_epoch=current_run_epoch,
        )
        or active_profile.interaction_id != intent["source_interaction_id"]
        or active_profile.profile.fingerprint != intent["execution_profile_fingerprint"]
    ):
        raise SessionRuntimePublicationConflict(
            "User-input publication has no exact active invocation authority."
        )
    if request.kind == "user-input-open":
        if request.transcript_messages or request.referenced_events:
            raise ValueError("Opening user input cannot append transcript or reference events.")
        opening_keys = {
            tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY,
            PENDING_USER_INPUT_CHECKPOINT_KEY,
        }
        if set(operations) not in {
            frozenset(opening_keys),
            frozenset(opening_keys | {CHECKPOINT_SCHEMA_VERSION_KEY}),
        }:
            raise ValueError(
                "Opening user input must replace exactly one pending tool round; "
                f"got {sorted(operations)}."
            )
        if any(
            key in current
            for key in (
                PENDING_USER_INPUT_CHECKPOINT_KEY,
                USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
                USER_INPUT_SUPERSESSION_INTENT_KEY,
            )
        ):
            raise SessionRuntimePublicationConflict(
                "A session already carries user-input lifecycle authority."
            )
        raw_round = current.get(tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY)
        round_operation = operations[tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY]
        pending_operation = operations[PENDING_USER_INPUT_CHECKPOINT_KEY]
        if (
            type(raw_round) is not dict
            or round_operation.action != "delete"
            or round_operation.expected_value_digest
            != runtime_publication_checkpoint_value_digest(raw_round)
            or intent.get("source_round_digest")
            != runtime_publication_checkpoint_value_digest(raw_round)
            or pending_operation.action != "set"
            or pending_operation.expected_value_digest is not None
            or type(pending_operation.value) is not dict
        ):
            raise ValueError("User-input opening mutation is not bound to its source round.")
        try:
            pending = PendingUserInput.model_validate(pending_operation.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("User-input opening checkpoint is malformed.") from exc
        if any(
            intent.get(key) != value for key, value in pending_user_input_identity(pending).items()
        ):
            raise ValueError("User-input opening checkpoint conflicts with its intent.")
        if (
            pending.session_id != session_id
            or pending.session_instance_id != session_instance_id
            or pending.source_interaction_id != request.interaction_id
            or pending.source_run_epoch != current_run_epoch
            or pending.source_run_epoch != intent["source_run_epoch"]
            or raw_round.get("tool_round_id") != pending.tool_round_id
            or raw_round.get("model_step_id") != pending.model_step_id
            or raw_round.get("model_attempt_id") != pending.model_attempt_id
            or raw_round.get("source_run_epoch") != pending.source_run_epoch
            or raw_round.get("execution_profile_fingerprint")
            != pending.execution_profile_fingerprint
        ):
            raise ValueError("User-input opening authority conflicts with its pending round.")
        if len(request.events) != 2 or [event.type for event in request.events] != [
            EventType.SESSION_CHECKPOINTED,
            EventType.SESSION_AWAITING_USER_INPUT,
        ]:
            raise ValueError("User-input opening requires checkpoint and awaiting events.")
        checkpoint_event, awaiting_event = request.events
        if (
            checkpoint_event.payload.get("checkpoint") != PENDING_USER_INPUT_CHECKPOINT_KEY
            or awaiting_event.payload.get("execution_profile_fingerprint")
            != intent["execution_profile_fingerprint"]
            or any(
                event.payload.get(field_name) != intent[field_name]
                for event in request.events
                for field_name in ("source_run_epoch", "pause_digest")
            )
        ):
            raise ValueError("User-input opening events conflict with their pause authority.")
    else:
        closing_keys = {
            PENDING_USER_INPUT_CHECKPOINT_KEY,
            USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
        }
        from cayu.sessions._foreground_child_checkpoint import (
            FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
            ForegroundChildPostActionContinuation,
            post_action_continuation_round_from_checkpoint,
        )

        close_continuation = operations.get(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
        if close_continuation is not None:
            marker = ForegroundChildPostActionContinuation.model_validate(close_continuation.value)
            round_evidence = post_action_continuation_round_from_checkpoint(current)
            if (
                close_continuation.action != "set"
                or marker.close_publication_id != request.publication_id
                or marker.wait.child_session_id != session_id
                or marker.wait.child_session_instance_id != session_instance_id
                or marker.wait.child_interaction_id != request.interaction_id
                or marker.action_kind != "user_input"
                or marker.action_id != intent.get("input_id")
                or round_evidence is None
                or marker.pending_tool_round != round_evidence.model_dump(mode="json")
                or marker.completed_model_step
                != current[PENDING_USER_INPUT_CHECKPOINT_KEY].get("model_step")
                or marker.continuation_revision != marker.wait.revision
            ):
                raise ValueError("User-input close continuation conflicts with its durable round.")
            closing_keys.add(FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY)
        if "foreground_parent_continuation" in operations:
            from cayu.sessions._foreground_child_checkpoint import gate_close_continuation

            parent_marker = operations["foreground_parent_continuation"]
            round_evidence = post_action_continuation_round_from_checkpoint(current)
            if round_evidence is None or parent_marker.action != "set":
                raise ValueError("User-input gate closure lacks its original round.")
            expected_parent = gate_close_continuation(
                current,
                pending=round_evidence,
                publication_id=request.publication_id,
                metadata=parent_marker.value["request"]["metadata"],
            )
            if expected_parent != parent_marker.value:
                raise ValueError("User-input gate closure has conflicting continuation semantics.")
            for key in ("foreground_child_wait", "foreground_child_terminal"):
                if key not in operations or operations[key].action != "delete":
                    raise ValueError(
                        "User-input gate closure must retire its exact delegated wait."
                    )
            closing_keys.update(
                {
                    "foreground_parent_continuation",
                    "foreground_child_wait",
                    "foreground_child_terminal",
                }
            )
        if set(operations) not in {
            frozenset(closing_keys),
            frozenset(closing_keys | {CHECKPOINT_SCHEMA_VERSION_KEY}),
        }:
            raise ValueError("Closing user input must clear exactly its pause and answer claim.")
        try:
            raw_pending = current[PENDING_USER_INPUT_CHECKPOINT_KEY]
            raw_resolution_intent = current[USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY]
            pending = PendingUserInput.model_validate(raw_pending)
            resolution_intent = UserInputResolutionIntent.model_validate(raw_resolution_intent)
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionRuntimePublicationConflict(
                "User-input closure has no exact active pause and answer claim."
            ) from exc
        require_resolution_intent_matches_pending(resolution_intent, pending=pending)
        if any(
            intent.get(key) != value for key, value in pending_user_input_identity(pending).items()
        ):
            raise ValueError("User-input closure intent conflicts with its active pause.")
        if (
            pending.session_id != session_id
            or pending.session_instance_id != session_instance_id
            or pending.source_interaction_id != request.interaction_id
            or resolution_intent.claim_run_epoch != current_run_epoch
            or intent.get("claim_run_epoch") != resolution_intent.claim_run_epoch
            or intent.get("answer_request_digest") != resolution_intent.answer_request_digest
            or intent.get("execution_state") != resolution_intent.execution_state
            or resolution_intent.execution_state != "executing"
            or intent.get("resolution_request_digest")
            != resolution_intent.resolution_request_digest
        ):
            raise ValueError("User-input closure has invalid resolution authority.")
        session_authority_rules._require_raw_sha256_digest(intent["answer_request_digest"])
        session_authority_rules._require_raw_sha256_digest(intent["resolution_request_digest"])
        expected_call_ids = [call.tool_call_id for call in pending.tool_calls]
        if intent.get("tool_call_ids") != expected_call_ids:
            raise ValueError("User-input closure call identities conflict with its pause.")
        referenced_event_ids = intent.get("referenced_event_ids")
        if type(referenced_event_ids) is not list or referenced_event_ids != list(
            publication_contracts._runtime_publication_referenced_event_ids(
                request.referenced_events
            )
        ):
            raise ValueError("User-input closure referenced events conflict with its intent.")
        for key, raw_value in (
            (PENDING_USER_INPUT_CHECKPOINT_KEY, raw_pending),
            (USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY, raw_resolution_intent),
        ):
            operation = operations[key]
            if (
                operation.action != "delete"
                or operation.expected_value_digest
                != runtime_publication_checkpoint_value_digest(raw_value)
            ):
                raise ValueError("User-input closure is not fenced to its exact checkpoint.")
        if len(request.events) != 1 or request.events[0].type is not EventType.SESSION_CHECKPOINTED:
            raise ValueError("User-input closure requires one checkpoint event.")
        closure_event = request.events[0]
        if (
            closure_event.payload.get("checkpoint") != PENDING_USER_INPUT_CHECKPOINT_KEY
            or closure_event.payload.get("transition") != "answered"
            or closure_event.payload.get("source_run_epoch") != intent["source_run_epoch"]
            or closure_event.payload.get("input_id") != pending.input_id
            or closure_event.payload.get("tool_call_id") != pending.tool_call_id
            or closure_event.payload.get("pause_digest") != intent["pause_digest"]
            or closure_event.payload.get("resolution_request_digest")
            != intent["resolution_request_digest"]
        ):
            raise ValueError("User-input closure event conflicts with its intent.")
        expected_message_count = 2 if pending.assistant_message_state == "quarantined" else 1
        if len(request.transcript_messages) != expected_message_count:
            raise ValueError("User-input closure transcript conflicts with its assistant state.")
        result_message = request.transcript_messages[-1]
        result_parts = [part for part in result_message.content if isinstance(part, ToolResultPart)]
        if result_message.role is not MessageRole.TOOL or len(result_parts) != len(
            result_message.content
        ):
            raise ValueError("User-input closure must end with grouped tool results.")
        if [part.tool_call_id for part in result_parts] != expected_call_ids:
            raise ValueError("User-input closure result order conflicts with its pause.")
        if expected_message_count == 2:
            assistant_message = request.transcript_messages[0]
            assistant_call_ids = [
                part.tool_call_id
                for part in assistant_message.content
                if isinstance(part, ToolCallPart)
            ]
            if (
                assistant_message.role is not MessageRole.ASSISTANT
                or assistant_call_ids != expected_call_ids
            ):
                raise ValueError(
                    "User-input closure assistant projection conflicts with its pause."
                )
        if durable_events_by_id is not None:
            if set(durable_events_by_id) != set(referenced_event_ids):
                raise ValueError("User-input closure is missing durable lifecycle evidence.")
            for event in durable_events_by_id.values():
                if (
                    event.type not in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                    or event.payload.get("input_id") != pending.input_id
                    or event.payload.get("tool_call_id") not in expected_call_ids
                    or event.payload.get("tool_round_id") != pending.tool_round_id
                    or event.payload.get("model_step_id") != pending.model_step_id
                    or event.payload.get("model_attempt_id") != pending.model_attempt_id
                ):
                    raise ValueError(
                        "User-input closure references conflicting lifecycle evidence."
                    )

    for event in request.events:
        if (
            event.session_id != session_id
            or event.interaction_id != request.interaction_id
            or event.payload.get("input_id") != intent["input_id"]
            or event.payload.get("tool_call_id") != intent["tool_call_id"]
            or event.payload.get("tool_round_id") != intent["tool_round_id"]
            or event.payload.get("model_step_id") != intent["model_step_id"]
            or event.payload.get("model_attempt_id") != intent["model_attempt_id"]
        ):
            raise ValueError("User-input publication event conflicts with its authority tuple.")


def _validate_wake_checkpoint_mutation(
    request: RuntimePublicationRequest,
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    session_instance_id: str,
    current_run_epoch: int,
) -> None:
    """Validate the closed atomic publication contract for ``sleep_until`` pauses.

    ``wake-open`` replaces exactly one pending tool round with ``pending_wake``.
    ``wake-close`` replaces exactly that ``pending_wake`` with its continuation
    under ``pending_user_input``, which the shared resolution then claims.
    """

    from cayu.approvals.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
        USER_INPUT_SUPERSESSION_INTENT_KEY,
        pending_user_input_identity,
    )
    from cayu.approvals.wakes import (
        PENDING_WAKE_CHECKPOINT_KEY,
        PendingSessionWake,
        pending_session_wake_digest,
        pending_session_wake_identity,
    )

    if request.operation_record_mutations or request.transcript_messages:
        raise ValueError("A wake publication cannot mutate records or the transcript.")
    if request.referenced_events:
        raise ValueError("A wake publication cannot reference other events.")
    operations = {operation.key: operation for operation in request.mutation.operations}
    schema_operation = operations.pop(CHECKPOINT_SCHEMA_VERSION_KEY, None)
    if schema_operation is not None and (
        schema_operation.action != "set"
        or schema_operation.value != CURRENT_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Wake checkpoint schema publication is invalid.")
    opening = request.kind == "wake-open"
    source_key = (
        tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
        if opening
        else PENDING_WAKE_CHECKPOINT_KEY
    )
    target_key = PENDING_WAKE_CHECKPOINT_KEY if opening else PENDING_USER_INPUT_CHECKPOINT_KEY
    if set(operations) != {source_key, target_key}:
        raise ValueError(f"A {request.kind} publication must replace {source_key} only.")
    current = {} if checkpoint is None else checkpoint
    if any(
        key in current
        for key in (
            target_key,
            USER_INPUT_RESOLUTION_INTENT_CHECKPOINT_KEY,
            USER_INPUT_SUPERSESSION_INTENT_KEY,
        )
    ) or (opening and PENDING_USER_INPUT_CHECKPOINT_KEY in current):
        raise SessionRuntimePublicationConflict("A session already carries pause authority.")
    raw_source = current.get(source_key)
    source_operation, target_operation = operations[source_key], operations[target_key]
    if (
        type(raw_source) is not dict
        or source_operation.action != "delete"
        or source_operation.expected_value_digest
        != runtime_publication_checkpoint_value_digest(raw_source)
        or target_operation.action != "set"
        or target_operation.expected_value_digest is not None
        or type(target_operation.value) is not dict
    ):
        raise ValueError("Wake publication is not bound to its exact source state.")
    try:
        wake = PendingSessionWake.model_validate(target_operation.value if opening else raw_source)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wake publication checkpoint is malformed.") from exc
    pending = wake.continuation
    if not opening and target_operation.value != pending.model_dump(mode="json"):
        raise ValueError("Wake closure must hand over its exact continuation.")
    intent = request.intent
    # Closing opens the continuation's shared pause, so its receipt carries that
    # pause's identity and binds the wake it closes.
    identity = (
        pending_session_wake_identity(wake)
        if opening
        else {
            **pending_user_input_identity(pending),
            "wake_pause_digest": pending_session_wake_digest(wake),
        }
    )
    expected_fields = set(identity) | {"event_ids"}
    expected_fields |= {"source_round_digest"} if opening else {"transition"}
    if (
        set(intent) != expected_fields
        or any(intent.get(key) != value for key, value in identity.items())
        or request.publication_id != f"{request.kind}:{wake.wake_id}"
        or request.interaction_id != pending.source_interaction_id
        or pending.session_id != session_id
        or pending.session_instance_id != session_instance_id
    ):
        raise ValueError("Wake publication identity conflicts with its session.")
    if opening:
        if (
            intent["source_round_digest"] != runtime_publication_checkpoint_value_digest(raw_source)
            or pending.source_run_epoch != current_run_epoch
            or raw_source.get("tool_round_id") != pending.tool_round_id
            or raw_source.get("model_step_id") != pending.model_step_id
            or raw_source.get("model_attempt_id") != pending.model_attempt_id
            or raw_source.get("source_run_epoch") != pending.source_run_epoch
            or raw_source.get("execution_profile_fingerprint")
            != pending.execution_profile_fingerprint
        ):
            raise ValueError("Wake opening conflicts with its pending round.")
        active_profile = active_invocation_execution_profile_from_checkpoint(current)
        if (
            active_profile is None
            or not active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session_id,
                run_epoch=current_run_epoch,
            )
            or active_profile.interaction_id != pending.source_interaction_id
            or active_profile.profile.fingerprint != pending.execution_profile_fingerprint
        ):
            raise SessionRuntimePublicationConflict(
                "Wake opening has no exact active invocation authority."
            )
        transition = "asleep"
    else:
        transition = intent["transition"]
        if transition != "woke":
            raise ValueError("Wake closure transition is invalid.")
    event_ids = intent.get("event_ids")
    if (
        len(request.events) != 1
        or event_ids != [event.id for event in request.events]
        or request.events[0].type is not EventType.SESSION_CHECKPOINTED
    ):
        raise ValueError("A wake publication requires exactly one checkpoint event.")
    event = request.events[0]
    if (
        event.session_id != session_id
        or event.interaction_id != request.interaction_id
        or event.payload.get("checkpoint") != PENDING_WAKE_CHECKPOINT_KEY
        or event.payload.get("transition") != transition
        or event.payload.get("wake_id") != wake.wake_id
        or event.payload.get("tool_call_id") != pending.tool_call_id
        or event.payload.get("pause_digest") != identity["pause_digest"]
        or event.payload.get("source_run_epoch") != pending.source_run_epoch
        or (not opening and event.payload.get("input_id") != pending.input_id)
    ):
        raise ValueError("Wake publication event conflicts with its authority.")
