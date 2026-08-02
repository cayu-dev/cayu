"""Secret-safe ingress preparation for session-owned requests and state."""

from __future__ import annotations

from typing import Any

from cayu.core.messages import Message
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
    redact_untrusted_message_for_boundary,
)
from cayu.runtime.approvals import ResolutionActor
from cayu.runtime.public_authority import public_authority_alias_is_reserved
from cayu.runtime.sessions import (
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    ForkSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    copy_compact_session_request,
    copy_enqueue_session_message_request,
    copy_fork_session_request,
    copy_interrupt_session_request,
    copy_resume_request,
    copy_run_request,
    run_request_authority_is_runtime_generated,
)
from cayu.runtime.structured_output import require_secret_free_structured_output_spec
from cayu.vaults import SecretRedactor


def prepare_run_request(
    request: RunRequest,
    *,
    redactor: SecretRedactor,
) -> RunRequest:
    """Return a request safe to use as the source of a durable session."""

    request = copy_run_request(request)
    if (
        request.session_id is not None
        and public_authority_alias_is_reserved(request.session_id)
        and not run_request_authority_is_runtime_generated(
            request,
            field_name="session_id",
            value=request.session_id,
        )
    ):
        raise ValueError("session_id uses the reserved public-authority alias namespace.")
    require_secret_free_structured_output_spec(
        request.structured_output,
        redactor=redactor,
        field_name="RunRequest.structured_output",
    )
    for field_name in (
        "agent_name",
        "task_worker_id",
        "provider_name",
        "model",
        "environment_name",
    ):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
        )
    for field_name in (
        "session_id",
        "task_id",
        "parent_session_id",
        "causal_budget_id",
    ):
        value = getattr(request, field_name)
        require_runtime_generated_or_secret_free_session_authority(
            value,
            runtime_generated=bool(
                value is not None
                and run_request_authority_is_runtime_generated(
                    request,
                    field_name=field_name,
                    value=value,
                )
            ),
            field_name=field_name,
            redactor=redactor,
        )
    for key, value in request.labels.items():
        if redactor.redact_text(key) != key or redactor.redact_text(value) != value:
            raise ValueError(
                "labels contain a workload secret and cannot be used as durable session authority."
            )
    return request.model_copy(
        update={
            "messages": redact_messages(
                request.messages,
                redactor=redactor,
                field_name="messages",
            ),
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            ),
        },
    )


def prepare_resume_request(
    request: ResumeRequest,
    *,
    redactor: SecretRedactor,
    store_resolved_session_id: str | None = None,
) -> ResumeRequest:
    request = copy_resume_request(request)
    require_secret_free_structured_output_spec(
        request.structured_output,
        redactor=redactor,
        field_name="ResumeRequest.structured_output",
    )
    require_store_resolved_or_secret_free_session_authority(
        request.session_id,
        store_resolved_value=store_resolved_session_id,
        field_name="session_id",
        redactor=redactor,
    )
    for field_name in ("model",):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
        )
    return request.model_copy(
        update={
            "messages": redact_messages(
                request.messages,
                redactor=redactor,
                field_name="messages",
            ),
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            ),
        },
    )


def prepare_compact_session_request(
    request: CompactSessionRequest,
    *,
    redactor: SecretRedactor,
    store_resolved_session_id: str | None = None,
) -> CompactSessionRequest:
    request = copy_compact_session_request(request)
    require_store_resolved_or_secret_free_session_authority(
        request.session_id,
        store_resolved_value=store_resolved_session_id,
        field_name="session_id",
        redactor=redactor,
    )
    for field_name in ("idempotency_key",):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
        )
    for field_name, value in (
        ("limits", request.limits.model_dump(mode="json")),
        (
            "budget_limits",
            [limit.model_dump(mode="json") for limit in request.budget_limits],
        ),
    ):
        if redactor.redact_json_values(value) != value:
            raise ValueError(
                f"{field_name} contains a workload secret and cannot be changed at "
                "the durable compaction boundary."
            )
    return request.model_copy(
        update={
            "instructions": (
                None if request.instructions is None else redactor.redact_text(request.instructions)
            ),
            "requested_by": redact_resolution_actor(
                request.requested_by,
                field_name="requested_by",
                redactor=redactor,
            ),
        },
    )


def prepare_interrupt_session_request(
    request: InterruptSessionRequest,
    *,
    redactor: SecretRedactor,
    store_resolved_session_id: str | None = None,
) -> InterruptSessionRequest:
    request = copy_interrupt_session_request(request)
    require_store_resolved_or_secret_free_session_authority(
        request.session_id,
        store_resolved_value=store_resolved_session_id,
        field_name="session_id",
        redactor=redactor,
    )
    return request.model_copy(
        update={
            "reason": (None if request.reason is None else redactor.redact_text(request.reason)),
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            ),
            "requested_by": redact_resolution_actor(
                request.requested_by,
                field_name="requested_by",
                redactor=redactor,
            ),
        },
    )


def prepare_fork_session_request(
    request: ForkSessionRequest,
    *,
    redactor: SecretRedactor,
    store_resolved_source_session_id: str | None = None,
) -> ForkSessionRequest:
    request = copy_fork_session_request(request)
    if public_authority_alias_is_reserved(request.session_id):
        raise ValueError("session_id uses the reserved public-authority alias namespace.")
    require_store_resolved_or_secret_free_session_authority(
        request.source_session_id,
        store_resolved_value=store_resolved_source_session_id,
        field_name="source_session_id",
        redactor=redactor,
    )
    for field_name in ("session_id", "agent_name", "model", "environment_name"):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
        )
    return request.model_copy(
        update={
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            )
        },
    )


def prepare_enqueue_message_request(
    request: EnqueueSessionMessageRequest,
    *,
    redactor: SecretRedactor,
    store_resolved_session_id: str | None = None,
) -> EnqueueSessionMessageRequest:
    request = copy_enqueue_session_message_request(request)
    require_store_resolved_or_secret_free_session_authority(
        request.session_id,
        store_resolved_value=store_resolved_session_id,
        field_name="session_id",
        redactor=redactor,
        authority_kind="durable queued-message authority",
    )
    for field_name in ("idempotency_key",):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
            authority_kind="durable queued-message authority",
        )
    return EnqueueSessionMessageRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        content=redactor.redact_text(request.content),
        delivery_mode=request.delivery_mode,
        requested_by=redact_resolution_actor(
            request.requested_by,
            field_name="requested_by",
            redactor=redactor,
        ),
    )


def require_secret_free_session_authority(
    value: str | None,
    *,
    field_name: str,
    redactor: SecretRedactor,
    authority_kind: str = "durable session authority",
) -> None:
    if type(value) is str and redactor.redact_text(value) != value:
        raise ValueError(
            f"{field_name} contains a workload secret and cannot be used as {authority_kind}."
        )


def require_store_resolved_or_secret_free_session_authority(
    value: str,
    *,
    store_resolved_value: str | None,
    field_name: str,
    redactor: SecretRedactor,
    authority_kind: str = "durable session authority",
) -> None:
    """Accept exact store-resolved authority or validate ordinary caller authority."""

    if store_resolved_value is not None:
        if value != store_resolved_value:
            raise ValueError(f"{field_name} does not match store-resolved session authority.")
        if redactor.is_exact_secret(value):
            raise ValueError(
                f"{field_name} contains a workload secret and cannot be used as {authority_kind}."
            )
        return
    require_secret_free_session_authority(
        value,
        field_name=field_name,
        redactor=redactor,
        authority_kind=authority_kind,
    )


def require_runtime_generated_or_secret_free_session_authority(
    value: str | None,
    *,
    runtime_generated: bool,
    field_name: str,
    redactor: SecretRedactor,
) -> None:
    """Accept exact runtime-generated authority without trusting caller input."""

    if runtime_generated:
        if type(value) is not str:
            raise ValueError(f"{field_name} does not match runtime-generated authority.")
        if redactor.is_exact_secret(value):
            raise ValueError(
                f"{field_name} contains a workload secret and cannot be used as "
                "durable session authority."
            )
        return
    require_secret_free_session_authority(
        value,
        field_name=field_name,
        redactor=redactor,
    )


def redact_json_object(
    value: dict[str, Any],
    *,
    field_name: str,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    redactor.require_no_secret_keys(
        value,
        field_name=field_name,
        match_short_substrings=True,
    )
    redacted = redactor.redact_json_values(value)
    if type(redacted) is not dict:
        raise AssertionError(f"{field_name} redaction returned a non-object.")
    return redacted


def redact_messages(
    messages: list[Message],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> list[Message]:
    return [
        redact_untrusted_message_for_boundary(
            message,
            redactor=redactor,
            field_name=field_name,
        )
        for message in messages
    ]


def redact_transcript(
    messages: list[Message],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> tuple[list[Message], bool]:
    """Detach raw transcript state and return explicit positive validation."""

    redacted_messages: list[Message] = []
    for index, message in enumerate(messages):
        try:
            redacted_messages.append(
                redact_runtime_message_for_boundary(
                    message,
                    redactor=redactor,
                    field_name=f"{field_name}[{index}]",
                )
            )
        except (TypeError, ValueError):
            messages.clear()
            redacted_messages.clear()
            return [], False
    messages.clear()
    return redacted_messages, True


def fork_checkpoint_is_secret_free(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
) -> bool:
    """Return whether legacy executable state can be copied unchanged."""

    if checkpoint is None:
        return True
    try:
        redactor.require_no_secret_keys(
            checkpoint,
            field_name="source_session.checkpoint",
            match_short_substrings=True,
        )
        return redactor.redact_json_values(checkpoint) == checkpoint
    except ValueError:
        return False


def fork_transcript_is_secret_free(
    messages: tuple[Message, ...],
    *,
    redactor: SecretRedactor,
) -> bool:
    """Return whether an exact legacy transcript prefix can be copied unchanged."""

    for index, message in enumerate(messages):
        try:
            redacted_message = redact_runtime_message_for_boundary(
                message,
                redactor=redactor,
                field_name=f"source_session.transcript[{index}]",
            )
        except (TypeError, ValueError):
            return False
        if redacted_message != message:
            return False
    return True


def redact_resolution_actor(
    actor: ResolutionActor | None,
    *,
    field_name: str,
    redactor: SecretRedactor,
) -> ResolutionActor | None:
    if actor is None:
        return None
    claims = redact_json_object(
        actor.claims,
        field_name=f"{field_name}.claims",
        redactor=redactor,
    )
    return actor.model_copy(
        update={
            "subject": redactor.redact_text(actor.subject),
            "tenant": (None if actor.tenant is None else redactor.redact_text(actor.tenant)),
            "claims": claims,
        },
        deep=True,
    )
