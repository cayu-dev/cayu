"""Secret-safe ingress preparation for session-owned requests and state."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.core.messages import Message
from cayu.runtime._message_redaction import (
    redact_runtime_message_for_boundary,
    redact_untrusted_message_for_boundary,
)
from cayu.runtime.approvals import ResolutionActor
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_METADATA_KEY,
    ActiveInvocationExecutionProfile,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileIdentity,
)
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    public_authority_alias_is_reserved,
)
from cayu.runtime.sessions import (
    FORK_EXECUTION_PROFILE_METADATA_KEY,
    FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY,
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    PROMPT_ANATOMY_TRANSITION_METADATA_KEY,
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    ForkExecutionProfileSource,
    ForkSessionRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    apply_runtime_session_create_claim,
    copy_compact_session_request,
    copy_enqueue_session_message_request,
    copy_fork_session_request,
    copy_interrupt_session_request,
    copy_resume_request,
    copy_run_request,
    copy_session,
    effective_fork_source_execution_profile,
    run_request_authority_is_runtime_generated,
    strip_runtime_session_create_claim_before_redaction,
)
from cayu.runtime.structured_output import require_secret_free_structured_output_spec
from cayu.runtime.tool_policy import TAINT_LABELS_METADATA_KEY, taint_labels_from_metadata
from cayu.vaults import SecretRedactor


class ForkAuthorityError(ValueError):
    """Internal fork-authority rejection with a fixed public-safe message."""


class ForkSourceNotFoundError(KeyError):
    """Internal missing-source signal that never embeds private authority."""


class ForkActiveModelStageError(ValueError):
    """Internal active-stage signal that never embeds private authority."""


@dataclass(frozen=True, slots=True)
class PreparedForkSessionRequest:
    """A redacted fork request plus its complete pre-redaction replay identity."""

    request: ForkSessionRequest
    request_sha256: str
    accepted_request_sha256s: tuple[str, ...]


def prepare_fork_source_execution_profile(
    source_session: Session,
    source_checkpoint: Mapping[str, Any] | None,
) -> tuple[
    ForkExecutionProfileSource,
    ExecutionProfileIdentity,
    ActiveInvocationExecutionProfile | None,
]:
    """Resolve profile authority without retaining a rejected checkpoint."""

    resolution: (
        tuple[
            ForkExecutionProfileSource,
            ExecutionProfileIdentity,
            ActiveInvocationExecutionProfile | None,
        ]
        | None
    ) = None
    failure: ForkAuthorityError | None = None
    try:
        resolution = effective_fork_source_execution_profile(
            source_session,
            source_checkpoint,
        )
    except Exception as exc:
        # Durable checkpoints can contain workload-private state outside the
        # profile record. Strip every unwound parser frame before replacing the
        # failure so traceback-local capture cannot recover that state.
        if exc.__traceback__ is not None:
            traceback_module.clear_frames(exc.__traceback__)
        failure = ForkAuthorityError(
            "Source session has no valid durable execution-profile identity."
        )
    finally:
        source_checkpoint = None
    if failure is not None:
        raise failure from None
    if resolution is None:  # pragma: no cover - defensive totality guard
        raise AssertionError("Fork source profile resolution returned no result.")
    return resolution


def prepare_run_request(
    request: RunRequest,
    *,
    redactor: SecretRedactor,
) -> RunRequest:
    """Return a request safe to use as the source of a durable session."""

    request = strip_runtime_session_create_claim_before_redaction(copy_run_request(request))
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
        "environment_name",
    ):
        require_secret_free_session_authority(
            getattr(request, field_name),
            field_name=field_name,
            redactor=redactor,
        )
    if request.target is not None:
        for field_name, value in (
            ("target.provider_name", request.target.provider_name),
            ("target.model", request.target.model),
        ):
            require_secret_free_session_authority(
                value,
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
    for origin in (request.invocation_origin, request._verified_invocation_origin):
        if origin is None:
            continue
        for value in (origin.subject, origin.tenant):
            if value is not None and redactor.redact_text(value) != value:
                raise ValueError(
                    "invocation origin contains a workload secret and cannot be used "
                    "as durable session authority."
                )
    _require_secret_free_targeted_tool_grants(
        request.tool_grants,
        redactor=redactor,
        field_name="RunRequest.tool_grants",
    )
    redacted_messages = redact_messages(
        request.messages,
        redactor=redactor,
        field_name="messages",
    )
    prepared = request.model_copy(
        update={
            "messages": redacted_messages,
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            ),
        },
    )
    prepared._input_redactions_applied = request._input_redactions_applied or any(
        original != redacted
        for original, redacted in zip(request.messages, redacted_messages, strict=True)
    )
    return apply_runtime_session_create_claim(prepared)


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
    if request.target is not None:
        require_secret_free_session_authority(
            request.target.provider_name,
            field_name="target.provider_name",
            redactor=redactor,
        )
        require_secret_free_session_authority(
            request.target.model,
            field_name="target.model",
            redactor=redactor,
        )
    profile_adoption = request.profile_adoption
    if profile_adoption is not None:
        require_secret_free_session_authority(
            profile_adoption.idempotency_key,
            field_name="profile_adoption.idempotency_key",
            redactor=redactor,
            authority_kind="durable execution-profile adoption authority",
        )
        requested_by = redact_resolution_actor(
            profile_adoption.requested_by,
            field_name="profile_adoption.requested_by",
            redactor=redactor,
        )
        if requested_by is None:
            raise AssertionError("Execution-profile adoption lost its required actor.")
        profile_adoption = ExecutionProfileAdoptionIntent(
            idempotency_key=profile_adoption.idempotency_key,
            reason=redactor.redact_text(profile_adoption.reason),
            requested_by=requested_by,
        )
    _require_secret_free_targeted_tool_grants(
        request.tool_grants,
        redactor=redactor,
        field_name="ResumeRequest.tool_grants",
    )
    redacted_messages = redact_messages(
        request.messages,
        redactor=redactor,
        field_name="messages",
    )
    prepared = request.model_copy(
        update={
            "messages": redacted_messages,
            "metadata": redact_json_object(
                request.metadata,
                field_name="metadata",
                redactor=redactor,
            ),
            "profile_adoption": profile_adoption,
        },
    )
    prepared._input_redactions_applied = request._input_redactions_applied or any(
        original != redacted
        for original, redacted in zip(request.messages, redacted_messages, strict=True)
    )
    return prepared


def _require_secret_free_targeted_tool_grants(
    grants: tuple[object, ...],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Reject grant identities that would make workload secrets durable authority."""

    for index, grant in enumerate(grants):
        for attribute in ("request_id", "tool_id", "origin"):
            value = getattr(grant, attribute, None)
            if value is not None and redactor.redact_text(value) != value:
                raise ValueError(
                    f"{field_name}[{index}].{attribute} contains a workload secret and "
                    "cannot be used as durable targeted-tool authority."
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
    try:
        if public_authority_alias_is_reserved(request.session_id):
            raise ForkAuthorityError(
                "session_id uses the reserved public-authority alias namespace."
            )
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
        profile_adoption = request.profile_adoption
        if profile_adoption is not None:
            require_secret_free_session_authority(
                profile_adoption.idempotency_key,
                field_name="profile_adoption.idempotency_key",
                redactor=redactor,
                authority_kind="durable execution-profile adoption authority",
            )
            requested_by = redact_resolution_actor(
                profile_adoption.requested_by,
                field_name="profile_adoption.requested_by",
                redactor=redactor,
            )
            if requested_by is None:
                raise AssertionError("Execution-profile adoption lost its required actor.")
            profile_adoption = ExecutionProfileAdoptionIntent(
                idempotency_key=profile_adoption.idempotency_key,
                reason=redactor.redact_text(profile_adoption.reason),
                requested_by=requested_by,
            )
        _require_secret_free_fork_policy_metadata(request.metadata, redactor=redactor)
        return request.model_copy(
            update={
                "metadata": redact_json_object(
                    request.metadata,
                    field_name="metadata",
                    redactor=redactor,
                ),
                "profile_adoption": profile_adoption,
            },
        )
    except ForkAuthorityError:
        raise
    except ValueError as exc:
        raise ForkAuthorityError(str(exc)) from None


def prepare_fork_session_request_with_identity(
    request: ForkSessionRequest,
    *,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    store_resolved_source_session_id: str | None = None,
) -> PreparedForkSessionRequest:
    """Prepare a fork while binding replay to the complete raw request.

    Descriptive fork metadata and adoption attribution may legitimately contain
    values that the public durable projection redacts.  Their exact replay
    identity is therefore derived before redaction.  When secret redaction is
    configured, the application's durable authority keyring supplies a keyed
    representation so the digest cannot become an offline secret oracle.
    Retained keyring entries keep retries valid across key rotation.
    """

    raw_request: ForkSessionRequest | None = None
    material = b""
    aliases: tuple[str, ...] = ()
    try:
        raw_request = copy_fork_session_request(request)
        request_document: dict[str, Any] = raw_request.model_dump(
            mode="json",
            warnings=False,
        )
        initial_invocation = raw_request._fork_group_initial_invocation
        if initial_invocation is not None:
            request_document = {
                "request": request_document,
                "fork_group_initial_invocation_sha256": (initial_invocation.request_sha256),
            }
        material = canonical_durable_json_bytes(
            request_document,
            "fork_session_request",
        )
        if public_authority_alias_codec is None:
            digests = (sha256(material).hexdigest(),)
        else:
            aliases = public_authority_alias_codec.aliases(
                material.decode("utf-8"),
                field_name="fork_request_digest",
            )
            digests = tuple(sha256(alias.encode("ascii")).hexdigest() for alias in aliases)
        prepared = prepare_fork_session_request(
            raw_request,
            redactor=redactor,
            store_resolved_source_session_id=store_resolved_source_session_id,
        )
        return PreparedForkSessionRequest(
            request=prepared,
            request_sha256=digests[0],
            accepted_request_sha256s=digests,
        )
    finally:
        del request
        raw_request = None
        material = b""
        aliases = ()


def _require_secret_free_fork_policy_metadata(
    metadata: dict[str, Any],
    *,
    redactor: SecretRedactor,
) -> None:
    """Reject fork policy authority before descriptive metadata is redacted."""

    if MODEL_TARGET_PROJECTION_METADATA_KEY in metadata:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{MODEL_TARGET_PROJECTION_METADATA_KEY!r}] is runtime-owned "
            "model-target authority."
        ) from None
    if PROMPT_ANATOMY_TRANSITION_METADATA_KEY in metadata:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{PROMPT_ANATOMY_TRANSITION_METADATA_KEY!r}] is runtime-owned "
            "prompt-transition authority."
        ) from None
    if FORK_EXECUTION_PROFILE_METADATA_KEY in metadata:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{FORK_EXECUTION_PROFILE_METADATA_KEY!r}] is runtime-owned "
            "fork-profile authority."
        ) from None
    if FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY in metadata:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{FORK_GROUP_SOURCE_SNAPSHOT_METADATA_KEY!r}] is runtime-owned "
            "fork-group source authority."
        ) from None
    if EXECUTION_PROFILE_METADATA_KEY in metadata:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{EXECUTION_PROFILE_METADATA_KEY!r}] is runtime-owned "
            "execution-profile authority."
        ) from None
    try:
        labels = taint_labels_from_metadata(metadata)
    except (TypeError, ValueError):
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{TAINT_LABELS_METADATA_KEY!r}] is not valid durable session "
            "policy authority."
        ) from None
    contains_secret = any(redactor.redact_text(label) != label for label in labels)
    labels = frozenset()
    if contains_secret:
        metadata.clear()
        raise ForkAuthorityError(
            f"metadata[{TAINT_LABELS_METADATA_KEY!r}] contains a workload secret "
            "and cannot be used as durable session policy authority."
        ) from None


def prepare_fork_source_session(
    source_session: Session,
    *,
    expected_source_session_id: str,
    store_resolved_source_session_id: str | None,
    redactor: SecretRedactor,
) -> Session:
    """Validate and detach the durable source authority used by a fork."""

    source_session = copy_session(source_session)
    error: str | None = None
    if source_session.id != expected_source_session_id:
        error = "source_session.id does not match requested session authority."
    store_resolved = store_resolved_source_session_id == source_session.id
    if error is None:
        for field_name in (
            "id",
            "agent_name",
            "provider_name",
            "model",
            "parent_session_id",
            "causal_budget_id",
            "runtime_name",
            "runtime_version",
            "environment_name",
        ):
            value = getattr(source_session, field_name)
            if (
                store_resolved
                and field_name in {"id", "parent_session_id", "causal_budget_id"}
                and type(value) is str
            ):
                if redactor.is_exact_secret(value):
                    error = (
                        f"source_session.{field_name} contains a workload secret and "
                        "cannot be used as durable session authority."
                    )
                    break
                continue
            if type(value) is str and redactor.redact_text(value) != value:
                error = (
                    f"source_session.{field_name} contains a workload secret and "
                    "cannot be used as durable session authority."
                )
                break
    if error is None and _labels_contain_secret(source_session.labels, redactor=redactor):
        error = (
            "source_session.labels contain a workload secret and cannot be used as "
            "durable session authority."
        )
    if error is not None:
        del source_session
        field_name = value = ""
        expected_source_session_id = ""
        store_resolved_source_session_id = None
        raise ForkAuthorityError(error) from None
    return source_session


def prepare_fork_registered_authority(
    *,
    agent_name: str,
    agent_provider_name: str | None,
    model: str,
    environment_name: str | None,
    redactor: SecretRedactor,
) -> tuple[str, str | None, str, str | None]:
    """Validate registration-derived fork authority before it crosses a boundary."""

    error: str | None = None
    for field_name, value in (
        ("agent_name", agent_name),
        ("provider_name", agent_provider_name),
        ("model", model),
        ("environment_name", environment_name),
    ):
        if type(value) is str and redactor.redact_text(value) != value:
            error = (
                f"{field_name} contains a workload secret and cannot be used as "
                "durable session authority."
            )
            break
    if error is not None:
        agent_name = model = field_name = value = ""
        agent_provider_name = environment_name = None
        raise ForkAuthorityError(error) from None
    return agent_name, agent_provider_name, model, environment_name


def prepare_derived_fork_session(
    fork_session: Session,
    *,
    source_session: Session,
    runtime_generated_session_id: str | None,
    store_resolved_source_session_id: str | None,
    redactor: SecretRedactor,
) -> Session:
    """Validate the complete derived child before the fork mutation."""

    fork_session = copy_session(fork_session)
    source_session = copy_session(source_session)
    error: str | None = None
    runtime_generated = runtime_generated_session_id is not None
    if runtime_generated and fork_session.id != runtime_generated_session_id:
        error = "session_id does not match runtime-generated authority."
    elif runtime_generated:
        if redactor.is_exact_secret(fork_session.id):
            error = (
                "session_id contains a workload secret and cannot be used as durable "
                "session authority."
            )
    elif redactor.redact_text(fork_session.id) != fork_session.id:
        error = (
            "session_id contains a workload secret and cannot be used as durable session authority."
        )

    store_resolved = store_resolved_source_session_id == source_session.id
    if error is None:
        for field_name, value, expected in (
            ("parent_session_id", fork_session.parent_session_id, source_session.id),
            (
                "causal_budget_id",
                fork_session.causal_budget_id,
                source_session.causal_budget_id,
            ),
        ):
            if value != expected:
                error = f"{field_name} does not match store-derived source authority."
                break
            if store_resolved and type(value) is str:
                if redactor.is_exact_secret(value):
                    error = (
                        f"{field_name} contains a workload secret and cannot be used as "
                        "durable session authority."
                    )
                    break
                continue
            if type(value) is str and redactor.redact_text(value) != value:
                error = (
                    f"{field_name} contains a workload secret and cannot be used as "
                    "durable session authority."
                )
                break

    if error is None:
        for field_name in (
            "agent_name",
            "provider_name",
            "model",
            "runtime_name",
            "runtime_version",
            "environment_name",
        ):
            value = getattr(fork_session, field_name)
            if type(value) is str and redactor.redact_text(value) != value:
                error = (
                    f"{field_name} contains a workload secret and cannot be used as "
                    "durable session authority."
                )
                break
    if error is None and _labels_contain_secret(fork_session.labels, redactor=redactor):
        error = "labels contain a workload secret and cannot be used as durable session authority."
    if error is None and _json_contains_secret_key(
        fork_session.metadata,
        redactor=redactor,
    ):
        error = (
            "metadata contains a workload secret and cannot be used as durable session authority."
        )
    if (
        error is None
        and redactor.redact_json_values(fork_session.metadata) != fork_session.metadata
    ):
        error = (
            "metadata contains a workload secret and cannot be used as durable session authority."
        )
    if error is not None:
        fork_session.metadata.clear()
        del fork_session, source_session
        field_name = value = expected = ""
        runtime_generated_session_id = store_resolved_source_session_id = None
        raise ForkAuthorityError(error) from None
    return fork_session


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
    redacted_content = redactor.redact_text(request.content)
    prepared = EnqueueSessionMessageRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        content=redacted_content,
        delivery_mode=request.delivery_mode,
        requested_by=redact_resolution_actor(
            request.requested_by,
            field_name="requested_by",
            redactor=redactor,
        ),
    )
    prepared._input_redactions_applied = (
        request._input_redactions_applied or redacted_content != request.content
    )
    return prepared


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


def _labels_contain_secret(
    labels: dict[str, str],
    *,
    redactor: SecretRedactor,
) -> bool:
    for key, value in labels.items():
        if redactor.redact_text(key) != key or redactor.redact_text(value) != value:
            return True
    return False


def _json_contains_secret_key(value: Any, *, redactor: SecretRedactor) -> bool:
    if type(value) is dict:
        return any(
            redactor.redact_text(key) != key or _json_contains_secret_key(item, redactor=redactor)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_json_contains_secret_key(item, redactor=redactor) for item in value)
    return False


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
    reject_secret_bearing_runtime_projection_authority: bool = False,
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
                    reject_secret_bearing_runtime_projection_authority=(
                        reject_secret_bearing_runtime_projection_authority
                    ),
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
