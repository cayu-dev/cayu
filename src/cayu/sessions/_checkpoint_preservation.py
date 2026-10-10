"""Shared checkpoint visibility and protected-root preservation.

Native stores and the runtime adapter compose these rules inside their existing
transactions. Domain owners retain their own authority checks; the scopes here
are the single lifecycle/workspace capabilities shared by producers and stores.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from cayu._validation import copy_durable_json_object
from cayu.sessions._browser_control_checkpoint import (
    browser_control_checkpoint_visible,
    project_browser_control_checkpoint,
)
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY, copy_model_failover_state
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    BROWSER_CONTROLS_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
    decode_runtime_checkpoint,
)

_COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION = 2
_MAX_COMPLETION_RESULT_EVENT_PUBLICATIONS = 64
_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX = "completion-result-publication:v1:"
_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX = "completion-result-owner:v1:"


def _completion_result_event_publication_owner_expiry(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("Completion-result event publication owner is malformed.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Completion-result event publication owner is malformed.") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Completion-result event publication owner is malformed.")
    normalized = parsed.astimezone(UTC)
    if normalized.isoformat() != value:
        raise ValueError("Completion-result event publication owner is malformed.")
    return normalized


def _is_completion_result_event_publication_owner_id(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == len(_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX) + 64
        and value.startswith(_COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX)
        and all(
            character in "0123456789abcdef"
            for character in value.removeprefix(
                _COMPLETION_RESULT_EVENT_PUBLICATION_OWNER_ID_PREFIX
            )
        )
    )


def _completion_result_event_publication_reservations(
    checkpoint: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if checkpoint is None:
        return {}
    if COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY not in checkpoint:
        return {}
    raw = checkpoint[COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY]
    if type(raw) is not dict or set(raw) != {"schema_version", "reservations"}:
        raise ValueError("Completion-result event publication authority is malformed.")
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != _COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION
    ):
        raise ValueError(
            "Completion-result event publication authority has an unsupported version."
        )
    reservations = raw.get("reservations")
    if type(reservations) is not dict or len(reservations) > (
        _MAX_COMPLETION_RESULT_EVENT_PUBLICATIONS
    ):
        raise ValueError("Completion-result event publication reservations are malformed.")
    copied: dict[str, dict[str, Any]] = {}
    for publication_id, record in reservations.items():
        if (
            type(publication_id) is not str
            or len(publication_id) != len(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX) + 64
            or not publication_id.startswith(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX)
            or type(record) is not dict
            or set(record) != {"schema_version", "publication_id", "authority_sha256", "owners"}
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != _COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION
            or record.get("publication_id") != publication_id
        ):
            raise ValueError("Completion-result event publication reservation is malformed.")
        authority_sha256 = record.get("authority_sha256")
        if (
            type(authority_sha256) is not str
            or len(authority_sha256) != 64
            or any(character not in "0123456789abcdef" for character in authority_sha256)
            or publication_id.removeprefix(_COMPLETION_RESULT_EVENT_PUBLICATION_ID_PREFIX)
            != authority_sha256
        ):
            raise ValueError("Completion-result event publication reservation is malformed.")
        owners = record.get("owners")
        if (
            type(owners) is not dict
            or not owners
            or len(owners) > (_MAX_COMPLETION_RESULT_EVENT_PUBLICATIONS)
        ):
            raise ValueError("Completion-result event publication reservation is malformed.")
        copied_owners: dict[str, dict[str, Any]] = {}
        for owner_id, owner in owners.items():
            if (
                not _is_completion_result_event_publication_owner_id(owner_id)
                or type(owner) is not dict
                or set(owner) != {"schema_version", "owner_id", "expires_at"}
                or type(owner.get("schema_version")) is not int
                or owner.get("schema_version")
                != _COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION
                or owner.get("owner_id") != owner_id
            ):
                raise ValueError("Completion-result event publication owner is malformed.")
            expires_at = _completion_result_event_publication_owner_expiry(owner.get("expires_at"))
            copied_owners[owner_id] = {
                "schema_version": _COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION,
                "owner_id": owner_id,
                "expires_at": expires_at.isoformat(),
            }
        copied[publication_id] = {
            "schema_version": _COMPLETION_RESULT_EVENT_PUBLICATIONS_SCHEMA_VERSION,
            "publication_id": publication_id,
            "authority_sha256": authority_sha256,
            "owners": copied_owners,
        }
    return copied


_INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED: ContextVar[bool] = ContextVar(
    "cayu_invocation_lifecycle_authority_mutation_allowed",
    default=False,
)
_INVOCATION_LIFECYCLE_AUTHORITY_READ_ALLOWED: ContextVar[bool] = ContextVar(
    "cayu_invocation_lifecycle_authority_read_allowed",
    default=False,
)
_WORKSPACE_OBSERVATION_AUTHORITY_MUTATION_ALLOWED: ContextVar[bool] = ContextVar(
    "cayu_workspace_observation_authority_mutation_allowed",
    default=False,
)

_EXECUTION_SNAPSHOT_AUTHORITY_MUTATION_ALLOWED: ContextVar[bool] = ContextVar(
    "cayu_execution_snapshot_authority_mutation_allowed", default=False
)


@contextmanager
def _execution_snapshot_authority_mutation_scope():
    """Allow the snapshot owner to compare full state and mutate only its root."""
    token = _EXECUTION_SNAPSHOT_AUTHORITY_MUTATION_ALLOWED.set(True)
    try:
        yield
    finally:
        _EXECUTION_SNAPSHOT_AUTHORITY_MUTATION_ALLOWED.reset(token)


@contextmanager
def _invocation_lifecycle_authority_mutation_scope():
    """Allow one typed command to see and replace private lifecycle roots."""

    token = _INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.set(True)
    try:
        yield
    finally:
        _INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.reset(token)


@contextmanager
def _invocation_lifecycle_authority_read_scope():
    """Allow one runtime-owned read callback to inspect private lifecycle roots."""

    token = _INVOCATION_LIFECYCLE_AUTHORITY_READ_ALLOWED.set(True)
    try:
        yield
    finally:
        _INVOCATION_LIFECYCLE_AUTHORITY_READ_ALLOWED.reset(token)


@contextmanager
def _workspace_observation_authority_mutation_scope():
    """Allow one runtime-owned recovery transform to replace workspace authority."""

    token = _WORKSPACE_OBSERVATION_AUTHORITY_MUTATION_ALLOWED.set(True)
    try:
        yield
    finally:
        _WORKSPACE_OBSERVATION_AUTHORITY_MUTATION_ALLOWED.reset(token)


def _replace_checkpoint_preserving_completion_result_event_publications(
    current: dict[str, Any] | None,
    replacement: dict[str, Any],
    *,
    preserve_completion_result_publications: bool = True,
    preserve_session_exports: bool = True,
    preserve_session_continuations: bool = True,
    session_id: str,
    decoded_replacement: bool = False,
) -> dict[str, Any]:
    """Replace caller state while retaining decoded runtime-owned checkpoint authority."""

    from cayu.collaboration import _session_export_store as session_exports
    from cayu.sessions import _producer_checkpoint as producers

    # Validate before decoding can normalize caller-controlled authority.
    from cayu.sessions import _session_continuation_store as continuations

    producer_root = producers.project_checkpoint_root(current, replacement, session_id=session_id)
    continuation_root = (
        continuations.project_checkpoint_root(current, replacement, session_id=session_id)
        if preserve_session_continuations
        else None
    )
    export_root = (
        session_exports.project_checkpoint_root(current, replacement, session_id=session_id)
        if preserve_session_exports
        else None
    )
    authoritative_current = current
    if current is not None and not (
        type(current.get(CHECKPOINT_SCHEMA_VERSION_KEY)) is int
        and current[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    ):
        authoritative_current = decode_runtime_checkpoint(current, session_id=session_id)
    lifecycle_mutation_allowed = _INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.get()
    workspace_mutation_allowed = _WORKSPACE_OBSERVATION_AUTHORITY_MUTATION_ALLOWED.get()
    snapshot_mutation_allowed = _EXECUTION_SNAPSHOT_AUTHORITY_MUTATION_ALLOWED.get()
    if decoded_replacement:
        # The runtime adapter owns this freshly decoded result. Projection below
        # still enforces private-root authority and the final document ceiling.
        updated = replacement
    elif lifecycle_mutation_allowed:
        # Typed lifecycle commands already own the current-schema authority
        # mutation. Keeping this private path literal also permits migration
        # fixtures to inject historical durable representations.
        updated = copy_durable_json_object(replacement, "checkpoint")
    elif workspace_mutation_allowed or (
        authoritative_current is not None
        and any(
            key in authoritative_current
            for key in (
                ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
                INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
                WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
                BROWSER_CONTROLS_CHECKPOINT_KEY,
                MODEL_FAILOVER_CHECKPOINT_KEY,
                "execution_snapshots",
            )
        )
    ):
        updated = decode_runtime_checkpoint(replacement, session_id=session_id)
        if updated is None:
            raise AssertionError("Checkpoint replacement decoded to no state.")
    else:
        # A raw SessionStore remains an opaque ordinary-checkpoint boundary.
        # Schema normalization belongs to the runtime wrapper unless private
        # runtime authority is already attached to this checkpoint.
        updated = copy_durable_json_object(replacement, "checkpoint")
    # Operator identities and takeover state are not generic checkpoint data.
    # Ordinary replacement may neither introduce nor erase this private root.
    # Lifecycle authority alone is not browser-control mutation authority.
    browser_controls = project_browser_control_checkpoint(
        authoritative_current, updated, session_id=session_id
    )
    updated.pop(BROWSER_CONTROLS_CHECKPOINT_KEY, None)
    if browser_controls is not None:
        updated[BROWSER_CONTROLS_CHECKPOINT_KEY] = browser_controls
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    # Model selection belongs to native stage, fork and admission transactions.
    # Generic replacement (including lifecycle callbacks) cannot manufacture,
    # reset or erase a selected route. Forks have no current route to preserve.
    updated.pop(MODEL_FAILOVER_CHECKPOINT_KEY, None)
    if authoritative_current is not None and MODEL_FAILOVER_CHECKPOINT_KEY in authoritative_current:
        route = copy_model_failover_state(authoritative_current[MODEL_FAILOVER_CHECKPOINT_KEY])
        if route.session_id != session_id:
            raise ValueError("Model failover checkpoint belongs to another session.")
        updated[MODEL_FAILOVER_CHECKPOINT_KEY] = route.payload()
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    if preserve_completion_result_publications:
        updated.pop(COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, None)
        if (
            authoritative_current is not None
            and COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY in authoritative_current
        ):
            _completion_result_event_publication_reservations(authoritative_current)
            updated[COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY] = copy_durable_json_object(
                authoritative_current[COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY],
                "completion_result_event_publications",
            )
    if not workspace_mutation_allowed:
        updated.pop(WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY, None)
        if (
            authoritative_current is not None
            and WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY in authoritative_current
        ):
            updated[WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY] = copy_durable_json_object(
                authoritative_current[WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY],
                WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY,
            )
    if not snapshot_mutation_allowed:
        updated.pop("execution_snapshots", None)
        if authoritative_current is not None and "execution_snapshots" in authoritative_current:
            updated["execution_snapshots"] = copy_durable_json_object(
                authoritative_current["execution_snapshots"], "execution_snapshots"
            )
            updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    if not lifecycle_mutation_allowed:
        preserved_lifecycle_authority = False
        for authority_key in (
            ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
            INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
            INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
            SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        ):
            updated.pop(authority_key, None)
            if authoritative_current is not None and authority_key in authoritative_current:
                updated[authority_key] = copy_durable_json_object(
                    authoritative_current[authority_key],
                    authority_key,
                )
                preserved_lifecycle_authority = True
        if preserved_lifecycle_authority:
            # The replacement was decoded before private roots were restored, so
            # legacy collisions were discarded and future schemas were rejected.
            # Retained authority therefore remains attached only to current-schema
            # ordinary state.
            assert updated[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    updated.pop(session_exports.ROOT_KEY, None)
    if export_root is not None:
        updated[session_exports.ROOT_KEY] = export_root
    updated.pop(continuations.ROOT_KEY, None)
    if continuation_root is not None:
        updated[continuations.ROOT_KEY] = continuation_root
    updated.pop(producers.ROOT_KEY, None)
    if producer_root is not None:
        updated[producers.ROOT_KEY] = producer_root
    # Restored private authority counts toward the same complete document
    # ceiling as the callback's ordinary state, before either side is written.
    return copy_durable_json_object(updated, "checkpoint")


def _copy_checkpoint_for_transform(
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
    decoded: bool = False,
) -> dict[str, Any] | None:
    """Validate and detach callback-visible state from store-owned authority."""

    from cayu.collaboration import _session_export_store as session_exports
    from cayu.sessions import _producer_checkpoint as producers
    from cayu.sessions import _session_continuation_store as continuations

    if checkpoint is None:
        return None
    if _EXECUTION_SNAPSHOT_AUTHORITY_MUTATION_ALLOWED.get():
        # Exact controller-position comparison includes all private domain
        # roots. Their independent projectors still preserve mutation authority.
        return (
            deepcopy(checkpoint) if decoded else copy_durable_json_object(checkpoint, "checkpoint")
        )
    lifecycle_authority_allowed = (
        _INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.get()
        or _INVOCATION_LIFECYCLE_AUTHORITY_READ_ALLOWED.get()
    )
    if decoded:
        copied = dict(checkpoint)
    elif lifecycle_authority_allowed or CHECKPOINT_SCHEMA_VERSION_KEY not in checkpoint:
        copied = copy_durable_json_object(checkpoint, "checkpoint")
    else:
        # Generic callbacks are an untrusted checkpoint entrance. Validate and
        # migrate store-owned state before invoking them so a no-op callback
        # cannot observe or silently preserve a future/incompatible schema.
        copied = decode_runtime_checkpoint(checkpoint, session_id=session_id)
        if copied is None:
            raise AssertionError("Stored checkpoint decoded to no state.")
    if not lifecycle_authority_allowed and CHECKPOINT_SCHEMA_VERSION_KEY in checkpoint:
        copied.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, None)
        copied.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
        copied.pop(INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY, None)
        copied.pop(SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY, None)
    if not browser_control_checkpoint_visible(session_id=session_id):
        copied.pop(BROWSER_CONTROLS_CHECKPOINT_KEY, None)
    if not lifecycle_authority_allowed:
        copied.pop(MODEL_FAILOVER_CHECKPOINT_KEY, None)
    if not session_exports.checkpoint_visible(session_id=session_id):
        copied.pop(session_exports.ROOT_KEY, None)
    # Typed lifecycle callbacks must retain receipts needed by pending
    # continuations. Read visibility does not grant index mutation authority;
    # project_checkpoint_root still requires the continuation owner's scope.
    if not continuations.checkpoint_visible() and not lifecycle_authority_allowed:
        copied.pop(continuations.ROOT_KEY, None)
    if not producers.checkpoint_visible(session_id=session_id) and not lifecycle_authority_allowed:
        copied.pop(producers.ROOT_KEY, None)
    if not lifecycle_authority_allowed:
        copied.pop("execution_snapshots", None)
    return deepcopy(copied) if decoded else copied


def _checkpoint_transform_result_preserving_completion_result_event_publications(
    current: dict[str, Any] | None,
    transformed: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    """Own callback output while retaining the store's private publication root."""

    return _replace_checkpoint_preserving_completion_result_event_publications(
        current,
        copy_durable_json_object(transformed, "checkpoint"),
        session_id=session_id,
    )
