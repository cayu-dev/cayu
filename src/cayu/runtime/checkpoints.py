"""Versioned root checkpoint payloads owned by the Cayu runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.runtime._model_completion_publication import ModelStepPublicationCheckpoint

if TYPE_CHECKING:
    from cayu.runtime.sessions import CheckpointRootFieldProjection

CHECKPOINT_SCHEMA_VERSION_KEY = "checkpoint_schema_version"
WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY = "workspace_observations"
ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY = "active_invocation_execution_profile"
INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY = "invocation_lifecycle_receipt"
INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE = "cayu.invocation-lifecycle-command-receipt-ledger"
INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION = 1
AUTOMATIC_RECALL_CHECKPOINT_KEY = "automatic_recall"
COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY = "completion_result_event_publications"
RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_KEY = "runtime_authored_user_message"
RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_VERSION = 1
AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY = "ambiguous_pending_user_input"
CURRENT_CHECKPOINT_SCHEMA_VERSION = 6
MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION = 1
_VERSIONLESS_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_EVIDENCE_SESSION_ID_MAX_BYTES = 256


def _bounded_checkpoint_evidence_session_id(session_id: str) -> str:
    session_id = require_clean_nonblank(session_id, "session_id")
    encoded = session_id.encode("utf-8")
    if len(encoded) <= _CHECKPOINT_EVIDENCE_SESSION_ID_MAX_BYTES:
        return session_id
    return f"sha256:{sha256(encoded).hexdigest()}"


class CheckpointCompatibilityError(RuntimeError):
    """A root checkpoint cannot be interpreted safely by this runtime."""

    def __init__(
        self,
        *,
        reason: str,
        session_id: str,
        observed_version: int | None,
        supported_min_version: int = MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION,
        supported_max_version: int = CURRENT_CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        self.reason = reason
        self.checkpoint_kind = "root"
        self.observed_version = observed_version
        self.supported_min_version = supported_min_version
        self.supported_max_version = supported_max_version
        self.session_id = _bounded_checkpoint_evidence_session_id(session_id)
        self.recovery_disposition = "cannot_migrate"
        self.resumable_in_place = False
        if reason == "checkpoint_schema_version_too_new":
            message = (
                f"Session {self.session_id!r} uses a newer root checkpoint schema; "
                "upgrade Cayu before resuming it."
            )
        else:
            message = f"Session {self.session_id!r} has an invalid root checkpoint schema version."
        super().__init__(message)

    def safe_evidence(self) -> dict[str, Any]:
        """Return bounded, content-free evidence for events and operator surfaces."""

        return {
            "checkpoint_kind": self.checkpoint_kind,
            "observed_version": self.observed_version,
            "reason": self.reason,
            "recovery_disposition": self.recovery_disposition,
            "resumable_in_place": self.resumable_in_place,
            "session_id": self.session_id,
            "supported_max_version": self.supported_max_version,
            "supported_min_version": self.supported_min_version,
        }


class CheckpointMigrationDefinitionError(ValueError):
    """The runtime's ordered checkpoint migration chain is invalid."""


@dataclass(frozen=True)
class CheckpointMigration:
    """One pure, unit-step root checkpoint upcaster."""

    source_version: int
    target_version: int
    migrate: Callable[[dict[str, Any]], dict[str, Any]]

    def __post_init__(self) -> None:
        if type(self.source_version) is not int or self.source_version < 1:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint migration source_version must be a positive integer."
            )
        if type(self.target_version) is not int or self.target_version < 1:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint migration target_version must be a positive integer."
            )
        if self.target_version != self.source_version + 1:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint migrations must advance exactly one version."
            )
        if not callable(self.migrate):
            raise CheckpointMigrationDefinitionError(
                "Checkpoint migration migrate must be callable."
            )


class CheckpointMigrator:
    """Decode and upcast root checkpoints through one ordered chain."""

    def __init__(
        self,
        *,
        current_version: int,
        migrations: Iterable[CheckpointMigration] = (),
        min_supported_version: int = MIN_SUPPORTED_CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        if type(current_version) is not int or current_version < 1:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint current_version must be a positive integer."
            )
        if type(min_supported_version) is not int or min_supported_version < 1:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint min_supported_version must be a positive integer."
            )
        if min_supported_version > current_version:
            raise CheckpointMigrationDefinitionError(
                "Checkpoint min_supported_version cannot exceed current_version."
            )
        by_source: dict[int, CheckpointMigration] = {}
        for migration in migrations:
            if type(migration) is not CheckpointMigration:
                raise CheckpointMigrationDefinitionError(
                    "Checkpoint migrations must contain CheckpointMigration values."
                )
            if migration.source_version in by_source:
                raise CheckpointMigrationDefinitionError(
                    "Checkpoint migration chain contains a duplicate migration "
                    f"from version {migration.source_version}."
                )
            by_source[migration.source_version] = migration
        for source_version in range(min_supported_version, current_version):
            if source_version not in by_source:
                raise CheckpointMigrationDefinitionError(
                    "Checkpoint migration chain is missing migration "
                    f"from version {source_version}."
                )
        if any(
            source_version < min_supported_version or source_version >= current_version
            for source_version in by_source
        ):
            raise CheckpointMigrationDefinitionError(
                "Checkpoint migration chain contains a step outside its supported range."
            )
        self.current_version = current_version
        self.min_supported_version = min_supported_version
        self._migrations = by_source

    def decode(
        self,
        checkpoint: dict[str, Any] | None,
        *,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Decode one root checkpoint and apply every required unit upcaster."""

        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint is None:
            return None
        if type(checkpoint) is not dict:
            raise CheckpointCompatibilityError(
                reason="invalid_root_checkpoint",
                session_id=session_id,
                observed_version=None,
                supported_min_version=self.min_supported_version,
                supported_max_version=self.current_version,
            )
        raw_version = checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY)
        if CHECKPOINT_SCHEMA_VERSION_KEY not in checkpoint:
            observed_version = _VERSIONLESS_CHECKPOINT_SCHEMA_VERSION
        elif type(raw_version) is not int or raw_version < 1:
            raise CheckpointCompatibilityError(
                reason="invalid_checkpoint_schema_version",
                session_id=session_id,
                observed_version=None,
                supported_min_version=self.min_supported_version,
                supported_max_version=self.current_version,
            )
        else:
            observed_version = raw_version
        if observed_version > self.current_version:
            raise CheckpointCompatibilityError(
                reason="checkpoint_schema_version_too_new",
                session_id=session_id,
                observed_version=observed_version,
                supported_min_version=self.min_supported_version,
                supported_max_version=self.current_version,
            )
        if observed_version < self.min_supported_version:
            raise CheckpointCompatibilityError(
                reason="checkpoint_schema_version_too_old",
                session_id=session_id,
                observed_version=observed_version,
                supported_min_version=self.min_supported_version,
                supported_max_version=self.current_version,
            )
        decoded = copy_durable_json_object(checkpoint, "checkpoint")
        decoded[CHECKPOINT_SCHEMA_VERSION_KEY] = observed_version
        while observed_version < self.current_version:
            migration = self._migrations[observed_version]
            try:
                migrated = migration.migrate(copy_durable_json_object(decoded, "checkpoint"))
                migrated = copy_durable_json_object(migrated, "checkpoint")
            except Exception:
                raise CheckpointMigrationDefinitionError(
                    f"Checkpoint migration from version {migration.source_version} failed."
                ) from None
            target_version = migrated.get(CHECKPOINT_SCHEMA_VERSION_KEY)
            if type(target_version) is not int or target_version != migration.target_version:
                raise CheckpointMigrationDefinitionError(
                    "Checkpoint migration from version "
                    f"{migration.source_version} must return checkpoint schema "
                    f"version {migration.target_version}."
                )
            decoded = migrated
            observed_version = target_version
        return decoded


def _migrate_checkpoint_v1_to_v2(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Add the durable tool-argument quarantine publication state."""

    migrated = copy_durable_json_object(checkpoint, "checkpoint")
    publication = migrated.get("last_model_step_publication")
    if type(publication) is dict:
        if (
            publication.get("record_type") == "cayu.model-step-publication"
            and publication.get("schema_version") == 1
        ):
            publication["schema_version"] = 2
        publication.setdefault("assistant_message_deferred", False)
    for checkpoint_key in ("pending_tool_round", "pending_user_input"):
        pending_state = migrated.get(checkpoint_key)
        if type(pending_state) is dict:
            pending_state.setdefault("assistant_message_state", "published")
            pending_state.setdefault("quarantined_assistant_message", None)
    migrated[CHECKPOINT_SCHEMA_VERSION_KEY] = 2
    return migrated


def _migrate_checkpoint_v2_to_v3(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reserve active-invocation execution-profile authority for v3 readers."""

    migrated = copy_durable_json_object(checkpoint, "checkpoint")
    migrated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, None)
    migrated.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
    migrated[CHECKPOINT_SCHEMA_VERSION_KEY] = 3
    return migrated


def _migrate_checkpoint_v3_to_v4(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reserve durable workspace-observation recovery state for v4 readers."""

    migrated = copy_durable_json_object(checkpoint, "checkpoint")
    # A v3 writer could not have produced this runtime-owned root.  Discarding
    # a caller-forged value prevents an old arbitrary checkpoint field from
    # becoming recovery authority after upgrade.
    migrated.pop(WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY, None)
    migrated[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
    return migrated


def _empty_invocation_lifecycle_receipt_history() -> dict[str, Any]:
    """Return authenticated evidence that pre-v5 lifecycle provenance is ambiguous."""

    receipt_history = {
        "record_type": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_RECORD_TYPE,
        "schema_version": INVOCATION_LIFECYCLE_RECEIPT_LEDGER_SCHEMA_VERSION,
        "receipts": [],
        "release_capacity_command_identity": None,
    }
    return {
        **receipt_history,
        "record_sha256": sha256(
            canonical_durable_json_bytes(
                receipt_history,
                "invocation lifecycle receipt ledger",
            )
        ).hexdigest(),
    }


def _migrate_checkpoint_v4_to_v5(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reserve the invocation-lifecycle authority plane for v5 readers."""

    migrated = copy_durable_json_object(checkpoint, "checkpoint")
    # A v4 writer treated this name as ordinary caller/application checkpoint
    # state.  It cannot become positive lifecycle authority merely because a
    # newer runtime recognizes the same spelling.
    migrated.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
    # Version 4 protected the active-profile spelling only after migration, but
    # its supported generic checkpoint entrances could subsequently replace the
    # value.  Remove that value from the live authority key.  A valid empty
    # receipt ledger records only that lifecycle authority history existed; it
    # contains no command receipt that could authorize admission, recovery,
    # settlement, cleanup, or replay.
    if ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY in migrated:
        migrated.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
        migrated[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = (
            _empty_invocation_lifecycle_receipt_history()
        )
    migrated[CHECKPOINT_SCHEMA_VERSION_KEY] = 5
    return migrated


def _migrate_checkpoint_v5_to_v6(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Retain pre-authority user-input pauses as bounded ambiguous state."""

    migrated = copy_durable_json_object(checkpoint, "checkpoint")
    # Version 5 did not own this spelling. Never authenticate a value supplied
    # by an older generic checkpoint writer merely because v6 recognizes it.
    migrated.pop(AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY, None)
    migrated.pop("user_input_resolution_intent", None)
    pending_interrupt = migrated.get("pending_session_interrupt")
    if type(pending_interrupt) is dict:
        # Version 5 did not own either nested supersession spelling. A generic
        # writer could place caller-shaped values under the otherwise-valid
        # pending interrupt root, so neither value may become v6 terminal
        # authority merely by surviving migration.
        pending_interrupt.pop("user_input_supersession_intent", None)
        pending_interrupt.pop("ambiguous_user_input_supersession_intent", None)
    pending_user_input = migrated.get("pending_user_input")
    if type(pending_user_input) is dict:
        # The old pause lacks the session incarnation, run epoch, interaction,
        # execution profile, and immutable opening receipt required to resume it.
        # Version 5 could not author the new nested schema or answer-claim root,
        # either, so a lookalike value remains untrusted. Preserve only
        # content-bound, secret-free evidence that an explicit operator
        # transition must retire; never invent executable authority.
        pending_digest = sha256(
            canonical_durable_json_bytes(pending_user_input, "pending_user_input")
        ).hexdigest()
        migrated.pop("pending_user_input")
        migrated[AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY] = {
            "schema_version": 1,
            "source_checkpoint_digest": pending_digest,
            "reason": "missing_exact_pause_authority",
        }
    migrated[CHECKPOINT_SCHEMA_VERSION_KEY] = 6
    return migrated


_RUNTIME_CHECKPOINT_MIGRATOR = CheckpointMigrator(
    current_version=CURRENT_CHECKPOINT_SCHEMA_VERSION,
    migrations=(
        CheckpointMigration(
            source_version=1,
            target_version=2,
            migrate=_migrate_checkpoint_v1_to_v2,
        ),
        CheckpointMigration(
            source_version=2,
            target_version=3,
            migrate=_migrate_checkpoint_v2_to_v3,
        ),
        CheckpointMigration(
            source_version=3,
            target_version=4,
            migrate=_migrate_checkpoint_v3_to_v4,
        ),
        CheckpointMigration(
            source_version=4,
            target_version=5,
            migrate=_migrate_checkpoint_v4_to_v5,
        ),
        CheckpointMigration(
            source_version=5,
            target_version=6,
            migrate=_migrate_checkpoint_v5_to_v6,
        ),
    ),
)


def validate_runtime_checkpoint_version_projection(
    *,
    session_id: str,
    version_is_present: bool,
    version: Any,
) -> None:
    """Validate a bounded root-version projection without checkpoint contents."""

    projected = {CHECKPOINT_SCHEMA_VERSION_KEY: version} if version_is_present else {}
    decode_runtime_checkpoint(projected, session_id=session_id)


def validate_runtime_checkpoint_root_projection(
    session_id: str,
    projection: CheckpointRootFieldProjection,
) -> None:
    """Validate one store-projected root version without nested checkpoint state."""

    version: Any = None
    scalar_text = projection.scalar_text
    if (
        projection.is_present
        and projection.json_type in {"integer", "number"}
        and not projection.scalar_text_truncated
        and type(scalar_text) is str
    ):
        digits = scalar_text[1:] if scalar_text.startswith("-") else scalar_text
        if digits and digits.isascii() and digits.isdigit():
            version = int(scalar_text)
    validate_runtime_checkpoint_version_projection(
        session_id=session_id,
        version_is_present=projection.is_present,
        version=version,
    )


def decode_runtime_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    session_id: str,
) -> dict[str, Any] | None:
    """Decode one root checkpoint into the current runtime-owned schema."""

    try:
        source_version = None
        if type(checkpoint) is dict:
            raw_source_version = checkpoint.get(CHECKPOINT_SCHEMA_VERSION_KEY)
            if CHECKPOINT_SCHEMA_VERSION_KEY not in checkpoint:
                source_version = _VERSIONLESS_CHECKPOINT_SCHEMA_VERSION
            elif type(raw_source_version) is int:
                source_version = raw_source_version
        decoded = _RUNTIME_CHECKPOINT_MIGRATOR.decode(
            checkpoint,
            session_id=session_id,
        )
        if decoded is not None and source_version in {3, 4}:
            # Versions 3 and 4 recognized active-invocation authority, but their
            # generic checkpoint writers could replace *or delete* that root.
            # Absence is therefore not positive evidence that a session was
            # never governed. Preserve no pre-v5 value as live authority and
            # retain only a non-executable history tombstone.
            decoded.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY, None)
            decoded[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = (
                _empty_invocation_lifecycle_receipt_history()
            )
        return decoded
    except CheckpointCompatibilityError as error:
        checkpoint = None
        error.__traceback__ = None
        raise error from None


def runtime_checkpoint_writer_view(
    checkpoint: dict[str, Any] | None,
    *,
    writer_version: int,
    session_id: str,
) -> dict[str, Any]:
    """Project current state into a supported staged writer's CAS schema."""

    decoded = decode_runtime_checkpoint(checkpoint, session_id=session_id)
    current = (
        {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
        if decoded is None
        else decoded
    )
    if writer_version == CURRENT_CHECKPOINT_SCHEMA_VERSION:
        return copy_durable_json_object(current, "checkpoint")
    if writer_version not in {1, 2, 3, 4, 5}:
        raise ValueError("Staged runtime publication uses an unsupported writer schema.")

    projected = copy_durable_json_object(current, "checkpoint")
    if (
        writer_version < CURRENT_CHECKPOINT_SCHEMA_VERSION
        and AMBIGUOUS_PENDING_USER_INPUT_CHECKPOINT_KEY in projected
    ):
        raise ValueError(
            "Ambiguous user-input pause authority cannot be represented by an older "
            f"v{writer_version} writer."
        )
    pending_user_input = projected.get("pending_user_input")
    if writer_version < CURRENT_CHECKPOINT_SCHEMA_VERSION and (
        (type(pending_user_input) is dict and "schema_version" in pending_user_input)
        or "user_input_resolution_intent" in projected
    ):
        raise ValueError(
            "Exact user-input pause authority cannot be represented by an older "
            f"v{writer_version} writer."
        )
    pending_interrupt = projected.get("pending_session_interrupt")
    if (
        writer_version < CURRENT_CHECKPOINT_SCHEMA_VERSION
        and type(pending_interrupt) is dict
        and (
            "user_input_supersession_intent" in pending_interrupt
            or "ambiguous_user_input_supersession_intent" in pending_interrupt
        )
    ):
        raise ValueError(
            "User-input supersession authority cannot be represented by an older "
            f"v{writer_version} writer."
        )
    if writer_version < 4 and WORKSPACE_OBSERVATIONS_CHECKPOINT_KEY in projected:
        raise ValueError(
            "Active workspace-observation recovery state cannot be represented by an "
            f"older v{writer_version} writer."
        )
    if writer_version < 5 and INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY in projected:
        raise ValueError(
            "Invocation lifecycle receipt authority cannot be represented by an older "
            f"v{writer_version} writer."
        )
    if writer_version == 4:
        projected[CHECKPOINT_SCHEMA_VERSION_KEY] = 4
        return projected
    if writer_version == 5:
        projected[CHECKPOINT_SCHEMA_VERSION_KEY] = 5
        return projected
    if writer_version == 3:
        projected[CHECKPOINT_SCHEMA_VERSION_KEY] = 3
        return projected
    if ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY in projected:
        raise ValueError(
            "Active invocation execution-profile authority cannot be represented by an "
            f"older v{writer_version} writer."
        )
    if writer_version == 2:
        projected[CHECKPOINT_SCHEMA_VERSION_KEY] = 2
        return projected
    publication = projected.get("last_model_step_publication")
    if type(publication) is dict:
        try:
            current_publication = ModelStepPublicationCheckpoint.model_validate(publication)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Current model-step publication state is not a recognized v2 pointer."
            ) from exc
        if current_publication.assistant_message_deferred:
            raise ValueError(
                "Current model-step publication state cannot be represented by a v1 writer."
            )
        publication = current_publication.model_dump(mode="json")
        publication.pop("assistant_message_deferred")
        publication["schema_version"] = 1
        projected["last_model_step_publication"] = publication
    for checkpoint_key in ("pending_tool_round", "pending_user_input"):
        pending_state = projected.get(checkpoint_key)
        if type(pending_state) is not dict:
            continue
        if (
            pending_state.get("assistant_message_state") != "published"
            or pending_state.get("quarantined_assistant_message") is not None
        ):
            raise ValueError(
                "Current assistant publication state cannot be represented by a v1 writer."
            )
        pending_state.pop("assistant_message_state")
        pending_state.pop("quarantined_assistant_message")
    projected[CHECKPOINT_SCHEMA_VERSION_KEY] = 1
    return projected
