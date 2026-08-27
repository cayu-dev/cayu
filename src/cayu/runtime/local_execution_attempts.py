from __future__ import annotations

import asyncio
import hashlib
import math
import os
import platform
import secrets
import traceback as traceback_module
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.capabilities import CapabilityClaim, CapabilityEvidence
from cayu.runners._subprocess import copy_runner_env
from cayu.runtime.invocation import copy_task_invocation

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp
    from cayu.runtime.tasks import Task, TaskStore


LOCAL_EXECUTION_ATTEMPT_SCHEMA_VERSION = 1
LOCAL_EXECUTION_CONTAINMENT_BACKEND = "linux_subreaper_v1"
DEFAULT_LOCAL_EXECUTION_STARTUP_TIMEOUT_SECONDS = 5.0
DEFAULT_LOCAL_EXECUTION_TERM_GRACE_SECONDS = 2.0
DEFAULT_LOCAL_EXECUTION_KILL_GRACE_SECONDS = 2.0
DEFAULT_LOCAL_EXECUTION_MAX_OUTPUT_BYTES = 1_048_576
MAX_LOCAL_EXECUTION_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_LOCAL_EXECUTION_ENVIRONMENT_BYTES = 16 * 1024 * 1024
MAX_LOCAL_EXECUTION_COMMAND_ITEMS = 1024
MAX_LOCAL_EXECUTION_RECOVERY_BATCH = 256
MAX_LOCAL_EXECUTION_IDENTITY_BYTES = 1024
MAX_LOCAL_EXECUTION_RECEIPT_BYTES = 65_536
_UNAVAILABLE_LOCAL_EXECUTION_HOST_IDENTITY = "unavailable"


_BoundaryP = ParamSpec("_BoundaryP")
_BoundaryResultT = TypeVar("_BoundaryResultT")


def _clear_local_execution_failure_tracebacks(error: BaseException) -> None:
    """Remove rejected request values from every reachable traceback frame."""

    pending = [error]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in observed:
            continue
        observed.add(id(current))
        if current.__traceback__ is not None:
            traceback_module.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _clean_local_execution_async_boundary(
    operation: Callable[_BoundaryP, Awaitable[_BoundaryResultT]],
) -> Callable[_BoundaryP, Awaitable[_BoundaryResultT]]:
    """Publish an async failure without retaining caller-owned request locals."""

    @wraps(operation)
    async def clean_boundary(
        *args: _BoundaryP.args,
        **kwargs: _BoundaryP.kwargs,
    ) -> _BoundaryResultT:
        published_failure: BaseException | None = None
        try:
            return await operation(*args, **kwargs)
        except BaseException as error:
            _clear_local_execution_failure_tracebacks(error)
            published_failure = error
        finally:
            del args, kwargs
        assert published_failure is not None
        published_failure.__traceback__ = None
        raise published_failure from None

    return clean_boundary


def _clean_local_execution_sync_boundary(
    operation: Callable[_BoundaryP, _BoundaryResultT],
) -> Callable[_BoundaryP, _BoundaryResultT]:
    """Publish a synchronous failure without retaining caller-owned request locals."""

    @wraps(operation)
    def clean_boundary(
        *args: _BoundaryP.args,
        **kwargs: _BoundaryP.kwargs,
    ) -> _BoundaryResultT:
        published_failure: BaseException | None = None
        try:
            return operation(*args, **kwargs)
        except BaseException as error:
            _clear_local_execution_failure_tracebacks(error)
            published_failure = error
        finally:
            del args, kwargs
        assert published_failure is not None
        published_failure.__traceback__ = None
        raise published_failure from None

    return clean_boundary


class LocalExecutionAttemptConflict(ValueError):
    """An exact local execution-attempt identity conflicts with durable state."""


class LocalExecutionAttemptUnavailable(RuntimeError):
    """The requested local execution containment strength is unavailable."""


class LocalExecutionAttemptUnsettled(RuntimeError):
    """A prior execution tree has not produced positive quiescence evidence."""


class LocalExecutionAttemptLifetime(StrEnum):
    PARENT_DEATH_CONTAINMENT = "parent_death_containment"
    GRACEFUL_CLEANUP = "graceful_cleanup"
    PERSISTENT_DETACHED = "persistent_detached"


class LocalExecutionAttemptPhase(StrEnum):
    PREPARED = "prepared"
    STARTING = "starting"
    RUNNING = "running"
    TERMINAL = "terminal"


class LocalExecutionAttemptQuiescence(StrEnum):
    NOT_DISPATCHED = "not_dispatched"
    TERMINAL_NOT_QUIESCENT = "terminal_not_quiescent"
    QUIESCENT = "quiescent"
    UNAVAILABLE = "unavailable"
    PERSISTENT_DETACHED = "persistent_detached"


class LocalExecutionAttemptEffectOutcome(StrEnum):
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class LocalExecutionAttemptListCursor(BaseModel):
    """Stable keyset cursor for bounded unsettled-attempt discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    created_at: datetime
    attempt_id: str

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _require_utc_datetime(value, "created_at")

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _require_bounded_identity(value, "attempt_id")


class LocalExecutionEffectPolicy(StrEnum):
    LOCAL_ONLY = "local_only"
    IDEMPOTENT_EXTERNAL = "idempotent_external"
    NON_IDEMPOTENT_EXTERNAL = "non_idempotent_external"


class LocalExecutionAttemptLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    deadline_seconds: float | None = Field(default=None, gt=0, le=31_536_000)
    startup_timeout_seconds: float = Field(
        default=DEFAULT_LOCAL_EXECUTION_STARTUP_TIMEOUT_SECONDS,
        gt=0,
        le=300,
    )
    term_grace_seconds: float = Field(
        default=DEFAULT_LOCAL_EXECUTION_TERM_GRACE_SECONDS,
        gt=0,
        le=300,
    )
    kill_grace_seconds: float = Field(
        default=DEFAULT_LOCAL_EXECUTION_KILL_GRACE_SECONDS,
        gt=0,
        le=300,
    )
    max_output_bytes: StrictInt = Field(
        default=DEFAULT_LOCAL_EXECUTION_MAX_OUTPUT_BYTES,
        ge=0,
        le=MAX_LOCAL_EXECUTION_OUTPUT_BYTES,
    )

    @field_validator(
        "deadline_seconds",
        "startup_timeout_seconds",
        "term_grace_seconds",
        "kill_grace_seconds",
        mode="before",
    )
    @classmethod
    def validate_seconds_type(cls, value: object, info) -> object:
        if value is not None and type(value) not in {int, float}:
            raise ValueError(f"{info.field_name} must be a number.")
        return value

    @field_validator(
        "deadline_seconds",
        "startup_timeout_seconds",
        "term_grace_seconds",
        "kill_grace_seconds",
    )
    @classmethod
    def validate_finite_seconds(cls, value: float | None, info) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class LocalExecutionAttemptRequest(BaseModel):
    """Ephemeral launch input; environment values are never durably persisted."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    effect_lineage_id: str
    argv: tuple[str, ...] = Field(min_length=1, max_length=MAX_LOCAL_EXECUTION_COMMAND_ITEMS)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    inherit_env: StrictBool = False
    lifetime: LocalExecutionAttemptLifetime = LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT
    effect_policy: LocalExecutionEffectPolicy = LocalExecutionEffectPolicy.NON_IDEMPOTENT_EXTERNAL
    idempotency_key: str | None = None
    workspace_identity: str | None = None
    execution_profile_fingerprint: str | None = None
    limits: LocalExecutionAttemptLimits = Field(default_factory=LocalExecutionAttemptLimits)

    @field_validator("effect_lineage_id")
    @classmethod
    def validate_effect_lineage_id(cls, value: str) -> str:
        return _require_bounded_identity(value, "effect_lineage_id")

    @field_validator(
        "idempotency_key",
        "workspace_identity",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_bounded_identity(value, info.field_name)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        owned: list[str] = []
        for item in value:
            if type(item) is not str or not item.strip():
                raise ValueError("argv entries must be non-empty strings.")
            owned.append(require_durable_text(item, "argv entry"))
        return tuple(owned)

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, "cwd")
        if not os.path.isabs(value):
            raise ValueError("cwd must be an absolute path.")
        return os.path.normpath(value)

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        copied = copy_runner_env(value, inherit_env=False)
        encoded_size = sum(
            len(name.encode("utf-8")) + len(item.encode("utf-8")) for name, item in copied.items()
        )
        if encoded_size > MAX_LOCAL_EXECUTION_ENVIRONMENT_BYTES:
            copied.clear()
            raise ValueError("env exceeds its UTF-8 byte limit.")
        return copied

    @model_validator(mode="after")
    def validate_effect_policy(self) -> LocalExecutionAttemptRequest:
        if (self.effect_policy is LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL) != (
            self.idempotency_key is not None
        ):
            raise ValueError("idempotency_key is required exactly for idempotent_external work.")
        launch_shape = {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": self.env,
        }
        if (
            len(
                canonical_durable_json_bytes(
                    launch_shape,
                    "local execution launch shape",
                )
            )
            > MAX_LOCAL_EXECUTION_ENVIRONMENT_BYTES
        ):
            raise ValueError("Local execution launch input exceeds its byte limit.")
        return self

    def effective_environment(self) -> dict[str, str]:
        return copy_runner_env(dict(self.env), inherit_env=bool(self.inherit_env))

    def durable_configuration(self) -> dict[str, Any]:
        """Return the secret-free fields that define retry/replacement authority."""

        environment = self.effective_environment()
        try:
            environment_sha256 = hashlib.sha256(
                canonical_durable_json_bytes(
                    environment,
                    "local execution environment",
                )
            ).hexdigest()
            idempotency_key_sha256 = (
                None
                if self.idempotency_key is None
                else hashlib.sha256(self.idempotency_key.encode("utf-8")).hexdigest()
            )
            return {
                "argv": list(self.argv),
                "cwd": self.cwd,
                "effect_lineage_id": self.effect_lineage_id,
                "effect_policy": self.effect_policy.value,
                "environment_names": sorted(environment),
                "environment_sha256": environment_sha256,
                "execution_profile_fingerprint": self.execution_profile_fingerprint,
                "idempotency_key_sha256": idempotency_key_sha256,
                "inherit_env": bool(self.inherit_env),
                "lifetime": self.lifetime.value,
                "limits": self.limits.model_dump(mode="json"),
                "workspace_identity": self.workspace_identity,
            }
        finally:
            environment.clear()


class LocalExecutionAttemptAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = LOCAL_EXECUTION_ATTEMPT_SCHEMA_VERSION
    attempt_id: str
    task_id: str
    task_created_at: datetime
    task_claim_updated_at: datetime
    task_claim_lease_expires_at: datetime
    task_invocation_sha256: str
    worker_id: str
    retry_series_id: str | None = None
    retry_attempt: StrictInt | None = Field(default=None, ge=1, le=100)
    session_id: str | None = None
    session_instance_id: str | None = None
    effect_lineage_id: str
    command_sha256: str
    execution_profile_fingerprint: str | None = None
    workspace_identity: str | None = None
    lifetime: LocalExecutionAttemptLifetime
    effect_policy: LocalExecutionEffectPolicy
    idempotency_key_sha256: str | None = None
    containment_backend: str
    request_sha256: str

    @field_validator(
        "attempt_id",
        "task_id",
        "task_invocation_sha256",
        "worker_id",
        "effect_lineage_id",
        "command_sha256",
        "containment_backend",
        "request_sha256",
    )
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        return _require_bounded_identity(value, info.field_name)

    @field_validator(
        "retry_series_id",
        "session_id",
        "session_instance_id",
        "execution_profile_fingerprint",
        "workspace_identity",
        "idempotency_key_sha256",
    )
    @classmethod
    def validate_optional_authority_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_bounded_identity(value, info.field_name)

    @field_validator(
        "task_invocation_sha256",
        "command_sha256",
        "idempotency_key_sha256",
        "request_sha256",
    )
    @classmethod
    def validate_authority_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_sha256(value, info.field_name)

    @field_validator(
        "task_created_at",
        "task_claim_updated_at",
        "task_claim_lease_expires_at",
    )
    @classmethod
    def normalize_task_authority_datetime(cls, value: datetime, info) -> datetime:
        return _require_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_task_claim_window(self) -> LocalExecutionAttemptAuthority:
        if self.task_claim_lease_expires_at <= self.task_claim_updated_at:
            raise ValueError("Local execution task claim window is invalid.")
        return self


class LocalExecutionProcessIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    pid: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    process_group: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    start_tick: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    proc_inode: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)


class LocalExecutionAttemptStart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    attempt_id: str
    request_sha256: str
    host_identity: str
    boot_id: str
    supervisor_nonce: str
    rendezvous_identity: str
    supervisor: LocalExecutionProcessIdentity
    root: LocalExecutionProcessIdentity | None = None
    started_at: datetime

    @field_validator(
        "attempt_id",
        "request_sha256",
        "host_identity",
        "boot_id",
        "supervisor_nonce",
        "rendezvous_identity",
    )
    @classmethod
    def validate_start_identity(cls, value: str, info) -> str:
        return _require_bounded_identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_start_digest(cls, value: str) -> str:
        return _require_sha256(value, "request_sha256")

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        return _require_utc_datetime(value, "started_at")


class LocalExecutionAttemptReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    attempt_id: str
    request_sha256: str
    host_identity: str
    boot_id: str
    supervisor_nonce: str
    root: LocalExecutionProcessIdentity | None = None
    terminal_reason: str
    exit_code: StrictInt | None = None
    term_sent: StrictBool = False
    kill_sent: StrictBool = False
    descendants_observed: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    quiescence: LocalExecutionAttemptQuiescence
    effect_outcome: LocalExecutionAttemptEffectOutcome
    settled_at: datetime
    receipt_sha256: str

    @field_validator(
        "attempt_id",
        "request_sha256",
        "host_identity",
        "boot_id",
        "supervisor_nonce",
        "terminal_reason",
        "receipt_sha256",
    )
    @classmethod
    def validate_receipt_identity(cls, value: str, info) -> str:
        return _require_bounded_identity(value, info.field_name)

    @field_validator("request_sha256", "receipt_sha256")
    @classmethod
    def validate_receipt_digest(cls, value: str, info) -> str:
        return _require_sha256(value, info.field_name)

    @field_validator("settled_at")
    @classmethod
    def normalize_settled_at(cls, value: datetime) -> datetime:
        return _require_utc_datetime(value, "settled_at")

    @model_validator(mode="after")
    def validate_receipt_state(self) -> LocalExecutionAttemptReceipt:
        if self.receipt_sha256 != local_execution_attempt_receipt_sha256(
            self.model_dump(mode="json", warnings=False)
        ):
            raise ValueError("Local execution receipt digest conflicted.")
        if self.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED and (
            self.root is not None
            or self.effect_outcome is not LocalExecutionAttemptEffectOutcome.NOT_STARTED
            or self.exit_code is not None
            or self.term_sent
            or self.kill_sent
            or self.descendants_observed != 0
        ):
            raise ValueError("A not-dispatched receipt contains dispatched-work evidence.")
        if (
            self.quiescence
            in {
                LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT,
                LocalExecutionAttemptQuiescence.UNAVAILABLE,
                LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED,
            }
            and self.effect_outcome is not LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        ):
            raise ValueError("Unproven quiescence requires an unknown effect outcome.")
        if self.quiescence is LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED and (
            self.root is None or self.exit_code is not None or self.term_sent or self.kill_sent
        ):
            raise ValueError("Persistent-detached evidence conflicts with its lifetime.")
        if (
            self.quiescence is LocalExecutionAttemptQuiescence.QUIESCENT
            and self.effect_outcome is LocalExecutionAttemptEffectOutcome.NOT_STARTED
        ):
            raise ValueError("Quiescent dispatched work cannot have a not-started outcome.")
        if self.effect_outcome is LocalExecutionAttemptEffectOutcome.SUCCEEDED and (
            self.root is None or self.exit_code != 0
        ):
            raise ValueError(
                "Successful local execution requires root identity and a zero exit status."
            )
        return self


class LocalExecutionAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    authority: LocalExecutionAttemptAuthority
    phase: LocalExecutionAttemptPhase
    quiescence: LocalExecutionAttemptQuiescence
    effect_outcome: LocalExecutionAttemptEffectOutcome
    start: LocalExecutionAttemptStart | None = None
    receipt: LocalExecutionAttemptReceipt | None = None
    recovery_owner_id: str | None = None
    recovery_owner_expires_at: datetime | None = None
    recovery_generation: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    created_at: datetime
    updated_at: datetime

    @field_validator("recovery_owner_id")
    @classmethod
    def validate_recovery_owner_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_identity(value, "recovery_owner_id")

    @field_validator("recovery_owner_expires_at", "created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _require_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle_state(self) -> LocalExecutionAttemptRecord:
        if self.updated_at < self.created_at:
            raise ValueError("Local execution attempt timestamps are not monotonic.")
        if (self.recovery_owner_id is None) != (self.recovery_owner_expires_at is None):
            raise ValueError("Local execution recovery ownership is incomplete.")
        if self.start is not None and (
            self.start.attempt_id != self.authority.attempt_id
            or self.start.request_sha256 != self.authority.request_sha256
        ):
            raise ValueError("Local execution start evidence conflicts with its authority.")
        if self.phase is LocalExecutionAttemptPhase.PREPARED:
            valid = (
                self.start is None
                and self.receipt is None
                and self.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
                and self.effect_outcome is LocalExecutionAttemptEffectOutcome.NOT_STARTED
            )
        elif self.phase is LocalExecutionAttemptPhase.STARTING:
            valid = (
                self.start is not None
                and self.start.root is None
                and self.receipt is None
                and self.quiescence is LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT
                and self.effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
            )
        elif self.phase is LocalExecutionAttemptPhase.RUNNING:
            valid = (
                self.start is not None
                and self.start.root is not None
                and self.receipt is None
                and self.quiescence is LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT
                and self.effect_outcome is LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
            )
        else:
            receipt = self.receipt
            valid = (
                receipt is not None
                and self.recovery_owner_id is None
                and self.recovery_owner_expires_at is None
                and self.quiescence is receipt.quiescence
                and self.effect_outcome is receipt.effect_outcome
                and receipt.attempt_id == self.authority.attempt_id
                and receipt.request_sha256 == self.authority.request_sha256
                and (
                    (
                        self.start is None
                        and receipt.root is None
                        and receipt.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
                    )
                    or (
                        self.start is not None
                        and receipt.host_identity == self.start.host_identity
                        and receipt.boot_id == self.start.boot_id
                        and receipt.supervisor_nonce == self.start.supervisor_nonce
                        and (self.start.root is None or receipt.root == self.start.root)
                    )
                )
            )
        if not valid:
            raise ValueError("Local execution attempt lifecycle evidence is inconsistent.")
        return self

    @property
    def retry_admissible(self) -> bool:
        if (
            self.phase is not LocalExecutionAttemptPhase.TERMINAL
            or self.receipt is None
            or self.quiescence
            not in {
                LocalExecutionAttemptQuiescence.NOT_DISPATCHED,
                LocalExecutionAttemptQuiescence.QUIESCENT,
            }
        ):
            return False
        # A caller declaration and key digest bind intent, but do not prove that
        # the downstream mutation actually consumed that key under an exact
        # idempotency contract. Generic local execution therefore keeps every
        # unknown external outcome fenced.
        return self.effect_outcome is not LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN

    @property
    def containment_settled(self) -> bool:
        return self.receipt is not None and self.quiescence in {
            LocalExecutionAttemptQuiescence.NOT_DISPATCHED,
            LocalExecutionAttemptQuiescence.QUIESCENT,
            LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED,
        }


def local_execution_attempt_list_cursor(
    record: LocalExecutionAttemptRecord,
) -> LocalExecutionAttemptListCursor:
    if not isinstance(record, LocalExecutionAttemptRecord):
        raise TypeError("record must be a LocalExecutionAttemptRecord.")
    return LocalExecutionAttemptListCursor(
        created_at=record.created_at,
        attempt_id=record.authority.attempt_id,
    )


def _copy_local_execution_attempt_list_cursor(
    cursor: LocalExecutionAttemptListCursor | None,
) -> LocalExecutionAttemptListCursor | None:
    if cursor is None:
        return None
    if not isinstance(cursor, LocalExecutionAttemptListCursor):
        raise TypeError("after must be a LocalExecutionAttemptListCursor.")
    return LocalExecutionAttemptListCursor(
        created_at=cursor.created_at,
        attempt_id=cursor.attempt_id,
    )


class LocalExecutionAttemptRecoveryClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    attempt_id: str
    request_sha256: str
    recovery_owner_id: str
    expected_recovery_generation: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    lease_seconds: StrictInt = Field(default=30, ge=1, le=300)

    @field_validator("attempt_id", "request_sha256", "recovery_owner_id")
    @classmethod
    def validate_claim_identity(cls, value: str, info) -> str:
        return _require_bounded_identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_claim_digest(cls, value: str) -> str:
        return _require_sha256(value, "request_sha256")


class LocalExecutionAttemptSettlement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    attempt_id: str
    request_sha256: str
    receipt: LocalExecutionAttemptReceipt
    recovery_owner_id: str | None = None
    expected_recovery_generation: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )

    @field_validator("attempt_id", "request_sha256")
    @classmethod
    def validate_settlement_identity(cls, value: str, info) -> str:
        return _require_bounded_identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_settlement_digest(cls, value: str) -> str:
        return _require_sha256(value, "request_sha256")

    @field_validator("recovery_owner_id")
    @classmethod
    def validate_settlement_owner(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_bounded_identity(value, "recovery_owner_id")


class LocalExecutionAttemptResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    attempt: LocalExecutionAttemptRecord
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: StrictBool = False
    stderr_truncated: StrictBool = False


def _copy_process_identity(
    identity: LocalExecutionProcessIdentity,
) -> LocalExecutionProcessIdentity:
    if not isinstance(identity, LocalExecutionProcessIdentity):
        raise TypeError("process identity must be a LocalExecutionProcessIdentity.")
    return LocalExecutionProcessIdentity(
        pid=identity.pid,
        process_group=identity.process_group,
        start_tick=identity.start_tick,
        proc_inode=identity.proc_inode,
    )


def _copy_local_execution_attempt_authority(
    authority: LocalExecutionAttemptAuthority,
) -> LocalExecutionAttemptAuthority:
    if not isinstance(authority, LocalExecutionAttemptAuthority):
        raise TypeError("authority must be a LocalExecutionAttemptAuthority.")
    return LocalExecutionAttemptAuthority(
        schema_version=authority.schema_version,
        attempt_id=authority.attempt_id,
        task_id=authority.task_id,
        task_created_at=authority.task_created_at,
        task_claim_updated_at=authority.task_claim_updated_at,
        task_claim_lease_expires_at=authority.task_claim_lease_expires_at,
        task_invocation_sha256=authority.task_invocation_sha256,
        worker_id=authority.worker_id,
        retry_series_id=authority.retry_series_id,
        retry_attempt=authority.retry_attempt,
        session_id=authority.session_id,
        session_instance_id=authority.session_instance_id,
        effect_lineage_id=authority.effect_lineage_id,
        command_sha256=authority.command_sha256,
        execution_profile_fingerprint=authority.execution_profile_fingerprint,
        workspace_identity=authority.workspace_identity,
        lifetime=authority.lifetime,
        effect_policy=authority.effect_policy,
        idempotency_key_sha256=authority.idempotency_key_sha256,
        containment_backend=authority.containment_backend,
        request_sha256=authority.request_sha256,
    )


def _copy_local_execution_attempt_start(
    start: LocalExecutionAttemptStart,
) -> LocalExecutionAttemptStart:
    if not isinstance(start, LocalExecutionAttemptStart):
        raise TypeError("start must be a LocalExecutionAttemptStart.")
    return LocalExecutionAttemptStart(
        attempt_id=start.attempt_id,
        request_sha256=start.request_sha256,
        host_identity=start.host_identity,
        boot_id=start.boot_id,
        supervisor_nonce=start.supervisor_nonce,
        rendezvous_identity=start.rendezvous_identity,
        supervisor=_copy_process_identity(start.supervisor),
        root=(None if start.root is None else _copy_process_identity(start.root)),
        started_at=start.started_at,
    )


def _copy_local_execution_attempt_receipt(
    receipt: LocalExecutionAttemptReceipt,
) -> LocalExecutionAttemptReceipt:
    if not isinstance(receipt, LocalExecutionAttemptReceipt):
        raise TypeError("receipt must be a LocalExecutionAttemptReceipt.")
    return LocalExecutionAttemptReceipt(
        attempt_id=receipt.attempt_id,
        request_sha256=receipt.request_sha256,
        host_identity=receipt.host_identity,
        boot_id=receipt.boot_id,
        supervisor_nonce=receipt.supervisor_nonce,
        root=(None if receipt.root is None else _copy_process_identity(receipt.root)),
        terminal_reason=receipt.terminal_reason,
        exit_code=receipt.exit_code,
        term_sent=receipt.term_sent,
        kill_sent=receipt.kill_sent,
        descendants_observed=receipt.descendants_observed,
        quiescence=receipt.quiescence,
        effect_outcome=receipt.effect_outcome,
        settled_at=receipt.settled_at,
        receipt_sha256=receipt.receipt_sha256,
    )


def _copy_local_execution_attempt_settlement(
    settlement: LocalExecutionAttemptSettlement,
) -> LocalExecutionAttemptSettlement:
    if not isinstance(settlement, LocalExecutionAttemptSettlement):
        raise TypeError("settlement must be a LocalExecutionAttemptSettlement.")
    return LocalExecutionAttemptSettlement(
        attempt_id=settlement.attempt_id,
        request_sha256=settlement.request_sha256,
        receipt=_copy_local_execution_attempt_receipt(settlement.receipt),
        recovery_owner_id=settlement.recovery_owner_id,
        expected_recovery_generation=settlement.expected_recovery_generation,
    )


class _AuthenticatedLocalExecutionAttemptSettlement(LocalExecutionAttemptSettlement):
    """Runtime-owned proof that receipt provenance was authenticated."""


def _authenticate_local_execution_attempt_settlement(
    settlement: LocalExecutionAttemptSettlement,
) -> LocalExecutionAttemptSettlement:
    copied = _copy_local_execution_attempt_settlement(settlement)
    return _AuthenticatedLocalExecutionAttemptSettlement(
        attempt_id=copied.attempt_id,
        request_sha256=copied.request_sha256,
        receipt=copied.receipt,
        recovery_owner_id=copied.recovery_owner_id,
        expected_recovery_generation=copied.expected_recovery_generation,
    )


def _copy_authenticated_local_execution_attempt_settlement(
    settlement: LocalExecutionAttemptSettlement,
) -> LocalExecutionAttemptSettlement:
    if type(settlement) is not _AuthenticatedLocalExecutionAttemptSettlement:
        raise LocalExecutionAttemptConflict(
            "Local execution settlement lacks runtime-authenticated receipt provenance."
        )
    return _authenticate_local_execution_attempt_settlement(settlement)


def _copy_local_execution_attempt_recovery_claim(
    claim: LocalExecutionAttemptRecoveryClaim,
) -> LocalExecutionAttemptRecoveryClaim:
    if not isinstance(claim, LocalExecutionAttemptRecoveryClaim):
        raise TypeError("claim must be a LocalExecutionAttemptRecoveryClaim.")
    return LocalExecutionAttemptRecoveryClaim(
        attempt_id=claim.attempt_id,
        request_sha256=claim.request_sha256,
        recovery_owner_id=claim.recovery_owner_id,
        expected_recovery_generation=claim.expected_recovery_generation,
        lease_seconds=claim.lease_seconds,
    )


def local_execution_attempt_request_sha256(authority: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_durable_json_bytes(authority, "local execution authority")
    ).hexdigest()


def local_execution_attempt_receipt_sha256(payload: dict[str, Any]) -> str:
    owned = copy_durable_json_object(payload, "local execution receipt")
    owned.pop("receipt_sha256", None)
    return hashlib.sha256(
        canonical_durable_json_bytes(owned, "local execution receipt")
    ).hexdigest()


def _require_utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _utc_json_timestamp(value: datetime, field_name: str) -> str:
    normalized = _require_utc_datetime(value, field_name)
    return normalized.isoformat().replace("+00:00", "Z")


def _require_bounded_identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_LOCAL_EXECUTION_IDENTITY_BYTES:
        raise ValueError(
            f"{field_name} cannot exceed {MAX_LOCAL_EXECUTION_IDENTITY_BYTES} UTF-8 bytes."
        )
    return value


def _require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _task_invocation_sha256(task: Any) -> str:
    invocation = task.invocation.model_dump(mode="json", warnings=False)
    return hashlib.sha256(canonical_durable_json_bytes(invocation, "task invocation")).hexdigest()


def require_local_execution_task_authority(
    task: Any | None,
    authority: LocalExecutionAttemptAuthority,
    *,
    now: datetime,
) -> None:
    """Validate every task-owned input that can change launch or retry admission."""

    now = _require_utc_datetime(now, "now")
    if task is None:
        raise LocalExecutionAttemptConflict("Local execution attempt task was not found.")
    retry = task.retry_series
    expected_retry_series_id = None if retry is None else retry.series_id
    expected_retry_attempt = None if retry is None else retry.attempt
    if (
        task.id != authority.task_id
        or task.created_at != authority.task_created_at
        or task.worker_id != authority.worker_id
        or getattr(task.status, "value", str(task.status)) not in {"claimed", "running"}
        or task.lease_expires_at is None
        or task.lease_expires_at <= now
        or task.session_id != authority.session_id
        or task.session_instance_id != authority.session_instance_id
        or _task_invocation_sha256(task) != authority.task_invocation_sha256
        or expected_retry_series_id != authority.retry_series_id
        or expected_retry_attempt != authority.retry_attempt
    ):
        raise LocalExecutionAttemptConflict(
            "Local execution attempt conflicts with exact task ownership."
        )


def require_local_execution_authority_integrity(
    authority: LocalExecutionAttemptAuthority,
) -> None:
    document = authority.model_dump(mode="json", warnings=False)
    expected_request_sha256 = document.pop("request_sha256")
    if (
        authority.containment_backend != LOCAL_EXECUTION_CONTAINMENT_BACKEND
        or local_execution_attempt_request_sha256(document) != expected_request_sha256
        or (
            authority.effect_policy is LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL
            and authority.idempotency_key_sha256 is None
        )
        or (
            authority.effect_policy is not LocalExecutionEffectPolicy.IDEMPOTENT_EXTERNAL
            and authority.idempotency_key_sha256 is not None
        )
    ):
        raise LocalExecutionAttemptConflict(
            "Local execution attempt authority failed integrity validation."
        )


def require_local_execution_recovery_eligible(
    task: Any,
    record: LocalExecutionAttemptRecord,
    *,
    now: datetime,
) -> None:
    now = _require_utc_datetime(now, "now")
    if task.id != record.authority.task_id or task.created_at != record.authority.task_created_at:
        raise LocalExecutionAttemptConflict(
            "Local execution recovery conflicts with the task incarnation."
        )
    if (
        getattr(task.status, "value", str(task.status)) in {"claimed", "running"}
        and task.worker_id == record.authority.worker_id
        and task.lease_expires_at is not None
        and task.lease_expires_at > now
    ):
        raise LocalExecutionAttemptConflict(
            "The original local execution owner still has a live task lease."
        )


def local_execution_effect_scope(
    authority: LocalExecutionAttemptAuthority,
) -> tuple[str, str]:
    return (
        (
            f"retry:{authority.retry_series_id}"
            if authority.retry_series_id is not None
            else f"task:{authority.task_id}"
        ),
        authority.effect_lineage_id,
    )


def prepare_local_execution_attempt_record(
    *,
    authority: LocalExecutionAttemptAuthority,
    task: Any,
    existing: LocalExecutionAttemptRecord | None,
    prior: LocalExecutionAttemptRecord | None,
    evidence_now: datetime,
    lease_now: datetime,
) -> LocalExecutionAttemptRecord:
    require_local_execution_authority_integrity(authority)
    evidence_now = _require_utc_datetime(evidence_now, "evidence_now")
    lease_now = _require_utc_datetime(lease_now, "lease_now")
    if existing is not None:
        if existing.authority != authority:
            raise LocalExecutionAttemptConflict(
                "Local execution attempt identity is bound to different authority."
            )
        return existing.model_copy(deep=True)
    if task is None:
        raise LocalExecutionAttemptConflict("Local execution attempt task was not found.")
    if (
        task.updated_at != authority.task_claim_updated_at
        or task.lease_expires_at != authority.task_claim_lease_expires_at
    ):
        raise LocalExecutionAttemptConflict(
            "Local execution attempt conflicts with the exact task claim generation."
        )
    require_local_execution_task_authority(task, authority, now=lease_now)
    if prior is not None and not prior.retry_admissible:
        raise LocalExecutionAttemptUnsettled(
            "A prior local execution attempt has not proven retry admissible."
        )
    return LocalExecutionAttemptRecord(
        authority=authority,
        phase=LocalExecutionAttemptPhase.PREPARED,
        quiescence=LocalExecutionAttemptQuiescence.NOT_DISPATCHED,
        effect_outcome=LocalExecutionAttemptEffectOutcome.NOT_STARTED,
        created_at=evidence_now,
        updated_at=evidence_now,
    )


def advance_local_execution_attempt_start(
    record: LocalExecutionAttemptRecord,
    start: LocalExecutionAttemptStart,
    *,
    evidence_now: datetime,
    lease_now: datetime,
) -> LocalExecutionAttemptRecord:
    evidence_now = _require_utc_datetime(evidence_now, "evidence_now")
    lease_now = _require_utc_datetime(lease_now, "lease_now")
    if (
        record.authority.attempt_id != start.attempt_id
        or record.authority.request_sha256 != start.request_sha256
    ):
        raise LocalExecutionAttemptConflict("Local execution start authority conflicted.")
    if (
        record.recovery_owner_id is not None
        and record.recovery_owner_expires_at is not None
        and record.recovery_owner_expires_at > lease_now
    ):
        raise LocalExecutionAttemptConflict("Local execution start lost ownership to recovery.")
    if record.receipt is not None:
        raise LocalExecutionAttemptConflict("A terminal local execution attempt cannot restart.")
    if record.start is not None:
        prior = record.start
        if (
            prior.attempt_id != start.attempt_id
            or prior.request_sha256 != start.request_sha256
            or prior.host_identity != start.host_identity
            or prior.boot_id != start.boot_id
            or prior.supervisor_nonce != start.supervisor_nonce
            or prior.rendezvous_identity != start.rendezvous_identity
            or prior.supervisor != start.supervisor
            or prior.started_at != start.started_at
            or (prior.root is not None and prior.root != start.root)
        ):
            raise LocalExecutionAttemptConflict("Local execution start evidence conflicted.")
        if prior.root is not None or start.root is None:
            return record.model_copy(deep=True)
    return record.model_copy(
        update={
            "phase": (
                LocalExecutionAttemptPhase.RUNNING
                if start.root is not None
                else LocalExecutionAttemptPhase.STARTING
            ),
            "quiescence": LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT,
            "effect_outcome": LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN,
            "start": start,
            "updated_at": evidence_now,
        }
    )


def settle_local_execution_attempt_record(
    record: LocalExecutionAttemptRecord,
    settlement: LocalExecutionAttemptSettlement,
    *,
    evidence_now: datetime,
    lease_now: datetime,
) -> LocalExecutionAttemptRecord:
    evidence_now = _require_utc_datetime(evidence_now, "evidence_now")
    lease_now = _require_utc_datetime(lease_now, "lease_now")
    if (
        record.authority.attempt_id != settlement.attempt_id
        or record.authority.request_sha256 != settlement.request_sha256
    ):
        raise LocalExecutionAttemptConflict("Local execution settlement authority conflicted.")
    if record.receipt is not None:
        if record.receipt != settlement.receipt:
            raise LocalExecutionAttemptConflict("Local execution settlement receipt conflicted.")
        return record.model_copy(deep=True)
    # A runtime-authenticated supervisor receipt is positive terminal evidence
    # and may settle across a stale inference-recovery claim. Inference-based
    # settlements carry an owner/generation and must still match exactly.
    if (
        record.recovery_owner_id is not None
        and settlement.recovery_owner_id is not None
        and (
            settlement.recovery_owner_id != record.recovery_owner_id
            or settlement.expected_recovery_generation != record.recovery_generation
            or record.recovery_owner_expires_at is None
            or record.recovery_owner_expires_at <= lease_now
        )
    ):
        raise LocalExecutionAttemptConflict(
            "Local execution settlement recovery ownership conflicted."
        )
    receipt = settlement.receipt
    receipt_document = receipt.model_dump(mode="json", warnings=False)
    if (
        receipt.receipt_sha256 != local_execution_attempt_receipt_sha256(receipt_document)
        or receipt.attempt_id != record.authority.attempt_id
        or receipt.request_sha256 != record.authority.request_sha256
        or receipt.quiescence
        not in {
            LocalExecutionAttemptQuiescence.NOT_DISPATCHED,
            LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT,
            LocalExecutionAttemptQuiescence.QUIESCENT,
            LocalExecutionAttemptQuiescence.UNAVAILABLE,
            LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED,
        }
        or (
            receipt.quiescence is LocalExecutionAttemptQuiescence.NOT_DISPATCHED
            and receipt.effect_outcome is not LocalExecutionAttemptEffectOutcome.NOT_STARTED
        )
        or (
            receipt.quiescence
            in {
                LocalExecutionAttemptQuiescence.TERMINAL_NOT_QUIESCENT,
                LocalExecutionAttemptQuiescence.UNAVAILABLE,
                LocalExecutionAttemptQuiescence.PERSISTENT_DETACHED,
            }
            and receipt.effect_outcome is not LocalExecutionAttemptEffectOutcome.OUTCOME_UNKNOWN
        )
        or (
            record.start is None
            and (
                receipt.root is not None
                or receipt.quiescence is not LocalExecutionAttemptQuiescence.NOT_DISPATCHED
            )
        )
        or (
            record.start is not None
            and (
                receipt.host_identity != record.start.host_identity
                or receipt.boot_id != record.start.boot_id
                or receipt.supervisor_nonce != record.start.supervisor_nonce
                or (record.start.root is not None and receipt.root != record.start.root)
            )
        )
    ):
        raise LocalExecutionAttemptConflict(
            "Local execution settlement receipt failed exact authority validation."
        )
    return record.model_copy(
        update={
            "phase": LocalExecutionAttemptPhase.TERMINAL,
            "quiescence": receipt.quiescence,
            "effect_outcome": receipt.effect_outcome,
            "receipt": receipt,
            "recovery_owner_id": None,
            "recovery_owner_expires_at": None,
            "updated_at": evidence_now,
        }
    )


def claim_local_execution_attempt_recovery_record(
    record: LocalExecutionAttemptRecord,
    claim: LocalExecutionAttemptRecoveryClaim,
    *,
    evidence_now: datetime,
    lease_now: datetime,
) -> LocalExecutionAttemptRecord:
    evidence_now = _require_utc_datetime(evidence_now, "evidence_now")
    lease_now = _require_utc_datetime(lease_now, "lease_now")
    if (
        record.authority.attempt_id != claim.attempt_id
        or record.authority.request_sha256 != claim.request_sha256
    ):
        raise LocalExecutionAttemptConflict("Local execution recovery authority conflicted.")
    if record.receipt is not None:
        return record.model_copy(deep=True)
    active_owner = (
        record.recovery_owner_id is not None
        and record.recovery_owner_expires_at is not None
        and record.recovery_owner_expires_at > lease_now
    )
    if record.recovery_generation != claim.expected_recovery_generation or (
        active_owner and record.recovery_owner_id != claim.recovery_owner_id
    ):
        raise LocalExecutionAttemptConflict("Local execution recovery generation conflicted.")
    return record.model_copy(
        update={
            "recovery_owner_id": claim.recovery_owner_id,
            "recovery_owner_expires_at": lease_now + timedelta(seconds=claim.lease_seconds),
            "recovery_generation": record.recovery_generation + 1,
            "updated_at": evidence_now,
        }
    )


def local_execution_attempt_capability_evidence(
    lifetime: LocalExecutionAttemptLifetime,
) -> CapabilityEvidence:
    supported = local_execution_parent_death_containment_platform_candidate()
    process_preflight_proved = False
    if supported:
        from cayu.mcp._stdio_process import (
            stdio_mcp_parent_death_containment_supported,
        )

        process_preflight_proved = stdio_mcp_parent_death_containment_supported()

    def available(name: str) -> CapabilityClaim:
        return CapabilityClaim(
            capability=name,
            state="available",
            proof_source="process_preflight",
            observation="available",
        )

    def declared(name: str) -> CapabilityClaim:
        return CapabilityClaim(
            capability=name,
            state="declared",
            proof_source="integration_declaration",
            observation="supported",
        )

    def unsupported(name: str, reason: str, remediation: str) -> CapabilityClaim:
        return CapabilityClaim(
            capability=name,
            state="unsupported",
            proof_source="integration_declaration",
            observation="unavailable",
            reason_code=reason,
            remediation_code=remediation,
        )

    graceful = (
        (
            available("graceful_cleanup")
            if process_preflight_proved
            else declared("graceful_cleanup")
        )
        if lifetime is not LocalExecutionAttemptLifetime.PERSISTENT_DETACHED and supported
        else unsupported(
            "graceful_cleanup",
            "complete_tree_cleanup_unavailable",
            "select_parent_death_containment_on_supported_linux",
        )
    )
    parent = (
        (
            available("parent_death_containment")
            if process_preflight_proved
            else declared("parent_death_containment")
        )
        if lifetime is LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT and supported
        else unsupported(
            "parent_death_containment",
            "parent_death_containment_unavailable",
            "select_parent_death_containment_on_supported_linux",
        )
    )
    hard = (
        (available("hard_deadline") if process_preflight_proved else declared("hard_deadline"))
        if lifetime is not LocalExecutionAttemptLifetime.PERSISTENT_DETACHED and supported
        else unsupported(
            "hard_deadline",
            "hard_deadline_unavailable",
            "select_parent_death_containment_on_supported_linux",
        )
    )
    detached = (
        (
            available("persistent_detached")
            if process_preflight_proved
            else declared("persistent_detached")
        )
        if lifetime is LocalExecutionAttemptLifetime.PERSISTENT_DETACHED and supported
        else unsupported(
            "persistent_detached",
            "persistent_detached_not_selected",
            "select_persistent_detached_on_posix",
        )
    )
    return CapabilityEvidence(
        subject="local_execution_attempt",
        claims=(graceful, hard, parent, detached),
    )


def local_execution_parent_death_containment_platform_candidate() -> bool:
    from cayu.mcp._stdio_process import (
        stdio_mcp_parent_death_containment_platform_candidate,
    )

    return (
        stdio_mcp_parent_death_containment_platform_candidate()
        and Path("/proc/self/stat").is_file()
    )


def local_execution_host_identity() -> str:
    configured_node_id = os.environ.get("CAYU_LOCAL_EXECUTION_NODE_ID")
    if configured_node_id is not None:
        try:
            machine_authority = require_durable_clean_nonblank(
                configured_node_id,
                "CAYU_LOCAL_EXECUTION_NODE_ID",
            )
        except (TypeError, ValueError):
            return _UNAVAILABLE_LOCAL_EXECUTION_HOST_IDENTITY
        source = "configured_node_id"
    else:
        machine_authority = ""
        source = ""
        for machine_id_path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
            try:
                candidate = machine_id_path.read_text(encoding="ascii").strip()
            except (OSError, UnicodeError):
                continue
            if candidate and len(candidate) <= 256:
                machine_authority = candidate
                source = "os_machine_id"
                break
        if not machine_authority:
            return _UNAVAILABLE_LOCAL_EXECUTION_HOST_IDENTITY
    material = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "machine_authority": machine_authority,
        "machine_authority_source": source,
        "system": platform.system(),
    }
    return hashlib.sha256(
        canonical_durable_json_bytes(material, "local execution host identity")
    ).hexdigest()


def local_execution_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        value = "unavailable"
    return require_durable_clean_nonblank(value, "boot_id")


def _task_authority(task: Task) -> tuple[str, str | None, int | None]:
    invocation = copy_task_invocation(task.invocation).model_dump(
        mode="json",
        warnings=False,
    )
    invocation_sha256 = hashlib.sha256(
        canonical_durable_json_bytes(invocation, "task invocation")
    ).hexdigest()
    retry = task.retry_series
    return (
        invocation_sha256,
        None if retry is None else retry.series_id,
        None if retry is None else retry.attempt,
    )


@_clean_local_execution_sync_boundary
def build_local_execution_attempt_authority(
    *,
    app: CayuApp,
    task: Task,
    worker_id: str,
    request: LocalExecutionAttemptRequest,
    attempt_id: str | None = None,
) -> LocalExecutionAttemptAuthority:
    request = _snapshot_local_execution_attempt_request(request)
    if task.worker_id != worker_id:
        raise LocalExecutionAttemptConflict("Task is not owned by the requested worker.")
    if task.lease_expires_at is None:
        raise LocalExecutionAttemptConflict("Task has no active claim lease.")
    durable_configuration = request.durable_configuration()
    if app.redact_json(durable_configuration) != durable_configuration:
        raise ValueError("Local execution durable authority contains a workload secret.")
    invocation_sha256, retry_series_id, retry_attempt = _task_authority(task)
    command_sha256 = hashlib.sha256(
        canonical_durable_json_bytes(
            durable_configuration,
            "local execution durable configuration",
        )
    ).hexdigest()
    idempotency_key_sha256 = (
        None
        if request.idempotency_key is None
        else hashlib.sha256(request.idempotency_key.encode("utf-8")).hexdigest()
    )
    authority_material = {
        "command_sha256": command_sha256,
        "containment_backend": LOCAL_EXECUTION_CONTAINMENT_BACKEND,
        "effect_lineage_id": request.effect_lineage_id,
        "effect_policy": request.effect_policy.value,
        "execution_profile_fingerprint": request.execution_profile_fingerprint,
        "idempotency_key_sha256": idempotency_key_sha256,
        "lifetime": request.lifetime.value,
        "retry_attempt": retry_attempt,
        "retry_series_id": retry_series_id,
        "schema_version": LOCAL_EXECUTION_ATTEMPT_SCHEMA_VERSION,
        "session_id": task.session_id,
        "session_instance_id": task.session_instance_id,
        "task_claim_updated_at": _utc_json_timestamp(
            task.updated_at,
            "task_claim_updated_at",
        ),
        "task_claim_lease_expires_at": _utc_json_timestamp(
            task.lease_expires_at,
            "task_claim_lease_expires_at",
        ),
        "task_created_at": _utc_json_timestamp(task.created_at, "task_created_at"),
        "task_id": task.id,
        "task_invocation_sha256": invocation_sha256,
        "worker_id": worker_id,
        "workspace_identity": request.workspace_identity,
    }
    if attempt_id is None:
        attempt_id = (
            "lex_"
            + hashlib.sha256(
                canonical_durable_json_bytes(
                    {
                        "authority": authority_material,
                        "schema": "cayu.local_execution.attempt_identity.v2",
                    },
                    "local execution attempt identity",
                )
            ).hexdigest()[:32]
        )
    authority_without_digest = {
        "attempt_id": attempt_id,
        **authority_material,
    }
    request_sha256 = local_execution_attempt_request_sha256(authority_without_digest)
    return LocalExecutionAttemptAuthority(
        schema_version=LOCAL_EXECUTION_ATTEMPT_SCHEMA_VERSION,
        attempt_id=attempt_id,
        task_id=task.id,
        task_created_at=task.created_at,
        task_claim_updated_at=task.updated_at,
        task_claim_lease_expires_at=task.lease_expires_at,
        task_invocation_sha256=invocation_sha256,
        worker_id=worker_id,
        retry_series_id=retry_series_id,
        retry_attempt=retry_attempt,
        session_id=task.session_id,
        session_instance_id=task.session_instance_id,
        effect_lineage_id=request.effect_lineage_id,
        command_sha256=command_sha256,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        workspace_identity=request.workspace_identity,
        lifetime=request.lifetime,
        effect_policy=request.effect_policy,
        idempotency_key_sha256=idempotency_key_sha256,
        containment_backend=LOCAL_EXECUTION_CONTAINMENT_BACKEND,
        request_sha256=request_sha256,
    )


def _snapshot_local_execution_attempt_request(
    request: LocalExecutionAttemptRequest,
) -> LocalExecutionAttemptRequest:
    if not isinstance(request, LocalExecutionAttemptRequest):
        raise TypeError("request must be a LocalExecutionAttemptRequest.")
    limits = LocalExecutionAttemptLimits(
        deadline_seconds=request.limits.deadline_seconds,
        startup_timeout_seconds=request.limits.startup_timeout_seconds,
        term_grace_seconds=request.limits.term_grace_seconds,
        kill_grace_seconds=request.limits.kill_grace_seconds,
        max_output_bytes=request.limits.max_output_bytes,
    )
    owned = LocalExecutionAttemptRequest(
        effect_lineage_id=request.effect_lineage_id,
        argv=request.argv,
        cwd=request.cwd,
        env=request.env,
        inherit_env=request.inherit_env,
        lifetime=request.lifetime,
        effect_policy=request.effect_policy,
        idempotency_key=request.idempotency_key,
        workspace_identity=request.workspace_identity,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        limits=limits,
    )
    environment = owned.effective_environment()
    try:
        return LocalExecutionAttemptRequest(
            effect_lineage_id=owned.effect_lineage_id,
            argv=owned.argv,
            cwd=owned.cwd,
            env=environment,
            inherit_env=False,
            lifetime=owned.lifetime,
            effect_policy=owned.effect_policy,
            idempotency_key=owned.idempotency_key,
            workspace_identity=owned.workspace_identity,
            execution_profile_fingerprint=owned.execution_profile_fingerprint,
            limits=limits,
        )
    finally:
        environment.clear()


class LocalExecutionAttemptCoordinator:
    """Task-backed owner for one complete local process-tree attempt."""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._task_store = task_store
        default_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        self._state_dir = (
            default_root / "cayu" / "local-execution-attempts"
            if state_dir is None
            else Path(state_dir).expanduser()
        )
        if not self._state_dir.is_absolute():
            raise ValueError("state_dir must be an absolute path.")
        self._recovery_after: LocalExecutionAttemptListCursor | None = None
        self._recovery_lock = asyncio.Lock()

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def capability_evidence(self) -> CapabilityEvidence:
        return local_execution_attempt_capability_evidence(
            LocalExecutionAttemptLifetime.PARENT_DEATH_CONTAINMENT
        )

    @_clean_local_execution_async_boundary
    async def run(
        self,
        *,
        app: CayuApp,
        task: Task,
        worker_id: str,
        request: LocalExecutionAttemptRequest,
    ) -> LocalExecutionAttemptResult:
        if not getattr(self._task_store, "supports_local_execution_attempts", False):
            raise NotImplementedError(
                "The configured TaskStore does not support local execution attempts."
            )
        from cayu.runtime._local_execution_attempt_owner import run_owned_local_execution_attempt

        request = _snapshot_local_execution_attempt_request(request)
        authority = build_local_execution_attempt_authority(
            app=app,
            task=task,
            worker_id=worker_id,
            request=request,
        )
        return await run_owned_local_execution_attempt(
            app=app,
            task_store=self._task_store,
            state_dir=self._state_dir,
            authority=authority,
            request=request,
        )

    async def recover(
        self,
        *,
        worker_id: str,
        limit: int = 32,
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        if not 1 <= limit <= MAX_LOCAL_EXECUTION_RECOVERY_BATCH:
            raise ValueError(f"limit must be between 1 and {MAX_LOCAL_EXECUTION_RECOVERY_BATCH}.")
        if not getattr(self._task_store, "supports_local_execution_attempts", False):
            raise NotImplementedError(
                "The configured TaskStore does not support local execution attempts."
            )
        if not local_execution_parent_death_containment_platform_candidate():
            raise LocalExecutionAttemptUnavailable(
                "General local execution recovery requires supported Linux process primitives."
            )

        recovered, _ = await self._recover_owned(
            worker_id=require_durable_clean_nonblank(worker_id, "worker_id"),
            limit=limit,
            deadline=None,
        )
        return recovered

    async def _recover_owned(
        self,
        *,
        worker_id: str,
        limit: int,
        deadline: float | None,
    ) -> tuple[tuple[LocalExecutionAttemptRecord, ...], bool]:
        if not 1 <= limit <= MAX_LOCAL_EXECUTION_RECOVERY_BATCH:
            raise ValueError(f"limit must be between 1 and {MAX_LOCAL_EXECUTION_RECOVERY_BATCH}.")
        if not getattr(self._task_store, "supports_local_execution_attempts", False):
            raise NotImplementedError(
                "The configured TaskStore does not support local execution attempts."
            )
        if not local_execution_parent_death_containment_platform_candidate():
            raise LocalExecutionAttemptUnavailable(
                "General local execution recovery requires supported Linux process primitives."
            )
        worker_id = require_durable_clean_nonblank(worker_id, "worker_id")
        from cayu.runtime._local_execution_attempt_owner import (
            recover_owned_local_execution_attempts,
        )

        lock_acquired = False
        if deadline is None:
            await self._recovery_lock.acquire()
            lock_acquired = True
        else:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return (), True
            try:
                await asyncio.wait_for(self._recovery_lock.acquire(), timeout=remaining)
                lock_acquired = True
            except TimeoutError:
                return (), True
        try:
            scan = await recover_owned_local_execution_attempts(
                task_store=self._task_store,
                state_dir=self._state_dir,
                worker_id=worker_id,
                limit=limit,
                after=self._recovery_after,
                deadline=deadline,
            )
            self._recovery_after = None if scan.reached_end else scan.after
            return scan.records, scan.deadline_elapsed
        finally:
            if lock_acquired:
                self._recovery_lock.release()

    async def drain(
        self, *, timeout_seconds: float = 5.0
    ) -> tuple[LocalExecutionAttemptRecord, ...]:
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")
        worker_id = f"local-drain-{os.getpid()}-{secrets.token_hex(8)}"
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        settled: dict[str, LocalExecutionAttemptRecord] = {}
        while asyncio.get_running_loop().time() < deadline:
            recovered, deadline_elapsed = await self._recover_owned(
                worker_id=worker_id,
                limit=32,
                deadline=deadline,
            )
            settled.update((record.authority.attempt_id, record) for record in recovered)
            if deadline_elapsed or asyncio.get_running_loop().time() >= deadline:
                break
            remaining = await self._task_store.list_unsettled_local_execution_attempts(limit=1)
            if not remaining:
                return tuple(settled.values())
            sleep_seconds = min(0.05, deadline - asyncio.get_running_loop().time())
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
        raise LocalExecutionAttemptUnsettled(
            "Local execution drain elapsed before every attempt proved settlement."
        )


__all__ = [
    "DEFAULT_LOCAL_EXECUTION_KILL_GRACE_SECONDS",
    "DEFAULT_LOCAL_EXECUTION_MAX_OUTPUT_BYTES",
    "DEFAULT_LOCAL_EXECUTION_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_LOCAL_EXECUTION_TERM_GRACE_SECONDS",
    "LOCAL_EXECUTION_ATTEMPT_SCHEMA_VERSION",
    "LocalExecutionAttemptAuthority",
    "LocalExecutionAttemptConflict",
    "LocalExecutionAttemptCoordinator",
    "LocalExecutionAttemptEffectOutcome",
    "LocalExecutionAttemptLifetime",
    "LocalExecutionAttemptLimits",
    "LocalExecutionAttemptListCursor",
    "LocalExecutionAttemptPhase",
    "LocalExecutionAttemptQuiescence",
    "LocalExecutionAttemptReceipt",
    "LocalExecutionAttemptRecord",
    "LocalExecutionAttemptRecoveryClaim",
    "LocalExecutionAttemptRequest",
    "LocalExecutionAttemptResult",
    "LocalExecutionAttemptSettlement",
    "LocalExecutionAttemptStart",
    "LocalExecutionAttemptUnavailable",
    "LocalExecutionAttemptUnsettled",
    "LocalExecutionEffectPolicy",
    "LocalExecutionProcessIdentity",
    "build_local_execution_attempt_authority",
    "local_execution_attempt_capability_evidence",
    "local_execution_attempt_list_cursor",
    "local_execution_attempt_receipt_sha256",
    "local_execution_attempt_request_sha256",
    "local_execution_boot_id",
    "local_execution_effect_scope",
    "local_execution_host_identity",
    "local_execution_parent_death_containment_platform_candidate",
    "require_local_execution_recovery_eligible",
    "require_local_execution_task_authority",
]
