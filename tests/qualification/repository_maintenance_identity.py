"""Immutable application reservation data, not authentication or execution authority."""

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator

from cayu import ExecutionDeadline


def _invalid() -> ValueError:
    return ValueError("Invalid maintenance reservation data.")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def _constant(_value):
    raise _invalid()


class MaintenanceTaskPhase(StrEnum):
    CODING = "coding"
    GIT_PREPARATION = "git_preparation"
    GIT_DELIVERY = "git_delivery"
    GITHUB_DELIVERY = "github_delivery"


class MaintenanceRunIntent(BaseModel):
    """Full accepted request; the host must supply authenticated identity/configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    tenant: StrictStr
    subject: StrictStr
    idempotency_key: StrictStr
    request_json: StrictStr

    @field_validator("tenant", "subject", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        bound = 128 if info.field_name == "idempotency_key" else 512
        if not value or len(value) > bound or value.strip() != value:
            raise _invalid()
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise _invalid()
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise _invalid() from None
        return value

    @field_validator("request_json")
    @classmethod
    def canonical_request(cls, value: str) -> str:
        try:
            if len(value.encode("utf-8")) > 65536:
                raise _invalid()
            parsed = json.loads(value, object_pairs_hook=_object, parse_constant=_constant)
            if type(parsed) is not dict:
                raise _invalid()
            canonical = json.dumps(
                parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            if len(canonical.encode("utf-8")) > 65536:
                raise _invalid()
        except (ValueError, UnicodeError, RecursionError):
            raise _invalid() from None
        return canonical

    @property
    def fingerprint(self) -> str:
        copied = copy_intent(self)
        encoded = json.dumps(
            [copied.tenant, copied.subject, copied.idempotency_key, copied.request_json],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def copy_intent(intent: MaintenanceRunIntent) -> MaintenanceRunIntent:
    """Revalidate before copying; never serialize rejected objects or subclasses."""
    if type(intent) is not MaintenanceRunIntent:
        raise _invalid()
    return MaintenanceRunIntent(
        tenant=intent.tenant,
        subject=intent.subject,
        idempotency_key=intent.idempotency_key,
        request_json=intent.request_json,
    )


class MaintenanceRunIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    intent: MaintenanceRunIntent
    public_id: StrictStr
    product_run_id: StrictStr
    session_id: StrictStr
    workflow_session_id: StrictStr
    task_id: StrictStr
    git_preparation_task_id: StrictStr
    git_delivery_task_id: StrictStr
    github_delivery_task_id: StrictStr
    coding_expires_at: StrictStr

    @field_validator("coding_expires_at")
    @classmethod
    def validate_coding_expiry(cls, value: str) -> str:
        try:
            if len(value) > 64:
                raise _invalid()
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
                raise _invalid()
            return parsed.astimezone(UTC).isoformat()
        except (ValueError, OverflowError):
            raise _invalid() from None

    def coding_deadline(self) -> ExecutionDeadline:
        """Restore the original expiry, never a fresh duration or execution permission."""
        current = copy_identity(self)
        return ExecutionDeadline(
            expires_at=datetime.fromisoformat(current.coding_expires_at),
            source="maintenance",
            scope="coding",
        )

    @field_validator("intent", mode="before")
    @classmethod
    def validate_intent(cls, value):
        if type(value) is dict:
            value = MaintenanceRunIntent.model_validate(value)
        return copy_intent(value)

    @field_validator(
        "public_id",
        "product_run_id",
        "session_id",
        "workflow_session_id",
        "task_id",
        "git_preparation_task_id",
        "git_delivery_task_id",
        "github_delivery_task_id",
    )
    @classmethod
    def validate_id(cls, value: str) -> str:
        try:
            if len(value) != 36 or str(UUID(value)) != value:
                raise _invalid()
        except ValueError:
            raise _invalid() from None
        return value

    @model_validator(mode="after")
    def validate_distinct(self):
        if (
            len(
                {
                    self.public_id,
                    self.product_run_id,
                    self.session_id,
                    self.workflow_session_id,
                    self.task_id,
                    self.git_preparation_task_id,
                    self.git_delivery_task_id,
                    self.github_delivery_task_id,
                }
            )
            != 8
        ):
            raise _invalid()
        return self


def copy_identity(identity: MaintenanceRunIdentity) -> MaintenanceRunIdentity:
    if type(identity) is not MaintenanceRunIdentity:
        raise _invalid()
    return MaintenanceRunIdentity.model_validate(
        {name: getattr(identity, name) for name in MaintenanceRunIdentity.model_fields}
    )


def task_id_for(identity: MaintenanceRunIdentity, phase: MaintenanceTaskPhase) -> str:
    """Resolve immutable phase data; this is not phase admission or authorization."""
    if type(phase) is not MaintenanceTaskPhase:
        raise _invalid()
    current = copy_identity(identity)
    return {
        MaintenanceTaskPhase.CODING: current.task_id,
        MaintenanceTaskPhase.GIT_PREPARATION: current.git_preparation_task_id,
        MaintenanceTaskPhase.GIT_DELIVERY: current.git_delivery_task_id,
        MaintenanceTaskPhase.GITHUB_DELIVERY: current.github_delivery_task_id,
    }[phase]


def allocate_identity(
    intent: MaintenanceRunIntent,
    *,
    workflow_session_id: str | None = None,
    coding_expires_at: str | None = None,
) -> MaintenanceRunIdentity:
    """For the trusted reservation owner; allocation alone does not reserve or authorize work."""
    return MaintenanceRunIdentity(
        intent=copy_intent(intent),
        public_id=str(uuid4()),
        product_run_id=str(uuid4()),
        session_id=str(uuid4()),
        workflow_session_id=str(uuid4()) if workflow_session_id is None else workflow_session_id,
        task_id=str(uuid4()),
        git_preparation_task_id=str(uuid4()),
        git_delivery_task_id=str(uuid4()),
        github_delivery_task_id=str(uuid4()),
        coding_expires_at=(datetime.now(UTC) + timedelta(seconds=180)).isoformat()
        if coding_expires_at is None
        else coding_expires_at,
    )
