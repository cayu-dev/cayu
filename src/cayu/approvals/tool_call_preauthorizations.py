"""Authorize one exact tool call for a task before any of its sessions exist.

A person who starts a piece of work can say up front "it may make this call":
one tool, with exactly these arguments, in this task. The record names that
person, is used at most once, and only stands in for an approval the policy
below would otherwise ask for. It never overrides a denial or a mandatory
approval.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from cayu._clock import utc_clock
from cayu.approvals.authorization_grants import (
    APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY,
    MANDATORY_CONTROL_OFFER_REASON,
    TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS,
    TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY,
    TOOL_MANDATORY_APPROVAL_METADATA_KEY,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantNotFound,
    _bounded_text,
    _GrantModel,
    _identifier,
    _optional_identifier,
    _optional_timestamp,
    normalize_grant_timestamp,
    tool_arguments_digest,
)
from cayu.runtime.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.sessions._run_identity import session_task_id
from cayu.tools.policy import (
    APPROVAL_GATE_METADATA_KEY,
    TOOL_POLICY_DRY_RUN_METADATA_KEY,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

#: Metadata key on the ALLOW result a preauthorization decided. Only
#: ``PreauthorizedToolCallPolicy`` writes it; it drops the key from ``then``.
TOOL_CALL_PREAUTHORIZATION_METADATA_KEY = "cayu:tool_call_preauthorization"
TOOL_CALL_PREAUTHORIZATION_SCHEMA_VERSION = 1
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ToolCallPreauthorizationUse(_GrantModel):
    """The one call that used a preauthorization."""

    session_id: str
    session_instance_id: str
    tool_round_id: str | None = None
    tool_call_id: str
    used_at: datetime

    @field_validator("session_id", "session_instance_id", "tool_call_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("tool_round_id")
    @classmethod
    def validate_round(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "tool_round_id")

    @field_validator("used_at")
    @classmethod
    def validate_used_at(cls, value: datetime) -> datetime:
        return normalize_grant_timestamp(value, "used_at")

    def same_call(self, other: ToolCallPreauthorizationUse) -> bool:
        return (
            self.session_id,
            self.session_instance_id,
            self.tool_round_id,
            self.tool_call_id,
        ) == (other.session_id, other.session_instance_id, other.tool_round_id, other.tool_call_id)


class ToolCallPreauthorization(_GrantModel):
    """One exact call a person authorized for one task, usable once.

    ``arguments_digest`` is :func:`tool_arguments_digest` of the exact
    arguments; build the record with :meth:`for_call` to compute it.
    ``agent_name``, when set, also binds the agent that may make the call.
    """

    schema_version: Literal[1] = TOOL_CALL_PREAUTHORIZATION_SCHEMA_VERSION
    preauthorization_id: str
    task_id: str
    tool_name: str
    arguments_digest: str
    agent_name: str | None = None
    authorized_by: ToolAuthorizationGrantActor
    created_at: datetime
    lifetime_seconds: StrictInt | None = Field(
        default=None, ge=1, le=TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS
    )
    expires_at: datetime | None = None
    justification: str | None = None
    use: ToolCallPreauthorizationUse | None = None
    revoked_at: datetime | None = None
    revoked_by: ToolAuthorizationGrantActor | None = None

    @field_validator("preauthorization_id", "task_id", "tool_name")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("agent_name")
    @classmethod
    def validate_agent(cls, value: str | None) -> str | None:
        return _optional_identifier(value, "agent_name")

    @field_validator("arguments_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if type(value) is not str or _DIGEST.fullmatch(value) is None:
            raise ValueError("arguments_digest must be 'sha256:' and 64 lowercase hex digits.")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return normalize_grant_timestamp(value, "created_at")

    @field_validator("expires_at", "revoked_at")
    @classmethod
    def validate_optional_timestamps(cls, value: datetime | None, info) -> datetime | None:
        return _optional_timestamp(value, info.field_name)

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_text(value, "justification")

    @model_validator(mode="after")
    def validate_record(self) -> ToolCallPreauthorization:
        expected = (
            None
            if self.lifetime_seconds is None
            else self.created_at + timedelta(seconds=self.lifetime_seconds)
        )
        if self.expires_at != expected:
            raise ValueError("expires_at must equal created_at plus lifetime_seconds.")
        if (self.revoked_by is not None) and self.revoked_at is None:
            raise ValueError("revoked_by requires revoked_at.")
        if self.use is not None and self.revoked_at is not None:
            raise ValueError("A used preauthorization cannot be revoked.")
        return self

    @classmethod
    def for_call(
        cls,
        *,
        preauthorization_id: str,
        task_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        authorized_by: ToolAuthorizationGrantActor,
        created_at: datetime,
        agent_name: str | None = None,
        lifetime_seconds: int | None = None,
        justification: str | None = None,
    ) -> ToolCallPreauthorization:
        """A preauthorization for exactly ``arguments``."""

        created_at = normalize_grant_timestamp(created_at, "created_at")
        return cls(
            preauthorization_id=preauthorization_id,
            task_id=task_id,
            tool_name=tool_name,
            arguments_digest=tool_arguments_digest(arguments),
            agent_name=agent_name,
            authorized_by=authorized_by,
            created_at=created_at,
            lifetime_seconds=lifetime_seconds,
            expires_at=(
                None
                if lifetime_seconds is None
                else created_at + timedelta(seconds=lifetime_seconds)
            ),
            justification=justification,
        )

    def available(self, now: datetime) -> bool:
        """Unused, unrevoked and unexpired at ``now``."""

        now = normalize_grant_timestamp(now, "now")
        return (
            self.use is None
            and self.revoked_at is None
            and (self.expires_at is None or now < self.expires_at)
        )

    def matches(
        self, *, task_id: str, tool_name: str, arguments_digest: str, agent_name: str
    ) -> bool:
        return (
            self.task_id == task_id
            and self.tool_name == tool_name
            and self.arguments_digest == arguments_digest
            and (self.agent_name is None or self.agent_name == agent_name)
        )

    def terms(self) -> dict[str, Any]:
        """What the record authorizes and who said so; equal terms make issuing idempotent.

        The issue time (and the expiry it sets) is left out, so a retry made later, or one
        racing the first request, is the same issue; the stored record keeps the first time.
        """

        return self.model_dump(
            mode="json",
            exclude={"use", "revoked_at", "revoked_by", "created_at", "expires_at"},
        )

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "preauthorization_id": self.preauthorization_id,
            "task_id": self.task_id,
            "authorized_by": self.authorized_by.model_dump(mode="json", exclude_none=True),
        }


class ToolCallPreauthorizationStore(ABC):
    """Durable preauthorizations. ``use`` must be atomic: one record, one call."""

    @abstractmethod
    async def issue(self, preauthorization: ToolCallPreauthorization) -> ToolCallPreauthorization:
        """Store a new record. The same id with the same terms returns the stored one;
        different terms raise ``ToolAuthorizationGrantConflict``."""

    @abstractmethod
    async def get(self, preauthorization_id: str) -> ToolCallPreauthorization:
        """The record, or ``ToolAuthorizationGrantNotFound``."""

    @abstractmethod
    async def list_for_task(self, task_id: str) -> tuple[ToolCallPreauthorization, ...]:
        """Every record for the task, ordered by ``(created_at, preauthorization_id)``."""

    @abstractmethod
    async def use(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments_digest: str,
        agent_name: str,
        use: ToolCallPreauthorizationUse,
    ) -> ToolCallPreauthorization | None:
        """Mark the first available match used by ``use`` and return it.

        A record this same call already used is returned again, so a retry of
        one execution is decided the same way. ``None`` when nothing applies.
        """

    @abstractmethod
    async def revoke(
        self,
        preauthorization_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        now: datetime,
    ) -> ToolCallPreauthorization:
        """Revoke an unused record; revoking a used one raises a conflict."""


# Transition rules every store applies inside its own atomic section, so the
# in-memory, SQLite and PostgreSQL stores decide identically.


def issued_transition(
    existing: ToolCallPreauthorization | None, record: ToolCallPreauthorization
) -> tuple[ToolCallPreauthorization, bool]:
    """The record to keep for ``issue`` and whether it must be written."""

    if type(record) is not ToolCallPreauthorization:
        raise TypeError("preauthorization must be a ToolCallPreauthorization.")
    if record.use is not None or record.revoked_at is not None:
        raise ValueError("Issue a preauthorization unused and unrevoked.")
    if existing is None:
        return record, True
    if existing.terms() != record.terms():
        raise ToolAuthorizationGrantConflict("A preauthorization with this id has different terms.")
    return existing, False


def used_transition(
    records: Iterable[ToolCallPreauthorization],
    *,
    task_id: str,
    tool_name: str,
    arguments_digest: str,
    agent_name: str,
    use: ToolCallPreauthorizationUse,
) -> tuple[ToolCallPreauthorization | None, bool]:
    """The record ``use`` decides from the task's records, and whether it changed."""

    candidates = sorted(
        (
            record
            for record in records
            if record.matches(
                task_id=task_id,
                tool_name=tool_name,
                arguments_digest=arguments_digest,
                agent_name=agent_name,
            )
        ),
        key=lambda record: (record.created_at, record.preauthorization_id),
    )
    for record in candidates:
        if record.use is not None and record.use.same_call(use):
            return record, False
    for record in candidates:
        if record.available(use.used_at):
            return record.model_copy(update={"use": use}), True
    return None, False


def revoked_transition(
    existing: ToolCallPreauthorization | None,
    *,
    preauthorization_id: str,
    revoked_by: ToolAuthorizationGrantActor | None,
    now: datetime,
) -> tuple[ToolCallPreauthorization, bool]:
    """The record ``revoke`` leaves and whether it must be written."""

    if existing is None:
        raise ToolAuthorizationGrantNotFound(preauthorization_id)
    if existing.use is not None:
        raise ToolAuthorizationGrantConflict("A used preauthorization cannot be revoked.")
    if existing.revoked_at is not None:
        return existing, False
    revoked = ToolCallPreauthorization.model_validate(
        {
            **existing.model_dump(),
            "revoked_at": normalize_grant_timestamp(now, "now"),
            "revoked_by": revoked_by,
        }
    )
    return revoked, True


def preauthorization_record_json(record: ToolCallPreauthorization) -> str:
    return record.model_dump_json()


def preauthorization_from_record_json(value: str | bytes) -> ToolCallPreauthorization:
    return ToolCallPreauthorization.model_validate_json(value)


class InMemoryToolCallPreauthorizationStore(ToolCallPreauthorizationStore):
    """Process-local store for tests and single-process applications."""

    def __init__(self) -> None:
        self._records: dict[str, ToolCallPreauthorization] = {}
        self._lock = asyncio.Lock()

    async def issue(self, preauthorization: ToolCallPreauthorization) -> ToolCallPreauthorization:
        async with self._lock:
            record, changed = issued_transition(
                self._records.get(getattr(preauthorization, "preauthorization_id", "")),
                preauthorization,
            )
            if changed:
                self._records[record.preauthorization_id] = record
            return record

    async def get(self, preauthorization_id: str) -> ToolCallPreauthorization:
        record = self._records.get(preauthorization_id)
        if record is None:
            raise ToolAuthorizationGrantNotFound(preauthorization_id)
        return record

    async def list_for_task(self, task_id: str) -> tuple[ToolCallPreauthorization, ...]:
        return tuple(
            sorted(
                (record for record in self._records.values() if record.task_id == task_id),
                key=lambda record: (record.created_at, record.preauthorization_id),
            )
        )

    async def use(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments_digest: str,
        agent_name: str,
        use: ToolCallPreauthorizationUse,
    ) -> ToolCallPreauthorization | None:
        async with self._lock:
            record, changed = used_transition(
                self._records.values(),
                task_id=task_id,
                tool_name=tool_name,
                arguments_digest=arguments_digest,
                agent_name=agent_name,
                use=use,
            )
            if record is not None and changed:
                self._records[record.preauthorization_id] = record
            return record

    async def revoke(
        self,
        preauthorization_id: str,
        *,
        revoked_by: ToolAuthorizationGrantActor | None,
        now: datetime,
    ) -> ToolCallPreauthorization:
        async with self._lock:
            record, changed = revoked_transition(
                self._records.get(preauthorization_id),
                preauthorization_id=preauthorization_id,
                revoked_by=revoked_by,
                now=now,
            )
            if changed:
                self._records[preauthorization_id] = record
            return record


def _mandatory(metadata: Mapping[str, Any]) -> bool:
    """Whether a REQUIRE_APPROVAL result is a mandatory approval, in either form the
    runtime reads: the ``cayu:mandatory_approval`` mark, or a grant policy's offer
    refused because a mandatory control covers the call."""

    offer = metadata.get(TOOL_AUTHORIZATION_GRANT_OFFER_METADATA_KEY)
    return metadata.get(TOOL_MANDATORY_APPROVAL_METADATA_KEY) is True or (
        type(offer) is dict
        and offer.get("grantable") is False
        and offer.get("reason") == MANDATORY_CONTROL_OFFER_REASON
    )


class PreauthorizedToolCallPolicy(ToolPolicy):
    """Turn REQUIRE_APPROVAL into ALLOW when a person preauthorized this exact call.

    ``then`` decides first (every call requires approval when omitted). Only a
    REQUIRE_APPROVAL decision for a tool in ``tools`` is considered, and only
    in a session that runs under a task (the runtime-owned task mark). The
    call's tool, exact arguments (by digest) and task, and the agent when the
    record names one, must match an available record; the store then marks it
    used by this call before ALLOW is returned. ALLOW and DENY pass through,
    a mandatory approval (``cayu:mandatory_approval``, or a wrapped grant
    policy's mandatory control) and a re-check of
    arguments a person changed still ask a person, and if the store fails the
    call needs approval. A dry run (``preview_tool_call``) reports what would
    happen without using the record.
    """

    def __init__(
        self,
        *,
        store: ToolCallPreauthorizationStore,
        tools: Iterable[str],
        then: ToolPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> None:
        if not isinstance(store, ToolCallPreauthorizationStore):
            raise TypeError("store must be a ToolCallPreauthorizationStore.")
        if then is not None and not isinstance(then, ToolPolicy):
            raise TypeError("then must be a ToolPolicy.")
        if isinstance(tools, str):
            raise TypeError("tools must be an iterable of tool names.")
        names = frozenset(_identifier(name, "tool_name") for name in tools)
        if not names:
            raise ValueError("tools names at least one tool.")
        self._store = store
        self._tools = names
        self._then = then
        self._clock = utc_clock(clock)
        self._execution_profile_identity = copy_execution_profile_behavior_identity(
            execution_profile_identity
        )

    @property
    def store(self) -> ToolCallPreauthorizationStore:
        return self._store

    @property
    def tools(self) -> frozenset[str]:
        return self._tools

    @property
    def then(self) -> ToolPolicy | None:
        return self._then

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        return copy_execution_profile_behavior_identity(self._execution_profile_identity)

    def _execution_profile_material(self) -> dict[str, object]:
        """Behavior inputs only; the records are durable data."""

        return {"tools": sorted(self._tools)}

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if self._then is None:
            base = ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Tool requires approval unless a person preauthorized this exact call.",
            )
        else:
            base = await self._then.authorize(request.model_copy(deep=True))
            if TOOL_CALL_PREAUTHORIZATION_METADATA_KEY in base.metadata:
                metadata = dict(base.metadata)
                del metadata[TOOL_CALL_PREAUTHORIZATION_METADATA_KEY]
                base = base.model_copy(update={"metadata": metadata})
        if (
            base.decision is not ToolPolicyDecision.REQUIRE_APPROVAL
            or _mandatory(base.metadata)
            or request.tool_name not in self._tools
            or request.metadata.get(APPROVED_ARGUMENTS_REAUTHORIZATION_METADATA_KEY) is True
        ):
            return base
        task_id = session_task_id(request.session)
        if task_id is None:
            return base
        digest = tool_arguments_digest(
            request.arguments if request.call_arguments is None else request.call_arguments
        )
        now = self._clock()
        try:
            if request.metadata.get(TOOL_POLICY_DRY_RUN_METADATA_KEY) is True:
                record = next(
                    (
                        record
                        for record in await self._store.list_for_task(task_id)
                        if record.available(now)
                        and record.matches(
                            task_id=task_id,
                            tool_name=request.tool_name,
                            arguments_digest=digest,
                            agent_name=request.agent.name,
                        )
                    ),
                    None,
                )
            else:
                record = await self._store.use(
                    task_id=task_id,
                    tool_name=request.tool_name,
                    arguments_digest=digest,
                    agent_name=request.agent.name,
                    use=ToolCallPreauthorizationUse(
                        session_id=request.session.id,
                        session_instance_id=request.session.instance_id,
                        tool_round_id=request.tool_round_id,
                        tool_call_id=request.tool_call_id,
                        used_at=now,
                    ),
                )
        except Exception:
            # A store that can't answer never authorizes; a person decides instead.
            return base
        if record is None:
            return base
        return ToolPolicyResult(
            decision=ToolPolicyDecision.ALLOW,
            reason="A person preauthorized this exact call for this task.",
            metadata={
                # The gate stays, so a grant policy around this one still matches its
                # gate-specific mandatory controls against the call.
                **(
                    {APPROVAL_GATE_METADATA_KEY: base.metadata[APPROVAL_GATE_METADATA_KEY]}
                    if APPROVAL_GATE_METADATA_KEY in base.metadata
                    else {}
                ),
                TOOL_CALL_PREAUTHORIZATION_METADATA_KEY: record.evidence_payload(),
            },
        )
