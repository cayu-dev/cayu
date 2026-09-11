"""Store-owned per-call effect state, shared by dispatch and reconciliation.

This is an internal transaction boundary, not a public receipt validator. Its
callers must supply frozen runtime authority and already projected evidence.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_bounded_durable_json_bytes,
    canonical_durable_json_bytes,
    copy_durable_json_object,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.failure_evidence import FailureEvidence
from cayu.runtime.sessions import (
    MAX_SESSION_ID_BYTES,
    EventQuery,
    RuntimePublicationMutation,
    Session,
    SessionOperationPublication,
    SessionStore,
    apply_runtime_publication_checkpoint_mutation,
)
from cayu.runtime.tool_effects import (
    ToolEffectConflict,
    ToolEffectReceipt,
    ToolEffectReconciliationResult,
    _bounded_text,
    _copy_string_map,
    copy_tool_effect_receipt,
)

EffectState = Literal[
    "prepared",
    "executing",
    "outcome_unknown",
    "completed",
    "failed",
    "reconciled_completed",
    "reconciled_failed",
]
_TERMINAL = frozenset({"completed", "failed", "reconciled_completed", "reconciled_failed"})
_NEXT = {
    "prepared": frozenset({"executing", "failed"}),
    "executing": frozenset({"outcome_unknown", "completed", "failed"}),
    "outcome_unknown": frozenset(
        {"outcome_unknown", "completed", "failed", "reconciled_completed", "reconciled_failed"}
    ),
}


class ToolEffectReconciliationRequired(RuntimeError):
    """An existing external call needs explicit receipt reconciliation, never replay."""

    def __init__(self) -> None:
        super().__init__(
            "An external tool effect is unresolved; explicit receipt reconciliation is required."
        )


class ToolEffectReconciliationCleanupFailure(ToolEffectReconciliationRequired):
    """Keep an unknown effect authoritative across a later bounded cleanup failure."""

    def __init__(
        self,
        primary: ToolEffectReconciliationRequired,
        cleanup: Exception,
    ) -> None:
        super().__init__()
        self.cleanup = cleanup
        self.failures = ExceptionGroup(
            "External tool uncertainty and observation cleanup failure.",
            [primary, cleanup],
        )


class ToolEffectIntent(BaseModel):
    """Immutable identity of the actual effective call, not its model proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr
    session_instance_id: StrictStr
    source_run_epoch: StrictInt = Field(ge=0)
    interaction_id: StrictStr
    model_step_id: StrictStr
    model_attempt_id: StrictStr
    tool_round_id: StrictStr
    tool_call_id: StrictStr
    agent_name: StrictStr
    tool_name: StrictStr
    idempotency_key: StrictStr
    effect: Literal["external"] = "external"
    execution_profile_fingerprint: StrictStr
    schema_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    arguments_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: StrictStr | None = None
    pause_id: StrictStr | None = None
    environment_name: StrictStr | None = None
    allocation_fingerprint: StrictStr | None = None
    reconciler_fingerprint: StrictStr | None = None
    targeted_invocation_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("*", mode="after")
    @classmethod
    def bound_identity(cls, value, info):
        if isinstance(value, str):
            return _bounded_text(
                value,
                info.field_name,
                maximum=MAX_SESSION_ID_BYTES if info.field_name == "session_id" else 256,
                identifier=True,
            )
        return value


class ToolEffectTerminal(BaseModel):
    """Selected result identity; its material lives in the durable stage/event."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    event_id: StrictStr = Field(min_length=1, max_length=256)
    result_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    receipt: ToolEffectReceipt | None = None
    reconciliation_request_digest: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("receipt", mode="before")
    @classmethod
    def detach_receipt(cls, value):
        if value is None:
            return None
        if type(value) is dict:
            return ToolEffectReceipt.model_validate(value)
        return copy_tool_effect_receipt(value)


class ToolEffectObservation(BaseModel):
    """Latest accepted nonterminal decision, never permission to redispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    event_id: StrictStr = Field(min_length=1, max_length=256)
    request_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    result: ToolEffectReconciliationResult

    @field_validator("result", mode="before")
    @classmethod
    def detach_result(cls, value):
        if type(value) is dict:
            copied = ToolEffectReconciliationResult.model_validate(value)
        elif type(value) is ToolEffectReconciliationResult:
            copied = ToolEffectReconciliationResult(
                **{
                    name: getattr(value, name)
                    for name in ToolEffectReconciliationResult.model_fields
                }
            )
        else:
            raise TypeError("Effect observation requires an exact reconciliation result.")
        if copied.receipt is not None:
            raise ValueError("Effect observation cannot select terminal receipt evidence.")
        return copied


class ToolEffectReconciliationAttempt(BaseModel):
    """One admitted request, retained until selected or durably superseded."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    event_id: StrictStr = Field(min_length=1, max_length=256)
    request_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: StrictInt = Field(ge=0)
    source_run_epoch: StrictInt = Field(ge=0)
    run_epoch: StrictInt = Field(ge=0)
    lookup: bool = Field(strict=True)


class ToolEffectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    intent: ToolEffectIntent
    revision: StrictInt = Field(ge=0)
    state: EffectState
    dispatch_id: StrictStr | None = None
    terminal: ToolEffectTerminal | None = None
    observation: ToolEffectObservation | None = None
    reconciliation_attempt: ToolEffectReconciliationAttempt | None = None
    resource_versions: dict[str, str] = Field(default_factory=dict)
    child_recovery_arguments: dict[str, Any] | None = Field(default=None, repr=False)
    publication_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("child_recovery_arguments", mode="before")
    @classmethod
    def detach_child_arguments(cls, value):
        if value is None:
            return None
        return copy_durable_json_object(value, "child_recovery_arguments")

    @field_validator("resource_versions", mode="before")
    @classmethod
    def detach_resources(cls, value):
        return _copy_string_map(value, "effect_resource_versions", maximum_items=32)

    @model_validator(mode="after")
    def validate_state(self) -> ToolEffectRecord:
        if self.child_recovery_arguments is not None and (
            sha256(
                canonical_durable_json_bytes(
                    self.child_recovery_arguments, "child_recovery_arguments"
                )
            ).hexdigest()
            != self.intent.arguments_digest
        ):
            raise ValueError("Child recovery arguments conflict with the prepared effect.")
        if self.reconciliation_attempt is not None and (
            self.state != "outcome_unknown"
            or self.revision != self.reconciliation_attempt.source_revision + 1
            or self.reconciliation_attempt.run_epoch < self.reconciliation_attempt.source_run_epoch
        ):
            raise ValueError("Effect reconciliation admission conflicts with its state.")
        undispatched_failure = self.state == "failed" and self.dispatch_id is None
        if undispatched_failure and (
            self.revision != 1 or self.observation is not None or self.resource_versions
        ):
            raise ValueError("Undispatched effect cannot carry execution evidence.")
        if (self.state == "prepared" or undispatched_failure) != (self.dispatch_id is None):
            raise ValueError("Effect dispatch identity conflicts with its state.")
        if (self.state in _TERMINAL) != (self.terminal is not None):
            raise ValueError("Effect terminal evidence conflicts with its state.")
        if self.observation is not None:
            if self.state in {"prepared", "executing"}:
                raise ValueError("Effect observation requires an unresolved or settled call.")
            if any(
                self.resource_versions.get(key) != value
                for key, value in self.observation.result.resource_versions.items()
            ):
                raise ValueError("Effect observation lost retained resource evidence.")
        if self.terminal is not None:
            reconciled = self.state.startswith("reconciled_")
            receipt = self.terminal.receipt
            if reconciled != (receipt is not None):
                raise ValueError("Effect reconciliation requires exactly one receipt.")
            if reconciled != (self.terminal.reconciliation_request_digest is not None):
                raise ValueError("Effect reconciliation requires exact request identity.")
            if receipt is not None and (
                self.state != f"reconciled_{receipt.outcome}"
                or receipt.tool_name != self.intent.tool_name
                or receipt.tool_call_id != self.intent.tool_call_id
                or receipt.idempotency_key != self.intent.idempotency_key
            ):
                raise ValueError("Effect receipt conflicts with its prepared call.")
            if receipt is not None and any(
                self.resource_versions.get(key) != value
                for key, value in receipt.resource_versions.items()
            ):
                raise ValueError("Effect receipt lost retained resource evidence.")
        return self


def _digest(value: object) -> str:
    return sha256(
        canonical_bounded_durable_json_bytes(
            value,
            "tool_effect_state",
            max_bytes=256 * 1024,
            max_nodes=16384,
            max_nesting=40,
        )
    ).hexdigest()


def _copy_model(value, model):
    if type(value) is not model:
        raise TypeError("Effect state requires an exact runtime model.")
    fields = {name: getattr(value, name) for name in model.model_fields}
    if model is ToolEffectRecord:
        fields["intent"] = _copy_model(value.intent, ToolEffectIntent)
        if value.terminal is not None:
            fields["terminal"] = _copy_model(value.terminal, ToolEffectTerminal)
        if value.observation is not None:
            fields["observation"] = _copy_model(value.observation, ToolEffectObservation)
        if value.reconciliation_attempt is not None:
            fields["reconciliation_attempt"] = _copy_model(
                value.reconciliation_attempt, ToolEffectReconciliationAttempt
            )
    return model(**fields)


def effect_storage_key(intent: ToolEffectIntent) -> str:
    intent = _copy_model(intent, ToolEffectIntent)
    return _call_storage_key(
        intent.session_id,
        intent.session_instance_id,
        intent.tool_round_id,
        intent.tool_call_id,
    )


def _call_storage_key(session_id: str, instance_id: str, round_id: str, call_id: str) -> str:
    return "tool-effect:v1:" + _digest([session_id, instance_id, round_id, call_id])


class _ExactReplay(Exception):
    pass


def validate_tool_effect_uncertainty_event(event: Event, record: ToolEffectRecord) -> None:
    """Authenticate historical uncertainty without changing current settlement.

    The record must be resolved from the store by the caller. A later receipt
    may settle the same dispatch without invalidating its earlier observation.
    """
    if (
        record.dispatch_id is None
        or event.type is not EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
        or event.session_id != record.intent.session_id
        or event.agent_name != record.intent.agent_name
        or event.environment_name != record.intent.environment_name
        or event.tool_name != record.intent.tool_name
        or event.interaction_id != record.intent.interaction_id
        or type(event.payload.get("schema_version")) is not int
        or event.payload.get("schema_version") != 1
        or event.payload.get("state") != "outcome_unknown"
        or any(
            event.payload.get(name) != getattr(record.intent, name)
            for name in (
                "model_step_id",
                "model_attempt_id",
                "tool_round_id",
                "tool_call_id",
                "approval_id",
            )
        )
        or event.payload.get("dispatch_id") != record.dispatch_id
        or event.payload.get("intent_digest") != _digest(record.intent.model_dump(mode="json"))
    ):
        raise ToolEffectConflict("Uncertainty event conflicts with its effect intent.")


class ToolEffectStateOwner:
    def __init__(self, store: SessionStore) -> None:
        self._store = store

    async def load_uncertainty_events(
        self,
        session: Session,
        *,
        tool_round_id: str,
        tool_call_ids: tuple[str, ...],
    ) -> list[Event]:
        """Read committed uncertainty evidence for this exact pending round.

        Delivery is owned by RuntimeEventWriter, not this state machine. Readback
        also covers a commit whose caller lost its acknowledgement or was cancelled.
        """
        expected = {}
        for call_id in tool_call_ids:
            record = await self.resolve_call(
                session, tool_round_id=tool_round_id, tool_call_id=call_id
            )
            if record is not None and record.state == "outcome_unknown":
                expected[call_id] = record
        if not expected:
            return []
        found: dict[str, Event] = {}
        for step_id in dict.fromkeys(record.intent.model_step_id for record in expected.values()):
            after_sequence = None
            while True:
                rows = await self._store.query_events(
                    EventQuery(
                        session_id=session.id,
                        model_step_id=step_id,
                        event_type=EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
                        after_sequence=after_sequence,
                        limit=100,
                    )
                )
                for row in rows:
                    event = row.event
                    if event.payload.get("tool_round_id") != tool_round_id:
                        continue
                    call_id = event.payload.get("tool_call_id")
                    if type(call_id) is not str or call_id not in expected:
                        continue
                    record = expected[call_id]
                    if call_id in found:
                        raise ToolEffectConflict(
                            "Uncertainty event conflicts with its effect intent."
                        )
                    validate_tool_effect_uncertainty_event(event, record)
                    found[call_id] = event
                if len(rows) < 100:
                    break
                after_sequence = rows[-1].sequence
        if found.keys() != expected.keys():
            raise ToolEffectConflict("Unresolved effect has no exact uncertainty event.")
        return [found[call_id] for call_id in expected]

    async def resolve_call(
        self,
        session: Session,
        *,
        tool_round_id: str,
        tool_call_id: str,
    ) -> ToolEffectRecord | None:
        """Resolve store-owned evidence; this does not authenticate a supplied receipt."""
        key = _call_storage_key(session.id, session.instance_id, tool_round_id, tool_call_id)
        raw = await self._store.load_session_operation(session.id, key)
        if raw is None:
            return None
        record = ToolEffectRecord.model_validate(raw)
        if (
            record.intent.session_id != session.id
            or record.intent.session_instance_id != session.instance_id
            or record.intent.tool_round_id != tool_round_id
            or record.intent.tool_call_id != tool_call_id
        ):
            raise ToolEffectConflict("Stored effect record conflicts with its call scope.")
        return record

    async def preserve_unresolved(
        self,
        session: Session,
        *,
        tool_round_id: str,
        tool_call_ids: tuple[str, ...],
        failure_evidence: FailureEvidence | None = None,
    ) -> bool:
        """Fence consumed calls before any generic interrupted-round synthesis.

        A selected terminal is positive durable evidence. Prepared calls remain
        fenced until recovery atomically proves non-dispatch and selects a result.
        Only consumed, nonterminal calls become unknown.
        The returned boolean reports unresolved work; it is not replay authority.
        """
        unresolved = False
        for call_id in tool_call_ids:
            record = await self.resolve_call(
                session, tool_round_id=tool_round_id, tool_call_id=call_id
            )
            if record is None or record.state not in {"prepared", "executing", "outcome_unknown"}:
                continue
            if record.state == "executing":
                await self.transition(
                    record,
                    state="outcome_unknown",
                    run_epoch=session.run_epoch,
                    failure_evidence=failure_evidence,
                )
            unresolved = True
        return unresolved

    async def require_unverified_recovery_allowed(
        self,
        session: Session,
        *,
        tool_round_id: str,
        tool_call_id: str,
    ) -> None:
        """Prevent caller-supplied outcomes from replacing protected effect evidence.

        This is the existing manual-result entrance, not receipt validation.
        Unknown effects require the registered validator; an already selected
        terminal must be resumed from its original evidence, never overwritten.
        A prepared-but-unconsumed effect cannot accept a claim that it executed.
        Calls outside this effect protocol retain their existing manual contract.
        """
        record = await self.resolve_call(
            session, tool_round_id=tool_round_id, tool_call_id=tool_call_id
        )
        if record is None:
            return
        if record.state == "executing":
            await self.transition(record, state="outcome_unknown", run_epoch=session.run_epoch)
        if record.state in {"executing", "outcome_unknown"}:
            raise ToolEffectReconciliationRequired()
        raise ToolEffectConflict("Manual recovery cannot replace durable external-effect evidence.")

    async def load(self, intent: ToolEffectIntent) -> ToolEffectRecord | None:
        intent = _copy_model(intent, ToolEffectIntent)
        raw = await self._store.load_session_operation(
            intent.session_id, effect_storage_key(intent)
        )
        if raw is None:
            return None
        record = ToolEffectRecord.model_validate(raw)
        if record.intent != intent:
            raise ToolEffectConflict("Stored effect intent differs from the expected call.")
        return record

    async def prepare(
        self,
        intent: ToolEffectIntent,
        *,
        run_epoch: int,
        child_recovery_arguments: dict[str, Any] | None = None,
    ) -> ToolEffectRecord:
        intent = _copy_model(intent, ToolEffectIntent)
        return await self._publish(
            intent=intent,
            expected=None,
            state="prepared",
            dispatch_id=None,
            terminal=None,
            run_epoch=run_epoch,
            mutation=RuntimePublicationMutation(),
            events=(),
            child_recovery_arguments=child_recovery_arguments,
        )

    async def begin(
        self,
        intent: ToolEffectIntent,
        *,
        run_epoch: int,
        child_recovery_arguments: dict[str, Any] | None = None,
    ) -> ToolEffectRecord:
        """Durably prepare and consume one call before its external implementation starts.

        A prior executing/unknown/terminal record cannot be mistaken for a fresh
        dispatch grant. Even exact duplicate preparation does not permit a
        second consumer of the prepared revision.
        """
        prepared = await self.prepare(
            intent, run_epoch=run_epoch, child_recovery_arguments=child_recovery_arguments
        )
        return await self.transition(prepared, state="executing", run_epoch=run_epoch)

    async def start_reconciliation(
        self,
        expected: ToolEffectRecord,
        *,
        source_run_epoch: int,
        run_epoch: int,
        request_digest: str,
        lookup: bool,
        event: Event,
    ) -> ToolEffectRecord:
        """Bind admission and its event before an application callback can start."""
        expected = _copy_model(expected, ToolEffectRecord)
        if expected.state != "outcome_unknown":
            raise ToolEffectConflict("Reconciliation admission requires an uncertain effect.")
        attempt = ToolEffectReconciliationAttempt(
            event_id=event.id,
            request_digest=request_digest,
            source_revision=expected.revision,
            source_run_epoch=source_run_epoch,
            run_epoch=run_epoch,
            lookup=lookup,
        )
        return await self._publish(
            intent=expected.intent,
            expected=expected,
            state=expected.state,
            dispatch_id=expected.dispatch_id,
            terminal=None,
            reconciliation_attempt=attempt,
            run_epoch=run_epoch,
            mutation=RuntimePublicationMutation(),
            events=(event,),
        )

    async def transition(
        self,
        expected: ToolEffectRecord,
        *,
        state: EffectState,
        run_epoch: int,
        terminal: ToolEffectTerminal | None = None,
        observation: ToolEffectObservation | None = None,
        mutation: RuntimePublicationMutation | None = None,
        events: tuple[Event, ...] = (),
        failure_evidence: FailureEvidence | None = None,
    ) -> ToolEffectRecord:
        expected = _copy_model(expected, ToolEffectRecord)
        if state not in _NEXT.get(expected.state, frozenset()):
            raise ToolEffectConflict("Effect state transition is not permitted.")
        if (expected.state == state) != (observation is not None):
            raise ToolEffectConflict("Effect observation requires an explicit unresolved decision.")
        if observation is not None:
            observation = _copy_model(observation, ToolEffectObservation)
            if expected.observation is not None and (
                expected.observation.request_digest == observation.request_digest
            ):
                raise ToolEffectConflict("Effect observation requires a new explicit decision.")
        attempt = expected.reconciliation_attempt
        selected_request = (
            observation.request_digest
            if observation is not None
            else None
            if terminal is None
            else terminal.reconciliation_request_digest
        )
        if (
            attempt is not None
            and selected_request is not None
            and (selected_request != attempt.request_digest)
        ):
            raise ToolEffectConflict("Effect settlement belongs to another admitted request.")
        return await self._publish(
            intent=expected.intent,
            expected=expected,
            state=state,
            dispatch_id=str(uuid4()) if state == "executing" else expected.dispatch_id,
            terminal=terminal,
            observation=observation,
            run_epoch=run_epoch,
            mutation=RuntimePublicationMutation() if mutation is None else mutation,
            events=events,
            failure_evidence=failure_evidence,
        )

    async def _publish(
        self,
        *,
        intent: ToolEffectIntent,
        expected: ToolEffectRecord | None,
        state: EffectState,
        dispatch_id: str | None,
        terminal: ToolEffectTerminal | None,
        run_epoch: int,
        mutation: RuntimePublicationMutation,
        events: tuple[Event, ...],
        observation: ToolEffectObservation | None = None,
        reconciliation_attempt: ToolEffectReconciliationAttempt | None = None,
        failure_evidence: FailureEvidence | None = None,
        child_recovery_arguments: dict[str, Any] | None = None,
    ) -> ToolEffectRecord:
        if type(run_epoch) is not int or run_epoch < 0:
            raise ValueError("Effect publication requires an exact run epoch.")
        mutation = RuntimePublicationMutation(operations=mutation.operations)
        events = tuple(copy_event(event) for event in events)
        if failure_evidence is not None:
            failure_evidence = _copy_model(failure_evidence, FailureEvidence)
            if (
                expected is None
                or expected.state != "executing"
                or state != "outcome_unknown"
                or failure_evidence.session_id != intent.session_id
                or failure_evidence.run_epoch != run_epoch
            ):
                raise ToolEffectConflict("Failure evidence does not match the uncertain dispatch.")
        if expected is not None and expected.state == "executing" and state == "outcome_unknown":
            events = (
                *events,
                Event(
                    type=EventType.TOOL_EFFECT_OUTCOME_UNKNOWN,
                    session_id=intent.session_id,
                    interaction_id=intent.interaction_id,
                    agent_name=intent.agent_name,
                    environment_name=intent.environment_name,
                    tool_name=intent.tool_name,
                    payload={
                        "schema_version": 1,
                        "state": "outcome_unknown",
                        **(
                            {"failure_evidence": failure_evidence.model_dump(mode="json")}
                            if failure_evidence is not None
                            else {}
                        ),
                        "record_revision": expected.revision + 1,
                        "intent_digest": _digest(intent.model_dump(mode="json")),
                        "dispatch_id": dispatch_id,
                        **{
                            name: getattr(intent, name)
                            for name in (
                                "model_step_id",
                                "model_attempt_id",
                                "tool_round_id",
                                "tool_call_id",
                                "approval_id",
                            )
                        },
                    },
                ),
            )
        if any(event.session_id != intent.session_id for event in events):
            raise ToolEffectConflict("Effect publication event belongs to another session.")
        terminal = None if terminal is None else _copy_model(terminal, ToolEffectTerminal)
        reconciliation_attempt = (
            None
            if reconciliation_attempt is None
            else _copy_model(reconciliation_attempt, ToolEffectReconciliationAttempt)
        )
        observation = (
            (None if expected is None else expected.observation)
            if observation is None
            else _copy_model(observation, ToolEffectObservation)
        )
        resources = {} if expected is None else dict(expected.resource_versions)
        incoming = observation.result.resource_versions if observation is not None else {}
        if terminal is not None and terminal.receipt is not None:
            incoming = {**incoming, **terminal.receipt.resource_versions}
        for key, value in incoming.items():
            if key in resources and resources[key] != value:
                raise ToolEffectConflict(
                    "Effect evidence conflicts with a retained resource version."
                )
            resources[key] = value
        resources = _copy_string_map(resources, "effect_resource_versions", maximum_items=32)
        child_recovery_arguments = (
            child_recovery_arguments if expected is None else expected.child_recovery_arguments
        )
        child_recovery_arguments = (
            None
            if child_recovery_arguments is None
            else copy_durable_json_object(child_recovery_arguments, "child_recovery_arguments")
        )
        material = {
            "intent": intent.model_dump(mode="json"),
            "state": state,
            "dispatch_id": dispatch_id,
            # Argument content is already bound by intent.arguments_digest.
            # Do not duplicate potentially large private inputs in the bounded
            # publication material on every state transition.
            "expected": (
                None
                if expected is None
                else expected.model_dump(mode="json", exclude={"child_recovery_arguments"})
            ),
            "terminal": None if terminal is None else terminal.model_dump(mode="json"),
            "observation": None if observation is None else observation.model_dump(mode="json"),
            "resource_versions": resources,
            "has_child_recovery_arguments": child_recovery_arguments is not None,
            "reconciliation_attempt": (
                None
                if reconciliation_attempt is None
                else reconciliation_attempt.model_dump(mode="json")
            ),
            "mutation_digest": sha256(
                canonical_durable_json_bytes(
                    mutation.model_dump(mode="json"),
                    "effect_checkpoint_mutation",
                )
            ).hexdigest(),
            "event_digests": [
                sha256(
                    canonical_durable_json_bytes(
                        event.model_dump(mode="json"),
                        "effect_event",
                    )
                ).hexdigest()
                for event in events
            ],
            "run_epoch": run_epoch,
        }
        prior_attempt = None if expected is None else expected.reconciliation_attempt
        if prior_attempt is not None and (
            reconciliation_attempt is not None
            or (terminal is not None and terminal.reconciliation_request_digest is None)
        ):
            # The successor owns this audit atomically. It does not depend on
            # the superseded callback returning or retaining a live waiter.
            payload = {
                "schema_version": 1,
                "kind": "reconciliation_superseded",
                "intent_digest": _digest(intent.model_dump(mode="json")),
                "request_digest": prior_attempt.request_digest,
                "attempt_digest": _digest(prior_attempt.model_dump(mode="json")),
                "selection_digest": _digest(material),
                "source_run_epoch": prior_attempt.run_epoch,
                **{
                    name: getattr(intent, name)
                    for name in (
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "tool_call_id",
                        "approval_id",
                    )
                },
            }
            audit = Event(
                id="tool-effect-reconciliation-conflict:v1:" + _digest(payload),
                type=EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT,
                session_id=intent.session_id,
                interaction_id=intent.interaction_id,
                agent_name=intent.agent_name,
                environment_name=intent.environment_name,
                tool_name=intent.tool_name,
                payload=payload,
            )
            events = (*events, audit)
            # First commit owns audit time. Reconstructing the exact successor
            # must not invent a conflict merely because its audit is sampled later.
            material["supersession_audit_digest"] = _digest(
                audit.model_dump(mode="json", exclude={"timestamp"})
            )
        proposed = ToolEffectRecord(
            intent=intent,
            revision=0 if expected is None else expected.revision + 1,
            state=state,
            dispatch_id=dispatch_id,
            terminal=terminal,
            observation=observation,
            resource_versions=resources,
            child_recovery_arguments=child_recovery_arguments,
            reconciliation_attempt=reconciliation_attempt,
            publication_digest=_digest(material),
        )
        operation = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: self._publish_owned(
                    expected=expected,
                    proposed=proposed,
                    run_epoch=run_epoch,
                    mutation=mutation,
                    events=events,
                )
            )
        )
        outcome = await await_shielded_task_outcome(operation)
        captured = outcome.result
        error = outcome.error if captured is None else captured.error
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            if error is not None and error is not outcome.cancellation:
                raise outcome.cancellation from error
            raise outcome.cancellation
        if error is not None:
            if isinstance(error, asyncio.CancelledError):
                raise unexpected_child_cancellation_error(
                    error,
                    operation="Tool effect publication",
                ) from error
            raise error
        if captured is None or captured.result is None:
            raise RuntimeError("Effect publication returned no record.")
        return captured.result

    async def _publish_owned(
        self,
        *,
        expected: ToolEffectRecord | None,
        proposed: ToolEffectRecord,
        run_epoch: int,
        mutation: RuntimePublicationMutation,
        events: tuple[Event, ...],
    ) -> ToolEffectRecord:
        intent = proposed.intent
        key = effect_storage_key(intent)
        proposed_json = proposed.model_dump(mode="json")

        def transform(session: Session, checkpoint: dict[str, Any] | None, raw: dict | None):
            if session.instance_id != intent.session_instance_id:
                raise ToolEffectConflict("Effect session incarnation changed.")
            current = None if raw is None else ToolEffectRecord.model_validate(raw)
            if current is None and intent.source_run_epoch != session.run_epoch:
                raise ToolEffectConflict("Effect preparation has a stale source epoch.")
            if current == proposed:
                raise _ExactReplay
            if current != expected:
                raise ToolEffectConflict("Effect compare-and-set lost its exact prior state.")
            updated = apply_runtime_publication_checkpoint_mutation(mutation, checkpoint)
            if proposed.reconciliation_attempt is not None:
                _validate_selected_reconciliation_attempt(proposed, events)
            if proposed.terminal is not None:
                _validate_selected_terminal(proposed, updated, events)
            if proposed.observation is not None and (
                expected is None or proposed.observation != expected.observation
            ):
                _validate_selected_observation(proposed, events)
            return SessionOperationPublication(
                checkpoint={} if updated is None else updated,
                operation_records={key: copy_durable_json_object(proposed_json, "effect_record")},
            )

        try:
            await self._store.publish_session_operation(
                intent.session_id,
                idempotency_key=key,
                operation_transform=transform,
                events=list(events),
                expected_run_epoch=run_epoch,
            )
        except _ExactReplay:
            return _copy_model(proposed, ToolEffectRecord)
        except ToolEffectConflict:
            raise
        except Exception as publication_error:
            try:
                observed = await self.load(intent)
            except Exception as readback_error:
                raise ExceptionGroup(
                    "Effect publication and exact readback failed.",
                    [publication_error, readback_error],
                ) from None
            if observed != proposed:
                raise
        return _copy_model(proposed, ToolEffectRecord)


def _validate_selected_reconciliation_attempt(
    record: ToolEffectRecord, events: tuple[Event, ...]
) -> None:
    attempt = record.reconciliation_attempt
    assert attempt is not None
    intent = record.intent
    candidates = [event for event in events if event.id == attempt.event_id]
    if len(candidates) != 1:
        raise ToolEffectConflict("Effect admission has no exact atomic start event.")
    event = candidates[0]
    if (
        event.type != EventType.TOOL_EFFECT_RECONCILIATION_STARTED
        or event.session_id != intent.session_id
        or event.interaction_id != intent.interaction_id
        or event.agent_name != intent.agent_name
        or event.tool_name != intent.tool_name
        or _digest(event.payload)
        != _digest(
            {
                "schema_version": 1,
                "request_digest": attempt.request_digest,
                "intent_digest": _digest(intent.model_dump(mode="json")),
                "dispatch_id": record.dispatch_id,
                "expected_revision": attempt.source_revision,
                "expected_run_epoch": attempt.source_run_epoch,
                "lookup": attempt.lookup,
                "execution_profile_fingerprint": intent.execution_profile_fingerprint,
                **{
                    name: getattr(intent, name)
                    for name in (
                        "model_step_id",
                        "model_attempt_id",
                        "tool_round_id",
                        "tool_call_id",
                        "approval_id",
                    )
                },
            }
        )
    ):
        raise ToolEffectConflict("Effect admission event conflicts with its selected evidence.")


def _validate_selected_observation(record: ToolEffectRecord, events: tuple[Event, ...]) -> None:
    observation = record.observation
    assert observation is not None
    candidates = [event for event in events if event.id == observation.event_id]
    if len(candidates) != 1:
        raise ToolEffectConflict("Effect observation has no exact atomic evidence event.")
    event = candidates[0]
    intent = record.intent
    rejected = observation.result.outcome == "conflict"
    if (
        event.type
        != (
            EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT
            if rejected
            else EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED
        )
        or event.session_id != intent.session_id
        or event.interaction_id != intent.interaction_id
        or event.agent_name != intent.agent_name
        or event.tool_name != intent.tool_name
        or event.payload
        != {
            "schema_version": 1,
            **({"kind": "validator_rejected"} if rejected else {}),
            "execution_profile_fingerprint": intent.execution_profile_fingerprint,
            **({"approval_id": intent.approval_id} if intent.approval_id is not None else {}),
            **({"input_id": intent.pause_id} if intent.pause_id is not None else {}),
            **{
                name: getattr(intent, name)
                for name in (
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "tool_call_id",
                    "idempotency_key",
                )
            },
            "request_digest": observation.request_digest,
            "result": observation.result.model_dump(mode="json"),
            "resource_versions": record.resource_versions,
        }
    ):
        raise ToolEffectConflict("Effect observation event conflicts with its selected evidence.")


def is_command_policy_refusal_terminal(event: Event) -> bool:
    """Classify runtime-owned terminal material, never a caller's ToolResult.

    The executor produces this event only from the invocation's source-bound
    policy-denial signal. Reconstructed events must come from the same durable
    staging/publication owner as other effect terminals.
    """

    result = event.payload.get("result")
    return (
        event.type is EventType.TOOL_CALL_BLOCKED
        and event.payload.get("denied_by") == "command_policy"
        and type(event.payload.get("decision")) is str
        and event.payload.get("decision") in {"deny", "require_command_approval"}
        and type(result) is dict
        and result.get("is_error") is True
    )


def _validate_selected_terminal(
    record: ToolEffectRecord,
    checkpoint: dict[str, Any] | None,
    events: tuple[Event, ...],
) -> None:
    from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint
    from cayu.runtime.user_input import pending_user_input_from_checkpoint

    terminal = record.terminal
    assert terminal is not None
    intent = record.intent
    candidates = [event for event in events if event.id == terminal.event_id]
    if intent.pause_id is not None:
        pending_input = pending_user_input_from_checkpoint(checkpoint)
        if pending_input is not None and (
            pending_input.input_id != intent.pause_id
            or any(
                getattr(pending_input, field) != getattr(intent, field)
                for field in ("model_step_id", "model_attempt_id", "tool_round_id")
            )
        ):
            raise ToolEffectConflict("Effect terminal belongs to a different user-input pause.")
        stages = () if pending_input is None else pending_input.staged_terminals
    else:
        pending = pending_tool_round_from_checkpoint(checkpoint)
        stages = () if pending is None else pending.staged_terminals
    if stages:
        candidates.extend(stage.event for stage in stages if stage.event.id == terminal.event_id)
    if not candidates:
        raise ToolEffectConflict("Effect settlement has no atomic terminal material.")
    expected_type = (
        EventType.TOOL_CALL_FAILED
        if record.state.endswith("failed")
        else EventType.TOOL_CALL_COMPLETED
    )
    for event in candidates:
        if record.dispatch_id is None:
            result = event.payload.get("result")
            structured = result.get("structured") if type(result) is dict else None
            if (
                record.state != "failed"
                or type(result) is not dict
                or result.get("is_error") is not True
                or type(structured) is not dict
                or structured.get("executed") is not False
                or structured.get("outcome_unknown") is not False
            ):
                raise ToolEffectConflict("Undispatched settlement requires non-execution evidence.")
        if (
            (
                event.type != expected_type
                and not (
                    record.state == "failed"
                    and record.dispatch_id is not None
                    and terminal.receipt is None
                    and is_command_policy_refusal_terminal(event)
                )
            )
            or event.session_id != intent.session_id
            or event.tool_name != intent.tool_name
            or event.agent_name != intent.agent_name
            or event.payload.get("approval_id") != intent.approval_id
            or event.payload.get("input_id") != intent.pause_id
            or any(
                event.payload.get(field) != getattr(intent, field)
                for field in (
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "tool_call_id",
                    "idempotency_key",
                )
            )
            or type(event.payload.get("result")) is not dict
            or sha256(
                canonical_durable_json_bytes(event.payload["result"], "effect_terminal_result")
            ).hexdigest()
            != terminal.result_digest
        ):
            raise ToolEffectConflict("Effect terminal material conflicts with its selection.")
        if terminal.receipt is not None:
            receipt = terminal.receipt
            result = event.payload["result"]
            if (
                result.get("content") != receipt.message
                or result.get("structured") != receipt.structured
                or result.get("is_error") is not (receipt.outcome == "failed")
                or result.get("artifacts") != []
            ):
                raise ToolEffectConflict("Effect terminal result differs from its receipt.")
