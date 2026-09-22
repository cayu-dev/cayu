"""Bounded collaboration waits over authenticated request outcomes.

The models in this module are data contracts only.  They do not authenticate
callers or run source work; the collaboration store and its registered source
owner perform those responsibilities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal, cast

from pydantic import Field, StrictInt, StrictStr, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    ContractValue,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
    snapshot_input,
)
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.requests import RequestCommand, RequestRef
from cayu.runtime._session_continuation import ContinuationTicket
from cayu.vaults.redaction import SecretRedactor

_CONTRACT_REDACTOR = SecretRedactor()


def _encoded(value: object) -> bytes:
    return canonical_durable_json_bytes(snapshot_input(value), "collaboration wait")


WaitPredicate = Literal["ALL_SUCCESS", "ALL_SETTLED", "ANY_SUCCESS", "QUORUM_SUCCESS"]
WaitState = Literal["pending", "elected", "cancelled", "expired", "unavailable"]
WaitEvidenceStatus = Literal["success", "failure", "settled", "ambiguous", "unavailable"]
WaitDelivery = Literal["none", "pending", "accepted", "excluded"]
WaitResult = Literal["success", "settled", "failure", "unavailable"]

MAX_WAIT_TARGETS = 64
MAX_WAIT_EVIDENCE = 64
MAX_WAIT_EVENTS = 128
# A wait embeds its bounded event/evidence history in one durable operation,
# but its future terminal publication still needs capacity reserved at
# registration.  The reservation is released only by terminal delivery or a
# terminal wait with no delivery handoff.
WAIT_RESERVED_BYTES = 4 * MAX_ENVELOPE_BYTES
WAIT_RESERVED_EVENTS = MAX_WAIT_EVENTS


def _utc(value: str, field: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware.")
    return parsed.astimezone(UTC).isoformat()


def deadline_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    return int(parsed.timestamp() * 1000)


class CollaborationWait(ContractValue):
    """Immutable finite wait registration.

    Version one qualifies collaboration request references as its source.  The
    target identity remains explicit so later source families can be added
    without changing the election contract.
    """

    operation: OperationRef
    source_owner: OwnerRef
    targets: tuple[RequestCommand, ...] = Field(min_length=1, max_length=MAX_WAIT_TARGETS)
    predicate: WaitPredicate
    predicate_version: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    threshold: StrictInt | None = Field(default=None, ge=1, le=MAX_WAIT_TARGETS)
    deadline: StrictStr
    failure_policy: StrictStr = Field(min_length=1, max_length=128)
    service_policy: StrictStr = Field(min_length=1, max_length=128)
    projection: ObjectRef | None = None
    wait_edge_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    initiator: InitiatorBinding
    delivery_ticket: ContinuationTicket | None = None

    unordered_fields = frozenset({"targets"})

    @model_validator(mode="before")
    @classmethod
    def normalize_exact_duplicate_targets(cls, value: object) -> object:
        """Remove identical unordered targets before base uniqueness checks."""
        if type(value) is not dict:
            return value
        raw_targets = cast("dict[str, object]", value).get("targets")
        if not isinstance(raw_targets, (list, tuple)) or len(raw_targets) > MAX_WAIT_TARGETS:
            return value
        unique: list[object] = []
        seen: set[bytes] = set()
        for target in raw_targets:
            try:
                key = canonical_durable_json_bytes(snapshot_input(target), "wait target")
            except (AttributeError, TypeError, ValueError):
                return value
            if key not in seen:
                seen.add(key)
                unique.append(target)
        if len(unique) == len(raw_targets):
            return value
        copied = dict(value)
        copied["targets"] = tuple(unique)
        return copied

    @classmethod
    def from_values(cls, **values: object) -> CollaborationWait:
        return cls.model_validate(values)

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(sorted(ref_key(target.intent.selection.reference) for target in self.targets))

    @property
    def source_keys(self) -> tuple[str, ...]:
        return tuple(sorted(source_key(target) for target in self.targets))

    @model_validator(mode="after")
    def validate_wait(self) -> CollaborationWait:
        # The raw tuple is bounded by the field constraint before this
        # normalization.  Exact duplicate descriptions are harmless and are
        # counted once; two descriptions of one request identity are a
        # content conflict, not an unordered-set normalization.
        unique_targets: list[RequestCommand] = []
        by_reference: dict[str, RequestCommand] = {}
        for target in self.targets:
            key = ref_key(target.intent.selection.reference)
            if (
                target.operation.application_scope != self.operation.application_scope
                or target.operation.namespace_incarnation != self.operation.namespace_incarnation
                or target.source != self.source_owner
            ):
                raise ValueError("Wait target belongs to another source namespace or owner.")
            previous = by_reference.get(key)
            if previous is None:
                by_reference[key] = target
                unique_targets.append(target)
            elif previous != target:
                raise ValueError("Wait targets conflict for one request identity.")
        if len(unique_targets) != len(self.targets):
            object.__setattr__(self, "targets", tuple(unique_targets))
        if self.deadline != _utc(self.deadline, "deadline"):
            raise ValueError("Wait deadline must use canonical UTC format.")
        if self.predicate_version != 1:
            raise ValueError("Unsupported wait predicate version.")
        if self.predicate == "QUORUM_SUCCESS":
            if self.threshold is None or self.threshold > len(unique_targets):
                raise ValueError("Quorum threshold must fit the target set.")
        elif self.threshold is not None:
            raise ValueError("Only QUORUM_SUCCESS accepts a threshold.")
        if len(set(self.target_keys)) != len(self.targets):
            raise ValueError("Wait targets must be distinct.")
        if self.initiator.issuer != self.source_owner:
            raise ValueError("Wait initiator must belong to the source owner.")
        if self.delivery_ticket is not None:
            ticket = self.delivery_ticket
            if (
                ticket.predicate_kind != self.predicate
                or ticket.predicate_version != self.predicate_version
                or ticket.threshold != self.threshold
                or ticket.deadline != self.deadline
                or ticket.failure_policy != self.failure_policy
                or ticket.service_policy != self.service_policy
                or ticket.wait_edge_revision != self.wait_edge_revision
                or set(ticket.targets)
                != {
                    request_object_ref(target.intent.selection.reference) for target in self.targets
                }
            ):
                raise ValueError("Session ticket differs from the wait contract.")
        encoded = _encoded(self)
        if len(encoded) > MAX_ENVELOPE_BYTES:
            raise ValueError("Wait registration exceeds its envelope.")
        # Admission must prove that the largest mandatory terminal snapshot
        # can fit, not merely that registration fits.  Evidence is source
        # owned and bounded independently from the request commands, so use a
        # conservative serialized probe for every distinct target.
        _require_terminal_capacity(self)
        return self


class WaitRegistration(ContractValue):
    mode: Literal["collaboration_wait"] = "collaboration_wait"
    wait: CollaborationWait
    registration_digest: Identifier
    registered_sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)

    @model_validator(mode="after")
    def exact_registration(self) -> WaitRegistration:
        expected = wait_identity_digest(self.wait)
        if self.registration_digest != expected:
            raise ValueError("Wait registration is not bound to its intent.")
        if (
            len(
                canonical_durable_json_bytes(
                    {"status": "match", "receipt": self.model_dump(mode="json")},
                    "collaboration wait exact lookup",
                )
            )
            > MAX_ENVELOPE_BYTES
        ):
            raise ValueError("Wait exact lookup exceeds its envelope.")
        return self


class WaitEvidence(ContractValue):
    target: RequestRef
    status: WaitEvidenceStatus
    # Assigned by the collaboration store when the evidence is ingested.
    # Source event order is deliberately not used for election ordering:
    # different source requests have independent event streams.
    ingestion_sequence: StrictInt = Field(default=0, ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    source_sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    accepted_at_ms: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    observed_at_ms: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    receipt_digest: Identifier
    commitment: Identifier | None = None

    @model_validator(mode="after")
    def bounded_evidence(self) -> WaitEvidence:
        if self.observed_at_ms < self.accepted_at_ms:
            raise ValueError("Evidence observation precedes source acceptance.")
        if len(_encoded(self)) > MAX_ENVELOPE_BYTES:
            raise ValueError("Wait evidence exceeds its envelope.")
        return self


class WaitElection(ContractValue):
    result: Literal["success", "settled", "failure", "unavailable"]
    selected: tuple[RequestRef, ...] = Field(max_length=MAX_WAIT_TARGETS)
    evidence: tuple[WaitEvidence, ...] = Field(max_length=MAX_WAIT_EVIDENCE)
    sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    elected_at_ms: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    outcome_digest: Identifier

    unordered_fields = frozenset({"selected", "evidence"})

    @model_validator(mode="after")
    def exact_election(self) -> WaitElection:
        if len(_encoded(self)) > MAX_ENVELOPE_BYTES:
            raise ValueError("Wait election exceeds its envelope.")
        if (
            self.outcome_digest
            != sha256(_encoded(self.model_copy(update={"outcome_digest": ""}))).hexdigest()
        ):
            raise ValueError("Wait election digest is not content-bound.")
        return self


class WaitEvent(ContractValue):
    sequence: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    kind: Literal["registered", "evidence", "elected", "cancelled", "expired", "released"]
    target: RequestRef | None = None
    evidence_digest: Identifier | None = None


def wait_operation_key(wait: CollaborationWait | OperationRef) -> tuple[str, int, str]:
    operation = wait.operation if isinstance(wait, CollaborationWait) else wait
    return operation.namespace_incarnation, operation.generation, operation.caller_key


def wait_identity_payload(wait: CollaborationWait) -> dict[str, object]:
    """Return immutable registration identity, excluding ticket lifecycle state."""
    payload = wait.model_dump(mode="json")
    ticket = payload.get("delivery_ticket")
    if isinstance(ticket, dict):
        ticket.pop("state", None)
        ticket.pop("revision", None)
    return payload


def wait_identity_digest(wait: CollaborationWait) -> str:
    return sha256(_encoded(wait_identity_payload(wait))).hexdigest()


class WaitSnapshot(ContractValue):
    mode: Literal["collaboration_wait"] = "collaboration_wait"
    registration: WaitRegistration
    state: WaitState
    delivery: WaitDelivery = "none"
    evidence: tuple[WaitEvidence, ...] = Field(default=(), max_length=MAX_WAIT_EVIDENCE)
    election: WaitElection | None = None
    revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    source_pins: tuple[str, ...] = Field(default=(), max_length=MAX_WAIT_TARGETS)
    delivery_receipt_digest: Identifier | None = None
    terminal_at_ms: StrictInt | None = Field(default=None, ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    reserved_bytes: StrictInt = Field(default=WAIT_RESERVED_BYTES, ge=0)
    reserved_events: StrictInt = Field(default=WAIT_RESERVED_EVENTS, ge=0)
    events: tuple[WaitEvent, ...] = Field(default=(), max_length=MAX_WAIT_EVENTS)

    @model_validator(mode="after")
    def coherent_snapshot(self) -> WaitSnapshot:
        if (
            self.registration.wait.operation.application_scope
            != self.registration.wait.source_owner.application_scope
        ):
            raise ValueError("Wait registration scope conflicts with its source owner.")
        if self.state == "elected" and self.election is None:
            raise ValueError("Elected wait requires an election.")
        if self.state != "elected" and self.election is not None:
            raise ValueError("Only an elected wait may carry an election.")
        if self.delivery != "none" and self.state not in {
            "elected",
            "cancelled",
            "expired",
            "unavailable",
        }:
            raise ValueError("Wait delivery state conflicts with wait lifecycle.")
        if self.delivery in {"accepted", "excluded"} and self.delivery_receipt_digest is None:
            raise ValueError("Settled wait delivery requires its receipt digest.")
        if self.delivery == "none" and self.delivery_receipt_digest is not None:
            raise ValueError("Unsettled wait cannot carry a delivery receipt.")
        if self.delivery == "accepted" and self.state != "elected":
            raise ValueError("Accepted wait delivery requires an elected wait.")
        if self.delivery == "excluded" and self.state not in {
            "cancelled",
            "expired",
            "unavailable",
        }:
            raise ValueError("Excluded wait delivery requires a refusal state.")
        if self.delivery == "pending":
            if self.registration.wait.delivery_ticket is None:
                raise ValueError("Pending wait delivery requires a session ticket.")
            if not self.source_pins:
                raise ValueError("Pending wait delivery must retain source pins.")
        if (
            self.registration.wait.delivery_ticket is not None
            and self.state in {"elected", "cancelled", "expired", "unavailable"}
            and self.delivery == "none"
        ):
            raise ValueError("Session-bound terminal wait requires pending delivery evidence.")
        if (
            self.state in {"elected", "cancelled", "expired", "unavailable"}
            and self.delivery == "none"
            and self.source_pins
        ):
            raise ValueError("Terminal wait without delivery cannot retain source pins.")
        if self.delivery in {"accepted", "excluded"} and self.source_pins:
            raise ValueError("Settled wait delivery cannot retain source pins.")
        if self.state in {"cancelled", "expired", "unavailable"} and self.terminal_at_ms is None:
            raise ValueError("Terminal refusal waits require an owner timestamp.")
        if self.state in {"pending", "elected"} and self.terminal_at_ms is not None:
            raise ValueError("Active waits cannot carry a refusal timestamp.")
        if self.state == "pending" and not self.source_pins:
            raise ValueError("Pending waits must retain source pins.")
        if len({ref_key(item.target) for item in self.evidence}) != len(self.evidence):
            raise ValueError("Wait evidence must contain one item per target.")
        if any(
            ref_key(item.target) not in self.registration.wait.target_keys for item in self.evidence
        ):
            raise ValueError("Wait evidence contains a foreign target.")
        if any(item not in self.registration.wait.source_keys for item in self.source_pins):
            raise ValueError("Wait source pins contain a foreign source.")
        if len(set(self.source_pins)) != len(self.source_pins):
            raise ValueError("Wait source pins must be distinct.")
        if any(
            a.sequence >= b.sequence for a, b in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("Wait events must be ordered.")
        if self.election is not None:
            election = self.election
            wait = self.registration.wait
            if {ref_key(item.target): item for item in election.evidence} != {
                ref_key(item.target): item for item in self.evidence
            } or len(election.evidence) != len(self.evidence):
                raise ValueError("Wait election evidence differs from retained evidence.")
            elected_events = tuple(event for event in self.events if event.kind == "elected")
            if len(elected_events) != 1 or elected_events[0].sequence != election.sequence:
                raise ValueError("Wait election lacks its exact event.")
            for item in self.evidence:
                if (
                    item.ingestion_sequence >= election.sequence
                    or item.observed_at_ms > election.elected_at_ms
                    or not any(
                        event.kind == "evidence"
                        and event.sequence == item.ingestion_sequence
                        and event.target == item.target
                        and event.evidence_digest == item.receipt_digest
                        for event in self.events
                    )
                ):
                    raise ValueError("Wait election evidence lacks its admitted event.")
            if election.result == "unavailable":
                if (
                    election.elected_at_ms < deadline_ms(wait.deadline)
                    or election.selected
                    or len(self.evidence) != len(wait.targets)
                    or evaluate_wait(wait, self.evidence) is not None
                ):
                    raise ValueError("Wait unavailable election conflicts with deadline evidence.")
            else:
                if (
                    election.result != evaluate_wait(wait, self.evidence)
                    or any(
                        item.accepted_at_ms >= deadline_ms(wait.deadline) for item in self.evidence
                    )
                    or {ref_key(ref) for ref in election.selected}
                    != {ref_key(item.target) for item in self.evidence if item.status == "success"}
                ):
                    raise ValueError("Wait election conflicts with its registered predicate.")
        if len(_encoded(self)) > MAX_ENVELOPE_BYTES:
            raise ValueError("Wait snapshot exceeds its envelope.")
        return self


class WaitControl(ContractValue):
    operation: OperationRef
    expected_revision: StrictInt = Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)
    action: Literal["cancel", "expire"]
    initiator: OwnerRef


def _require_terminal_capacity(wait: CollaborationWait) -> None:
    """Measure complete future records, including retained observation history.

    Construct only these internal probes without recursive model validation:
    validating their nested registration would re-enter this admission check.
    The normal contract serializer still enforces aggregate byte/node limits.
    No constructed probe is returned to a caller or persisted.
    """
    maximum = MAX_PORTABLE_JSON_INTEGER
    # Identifier permits up to 512 UTF-8 bytes; JSON control escapes expand
    # each byte to six. Source commitments need not be SHA-256 strings.
    identifier = "\x01" * 512
    refs = tuple(target.intent.selection.reference for target in wait.targets)
    evidence = tuple(
        WaitEvidence.model_construct(
            target=ref,
            status="unavailable",
            ingestion_sequence=maximum,
            source_sequence=maximum,
            accepted_at_ms=maximum,
            observed_at_ms=maximum,
            receipt_digest=identifier,
            commitment=identifier,
        )
        for ref in refs
    )
    # Each target can contribute an unknown observation and then one terminal
    # observation; subsequent unknowns/replayed terminal evidence do not append.
    events = (
        WaitEvent(sequence=1, kind="registered"),
        *(
            WaitEvent.model_construct(
                sequence=index + 2,
                kind="evidence",
                target=ref,
                evidence_digest=identifier,
            )
            for index, ref in enumerate((*refs, *refs))
        ),
        WaitEvent(sequence=len(refs) * 2 + 2, kind="elected"),
    )
    registration = WaitRegistration.model_construct(
        wait=wait, registration_digest="0" * 64, registered_sequence=1
    )
    election = WaitElection.model_construct(
        result="unavailable",
        selected=refs,
        evidence=evidence,
        sequence=maximum,
        elected_at_ms=maximum,
        outcome_digest="0" * 64,
    )
    pending = WaitSnapshot.model_construct(
        registration=registration,
        state="elected",
        delivery="pending" if wait.delivery_ticket is not None else "none",
        evidence=evidence,
        election=election,
        revision=maximum,
        source_pins=wait.source_keys if wait.delivery_ticket is not None else (),
        events=events,
    )
    released = pending.model_copy(
        update={
            "delivery": "accepted" if wait.delivery_ticket is not None else "none",
            "delivery_receipt_digest": identifier if wait.delivery_ticket is not None else None,
            "source_pins": (),
            "reserved_bytes": 0,
            "reserved_events": 0,
            "events": (*events, WaitEvent(sequence=len(events) + 1, kind="released")),
        }
    )
    # Refusal has no election, but does carry its own terminal timestamp.
    refused = pending.model_copy(
        update={"state": "unavailable", "election": None, "terminal_at_ms": maximum}
    )
    for probe in (pending, released, refused):
        if len(_encoded(probe)) > MAX_ENVELOPE_BYTES:
            raise ValueError("Wait terminal representation exceeds its envelope.")


def ref_key(reference: RequestRef | ObjectRef) -> str:
    value = reference.model_dump(mode="json")
    return sha256(canonical_durable_json_bytes(value, "wait target")).hexdigest()


def request_object_ref(reference: RequestRef) -> ObjectRef:
    return ObjectRef(
        owner=reference.owner,
        kind="collaboration.request",
        object_id=reference.request_id,
        incarnation=reference.incarnation,
    )


def source_key(command: RequestCommand) -> str:
    return sha256(contract_bytes(command, redactor=_CONTRACT_REDACTOR)).hexdigest()


def evidence_digest(evidence: WaitEvidence) -> str:
    return sha256(contract_bytes(evidence, redactor=_CONTRACT_REDACTOR)).hexdigest()


def election_digest(
    result: WaitResult,
    selected: tuple[RequestRef, ...],
    evidence: tuple[WaitEvidence, ...],
    sequence: int,
    elected_at_ms: int,
) -> str:
    def canonical(values):
        indexed = [
            (canonical_durable_json_bytes(snapshot_input(item), "wait election"), item)
            for item in values
        ]
        return tuple(item for _, item in sorted(indexed, key=lambda pair: pair[0]))

    probe = {
        "result": result,
        "selected": canonical(selected),
        "evidence": canonical(evidence),
        "sequence": sequence,
        "elected_at_ms": elected_at_ms,
        "outcome_digest": "",
    }
    return sha256(_encoded(probe)).hexdigest()


def evaluate_wait(wait: CollaborationWait, evidence: tuple[WaitEvidence, ...]) -> WaitResult | None:
    """Return the terminal result, or None while the predicate remains possible."""

    by_target = {ref_key(item.target): item for item in evidence}
    successes = sum(item.status == "success" for item in by_target.values())
    terminal = sum(item.status in {"success", "failure", "settled"} for item in by_target.values())
    possible = len(wait.targets) - sum(
        item.status in {"failure", "settled"} for item in by_target.values()
    )
    if wait.predicate == "ALL_SUCCESS":
        if any(item.status in {"failure", "settled"} for item in by_target.values()):
            return "failure"
        return "success" if successes == len(wait.targets) else None
    if wait.predicate == "ALL_SETTLED":
        return "settled" if terminal == len(wait.targets) else None
    required = 1 if wait.predicate == "ANY_SUCCESS" else wait.threshold
    assert required is not None
    if successes >= required:
        return "success"
    return "failure" if possible < required else None
