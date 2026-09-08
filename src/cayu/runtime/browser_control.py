"""Content-free browser-control records shared by runtime and operator owners.

These values describe expected authority; constructing one does not authenticate
an operator or authorize input. Store/guest owners must validate the authenticated
caller and compare the complete expected record at the actual mutation boundary.
Credentials, input content, pixels and DOM state have no place in these records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import MAX_PORTABLE_JSON_INTEGER, require_durable_clean_nonblank
from cayu.browser_profiles import _canonical_origin

_Identity = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
_Fingerprint = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
_Counter = Annotated[int, Field(strict=True, ge=1, le=MAX_PORTABLE_JSON_INTEGER)]
_Time = Annotated[int, Field(strict=True, ge=0, le=MAX_PORTABLE_JSON_INTEGER)]

BrowserControlState = Literal[
    "agent_controlled",
    "takeover_requested",
    "operator_controlled",
    "handback_pending",
    "control_uncertain",
    "closed",
    "allocation_lost",
]
BrowserCheckpointConsent = Literal["undecided", "allow", "deny"]


class _ControlModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
        revalidate_instances="always",
    )

    @field_validator("*", mode="after")
    @classmethod
    def portable_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return require_durable_clean_nonblank(value, "browser control field")
        return value

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        # Do not serialize caller-owned instances before validation: Pydantic
        # serialization warnings can render rejected objects and sibling values.
        del deep
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update({} if update is None else update)
        copied = type(self).model_validate(values)
        object.__setattr__(
            copied,
            "__pydantic_fields_set__",
            self.model_fields_set | (set() if update is None else set(update)),
        )
        return copied


class BrowserOperatorPurpose(_ControlModel):
    """Application-declared intervention scope, never page-authored instructions."""

    code: Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    expected_origins: tuple[Annotated[str, Field(strict=True, max_length=2048)], ...] = Field(
        min_length=1, max_length=32
    )

    @field_validator("expected_origins")
    @classmethod
    def canonical_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))) or any(
            _canonical_origin(value) != value for value in values
        ):
            raise ValueError("Expected browser origins must be canonical, unique and sorted.")
        return values


class BrowserControlAllocation(_ControlModel):
    """Admitted parent/allocation before the private guest reveals its worker ID."""

    session_id: _Identity
    session_instance_id: _Identity
    run_epoch: _Counter
    interaction_id: _Identity
    execution_profile_fingerprint: _Fingerprint
    environment_name: _Identity
    allocation_fingerprint: _Fingerprint
    browser_session_id: _Identity
    operator_purpose: BrowserOperatorPurpose
    profile_checkpoint_policy: Literal[
        "unavailable", "disabled", "on_close", "after_terminal_operation"
    ] = "unavailable"


class BrowserControlIdentity(BrowserControlAllocation):
    """Exact admitted parent and live worker; never a caller bearer token."""

    worker_instance_id: _Identity


class BrowserOperatorIdentity(_ControlModel):
    """Expected authenticated identity and application authorization policy."""

    subject: _Identity = Field(repr=False)
    tenant: _Identity | None = Field(repr=False)
    operator_session_id: _Identity = Field(repr=False)
    authorization_policy_fingerprint: _Fingerprint


class BrowserControlPage(_ControlModel):
    """Content-free page authority observed at request or handback."""

    page_id: _Identity
    revision: _Identity
    control_epoch: _Counter


class BrowserOperatorPageOperations(_ControlModel):
    """Cumulative input admissions for one page; never input content or a digest."""

    page_id: _Identity
    operations: _Counter


class BrowserObservedPageLocation(_ControlModel):
    """Origin-only observation, separate from page/input authority or instructions."""

    page: BrowserControlPage
    origin: Annotated[str, Field(strict=True, max_length=2048)] | None

    @field_validator("origin")
    @classmethod
    def exact_origin(cls, value: str | None) -> str | None:
        if value is not None and _canonical_origin(value) != value:
            raise ValueError("Browser location must be a canonical HTTPS origin.")
        return value


class BrowserControlPageAudit(_ControlModel):
    """Bounded native boundary evidence, never operator-supplied input authority."""

    request_id: Annotated[str, Field(strict=True, pattern=r"^bt_[0-9a-f]{32}$")]
    phase: Literal["acquired", "handed_back"]
    control_epoch: _Counter
    locations: tuple[BrowserObservedPageLocation, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def complete_page_set(self) -> Self:
        ids = tuple(location.page.page_id for location in self.locations)
        if ids != tuple(sorted(set(ids))) or (self.phase == "acquired" and not ids):
            raise ValueError("Browser audit requires a canonical boundary page set.")
        return self


class BrowserPagesIntent(_ControlModel):
    """Exact allocation expectation for content-free page discovery."""

    identity: BrowserControlIdentity
    expected_record_revision: _Counter


class BrowserViewIntent(_ControlModel):
    """Caller expectations only; authenticated operator identity is not accepted."""

    identity: BrowserControlIdentity
    expected_record_revision: _Counter
    page: BrowserControlPage


class BrowserHandbackIntent(_ControlModel):
    identity: BrowserControlIdentity
    expected_record_revision: _Counter
    expected_control_epoch: _Counter
    request_id: Annotated[str, Field(strict=True, pattern=r"^bt_[0-9a-f]{32}$")]


class BrowserSensitiveEntryIntent(BrowserHandbackIntent):
    """Exact operator request to suspend capture, never an input payload."""


class BrowserRenewIntent(BrowserHandbackIntent):
    """Extend only the still-live exact lease, without renewing the request maximum."""

    expected_lease_until_ms: _Time
    lease_until_ms: _Time


class BrowserTextInputIntent(BrowserHandbackIntent):
    """Content-free input authority; text travels only on the private channel.

    Non-text kinds are a closed native-key set, never arbitrary browser shortcuts.
    """

    input_sequence: _Counter
    page: BrowserControlPage
    input_kind: Literal["text", "tab", "backtab", "enter", "escape", "backspace"] = "text"


BrowserControlAction = Literal[
    "view",
    "takeover",
    "renew",
    "handback",
    "checkpoint",
    "sensitive_entry",
    "text_input",
    "key_input",
]


class BrowserControlPrincipal(_ControlModel):
    """Authenticated server provenance, not a bearer capability or permission.

    A missing authenticated tenant remains missing; the browser boundary never
    invents a default tenant or infers session membership from this value.
    """

    subject: _Identity = Field(repr=False)
    tenant: _Identity | None = Field(default=None, repr=False)


class BrowserControlPolicyRequest(_ControlModel):
    """One exact, content-free application authorization question."""

    principal: BrowserControlPrincipal = Field(repr=False)
    identity: BrowserControlIdentity
    operator_session_id: _Identity = Field(repr=False)
    action: BrowserControlAction
    record_revision: _Counter
    control_epoch: _Counter
    state: BrowserControlState


class BrowserControlPolicyResult(_ControlModel):
    """No permission is granted by omission or by a truthy extension value."""

    allowed: Annotated[bool, Field(strict=True)] = False


class BrowserControlPolicy(ABC):
    """Application-owned live-view and operator-action permission.

    Implementations must authorize the session/allocation and action explicitly.
    Authentication or a matching tenant alone is not an authorization decision.
    Decisions are scoped to the supplied revision; they are not reusable grants
    for later control generations. No page content, pixels, or input is supplied.
    """

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable, versioned, non-secret application policy identity."""

    @abstractmethod
    async def decide(self, request: BrowserControlPolicyRequest) -> BrowserControlPolicyResult:
        """Authorize this exact action, or deny without causing browser effects."""


class BrowserTakeoverIntent(_ControlModel):
    """Caller expectation without an operator identity or permission claim."""

    request_id: Annotated[str, Field(strict=True, pattern=r"^bt_[0-9a-f]{32}$")]
    identity: BrowserControlIdentity
    expected_record_revision: _Counter
    expected_control_epoch: _Counter
    pages: tuple[BrowserControlPage, ...] = Field(min_length=1, max_length=32)
    purpose_code: Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    requested_at_ms: _Time
    expires_at_ms: _Time
    maximum_until_ms: _Time
    checkpoint_consent: BrowserCheckpointConsent = "undecided"

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.requested_at_ms < self.expires_at_ms <= self.maximum_until_ms:
            raise ValueError("Browser takeover requires an ordered finite lifetime.")
        if self.maximum_until_ms - self.requested_at_ms > 3_600_000:
            raise ValueError("Browser takeover exceeds its one-hour maximum lifetime.")
        page_ids = tuple(page.page_id for page in self.pages)
        if page_ids != tuple(sorted(set(page_ids))):
            raise ValueError("Browser takeover pages must be unique and sorted.")
        return self


class BrowserTakeoverRequest(BrowserTakeoverIntent):
    """Intent bound to the runtime-derived, application-authorized operator."""

    operator: BrowserOperatorIdentity = Field(repr=False)


class BrowserControlRecord(_ControlModel):
    """One exact control generation, with no input payload or replayable input."""

    schema_version: Literal[1] = 1
    identity: BrowserControlIdentity
    revision: _Counter = 1
    control_epoch: _Counter = 1
    transport_generation: _Counter = 1
    state: BrowserControlState = "agent_controlled"
    request: BrowserTakeoverRequest | None = Field(default=None, repr=False)
    lease_until_ms: _Time | None = None
    pending_lease_until_ms: _Time | None = None
    fresh_observation_required: bool = Field(default=False, strict=True)
    sensitive_entry: bool = Field(default=False, strict=True)
    sensitive_entry_pending: bool = Field(default=False, strict=True)
    capture_restricted: bool = Field(default=False, strict=True)
    manual_mutation_uncertain: bool = Field(default=False, strict=True)
    checkpoint_consent: BrowserCheckpointConsent = "undecided"
    settled_input_sequence: Annotated[
        int, Field(strict=True, ge=0, le=MAX_PORTABLE_JSON_INTEGER)
    ] = 0
    acquisition_audit: BrowserControlPageAudit | None = None
    handback_audit: BrowserControlPageAudit | None = None
    # Includes pending/uncertain admissions. Only exact guest settlement permits
    # their handback; accounting must not disappear when a new takeover starts.
    operator_page_operations: tuple[BrowserOperatorPageOperations, ...] = Field(
        default=(), max_length=128
    )
    pending_input_sequence: _Counter | None = None
    pending_input_page: BrowserControlPage | None = None
    pending_input_kind: Literal["text", "tab", "backtab", "enter", "escape", "backspace"] | None = (
        None
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported browser control record version.")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        for phase, audit in (
            ("acquired", self.acquisition_audit),
            ("handed_back", self.handback_audit),
        ):
            if audit is None:
                continue
            offset = 1 if phase == "acquired" else 2
            if (
                self.request is None
                or audit.request_id != self.request.request_id
                or audit.phase != phase
                or audit.control_epoch != self.request.expected_control_epoch + offset
                or self.control_epoch < audit.control_epoch
            ):
                raise ValueError("Browser audit belongs to a different control boundary.")
            if (
                phase == "acquired"
                and tuple(item.page for item in audit.locations) != self.request.pages
            ):
                raise ValueError("Browser acquisition audit differs from its requested page set.")
        if self.handback_audit is not None and self.acquisition_audit is None:
            raise ValueError("Browser handback audit requires its acquisition evidence.")
        page_ids = tuple(item.page_id for item in self.operator_page_operations)
        total_operator_operations = sum(item.operations for item in self.operator_page_operations)
        if (
            page_ids != tuple(sorted(set(page_ids)))
            or total_operator_operations > MAX_PORTABLE_JSON_INTEGER
            or total_operator_operations
            != (self.pending_input_sequence or self.settled_input_sequence)
        ):
            raise ValueError("Browser operator accounting must be canonical and bounded.")
        if self.request is not None and self.request.identity != self.identity:
            raise ValueError("Browser control request belongs to another allocation.")
        if (
            self.request is not None
            and self.request.purpose_code != self.identity.operator_purpose.code
        ):
            raise ValueError("Browser control request differs from its application purpose.")
        if self.request is not None:
            if self.revision <= self.request.expected_record_revision:
                raise ValueError("Browser control must advance its request's source revision.")
            expected_epoch = self.request.expected_control_epoch
            if expected_epoch < self.transport_generation:
                raise ValueError("Browser takeover precedes its transport generation.")
            if self.state == "takeover_requested" and (
                self.control_epoch != expected_epoch or self.lease_until_ms is not None
            ):
                raise ValueError("Pending takeover cannot claim acquired input authority.")
            if self.state in {"operator_controlled", "handback_pending"} and (
                self.control_epoch != expected_epoch + 1
            ):
                raise ValueError("Acquired browser control requires its exact advanced epoch.")
            if self.state == "agent_controlled" and self.control_epoch != expected_epoch + 2:
                raise ValueError("Handed-back browser control requires its exact successor epoch.")
            if not expected_epoch <= self.control_epoch <= expected_epoch + 2:
                raise ValueError("Browser control epoch conflicts with its takeover request.")
        if (
            self.state
            in {
                "takeover_requested",
                "operator_controlled",
                "handback_pending",
            }
            and self.request is None
        ):
            raise ValueError("Browser control state requires an exact takeover request.")
        if self.request is None and (
            self.control_epoch != self.transport_generation
            or self.pending_input_sequence is not None
            or self.settled_input_sequence != 0
            or self.operator_page_operations
            or self.sensitive_entry
            or self.sensitive_entry_pending
            or self.manual_mutation_uncertain
            or self.checkpoint_consent != "undecided"
        ):
            raise ValueError("Browser control without takeover cannot claim operator history.")
        if self.lease_until_ms is not None and (
            self.request is None
            or self.lease_until_ms <= self.request.requested_at_ms
            or self.lease_until_ms > self.request.maximum_until_ms
        ):
            raise ValueError("Browser control lease exceeds its request authority.")
        if self.state == "operator_controlled" and self.lease_until_ms is None:
            raise ValueError("Operator control requires a finite lease.")
        if self.pending_lease_until_ms is not None and (
            self.request is None
            or self.lease_until_ms is None
            or self.state not in {"operator_controlled", "control_uncertain"}
            or not self.lease_until_ms
            < self.pending_lease_until_ms
            <= self.request.maximum_until_ms
            or self.pending_input_sequence is not None
            or self.sensitive_entry_pending
        ):
            raise ValueError("Browser lease renewal requires its exclusive live source lease.")
        if self.state in {"agent_controlled", "closed", "allocation_lost"}:
            if self.lease_until_ms is not None or self.pending_input_sequence is not None:
                raise ValueError("Settled browser control cannot retain live input authority.")
            if self.sensitive_entry:
                raise ValueError("Settled browser control cannot remain in sensitive entry.")
        if self.sensitive_entry and not self.capture_restricted:
            raise ValueError("Sensitive entry requires capture restrictions.")
        if self.sensitive_entry and self.state == "takeover_requested":
            raise ValueError("Pending takeover cannot start sensitive entry.")
        if self.sensitive_entry_pending and (
            self.sensitive_entry
            or not self.capture_restricted
            or self.state not in {"operator_controlled", "control_uncertain"}
            or self.pending_input_sequence is not None
        ):
            raise ValueError("Pending sensitive entry requires exclusive capture suspension.")
        if self.pending_input_sequence is not None and (
            self.pending_input_sequence != self.settled_input_sequence + 1
            or self.state not in {"operator_controlled", "handback_pending", "control_uncertain"}
        ):
            raise ValueError("Pending browser input must be the exact next owned sequence.")
        if (self.pending_input_page is None) != (self.pending_input_sequence is None):
            raise ValueError("Browser input requires both its pending sequence and page.")
        if (self.pending_input_kind is None) != (self.pending_input_sequence is None):
            raise ValueError("Browser input requires its exact pending kind.")
        return self


class BrowserControlConflict(ValueError):
    """Expected control material does not match the authoritative record."""


def closed_browser_control_successor(record: BrowserControlRecord) -> BrowserControlRecord | None:
    """Exact state shape for runtime-confirmed close; not authority to publish it."""
    record = BrowserControlRecord.model_validate(record)
    if record.state not in {"agent_controlled", "takeover_requested", "control_uncertain"}:
        return None
    return record.model_copy(
        update={
            "revision": record.revision + 1,
            "state": "closed",
            "fresh_observation_required": False,
        }
    )


def rebound_browser_control_successor(
    current: BrowserControlRecord, identity: BrowserControlIdentity
) -> BrowserControlRecord:
    """Exact surviving guest, newer invocation, and no native-input authority.

    Native acknowledgement still must establish the matching transport fence.
    A prior takeover (including uncertain input) requires explicit closure instead.
    """
    current = BrowserControlRecord.model_validate(current)
    identity = BrowserControlIdentity.model_validate(identity)
    if (
        identity.run_epoch <= current.identity.run_epoch
        or identity.model_dump(exclude={"run_epoch", "interaction_id"})
        != current.identity.model_dump(exclude={"run_epoch", "interaction_id"})
        or current.state not in {"agent_controlled", "control_uncertain"}
        or current.request is not None
        or current.sensitive_entry
        or current.sensitive_entry_pending
        or current.capture_restricted
        or current.manual_mutation_uncertain
        or current.pending_input_sequence is not None
        or current.settled_input_sequence != 0
        or current.operator_page_operations
    ):
        raise BrowserControlConflict(
            "Browser control recovery requires the same view-only allocation; "
            "resolve uncertain takeover through exact allocation closure."
        )
    return current.model_copy(
        update={
            "identity": identity,
            "revision": current.revision + 1,
            "control_epoch": current.control_epoch + 1,
            "transport_generation": current.transport_generation + 1,
            "state": "agent_controlled",
            "fresh_observation_required": True,
        }
    )


class BrowserControlCheckpoint(_ControlModel):
    """Bounded allocation records owned by one parent session transaction."""

    schema_version: Literal[1] = 1
    records: tuple[BrowserControlRecord, ...] = Field(default=(), max_length=32)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported browser control checkpoint version.")
        return value

    @model_validator(mode="after")
    def validate_records(self) -> Self:
        keys = tuple(record.identity.browser_session_id for record in self.records)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Browser control records must be unique and sorted.")
        parents = {
            (record.identity.session_id, record.identity.session_instance_id)
            for record in self.records
        }
        if len(parents) > 1:
            raise ValueError("Browser control checkpoint contains conflicting parent authority.")
        return self

    def replace_record(
        self,
        *,
        expected: BrowserControlRecord | None,
        desired: BrowserControlRecord,
    ) -> BrowserControlCheckpoint:
        """Compare the complete expected record; no field is a wildcard.

        This pure function must run *inside* the same store transaction as the
        governed operation. Calling it on an earlier read does not reserve work.
        """

        owned = BrowserControlCheckpoint.model_validate(self)
        desired = BrowserControlRecord.model_validate(desired)
        expected = None if expected is None else BrowserControlRecord.model_validate(expected)
        key = desired.identity.browser_session_id
        current = next(
            (record for record in owned.records if record.identity.browser_session_id == key),
            None,
        )
        if current != expected:
            raise BrowserControlConflict("Browser control changed before publication.")
        if expected is None:
            pristine = BrowserControlRecord(identity=desired.identity)
            if desired != pristine:
                raise BrowserControlConflict("Browser control initialization must be pristine.")
        elif desired.identity != expected.identity:
            if desired != rebound_browser_control_successor(expected, desired.identity):
                raise BrowserControlConflict("Browser reconnect must advance its exact fence.")
        elif (
            desired.revision != expected.revision + 1
            or desired.transport_generation != expected.transport_generation
        ):
            raise BrowserControlConflict(
                "Browser control publication must advance the exact owner."
            )
        remaining = tuple(
            record for record in owned.records if record.identity.browser_session_id != key
        )
        return BrowserControlCheckpoint(
            records=tuple(
                sorted((*remaining, desired), key=lambda record: record.identity.browser_session_id)
            )
        )


def request_browser_takeover(
    current: BrowserControlRecord,
    request: BrowserTakeoverRequest,
    *,
    now_ms: int,
) -> BrowserControlRecord:
    """Pure transition; the transaction owner supplies authenticated material/time.

    The owner must validate current page evidence, policy and operator authority
    before committing this value. This function cannot grant operator input.
    """

    current = BrowserControlRecord.model_validate(current)
    request = BrowserTakeoverRequest.model_validate(request)
    if type(now_ms) is not int or not 0 <= now_ms <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Browser control requires a valid store-owned time.")
    if current.request is not None and current.request.request_id == request.request_id:
        if current.request != request:
            raise BrowserControlConflict("Browser takeover request identity conflicts.")
        # Exact historical readback neither renews a lease nor delivers input.
        return current
    if (
        current.state != "agent_controlled"
        or current.identity != request.identity
        or request.purpose_code != current.identity.operator_purpose.code
        or current.revision != request.expected_record_revision
        or current.control_epoch != request.expected_control_epoch
    ):
        raise BrowserControlConflict("Browser takeover source authority changed.")
    if not request.requested_at_ms <= now_ms < request.expires_at_ms:
        raise BrowserControlConflict("Browser takeover request is not current.")
    return current.model_copy(
        update={
            "state": "takeover_requested",
            "revision": current.revision + 1,
            "request": request,
            "checkpoint_consent": request.checkpoint_consent,
            "acquisition_audit": None,
            "handback_audit": None,
        }
    )
