from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.egress.destinations import normalize_egress_hostname
from cayu.egress.policy import BrowserEgressPolicy, EgressPolicy, HttpEgressPolicy

EGRESS_AUTHORITY_SCHEMA_VERSION = 1
EGRESS_AUTHORITY_TEXT_MAX_BYTES = 256
EGRESS_AUTHORITY_MAX_POLICIES = 128
EGRESS_AUTHORITY_MAX_BINDINGS = 256
EGRESS_AUTHORITY_MAX_OPERATIONS = 512
EGRESS_AUTHORITY_MAX_DESTINATIONS = 1024
EGRESS_AUTHORITY_MAX_DENIED_PREFIXES = 1024
EGRESS_AUTHORITY_MAX_TOTAL_OPERATIONS = 4096
EGRESS_AUTHORITY_MAX_EFFECTIVE_PERMISSIONS = 4096
EGRESS_AUTHORITY_MAX_CANONICAL_BYTES = 1024 * 1024
EGRESS_AUTHORITY_MAX_COMPARISON_WORK = 1024 * 1024

_ADAPTER_VERIFIED_CUTOVER_RECEIPTS: dict[
    int,
    tuple[weakref.ReferenceType[EgressAuthorityCutoverReceipt], str],
] = {}


class EgressAuthorityCutoverStrategy(StrEnum):
    """How an adapter can replace one admitted egress generation."""

    VERIFIED_IN_PLACE = "verified_in_place"
    FRESH_AUTHORITY_PATH = "fresh_authority_path"
    ALLOCATION_REPLACEMENT = "allocation_replacement"
    UNSUPPORTED = "unsupported"


class EgressAuthorityChangeKind(StrEnum):
    """Typed comparison of a candidate authority against the active authority."""

    UNCHANGED = "unchanged"
    NARROWER = "narrower"
    WIDER = "wider"
    INCOMPARABLE = "incomparable"
    REFUSED = "refused"


class EgressAuthorityTransitionState(StrEnum):
    """Durable lifecycle states for one governed authority replacement."""

    AUTHORIZED = "authorized"
    INSTALLING = "installing"
    ACTIVE = "active"
    REFUSED = "refused"
    AMBIGUOUS = "ambiguous"


class EgressAuthorityOperation(BaseModel):
    """One bounded HTTP operation admitted by a built-in policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    method: str = Field(max_length=32)
    path: str = Field(max_length=2048)
    match: Literal["exact", "prefix"] = "exact"

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "method").upper()

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "path")
        if not value.startswith("/"):
            raise ValueError("Egress authority paths must start with '/'.")
        return value


class EgressAuthorityPolicyIdentity(BaseModel):
    """Secret-free canonical projection of one policy's operation semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    kind: Literal["http", "browser", "opaque"]
    allowed_destinations: tuple[str, ...] = ()
    operations: tuple[EgressAuthorityOperation, ...] = ()
    denied_path_prefixes: tuple[str, ...] = ()
    comparison_available: StrictBool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "policy.name")

    @field_validator("allowed_destinations", mode="before")
    @classmethod
    def validate_allowed_destinations(cls, value: Any) -> tuple[str, ...]:
        destinations = tuple(value)
        normalized = tuple(
            sorted(
                {
                    normalize_egress_hostname(item, field_name="allowed_destinations")
                    for item in destinations
                }
            )
        )
        if len(normalized) > EGRESS_AUTHORITY_MAX_DESTINATIONS:
            raise ValueError("Egress authority policy has too many destinations.")
        return normalized

    @field_validator("operations", mode="before")
    @classmethod
    def validate_operations(cls, value: Any) -> tuple[EgressAuthorityOperation, ...]:
        operations = tuple(value)
        if len(operations) > EGRESS_AUTHORITY_MAX_OPERATIONS:
            raise ValueError("Egress authority policy has too many operations.")
        copied = tuple(
            item
            if type(item) is EgressAuthorityOperation
            else EgressAuthorityOperation.model_validate(item)
            for item in operations
        )
        ordered = tuple(sorted(set(copied), key=lambda item: (item.method, item.path, item.match)))
        return ordered

    @field_validator("denied_path_prefixes", mode="before")
    @classmethod
    def validate_denied_prefixes(cls, value: Any) -> tuple[str, ...]:
        prefixes = tuple(value)
        if len(prefixes) > EGRESS_AUTHORITY_MAX_DENIED_PREFIXES:
            raise ValueError("Egress authority policy has too many denied path prefixes.")
        normalized: list[str] = []
        for index, prefix in enumerate(prefixes):
            item = require_durable_clean_nonblank(prefix, f"denied_path_prefixes[{index}]")
            if not item.startswith("/"):
                raise ValueError("Denied egress path prefixes must start with '/'.")
            normalized.append(item.rstrip("/") or "/")
        return tuple(sorted(set(normalized)))

    @model_validator(mode="after")
    def validate_comparison(self) -> EgressAuthorityPolicyIdentity:
        if self.kind == "opaque":
            if self.comparison_available:
                raise ValueError("Opaque egress policies cannot claim semantic comparison.")
            if self.allowed_destinations or self.operations or self.denied_path_prefixes:
                raise ValueError("Opaque egress policies cannot project operation semantics.")
        elif not self.comparison_available:
            raise ValueError("Built-in egress policies must expose comparison semantics.")
        elif not self.allowed_destinations:
            raise ValueError("Built-in egress policies must project allowed destinations.")
        return self


class EgressAuthorityBindingIdentity(BaseModel):
    """One destination/credential class bound to a named policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    destination: str = Field(max_length=253)
    policy_name: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    credential_kind: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    credential_authority_fingerprint: str

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value: str) -> str:
        return normalize_egress_hostname(value, field_name="destination")

    @field_validator("policy_name", "credential_kind")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("credential_authority_fingerprint")
    @classmethod
    def validate_credential_authority_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("credential_authority_fingerprint must be a lowercase SHA-256 digest.")
        return value


class EgressAuthorityIdentity(BaseModel):
    """Versioned, bounded, secret-free identity for one admitted egress generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.egress-authority"] = "cayu.egress-authority"
    schema_version: Literal[1] = EGRESS_AUTHORITY_SCHEMA_VERSION
    generation: StrictInt = Field(ge=1)
    authority_source: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    authority_scope: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    policy_version: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    runner_kind: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    cutover_strategy: EgressAuthorityCutoverStrategy
    policies: tuple[EgressAuthorityPolicyIdentity, ...]
    bindings: tuple[EgressAuthorityBindingIdentity, ...]
    comparison_available: StrictBool
    fingerprint: str

    @field_validator("authority_source", "authority_scope", "policy_version", "runner_kind")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("policies", mode="before")
    @classmethod
    def validate_policies(cls, value: Any) -> tuple[EgressAuthorityPolicyIdentity, ...]:
        policies = tuple(value)
        if not policies or len(policies) > EGRESS_AUTHORITY_MAX_POLICIES:
            raise ValueError("Egress authority must contain a bounded non-empty policy set.")
        copied = tuple(
            item
            if type(item) is EgressAuthorityPolicyIdentity
            else EgressAuthorityPolicyIdentity.model_validate(item)
            for item in policies
        )
        if len({item.name for item in copied}) != len(copied):
            raise ValueError("Egress authority policy names must be unique.")
        return tuple(sorted(copied, key=lambda item: item.name))

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: Any) -> tuple[EgressAuthorityBindingIdentity, ...]:
        bindings = tuple(value)
        if not bindings or len(bindings) > EGRESS_AUTHORITY_MAX_BINDINGS:
            raise ValueError("Egress authority must contain a bounded non-empty binding set.")
        copied = tuple(
            item
            if type(item) is EgressAuthorityBindingIdentity
            else EgressAuthorityBindingIdentity.model_validate(item)
            for item in bindings
        )
        return tuple(
            sorted(
                set(copied),
                key=lambda item: (
                    item.destination,
                    item.policy_name,
                    item.credential_kind,
                    item.credential_authority_fingerprint,
                ),
            )
        )

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> EgressAuthorityIdentity:
        policy_names = {policy.name for policy in self.policies}
        if any(binding.policy_name not in policy_names for binding in self.bindings):
            raise ValueError("Egress authority bindings must reference a declared policy.")
        comparison_available = all(policy.comparison_available for policy in self.policies)
        if self.comparison_available != comparison_available:
            raise ValueError("Egress authority comparison availability is inconsistent.")
        if sum(len(policy.operations) for policy in self.policies) > (
            EGRESS_AUTHORITY_MAX_TOTAL_OPERATIONS
        ):
            raise ValueError("Egress authority has too many aggregate operations.")
        policies_by_name = {policy.name: policy for policy in self.policies}
        effective_permissions = sum(
            len(policies_by_name[binding.policy_name].operations)
            for binding in self.bindings
            if binding.destination in policies_by_name[binding.policy_name].allowed_destinations
        )
        if effective_permissions > EGRESS_AUTHORITY_MAX_EFFECTIVE_PERMISSIONS:
            raise ValueError("Egress authority has too many effective permissions.")
        if sum(len(policy.allowed_destinations) for policy in self.policies) > (
            EGRESS_AUTHORITY_MAX_DESTINATIONS
        ):
            raise ValueError("Egress authority has too many aggregate destinations.")
        if sum(len(policy.denied_path_prefixes) for policy in self.policies) > (
            EGRESS_AUTHORITY_MAX_DENIED_PREFIXES
        ):
            raise ValueError("Egress authority has too many aggregate denied path prefixes.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json", exclude={"fingerprint"}),
                    "egress_authority",
                )
            )
            > EGRESS_AUTHORITY_MAX_CANONICAL_BYTES
        ):
            raise ValueError("Egress authority exceeds the canonical byte limit.")
        if self.fingerprint != _egress_authority_fingerprint(self):
            raise ValueError("Egress authority fingerprint does not match its contents.")
        return self


def _copy_egress_authority_identity(
    identity: EgressAuthorityIdentity,
) -> EgressAuthorityIdentity:
    """Defensively copy authority material without serializing caller-owned models."""

    if type(identity) is not EgressAuthorityIdentity:
        raise TypeError("identity must be an EgressAuthorityIdentity.")
    policies: list[EgressAuthorityPolicyIdentity] = []
    for policy in identity.policies:
        if type(policy) is not EgressAuthorityPolicyIdentity:
            raise TypeError("Egress authority policies must be validated policy identities.")
        operations: list[EgressAuthorityOperation] = []
        for operation in policy.operations:
            if type(operation) is not EgressAuthorityOperation:
                raise TypeError("Egress authority operations must be validated operations.")
            operations.append(
                EgressAuthorityOperation(
                    method=operation.method,
                    path=operation.path,
                    match=operation.match,
                )
            )
        policies.append(
            EgressAuthorityPolicyIdentity(
                name=policy.name,
                kind=policy.kind,
                allowed_destinations=policy.allowed_destinations,
                operations=tuple(operations),
                denied_path_prefixes=policy.denied_path_prefixes,
                comparison_available=policy.comparison_available,
            )
        )
    bindings: list[EgressAuthorityBindingIdentity] = []
    for binding in identity.bindings:
        if type(binding) is not EgressAuthorityBindingIdentity:
            raise TypeError("Egress authority bindings must be validated binding identities.")
        bindings.append(
            EgressAuthorityBindingIdentity(
                destination=binding.destination,
                policy_name=binding.policy_name,
                credential_kind=binding.credential_kind,
                credential_authority_fingerprint=(binding.credential_authority_fingerprint),
            )
        )
    return EgressAuthorityIdentity(
        record_type=identity.record_type,
        schema_version=identity.schema_version,
        generation=identity.generation,
        authority_source=identity.authority_source,
        authority_scope=identity.authority_scope,
        policy_version=identity.policy_version,
        runner_kind=identity.runner_kind,
        cutover_strategy=identity.cutover_strategy,
        policies=tuple(policies),
        bindings=tuple(bindings),
        comparison_available=identity.comparison_available,
        fingerprint=identity.fingerprint,
    )


class EgressAuthorityCutoverReceipt(BaseModel):
    """Bounded receipt shape for one exact target generation.

    Constructing this public model validates content only. Runtime activation
    additionally requires private evidence issued by the backend adapter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.egress-authority-cutover"] = "cayu.egress-authority-cutover"
    schema_version: Literal[1] = 1
    state: Literal[EgressAuthorityTransitionState.ACTIVE] = EgressAuthorityTransitionState.ACTIVE
    from_fingerprint: str
    to_fingerprint: str
    from_generation: StrictInt = Field(ge=1)
    to_generation: StrictInt = Field(ge=1)
    runner_kind: str = Field(max_length=EGRESS_AUTHORITY_TEXT_MAX_BYTES)
    strategy: EgressAuthorityCutoverStrategy
    environment_fingerprint: str
    same_allocation: Literal[True] = True
    workspace_continuity_verified: Literal[True] = True
    old_authority_revoked: Literal[True] = True
    old_path_closed: Literal[True] = True
    backend_verified: Literal[True] = True
    fingerprint: str

    @field_validator(
        "from_fingerprint",
        "to_fingerprint",
        "environment_fingerprint",
        "fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("runner_kind")
    @classmethod
    def validate_runner_kind(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "runner_kind")

    @model_validator(mode="after")
    def validate_receipt(self) -> EgressAuthorityCutoverReceipt:
        if self.to_generation <= self.from_generation:
            raise ValueError("Egress authority generations must increase at cutover.")
        if self.from_fingerprint == self.to_fingerprint:
            raise ValueError("Egress authority cutover requires a distinct target identity.")
        if self.strategy is not EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH:
            raise ValueError("Same-allocation receipts require a fresh authority path strategy.")
        if self.fingerprint != _egress_authority_cutover_receipt_fingerprint(self):
            raise ValueError("Egress authority cutover receipt fingerprint is inconsistent.")
        return self


def build_egress_authority_cutover_receipt(
    *,
    expected: EgressAuthorityIdentity,
    target: EgressAuthorityIdentity,
    environment_fingerprint: str,
) -> EgressAuthorityCutoverReceipt:
    """Build a structurally validated, non-authoritative receipt value."""

    if expected.runner_kind != target.runner_kind:
        raise ValueError("Egress authority cutover cannot change runner kind in place.")
    if target.cutover_strategy is not EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH:
        raise ValueError("Target authority does not declare fresh-path cutover support.")
    provisional = EgressAuthorityCutoverReceipt.model_construct(
        record_type="cayu.egress-authority-cutover",
        schema_version=1,
        state=EgressAuthorityTransitionState.ACTIVE,
        from_fingerprint=expected.fingerprint,
        to_fingerprint=target.fingerprint,
        from_generation=expected.generation,
        to_generation=target.generation,
        runner_kind=target.runner_kind,
        strategy=target.cutover_strategy,
        environment_fingerprint=environment_fingerprint,
        same_allocation=True,
        workspace_continuity_verified=True,
        old_authority_revoked=True,
        old_path_closed=True,
        backend_verified=True,
        fingerprint="0" * 64,
    )
    return EgressAuthorityCutoverReceipt.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "fingerprint": _egress_authority_cutover_receipt_fingerprint(provisional),
        }
    )


def _build_adapter_verified_egress_authority_cutover_receipt(
    *,
    expected: EgressAuthorityIdentity,
    target: EgressAuthorityIdentity,
    environment_fingerprint: str,
) -> EgressAuthorityCutoverReceipt:
    """Build proof at an adapter-owned backend verification boundary."""

    receipt = build_egress_authority_cutover_receipt(
        expected=expected,
        target=target,
        environment_fingerprint=environment_fingerprint,
    )
    identity = id(receipt)

    def forget(reference: weakref.ReferenceType[EgressAuthorityCutoverReceipt]) -> None:
        current = _ADAPTER_VERIFIED_CUTOVER_RECEIPTS.get(identity)
        if current is not None and current[0] is reference:
            _ADAPTER_VERIFIED_CUTOVER_RECEIPTS.pop(identity, None)

    reference = weakref.ref(receipt, forget)
    _ADAPTER_VERIFIED_CUTOVER_RECEIPTS[identity] = (
        reference,
        _egress_authority_cutover_receipt_fingerprint(receipt),
    )
    return receipt


def _egress_authority_cutover_receipt_is_adapter_verified(
    receipt: EgressAuthorityCutoverReceipt,
) -> bool:
    """Return positive in-process proof owned by a backend adapter."""

    attestation = _ADAPTER_VERIFIED_CUTOVER_RECEIPTS.get(id(receipt))
    return bool(
        attestation is not None
        and attestation[0]() is receipt
        and _egress_authority_cutover_receipt_fingerprint(receipt) == attestation[1]
    )


def build_egress_authority_identity(
    *,
    policies: Mapping[str, EgressPolicy],
    bindings: Sequence[EgressAuthorityBindingIdentity],
    generation: int,
    authority_source: str,
    authority_scope: str,
    policy_version: str,
    runner_kind: str,
    cutover_strategy: EgressAuthorityCutoverStrategy,
) -> EgressAuthorityIdentity:
    """Resolve runtime policy objects into a bounded durable authority identity."""

    policy_identities: list[EgressAuthorityPolicyIdentity] = []
    for name in sorted(policies):
        policy = policies[name]
        if not isinstance(policy, EgressPolicy):
            raise TypeError("Virtual-egress policies must implement EgressPolicy.")
        if policy.name != name:
            raise ValueError("Virtual-egress policy mapping keys must match policy.name.")
        if type(policy) is HttpEgressPolicy:
            operations = tuple(
                EgressAuthorityOperation(method=method, path=path, match="exact")
                for method, path in policy.allowed_endpoints
                if not any(_path_matches_prefix(path, prefix) for prefix in policy.denied_prefixes)
            )
            identity = EgressAuthorityPolicyIdentity(
                name=name,
                kind="http",
                allowed_destinations=tuple(policy.allowed_hosts),
                operations=operations,
                denied_path_prefixes=policy.denied_prefixes,
                comparison_available=True,
            )
        elif type(policy) is BrowserEgressPolicy:
            identity = EgressAuthorityPolicyIdentity(
                name=name,
                kind="browser",
                allowed_destinations=tuple(policy.allowed_hosts),
                operations=tuple(
                    EgressAuthorityOperation(method=method, path=path, match="prefix")
                    for method in ("GET", "HEAD")
                    for path in policy.allowed_path_prefixes
                    if not any(
                        _path_matches_prefix(path, denied) for denied in policy.denied_prefixes
                    )
                ),
                denied_path_prefixes=policy.denied_prefixes,
                comparison_available=True,
            )
        else:
            identity = EgressAuthorityPolicyIdentity(
                name=name,
                kind="opaque",
                comparison_available=False,
            )
        policy_identities.append(identity)

    provisional = EgressAuthorityIdentity.model_construct(
        record_type="cayu.egress-authority",
        schema_version=EGRESS_AUTHORITY_SCHEMA_VERSION,
        generation=generation,
        authority_source=authority_source,
        authority_scope=authority_scope,
        policy_version=policy_version,
        runner_kind=runner_kind,
        cutover_strategy=cutover_strategy,
        policies=tuple(policy_identities),
        bindings=tuple(
            sorted(
                {
                    item
                    if type(item) is EgressAuthorityBindingIdentity
                    else EgressAuthorityBindingIdentity.model_validate(item)
                    for item in bindings
                },
                key=lambda item: (
                    item.destination,
                    item.policy_name,
                    item.credential_kind,
                    item.credential_authority_fingerprint,
                ),
            )
        ),
        comparison_available=all(item.comparison_available for item in policy_identities),
        fingerprint="0" * 64,
    )
    return EgressAuthorityIdentity.model_validate(
        {
            **provisional.model_dump(mode="json"),
            "fingerprint": _egress_authority_fingerprint(provisional),
        }
    )


def compare_egress_authority(
    expected: EgressAuthorityIdentity,
    candidate: EgressAuthorityIdentity,
    *,
    refused: bool = False,
) -> EgressAuthorityChangeKind:
    """Compare allowed operations conservatively; uncertainty is incomparable."""

    if refused:
        return EgressAuthorityChangeKind.REFUSED
    expected = EgressAuthorityIdentity.model_validate(expected.model_dump(mode="json"))
    candidate = EgressAuthorityIdentity.model_validate(candidate.model_dump(mode="json"))
    if expected.fingerprint == candidate.fingerprint:
        return EgressAuthorityChangeKind.UNCHANGED
    if (
        expected.authority_source != candidate.authority_source
        or expected.authority_scope != candidate.authority_scope
        or expected.runner_kind != candidate.runner_kind
        or not expected.comparison_available
        or not candidate.comparison_available
    ):
        return EgressAuthorityChangeKind.INCOMPARABLE
    expected_permissions = _permissions(expected)
    candidate_permissions = _permissions(candidate)
    if (
        _permission_comparison_work(candidate_permissions, expected_permissions)
        > EGRESS_AUTHORITY_MAX_COMPARISON_WORK
        or _permission_comparison_work(expected_permissions, candidate_permissions)
        > EGRESS_AUTHORITY_MAX_COMPARISON_WORK
    ):
        return EgressAuthorityChangeKind.INCOMPARABLE
    candidate_within_expected = _permission_set_within(candidate_permissions, expected_permissions)
    expected_within_candidate = _permission_set_within(expected_permissions, candidate_permissions)
    if candidate_within_expected and expected_within_candidate:
        return EgressAuthorityChangeKind.UNCHANGED
    if candidate_within_expected:
        return EgressAuthorityChangeKind.NARROWER
    if expected_within_candidate:
        return EgressAuthorityChangeKind.WIDER
    return EgressAuthorityChangeKind.INCOMPARABLE


class _Permission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    credential_kind: str
    credential_authority_fingerprint: str
    destination: str
    method: str
    path: str
    match: Literal["exact", "prefix"]
    denied_path_prefixes: tuple[str, ...]


def _permissions(identity: EgressAuthorityIdentity) -> tuple[_Permission, ...]:
    policies = {policy.name: policy for policy in identity.policies}
    permissions: list[_Permission] = []
    for binding in identity.bindings:
        policy = policies[binding.policy_name]
        if binding.destination not in policy.allowed_destinations:
            continue
        for operation in policy.operations:
            permissions.append(
                _Permission(
                    credential_kind=binding.credential_kind,
                    credential_authority_fingerprint=(binding.credential_authority_fingerprint),
                    destination=binding.destination,
                    method=operation.method,
                    path=operation.path,
                    match=operation.match,
                    denied_path_prefixes=policy.denied_path_prefixes,
                )
            )
    return tuple(permissions)


def _permission_set_within(
    children: Sequence[_Permission],
    parents: Sequence[_Permission],
) -> bool:
    parents_by_key: dict[tuple[str, str, str, str], list[_Permission]] = {}
    for parent in parents:
        parents_by_key.setdefault(_permission_comparison_key(parent), []).append(parent)
    return all(
        any(
            _permission_covers(parent, child)
            for parent in parents_by_key.get(_permission_comparison_key(child), ())
        )
        for child in children
    )


def _permission_comparison_work(
    children: Sequence[_Permission],
    parents: Sequence[_Permission],
) -> int:
    parent_work: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    for parent in parents:
        key = _permission_comparison_key(parent)
        count, denied_prefixes = parent_work.get(key, (0, 0))
        parent_work[key] = (count + 1, denied_prefixes + len(parent.denied_path_prefixes))
    total = 0
    for child in children:
        count, denied_prefixes = parent_work.get(_permission_comparison_key(child), (0, 0))
        denied_comparison_factor = (
            1 if child.match == "exact" else max(len(child.denied_path_prefixes), 1)
        )
        total += count + denied_prefixes * denied_comparison_factor
        if total > EGRESS_AUTHORITY_MAX_COMPARISON_WORK:
            return total
    return total


def _permission_comparison_key(permission: _Permission) -> tuple[str, str, str, str]:
    return (
        permission.credential_kind,
        permission.credential_authority_fingerprint,
        permission.destination,
        permission.method,
    )


def _permission_covers(parent: _Permission, child: _Permission) -> bool:
    if (
        parent.credential_kind != child.credential_kind
        or parent.credential_authority_fingerprint != child.credential_authority_fingerprint
        or parent.destination != child.destination
        or parent.method != child.method
    ):
        return False
    if parent.match == "exact":
        path_covered = child.match == "exact" and parent.path == child.path
    else:
        path_covered = _path_matches_prefix(child.path, parent.path)
    if not path_covered:
        return False
    if child.match == "exact":
        return not any(
            _path_matches_prefix(child.path, prefix) for prefix in parent.denied_path_prefixes
        )
    return all(
        not _prefixes_intersect(child.path, denied)
        or any(
            _path_matches_prefix(denied, child_denied)
            for child_denied in child.denied_path_prefixes
        )
        for denied in parent.denied_path_prefixes
    )


def _path_matches_prefix(path: str, prefix: str) -> bool:
    return (
        path.startswith("/") if prefix == "/" else path == prefix or path.startswith(prefix + "/")
    )


def _prefixes_intersect(left: str, right: str) -> bool:
    return _path_matches_prefix(left, right) or _path_matches_prefix(right, left)


def _egress_authority_fingerprint(identity: EgressAuthorityIdentity) -> str:
    material = identity.model_dump(mode="json", exclude={"fingerprint"})
    return sha256(canonical_durable_json_bytes(material, "egress_authority")).hexdigest()


def _egress_authority_cutover_receipt_fingerprint(
    receipt: EgressAuthorityCutoverReceipt,
) -> str:
    material = receipt.model_dump(mode="json", exclude={"fingerprint"})
    return sha256(canonical_durable_json_bytes(material, "egress_authority_cutover")).hexdigest()
