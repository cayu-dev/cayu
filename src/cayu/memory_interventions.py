"""Portable declarations and evidence bindings for fixed memory interventions.

This module deliberately stops at the contract boundary.  It does not apply an
intervention, run an experiment, select a candidate, or mutate production
knowledge.  Runtime execution is owned by the intervention executor built on
top of these records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import StrEnum
from functools import partial
from itertools import islice
from typing import Any, Literal, Self, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_clean_nonblank,
    revalidate_model_input,
    revalidate_model_inputs,
)
from cayu.agent_snapshots import (
    AgentSnapshot,
    AgentSnapshotComponentKind,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotOverlayKind,
    AgentSnapshotResultBinding,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
)
from cayu.memory import AutomaticRecallMode, AutomaticRecallPolicy
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionStatus,
    MemoryEvidenceAlias,
)

MEMORY_INTERVENTION_SCHEMA_VERSION = 1
MEMORY_INTERVENTION_MAX_BYTES = 1 << 20
MEMORY_INTERVENTION_MAX_CHANGED_ITEMS = 128
MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS = 128
MEMORY_INTERVENTION_MAX_FIXTURE_BYTES = 16 << 20

_MAX_ID_CHARS = 256
_MAX_REASON_CHARS = 256
_SHA256_HEX = frozenset("0123456789abcdef")
_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


def _clean(value: str, field_name: str, *, max_chars: int = _MAX_ID_CHARS) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters.")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _content_fingerprint(value: object, field_name: str) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def memory_attribution_fingerprint(attribution: MemoryAttribution) -> str:
    """Return the canonical portable identity of one attribution projection."""

    if type(attribution) is not MemoryAttribution:
        raise TypeError("attribution must be an exact MemoryAttribution.")
    validated = MemoryAttribution.model_validate(attribution.model_dump(mode="python"))
    return _content_fingerprint(validated, "memory attribution")


def _ordered_input(value: object, field_name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be an ordered array.")
    return value


def _ordered_unique_fingerprints(value: object, field_name: str) -> tuple[str, ...]:
    items = _ordered_input(value, field_name)
    values = tuple(_fingerprint(item, field_name) for item in items)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{field_name} must be unique and sorted.")
    return values


def _bounded_tuple(
    values: Iterable[Any],
    *,
    maximum: int,
    field_name: str,
) -> tuple[Any, ...]:
    items = tuple(islice(values, maximum + 1))
    if len(items) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} values.")
    return items


def _require_nested_json_member_types(
    value: object,
    *,
    field_name: str,
    members: tuple[tuple[str, type | tuple[type, ...], str], ...],
) -> None:
    if isinstance(value, BaseModel) or not isinstance(value, dict):
        return
    document = cast("dict[str, Any]", value)
    for member_name, expected_type, json_type_name in members:
        member_value = document.get(member_name)
        expected_types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
        if all(type(member_value) is not member_type for member_type in expected_types):
            raise ValueError(f"{field_name}.{member_name} must be a JSON {json_type_name}.")


def _require_attribution_json_timestamp_types(value: object) -> None:
    if isinstance(value, BaseModel) or not isinstance(value, dict):
        return
    document = cast("dict[str, Any]", value)
    receipts = document.get("receipts")
    if isinstance(receipts, list | tuple):
        for index, receipt in enumerate(receipts):
            _require_nested_json_member_types(
                receipt,
                field_name=f"attribution.receipts[{index}]",
                members=(("created_at", (str, datetime), "string"),),
            )
    exposures = document.get("exposures")
    if isinstance(exposures, list | tuple):
        for exposure_index, exposure in enumerate(exposures):
            _require_nested_json_member_types(
                exposure,
                field_name=f"attribution.exposures[{exposure_index}]",
                members=(
                    ("created_at", (str, datetime), "string"),
                    ("updated_at", (str, datetime), "string"),
                ),
            )
            if not isinstance(exposure, dict):
                continue
            transitions = exposure.get("transitions")
            if isinstance(transitions, list | tuple):
                for transition_index, transition in enumerate(transitions):
                    _require_nested_json_member_types(
                        transition,
                        field_name=(
                            f"attribution.exposures[{exposure_index}]"
                            f".transitions[{transition_index}]"
                        ),
                        members=(("occurred_at", (str, datetime), "string"),),
                    )


class _MemoryInterventionModel(BaseModel):
    model_config = _MODEL_CONFIG

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def validate_schema_version_literal(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be a JSON integer.")
        return value

    @field_validator(
        "evidence_only",
        "production_mutation_allowed",
        "generic_experiment_comparability_required",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def validate_boolean_literals(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a JSON boolean.")
        return value


class _FingerprintRecord(_MemoryInterventionModel):
    fingerprint: StrictStr
    record_type: StrictStr

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    def identity_material(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"fingerprint"})

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        expected = _content_fingerprint(self.identity_material(), self.record_type)
        if self.fingerprint != expected:
            raise ValueError("Record fingerprint does not match its canonical contents.")
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            self.record_type,
        )
        if len(encoded) > MEMORY_INTERVENTION_MAX_BYTES:
            raise ValueError(
                f"Memory intervention record exceeds {MEMORY_INTERVENTION_MAX_BYTES} bytes."
            )
        return self


class MemoryInterventionKind(StrEnum):
    AS_DECLARED = "as_declared"
    AUTOMATIC_RECALL_OFF = "automatic_recall_off"
    OMIT_ITEMS = "omit_items"
    REPLACE_ITEMS = "replace_items"
    NEGATIVE_CONTROL = "negative_control"


class MemoryInterventionItemIdentityKind(StrEnum):
    ALIAS = "alias"
    FINGERPRINT = "fingerprint"


class MemoryInterventionChangeKind(StrEnum):
    OMIT = "omit"
    REPLACE = "replace"
    INJECT_NEGATIVE_CONTROL = "inject_negative_control"


class MemoryNegativeControlKind(StrEnum):
    IRRELEVANT = "irrelevant"
    STALE = "stale"
    CONFLICTING = "conflicting"
    ADVERSARIAL = "adversarial"


class MemoryInterventionEffectStatus(StrEnum):
    VERIFIED_NO_CHANGE = "verified_no_change"
    APPLIED = "applied"
    MATCHED_NO_ITEMS = "matched_no_items"
    INDETERMINATE = "indeterminate"
    CONFLICTING = "conflicting"


class MemoryInterventionComparabilityStatus(StrEnum):
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"


class MemoryInterventionMismatchReason(StrEnum):
    BASELINE_MEMORY_STATE = "baseline_memory_state"
    INTERVENTION_IDENTITY = "intervention_identity"
    RECALL_POLICY = "recall_policy"
    MATERIALIZATION_OVERLAY = "materialization_overlay"
    CHANGED_ITEM_REVISIONS = "changed_item_revisions"
    TRIAL_MODE = "trial_mode"
    REQUIRED_ATTRIBUTION_AVAILABILITY = "required_attribution_availability"


class MemoryInterventionBounds(_MemoryInterventionModel):
    """Precommitted upper bounds for one memory-only intervention."""

    max_changed_items: StrictInt = Field(ge=0, le=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS)
    max_fixture_bytes: StrictInt = Field(ge=0, le=MEMORY_INTERVENTION_MAX_FIXTURE_BYTES)
    max_effect_receipts: StrictInt = Field(
        default=MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
        ge=1,
        le=MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
    )


class MemoryInterventionItemIdentity(_MemoryInterventionModel):
    """One exact revision-bound item identity without retained memory text."""

    kind: MemoryInterventionItemIdentityKind
    revision_fingerprint: StrictStr
    alias: MemoryEvidenceAlias | None = None
    item_fingerprint: StrictStr | None = None

    @field_validator("revision_fingerprint", "item_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _fingerprint(value, info.field_name)

    @field_validator("alias", mode="before")
    @classmethod
    def copy_alias(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryEvidenceAlias)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.kind is MemoryInterventionItemIdentityKind.ALIAS:
            if self.alias is None or self.alias.kind != "item" or self.item_fingerprint is not None:
                raise ValueError("Alias item identity requires exactly one item alias.")
        elif self.alias is not None or self.item_fingerprint is None:
            raise ValueError("Fingerprint item identity requires exactly one item fingerprint.")
        return self

    def sort_key(self) -> tuple[str, str, str, str]:
        value = self.item_fingerprint if self.alias is None else self.alias.digest
        if value is None:  # guarded by validate_identity
            raise RuntimeError("Validated item identity lost its value.")
        key_id = "" if self.alias is None else self.alias.key_id
        return (self.kind.value, key_id, value, self.revision_fingerprint)


class MemoryInterventionFixtureRef(_MemoryInterventionModel):
    """Content-addressed evaluation input; never a store location or memory text."""

    fixture_id: StrictStr
    fixture_fingerprint: StrictStr
    representation_fingerprint: StrictStr
    size_bytes: StrictInt = Field(ge=0, le=MEMORY_INTERVENTION_MAX_FIXTURE_BYTES)

    @field_validator("fixture_id")
    @classmethod
    def validate_fixture_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("fixture_fingerprint", "representation_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)


class MemoryInterventionItemChange(_MemoryInterventionModel):
    """One bounded omission, replacement, or control-fixture injection."""

    kind: MemoryInterventionChangeKind
    source_item: MemoryInterventionItemIdentity | None = None
    fixture: MemoryInterventionFixtureRef | None = None

    @field_validator("source_item", mode="before")
    @classmethod
    def copy_source(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionItemIdentity)

    @field_validator("fixture", mode="before")
    @classmethod
    def copy_fixture(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionFixtureRef)

    @model_validator(mode="after")
    def validate_change(self) -> Self:
        if self.kind is MemoryInterventionChangeKind.OMIT:
            if self.source_item is None or self.fixture is not None:
                raise ValueError("Omission requires one source item and no fixture.")
        elif self.kind is MemoryInterventionChangeKind.REPLACE:
            if self.source_item is None or self.fixture is None:
                raise ValueError("Replacement requires one source item and one fixture.")
        elif self.source_item is not None or self.fixture is None:
            raise ValueError("Negative-control injection requires only one fixture.")
        return self

    def sort_key(self) -> tuple[str, str, str]:
        source = "" if self.source_item is None else ":".join(self.source_item.sort_key())
        fixture = "" if self.fixture is None else self.fixture.fixture_fingerprint
        return (self.kind.value, source, fixture)


class MemoryInterventionSpec(_FingerprintRecord):
    """Immutable declaration of one fixed intervention over a snapshot memory frontier."""

    record_type: Literal["cayu.memory-intervention-spec"] = "cayu.memory-intervention-spec"
    schema_version: Literal[1] = MEMORY_INTERVENTION_SCHEMA_VERSION
    spec_id: StrictStr
    snapshot_fingerprint: StrictStr
    memory_state_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    recall_policy_ref_fingerprint: StrictStr
    starting_recall_policy_fingerprint: StrictStr
    trial_recall_policy_fingerprint: StrictStr
    starting_recall_mode: AutomaticRecallMode
    trial_recall_mode: AutomaticRecallMode
    trial_state_mode: AgentSnapshotTrialStateMode
    authority_scope_fingerprint: StrictStr
    kind: MemoryInterventionKind
    bounds: MemoryInterventionBounds
    changes: tuple[MemoryInterventionItemChange, ...] = Field(
        default=(),
        max_length=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
    )
    negative_control_kind: MemoryNegativeControlKind | None = None
    proposer_fingerprint: StrictStr | None = None
    source_fingerprint: StrictStr | None = None
    reason: StrictStr | None = Field(default=None, max_length=_MAX_REASON_CHARS)
    evidence_only: Literal[True] = True
    production_mutation_allowed: Literal[False] = False

    @field_validator("spec_id")
    @classmethod
    def validate_spec_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator(
        "snapshot_fingerprint",
        "memory_state_fingerprint",
        "execution_profile_fingerprint",
        "recall_policy_ref_fingerprint",
        "starting_recall_policy_fingerprint",
        "trial_recall_policy_fingerprint",
        "authority_scope_fingerprint",
        "proposer_fingerprint",
        "source_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _fingerprint(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(value, info.field_name, max_chars=_MAX_REASON_CHARS)

    @field_validator("bounds", mode="before")
    @classmethod
    def copy_bounds(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionBounds)

    @field_validator("changes", mode="before")
    @classmethod
    def copy_changes(cls, value: object) -> object:
        _ordered_input(value, "changes")
        return revalidate_model_inputs(
            value,
            MemoryInterventionItemChange,
            maximum=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
            field_name="changes",
        )

    @model_validator(mode="after")
    def validate_intervention(self) -> Self:
        if self.recall_policy_ref_fingerprint != self.starting_recall_policy_fingerprint:
            raise ValueError(
                "Starting recall policy does not match the snapshot recall-policy frontier."
            )
        keys = tuple(change.sort_key() for change in self.changes)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Intervention changes must be unique and deterministically sorted.")
        source_keys = tuple(
            change.source_item.sort_key()
            for change in self.changes
            if change.source_item is not None
        )
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("Intervention source identities must be unique.")
        source_revisions = tuple(
            change.source_item.revision_fingerprint
            for change in self.changes
            if change.source_item is not None
        )
        if len(source_revisions) != len(set(source_revisions)):
            raise ValueError("Intervention source revision identities must be unique.")
        if len(self.changes) > self.bounds.max_changed_items:
            raise ValueError("Declared intervention exceeds max_changed_items.")
        fixture_bytes = sum(
            change.fixture.size_bytes for change in self.changes if change.fixture is not None
        )
        if fixture_bytes > self.bounds.max_fixture_bytes:
            raise ValueError("Declared intervention fixtures exceed max_fixture_bytes.")

        has_declaration = all(
            value is not None
            for value in (self.proposer_fingerprint, self.source_fingerprint, self.reason)
        )
        if self.kind in {
            MemoryInterventionKind.AS_DECLARED,
            MemoryInterventionKind.AUTOMATIC_RECALL_OFF,
        }:
            if self.changes or self.bounds.max_changed_items or self.bounds.max_fixture_bytes:
                raise ValueError("Policy-only interventions cannot declare item changes.")
            if any(
                value is not None
                for value in (
                    self.negative_control_kind,
                    self.proposer_fingerprint,
                    self.source_fingerprint,
                    self.reason,
                )
            ):
                raise ValueError("Policy-only interventions cannot declare item provenance.")
        elif not self.changes or not has_declaration:
            raise ValueError("Item interventions require changes, proposer, source, and reason.")

        expected_change_kind = {
            MemoryInterventionKind.OMIT_ITEMS: MemoryInterventionChangeKind.OMIT,
            MemoryInterventionKind.REPLACE_ITEMS: MemoryInterventionChangeKind.REPLACE,
            MemoryInterventionKind.NEGATIVE_CONTROL: (
                MemoryInterventionChangeKind.INJECT_NEGATIVE_CONTROL
            ),
        }.get(self.kind)
        if expected_change_kind is not None and any(
            change.kind is not expected_change_kind for change in self.changes
        ):
            raise ValueError("Mixed or mismatched intervention change kinds are forbidden.")
        if (self.kind is MemoryInterventionKind.NEGATIVE_CONTROL) != (
            self.negative_control_kind is not None
        ):
            raise ValueError("Only a negative-control intervention declares a control kind.")

        if self.kind is MemoryInterventionKind.AUTOMATIC_RECALL_OFF:
            if self.starting_recall_mode is AutomaticRecallMode.OFF:
                raise ValueError("Recall-off starting policy must not already be OFF.")
            if self.trial_recall_mode is not AutomaticRecallMode.OFF:
                raise ValueError("automatic_recall_off requires the exact OFF trial policy.")
            if self.starting_recall_policy_fingerprint == self.trial_recall_policy_fingerprint:
                raise ValueError("Recall-off must replace the frozen starting recall policy.")
        elif (
            self.starting_recall_policy_fingerprint != self.trial_recall_policy_fingerprint
            or self.starting_recall_mode is not self.trial_recall_mode
        ):
            raise ValueError("Memory-item interventions cannot change the recall policy.")
        return self

    @classmethod
    def create(
        cls,
        *,
        spec_id: str,
        snapshot: AgentSnapshot,
        starting_recall_policy: AutomaticRecallPolicy,
        trial_recall_policy: AutomaticRecallPolicy,
        trial_state_mode: AgentSnapshotTrialStateMode,
        kind: MemoryInterventionKind,
        bounds: MemoryInterventionBounds,
        changes: Sequence[MemoryInterventionItemChange] = (),
        negative_control_kind: MemoryNegativeControlKind | None = None,
        proposer_fingerprint: str | None = None,
        source_fingerprint: str | None = None,
        reason: str | None = None,
    ) -> MemoryInterventionSpec:
        if type(snapshot) is not AgentSnapshot:
            raise TypeError("snapshot must be an exact AgentSnapshot.")
        if type(starting_recall_policy) is not AutomaticRecallPolicy:
            raise TypeError("starting_recall_policy must be an exact AutomaticRecallPolicy.")
        if type(trial_recall_policy) is not AutomaticRecallPolicy:
            raise TypeError("trial_recall_policy must be an exact AutomaticRecallPolicy.")
        if type(bounds) is not MemoryInterventionBounds:
            raise TypeError("bounds must be an exact MemoryInterventionBounds.")
        snapshot = AgentSnapshot.model_validate(snapshot.model_dump(mode="python"))
        if snapshot.memory_state is None or snapshot.memory_state.recall_policy is None:
            raise ValueError("A memory intervention requires a snapshot recall-policy frontier.")
        starting = AutomaticRecallPolicy.model_validate(
            starting_recall_policy.model_dump(mode="python")
        )
        trial = AutomaticRecallPolicy.model_validate(trial_recall_policy.model_dump(mode="python"))
        bounded_changes = _bounded_tuple(
            changes,
            maximum=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
            field_name="changes",
        )
        ordered_changes = tuple(sorted(bounded_changes, key=lambda change: change.sort_key()))
        values: dict[str, Any] = {
            "spec_id": spec_id,
            "snapshot_fingerprint": snapshot.fingerprint,
            "memory_state_fingerprint": snapshot.memory_state.fingerprint,
            "execution_profile_fingerprint": snapshot.execution_profile.fingerprint,
            "recall_policy_ref_fingerprint": snapshot.memory_state.recall_policy.fingerprint,
            "starting_recall_policy_fingerprint": starting.fingerprint(),
            "trial_recall_policy_fingerprint": trial.fingerprint(),
            "starting_recall_mode": starting.mode,
            "trial_recall_mode": trial.mode,
            "trial_state_mode": trial_state_mode,
            "authority_scope_fingerprint": snapshot.authority_scope_fingerprint,
            "kind": kind,
            "bounds": bounds,
            "changes": ordered_changes,
            "negative_control_kind": negative_control_kind,
            "proposer_fingerprint": proposer_fingerprint,
            "source_fingerprint": source_fingerprint,
            "reason": reason,
        }
        provisional = cls.model_construct(fingerprint="0" * 64, **values)
        return cls(
            fingerprint=_content_fingerprint(provisional.identity_material(), cls.__name__),
            **values,
        )


class MemoryInterventionOperation(_FingerprintRecord):
    """Precommitment binding one spec to one isolated materialization and trial."""

    record_type: Literal["cayu.memory-intervention-operation"] = (
        "cayu.memory-intervention-operation"
    )
    schema_version: Literal[1] = MEMORY_INTERVENTION_SCHEMA_VERSION
    spec_fingerprint: StrictStr
    intervention_kind: MemoryInterventionKind
    candidate_id: StrictStr
    snapshot_fingerprint: StrictStr
    memory_state_fingerprint: StrictStr
    materialization_fingerprint: StrictStr
    memory_overlay_fingerprint: StrictStr
    state_scope_id: StrictStr
    trial_state_mode: AgentSnapshotTrialStateMode
    trial_binding_fingerprint: StrictStr
    case_id: StrictStr
    trial_id: StrictStr

    @field_validator(
        "spec_fingerprint",
        "snapshot_fingerprint",
        "memory_state_fingerprint",
        "materialization_fingerprint",
        "memory_overlay_fingerprint",
        "trial_binding_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("candidate_id", "state_scope_id", "case_id", "trial_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @property
    def operation_id(self) -> str:
        """Stable idempotency identity for the precommitted operation."""

        return self.fingerprint

    @classmethod
    def create(
        cls,
        *,
        spec: MemoryInterventionSpec,
        materialization: AgentSnapshotMaterialization,
        trial: AgentSnapshotTrialBinding,
    ) -> MemoryInterventionOperation:
        spec = MemoryInterventionSpec.model_validate(spec.model_dump(mode="python"))
        materialization = AgentSnapshotMaterialization.model_validate(
            materialization.model_dump(mode="python")
        )
        trial = AgentSnapshotTrialBinding.model_validate(trial.model_dump(mode="python"))
        if materialization.snapshot_fingerprint != spec.snapshot_fingerprint:
            raise ValueError("Intervention materialization belongs to another snapshot.")
        if materialization.state_mode is not spec.trial_state_mode:
            raise ValueError("Intervention materialization uses another trial-state mode.")
        if (
            trial.snapshot_fingerprint != spec.snapshot_fingerprint
            or trial.materialization_fingerprint != materialization.fingerprint
            or trial.candidate_id != materialization.candidate_id
        ):
            raise ValueError("Intervention trial does not match its materialization.")
        expected_scope_id = AgentSnapshotMaterializationRequest.derive_state_scope_id(
            snapshot_fingerprint=spec.snapshot_fingerprint,
            candidate_id=materialization.candidate_id,
            trial_id=trial.trial_id,
            state_mode=spec.trial_state_mode,
            state_partition_fingerprint=spec.fingerprint,
        )
        if (
            materialization.state_scope_id != expected_scope_id
            or materialization.state_partition_fingerprint != spec.fingerprint
        ):
            raise ValueError("Intervention materialization uses another trial state scope.")
        memory_component = next(
            (
                component
                for component in materialization.components
                if component.kind is AgentSnapshotComponentKind.MEMORY
            ),
            None,
        )
        if (
            memory_component is None
            or memory_component.overlay is None
            or memory_component.overlay.kind is not AgentSnapshotOverlayKind.MEMORY
        ):
            raise ValueError("A memory intervention requires one isolated memory overlay.")
        memory_overlay = memory_component.overlay
        if (
            memory_component.baseline_fingerprint != spec.memory_state_fingerprint
            or memory_overlay.baseline_fingerprint != spec.memory_state_fingerprint
        ):
            raise ValueError(
                "Intervention memory baseline differs from its declared starting state."
            )
        if trial.memory_overlay_fingerprint != memory_overlay.fingerprint:
            raise ValueError("Intervention trial and materialization memory overlays differ.")
        values: dict[str, Any] = {
            "spec_fingerprint": spec.fingerprint,
            "intervention_kind": spec.kind,
            "candidate_id": materialization.candidate_id,
            "snapshot_fingerprint": spec.snapshot_fingerprint,
            "memory_state_fingerprint": spec.memory_state_fingerprint,
            "materialization_fingerprint": materialization.fingerprint,
            "memory_overlay_fingerprint": memory_overlay.fingerprint,
            "state_scope_id": materialization.state_scope_id,
            "trial_state_mode": materialization.state_mode,
            "trial_binding_fingerprint": trial.fingerprint,
            "case_id": trial.case_id,
            "trial_id": trial.trial_id,
        }
        provisional = cls.model_construct(fingerprint="0" * 64, **values)
        return cls(
            fingerprint=_content_fingerprint(provisional.identity_material(), cls.__name__),
            **values,
        )


class MemoryInterventionEffectReceiptRef(_MemoryInterventionModel):
    """Bounded identity of one application-owned effect receipt."""

    owner_id: StrictStr
    receipt_fingerprint: StrictStr
    effect_fingerprint: StrictStr

    @field_validator("owner_id")
    @classmethod
    def validate_owner_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("receipt_fingerprint", "effect_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.owner_id, self.receipt_fingerprint, self.effect_fingerprint)

    def receipt_identity_key(self) -> tuple[str, str]:
        return (self.owner_id, self.receipt_fingerprint)


class MemoryInterventionReceipt(_FingerprintRecord):
    """Deterministic evidence of the exact outcome of one precommitted operation."""

    record_type: Literal["cayu.memory-intervention-receipt"] = "cayu.memory-intervention-receipt"
    schema_version: Literal[1] = MEMORY_INTERVENTION_SCHEMA_VERSION
    spec_fingerprint: StrictStr
    operation_fingerprint: StrictStr
    intervention_kind: MemoryInterventionKind
    snapshot_fingerprint: StrictStr
    starting_memory_state_fingerprint: StrictStr
    materialization_fingerprint: StrictStr
    memory_overlay_fingerprint: StrictStr
    state_scope_id: StrictStr
    status: MemoryInterventionEffectStatus
    result_memory_state_fingerprint: StrictStr | None = None
    result_recall_policy_fingerprint: StrictStr | None = None
    matched_item_count: StrictInt = Field(ge=0, le=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS)
    changed_item_revision_fingerprints: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
    )
    effect_fingerprints: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
    )
    application_effect_receipts: tuple[MemoryInterventionEffectReceiptRef, ...] = Field(
        default=(),
        max_length=MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
    )
    evidence_only: Literal[True] = True
    production_mutation_allowed: Literal[False] = False

    @field_validator(
        "spec_fingerprint",
        "operation_fingerprint",
        "snapshot_fingerprint",
        "starting_memory_state_fingerprint",
        "materialization_fingerprint",
        "memory_overlay_fingerprint",
        "result_memory_state_fingerprint",
        "result_recall_policy_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _fingerprint(value, info.field_name)

    @field_validator("state_scope_id")
    @classmethod
    def validate_scope(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("changed_item_revision_fingerprints", "effect_fingerprints", mode="before")
    @classmethod
    def validate_ordered_fingerprints(cls, value: object, info) -> tuple[str, ...]:
        return _ordered_unique_fingerprints(value, info.field_name)

    @field_validator("application_effect_receipts", mode="before")
    @classmethod
    def copy_effect_receipts(cls, value: object) -> object:
        _ordered_input(value, "application_effect_receipts")
        return revalidate_model_inputs(
            value,
            MemoryInterventionEffectReceiptRef,
            maximum=MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
            field_name="application_effect_receipts",
        )

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        receipt_keys = tuple(receipt.sort_key() for receipt in self.application_effect_receipts)
        if receipt_keys != tuple(sorted(set(receipt_keys))):
            raise ValueError("Application effect receipts must be unique and sorted.")
        receipt_identity_keys = tuple(
            receipt.receipt_identity_key() for receipt in self.application_effect_receipts
        )
        if len(receipt_identity_keys) != len(set(receipt_identity_keys)):
            raise ValueError("Application effect receipt identities must be unique.")
        if any(
            receipt.effect_fingerprint not in self.effect_fingerprints
            for receipt in self.application_effect_receipts
        ):
            raise ValueError(
                "Application effect receipts must reference declared effect fingerprints."
            )
        receipt_effect_fingerprints = {
            receipt.effect_fingerprint for receipt in self.application_effect_receipts
        }
        if (
            self.status is MemoryInterventionEffectStatus.APPLIED
            and receipt_effect_fingerprints != set(self.effect_fingerprints)
        ):
            raise ValueError(
                "Applied intervention requires application receipt proof for every effect."
            )
        if self.status is MemoryInterventionEffectStatus.VERIFIED_NO_CHANGE:
            if self.intervention_kind is not MemoryInterventionKind.AS_DECLARED:
                raise ValueError("Only as_declared can produce verified_no_change.")
            if self.matched_item_count or self.changed_item_revision_fingerprints:
                raise ValueError("Verified baseline cannot report changed items.")
            if (
                self.result_memory_state_fingerprint != self.starting_memory_state_fingerprint
                or self.result_recall_policy_fingerprint is None
                or self.effect_fingerprints
            ):
                raise ValueError(
                    "Verified baseline requires its exact unchanged memory and recall policy."
                )
        elif self.intervention_kind is MemoryInterventionKind.AS_DECLARED:
            raise ValueError("as_declared must produce verified_no_change.")
        if self.status is MemoryInterventionEffectStatus.MATCHED_NO_ITEMS:
            if self.intervention_kind not in {
                MemoryInterventionKind.OMIT_ITEMS,
                MemoryInterventionKind.REPLACE_ITEMS,
            }:
                raise ValueError("Only item matching can produce matched_no_items.")
            if (
                self.matched_item_count
                or self.changed_item_revision_fingerprints
                or self.effect_fingerprints
                or self.result_memory_state_fingerprint != self.starting_memory_state_fingerprint
                or self.result_recall_policy_fingerprint is None
            ):
                raise ValueError("matched_no_items requires an exact unchanged result.")
        if self.status is MemoryInterventionEffectStatus.APPLIED and (
            self.result_memory_state_fingerprint is None
            or self.result_recall_policy_fingerprint is None
            or not self.effect_fingerprints
        ):
            raise ValueError("Applied intervention requires exact resulting effect identities.")
        if self.status is MemoryInterventionEffectStatus.APPLIED:
            if self.result_memory_state_fingerprint == self.starting_memory_state_fingerprint:
                raise ValueError("Applied intervention must change the memory-state identity.")
            item_intervention = self.intervention_kind in {
                MemoryInterventionKind.OMIT_ITEMS,
                MemoryInterventionKind.REPLACE_ITEMS,
            }
            if item_intervention and (
                not self.matched_item_count
                or len(self.changed_item_revision_fingerprints) != self.matched_item_count
            ):
                raise ValueError("Applied item intervention requires every changed revision.")
            if not item_intervention and self.changed_item_revision_fingerprints:
                raise ValueError("Policy/control intervention cannot report source-item changes.")
        if self.status in {
            MemoryInterventionEffectStatus.INDETERMINATE,
            MemoryInterventionEffectStatus.CONFLICTING,
        } and (
            self.result_memory_state_fingerprint is not None
            or self.result_recall_policy_fingerprint is not None
            or self.changed_item_revision_fingerprints
            or self.effect_fingerprints
            or self.application_effect_receipts
        ):
            raise ValueError("Untrusted intervention outcomes cannot carry successful effects.")
        if len(self.changed_item_revision_fingerprints) > self.matched_item_count:
            raise ValueError("Changed item revisions exceed the matched item count.")
        return self

    @classmethod
    def create(
        cls,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        status: MemoryInterventionEffectStatus,
        result_memory_state_fingerprint: str | None = None,
        result_recall_policy_fingerprint: str | None = None,
        matched_item_count: int = 0,
        changed_item_revision_fingerprints: Iterable[str] = (),
        effect_fingerprints: Iterable[str] = (),
        application_effect_receipts: Iterable[MemoryInterventionEffectReceiptRef] = (),
    ) -> MemoryInterventionReceipt:
        spec = MemoryInterventionSpec.model_validate(spec.model_dump(mode="python"))
        operation = MemoryInterventionOperation.model_validate(operation.model_dump(mode="python"))
        if (
            operation.spec_fingerprint != spec.fingerprint
            or operation.intervention_kind is not spec.kind
            or operation.snapshot_fingerprint != spec.snapshot_fingerprint
            or operation.memory_state_fingerprint != spec.memory_state_fingerprint
        ):
            raise ValueError("Intervention operation conflicts with its declared spec.")
        changed_input = _bounded_tuple(
            changed_item_revision_fingerprints,
            maximum=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
            field_name="changed_item_revision_fingerprints",
        )
        effects_input = _bounded_tuple(
            effect_fingerprints,
            maximum=MEMORY_INTERVENTION_MAX_CHANGED_ITEMS,
            field_name="effect_fingerprints",
        )
        receipts_input = _bounded_tuple(
            application_effect_receipts,
            maximum=MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS,
            field_name="application_effect_receipts",
        )
        if len(changed_input) != len(set(changed_input)):
            raise ValueError("Changed-item revisions must be unique.")
        if len(effects_input) != len(set(effects_input)):
            raise ValueError("Effect fingerprints must be unique.")
        receipt_keys = tuple(receipt.sort_key() for receipt in receipts_input)
        if len(receipt_keys) != len(set(receipt_keys)):
            raise ValueError("Application effect receipts must be unique.")
        receipt_identity_keys = tuple(receipt.receipt_identity_key() for receipt in receipts_input)
        if len(receipt_identity_keys) != len(set(receipt_identity_keys)):
            raise ValueError("Application effect receipt identities must be unique.")
        if any(receipt.effect_fingerprint not in effects_input for receipt in receipts_input):
            raise ValueError(
                "Application effect receipts must reference declared effect fingerprints."
            )
        changed = tuple(sorted(changed_input))
        effects = tuple(sorted(effects_input))
        effect_receipts = tuple(sorted(receipts_input, key=lambda receipt: receipt.sort_key()))
        if matched_item_count > spec.bounds.max_changed_items:
            raise ValueError("Intervention outcome exceeds its declared changed-item bound.")
        if len(effect_receipts) > spec.bounds.max_effect_receipts:
            raise ValueError("Intervention outcome exceeds its effect-receipt bound.")
        declared_revisions = {
            change.source_item.revision_fingerprint
            for change in spec.changes
            if change.source_item is not None
        }
        if set(changed) - declared_revisions:
            raise ValueError("Intervention outcome reports undeclared changed-item revisions.")
        if (
            result_recall_policy_fingerprint is not None
            and result_recall_policy_fingerprint != spec.trial_recall_policy_fingerprint
        ):
            raise ValueError("Intervention outcome reports a different trial recall policy.")
        values: dict[str, Any] = {
            "spec_fingerprint": spec.fingerprint,
            "operation_fingerprint": operation.fingerprint,
            "intervention_kind": spec.kind,
            "snapshot_fingerprint": spec.snapshot_fingerprint,
            "starting_memory_state_fingerprint": spec.memory_state_fingerprint,
            "materialization_fingerprint": operation.materialization_fingerprint,
            "memory_overlay_fingerprint": operation.memory_overlay_fingerprint,
            "state_scope_id": operation.state_scope_id,
            "status": status,
            "result_memory_state_fingerprint": result_memory_state_fingerprint,
            "result_recall_policy_fingerprint": result_recall_policy_fingerprint,
            "matched_item_count": matched_item_count,
            "changed_item_revision_fingerprints": changed,
            "effect_fingerprints": effects,
            "application_effect_receipts": effect_receipts,
        }
        provisional = cls.model_construct(fingerprint="0" * 64, **values)
        return cls(
            fingerprint=_content_fingerprint(provisional.identity_material(), cls.__name__),
            **values,
        )


class MemoryInterventionTrialBinding(_FingerprintRecord):
    """Bind one intervention effect to Cayu trial, result, and attribution records."""

    record_type: Literal["cayu.memory-intervention-trial"] = "cayu.memory-intervention-trial"
    schema_version: Literal[1] = MEMORY_INTERVENTION_SCHEMA_VERSION
    spec: MemoryInterventionSpec
    operation: MemoryInterventionOperation
    receipt: MemoryInterventionReceipt
    trial: AgentSnapshotTrialBinding
    result: AgentSnapshotResultBinding
    terminal_evidence_available: StrictBool = False
    expected_receipt_count: StrictInt | None = Field(default=None, ge=0)
    expected_exposure_count: StrictInt | None = Field(default=None, ge=0)
    attribution: MemoryAttribution
    attribution_fingerprint: StrictStr

    @field_validator("spec", mode="before")
    @classmethod
    def copy_spec(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionSpec)

    @field_validator("operation", mode="before")
    @classmethod
    def copy_operation(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionOperation)

    @field_validator("receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionReceipt)

    @field_validator("trial", mode="before")
    @classmethod
    def copy_trial(cls, value: object) -> object:
        _require_nested_json_member_types(
            value,
            field_name="trial",
            members=(
                ("schema_version", int, "integer"),
                ("created_at", (str, datetime), "string"),
            ),
        )
        return revalidate_model_input(value, AgentSnapshotTrialBinding)

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        _require_nested_json_member_types(
            value,
            field_name="result",
            members=(
                ("schema_version", int, "integer"),
                ("recorded_at", (str, datetime), "string"),
            ),
        )
        return revalidate_model_input(value, AgentSnapshotResultBinding)

    @field_validator("attribution", mode="before")
    @classmethod
    def copy_attribution(cls, value: object) -> object:
        _require_attribution_json_timestamp_types(value)
        return revalidate_model_input(value, MemoryAttribution)

    @field_validator("attribution_fingerprint")
    @classmethod
    def validate_attribution_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if (
            self.operation.spec_fingerprint != self.spec.fingerprint
            or self.receipt.spec_fingerprint != self.spec.fingerprint
            or self.receipt.operation_fingerprint != self.operation.fingerprint
        ):
            raise ValueError("Intervention trial contains conflicting spec/effect identities.")
        if (
            self.operation.intervention_kind is not self.spec.kind
            or self.operation.snapshot_fingerprint != self.spec.snapshot_fingerprint
            or self.operation.memory_state_fingerprint != self.spec.memory_state_fingerprint
            or self.operation.trial_state_mode is not self.spec.trial_state_mode
        ):
            raise ValueError("Intervention operation conflicts with its declared starting state.")
        if (
            self.receipt.intervention_kind is not self.spec.kind
            or self.receipt.snapshot_fingerprint != self.spec.snapshot_fingerprint
            or self.receipt.starting_memory_state_fingerprint != self.spec.memory_state_fingerprint
            or self.receipt.materialization_fingerprint
            != self.operation.materialization_fingerprint
            or self.receipt.memory_overlay_fingerprint != self.operation.memory_overlay_fingerprint
            or self.receipt.state_scope_id != self.operation.state_scope_id
        ):
            raise ValueError("Intervention receipt conflicts with its precommitted operation.")
        if (
            self.receipt.result_recall_policy_fingerprint is not None
            and self.receipt.result_recall_policy_fingerprint
            != self.spec.trial_recall_policy_fingerprint
        ):
            raise ValueError("Intervention receipt reports another trial recall policy.")
        declared_revisions = {
            change.source_item.revision_fingerprint
            for change in self.spec.changes
            if change.source_item is not None
        }
        if (
            self.receipt.matched_item_count > self.spec.bounds.max_changed_items
            or set(self.receipt.changed_item_revision_fingerprints) - declared_revisions
            or len(self.receipt.application_effect_receipts) > self.spec.bounds.max_effect_receipts
        ):
            raise ValueError("Intervention receipt exceeds or conflicts with its declaration.")
        if (
            self.operation.trial_binding_fingerprint != self.trial.fingerprint
            or self.result.trial_fingerprint != self.trial.fingerprint
            or self.trial.snapshot_fingerprint != self.operation.snapshot_fingerprint
            or self.operation.materialization_fingerprint != self.trial.materialization_fingerprint
            or self.trial.memory_overlay_fingerprint != self.operation.memory_overlay_fingerprint
            or self.operation.candidate_id != self.trial.candidate_id
            or self.operation.case_id != self.trial.case_id
            or self.operation.trial_id != self.trial.trial_id
        ):
            raise ValueError("Intervention operation, trial, and result lineage conflict.")
        expected_attribution = memory_attribution_fingerprint(self.attribution)
        if self.attribution_fingerprint != expected_attribution:
            raise ValueError("Attribution fingerprint does not match its exact projection.")
        if self.result.memory_evidence_fingerprint != expected_attribution:
            raise ValueError("AgentSnapshot result does not bind the supplied memory attribution.")
        if (self.expected_receipt_count is None) is not (self.expected_exposure_count is None):
            raise ValueError("Terminal memory counts must be present or absent together.")
        counts_available = self.expected_receipt_count is not None
        if self.terminal_evidence_available is not counts_available:
            raise ValueError(
                "Terminal memory counts must be present exactly when terminal evidence is available."
            )
        return self

    @property
    def proves_no_memory_exposure(self) -> bool:
        """Whether complete attribution proves that this trial exposed no memory."""

        return (
            self.terminal_evidence_available
            and self.expected_receipt_count == 0
            and self.expected_exposure_count == 0
            and self.attribution.status is MemoryAttributionStatus.COMPLETE
            and self.attribution.observed_receipt_count == 0
            and self.attribution.observed_exposure_count == 0
            and self.attribution.observed_item_count == 0
        )

    @classmethod
    def create(
        cls,
        *,
        spec: MemoryInterventionSpec,
        operation: MemoryInterventionOperation,
        receipt: MemoryInterventionReceipt,
        trial: AgentSnapshotTrialBinding,
        result: AgentSnapshotResultBinding,
        attribution: MemoryAttribution,
        terminal_evidence_available: bool = False,
        expected_receipt_count: int | None = None,
        expected_exposure_count: int | None = None,
    ) -> MemoryInterventionTrialBinding:
        attribution_fingerprint = memory_attribution_fingerprint(attribution)
        values: dict[str, Any] = {
            "spec": spec,
            "operation": operation,
            "receipt": receipt,
            "trial": trial,
            "result": result,
            "terminal_evidence_available": terminal_evidence_available,
            "expected_receipt_count": expected_receipt_count,
            "expected_exposure_count": expected_exposure_count,
            "attribution": attribution,
            "attribution_fingerprint": attribution_fingerprint,
        }
        provisional = cls.model_construct(fingerprint="0" * 64, **values)
        return cls(
            fingerprint=_content_fingerprint(provisional.identity_material(), cls.__name__),
            **values,
        )


def _comparability_mismatches(
    baseline: MemoryInterventionTrialBinding,
    intervention: MemoryInterventionTrialBinding,
    required_attribution_statuses: tuple[MemoryAttributionStatus, ...],
) -> tuple[MemoryInterventionMismatchReason, ...]:
    if baseline.receipt.status in {
        MemoryInterventionEffectStatus.INDETERMINATE,
        MemoryInterventionEffectStatus.CONFLICTING,
    } or intervention.receipt.status in {
        MemoryInterventionEffectStatus.INDETERMINATE,
        MemoryInterventionEffectStatus.CONFLICTING,
    }:
        raise ValueError("Memory comparability requires determinate intervention effects.")
    reasons: set[MemoryInterventionMismatchReason] = set()
    baseline_spec = baseline.spec
    intervention_spec = intervention.spec
    if (
        baseline_spec.kind is not MemoryInterventionKind.AS_DECLARED
        or baseline_spec.snapshot_fingerprint != intervention_spec.snapshot_fingerprint
        or baseline_spec.memory_state_fingerprint != intervention_spec.memory_state_fingerprint
        or baseline.receipt.starting_memory_state_fingerprint
        != intervention.receipt.starting_memory_state_fingerprint
    ):
        reasons.add(MemoryInterventionMismatchReason.BASELINE_MEMORY_STATE)
    if (
        baseline_spec.fingerprint == intervention_spec.fingerprint
        or intervention_spec.kind is MemoryInterventionKind.AS_DECLARED
        or baseline.operation.candidate_id == intervention.operation.candidate_id
    ):
        reasons.add(MemoryInterventionMismatchReason.INTERVENTION_IDENTITY)
    if (
        baseline_spec.recall_policy_ref_fingerprint
        != intervention_spec.recall_policy_ref_fingerprint
        or baseline_spec.starting_recall_policy_fingerprint
        != intervention_spec.starting_recall_policy_fingerprint
    ):
        reasons.add(MemoryInterventionMismatchReason.RECALL_POLICY)
    if (
        baseline.operation.materialization_fingerprint
        == intervention.operation.materialization_fingerprint
        or baseline.operation.memory_overlay_fingerprint
        == intervention.operation.memory_overlay_fingerprint
        or baseline.operation.state_scope_id == intervention.operation.state_scope_id
    ):
        reasons.add(MemoryInterventionMismatchReason.MATERIALIZATION_OVERLAY)
    declared_revisions = {
        change.source_item.revision_fingerprint
        for change in intervention_spec.changes
        if change.source_item is not None
    }
    if set(intervention.receipt.changed_item_revision_fingerprints) != declared_revisions:
        reasons.add(MemoryInterventionMismatchReason.CHANGED_ITEM_REVISIONS)
    if baseline_spec.trial_state_mode is not intervention_spec.trial_state_mode:
        reasons.add(MemoryInterventionMismatchReason.TRIAL_MODE)
    if (
        baseline.attribution.status not in required_attribution_statuses
        or intervention.attribution.status not in required_attribution_statuses
        or not baseline.terminal_evidence_available
        or not intervention.terminal_evidence_available
        or baseline.expected_receipt_count is None
        or baseline.expected_exposure_count is None
        or intervention.expected_receipt_count is None
        or intervention.expected_exposure_count is None
        or baseline.attribution.observed_receipt_count < baseline.expected_receipt_count
        or baseline.attribution.observed_exposure_count < baseline.expected_exposure_count
        or intervention.attribution.observed_receipt_count < intervention.expected_receipt_count
        or intervention.attribution.observed_exposure_count < intervention.expected_exposure_count
    ):
        reasons.add(MemoryInterventionMismatchReason.REQUIRED_ATTRIBUTION_AVAILABILITY)
    return tuple(sorted(reasons, key=str))


class MemoryInterventionComparability(_FingerprintRecord):
    """Memory-specific comparability only; generic experiment dimensions stay external."""

    record_type: Literal["cayu.memory-intervention-comparability"] = (
        "cayu.memory-intervention-comparability"
    )
    schema_version: Literal[1] = MEMORY_INTERVENTION_SCHEMA_VERSION
    baseline: MemoryInterventionTrialBinding
    intervention: MemoryInterventionTrialBinding
    required_attribution_statuses: tuple[MemoryAttributionStatus, ...] = (
        MemoryAttributionStatus.COMPLETE,
    )
    status: MemoryInterventionComparabilityStatus
    mismatch_reasons: tuple[MemoryInterventionMismatchReason, ...]
    generic_experiment_comparability_required: Literal[True] = True
    evidence_only: Literal[True] = True

    @field_validator("baseline", "intervention", mode="before")
    @classmethod
    def copy_bindings(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionTrialBinding)

    @field_validator("required_attribution_statuses", mode="before")
    @classmethod
    def validate_required_statuses(cls, value: object) -> tuple[MemoryAttributionStatus, ...]:
        items = _ordered_input(value, "required_attribution_statuses")
        statuses = tuple(MemoryAttributionStatus(item) for item in items)
        if not statuses or statuses != tuple(sorted(set(statuses), key=str)):
            raise ValueError("Required attribution statuses must be nonempty, unique, and sorted.")
        return statuses

    @field_validator("mismatch_reasons", mode="before")
    @classmethod
    def validate_mismatch_reasons(
        cls, value: object
    ) -> tuple[MemoryInterventionMismatchReason, ...]:
        items = _ordered_input(value, "mismatch_reasons")
        reasons = tuple(MemoryInterventionMismatchReason(item) for item in items)
        if reasons != tuple(sorted(set(reasons), key=str)):
            raise ValueError("Mismatch reasons must be unique and sorted.")
        return reasons

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        expected = _comparability_mismatches(
            self.baseline,
            self.intervention,
            self.required_attribution_statuses,
        )
        if self.mismatch_reasons != expected:
            raise ValueError("Memory comparability reasons do not match the supplied evidence.")
        expected_status = (
            MemoryInterventionComparabilityStatus.COMPARABLE
            if not expected
            else MemoryInterventionComparabilityStatus.INCOMPARABLE
        )
        if self.status is not expected_status:
            raise ValueError("Memory comparability status does not match its mismatch reasons.")
        return self

    @classmethod
    def create(
        cls,
        *,
        baseline: MemoryInterventionTrialBinding,
        intervention: MemoryInterventionTrialBinding,
        required_attribution_statuses: tuple[MemoryAttributionStatus, ...] = (
            MemoryAttributionStatus.COMPLETE,
        ),
    ) -> MemoryInterventionComparability:
        if len(required_attribution_statuses) != len(set(required_attribution_statuses)):
            raise ValueError("Required attribution statuses must be unique.")
        statuses = tuple(sorted(required_attribution_statuses, key=str))
        reasons = _comparability_mismatches(baseline, intervention, statuses)
        values: dict[str, Any] = {
            "baseline": baseline,
            "intervention": intervention,
            "required_attribution_statuses": statuses,
            "status": (
                MemoryInterventionComparabilityStatus.COMPARABLE
                if not reasons
                else MemoryInterventionComparabilityStatus.INCOMPARABLE
            ),
            "mismatch_reasons": reasons,
        }
        provisional = cls.model_construct(fingerprint="0" * 64, **values)
        return cls(
            fingerprint=_content_fingerprint(provisional.identity_material(), cls.__name__),
            **values,
        )


MemoryInterventionRecord: TypeAlias = (
    MemoryInterventionSpec
    | MemoryInterventionOperation
    | MemoryInterventionReceipt
    | MemoryInterventionTrialBinding
    | MemoryInterventionComparability
)

_RECORD_TYPES: dict[str, type[_FingerprintRecord]] = {
    "cayu.memory-intervention-spec": MemoryInterventionSpec,
    "cayu.memory-intervention-operation": MemoryInterventionOperation,
    "cayu.memory-intervention-receipt": MemoryInterventionReceipt,
    "cayu.memory-intervention-trial": MemoryInterventionTrialBinding,
    "cayu.memory-intervention-comparability": MemoryInterventionComparability,
}


def memory_intervention_to_json(record: MemoryInterventionRecord) -> str:
    """Return deterministic compact JSON for one exact intervention record."""

    if type(record) not in _RECORD_TYPES.values():
        raise TypeError("record must be an exact MemoryInterventionRecord.")
    model_type = _RECORD_TYPES.get(record.record_type)
    if model_type is None or type(record) is not model_type:
        raise TypeError("record must be an exact MemoryInterventionRecord.")
    validated = model_type.model_validate(record.model_dump(mode="python"))
    document = validated.model_dump(mode="json")
    encoded = canonical_durable_json_bytes(document, "memory intervention record")
    if len(encoded) > MEMORY_INTERVENTION_MAX_BYTES:
        raise ValueError("Memory intervention record exceeds its portable byte limit.")
    return encoded.decode("utf-8")


def memory_intervention_from_json(source: str | bytes) -> MemoryInterventionRecord:
    """Parse one bounded record and fail closed on unknown type or schema version."""

    if not isinstance(source, str | bytes):
        raise TypeError("source must be JSON text or bytes.")
    if isinstance(source, str):
        try:
            raw = source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Memory intervention JSON must contain valid Unicode.") from exc
    else:
        raw = source
    if len(raw) > MEMORY_INTERVENTION_MAX_BYTES:
        raise ValueError("Memory intervention record exceeds its portable byte limit.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Memory intervention JSON must be valid UTF-8.") from exc
    try:
        decoded = json.loads(
            text,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="memory intervention JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="memory intervention JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="memory intervention JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError("Memory intervention JSON nesting exceeds the supported depth.") from exc
    document = copy_durable_json_object(decoded, "memory intervention JSON")
    record_type = document.get("record_type")
    model_type = _RECORD_TYPES.get(record_type) if type(record_type) is str else None
    if model_type is None:
        raise ValueError(f"Unsupported memory intervention record_type {record_type!r}.")
    version = document.get("schema_version")
    if type(version) is not int or version != MEMORY_INTERVENTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported memory intervention schema_version {version!r}; "
            f"this Cayu version supports only {MEMORY_INTERVENTION_SCHEMA_VERSION}."
        )
    return cast("MemoryInterventionRecord", model_type.model_validate(document))


__all__ = [
    "MEMORY_INTERVENTION_MAX_BYTES",
    "MEMORY_INTERVENTION_MAX_CHANGED_ITEMS",
    "MEMORY_INTERVENTION_MAX_EFFECT_RECEIPTS",
    "MEMORY_INTERVENTION_MAX_FIXTURE_BYTES",
    "MEMORY_INTERVENTION_SCHEMA_VERSION",
    "MemoryInterventionBounds",
    "MemoryInterventionChangeKind",
    "MemoryInterventionComparability",
    "MemoryInterventionComparabilityStatus",
    "MemoryInterventionEffectReceiptRef",
    "MemoryInterventionEffectStatus",
    "MemoryInterventionFixtureRef",
    "MemoryInterventionItemChange",
    "MemoryInterventionItemIdentity",
    "MemoryInterventionItemIdentityKind",
    "MemoryInterventionKind",
    "MemoryInterventionMismatchReason",
    "MemoryInterventionOperation",
    "MemoryInterventionReceipt",
    "MemoryInterventionRecord",
    "MemoryInterventionSpec",
    "MemoryInterventionTrialBinding",
    "MemoryNegativeControlKind",
    "memory_attribution_fingerprint",
    "memory_intervention_from_json",
    "memory_intervention_to_json",
]
