"""Bounded durable values for ordinary-model routing, not dispatch credentials.

Only the stage transaction may advance these records. Parsing a record or matching
its digest proves structure/content, never caller authority to dispatch a model.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Any, Literal, Self, cast

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
    require_clean_nonblank,
    require_durable_clean_nonblank,
)

MODEL_FAILOVER_CHECKPOINT_KEY = "model_failover"
_Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
_Counter = Annotated[StrictInt, Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)]


class ModelTarget(BaseModel):
    """An application-selected provider and model pair for one session epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider_name: str
    model: str

    @field_validator("provider_name", "model")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


def _copy_failover_model_target(value: object) -> ModelTarget:
    """Validate raw target fields before serializers can render rejected values."""

    provider_name: object
    model: object
    if type(value) is ModelTarget:
        provider_name, model = value.provider_name, value.model
    elif type(value) is dict:
        raw_fields = cast("dict[object, object]", value)
        if any(type(key) is not str for key in raw_fields) or raw_fields.keys() != {
            "provider_name",
            "model",
        }:
            raise ValueError("Failover targets must contain exactly provider_name and model.")
        provider_name, model = raw_fields["provider_name"], raw_fields["model"]
    else:
        raise ValueError("Failover targets must contain exactly provider_name and model.")
    fields = {}
    for name, raw in (("provider_name", provider_name), ("model", model)):
        if type(raw) is not str:
            raise ValueError("Failover target identities must be strings.")
        if len(raw) > 256:
            raise ValueError("Failover target identities cannot exceed 256 UTF-8 bytes.")
        cleaned = require_durable_clean_nonblank(raw, f"failover target {name}")
        if len(cleaned.encode("utf-8")) > 256:
            raise ValueError("Failover target identities cannot exceed 256 UTF-8 bytes.")
        fields[name] = cleaned
    return ModelTarget(**fields)


class ModelFailoverPolicy(BaseModel):
    """Bounded, explicit alternatives after an eligible ordinary model failure.

    The primary target is resolved by the run entrance. ``max_total_attempts``
    includes that primary's attempts and all same-provider retries, not just
    transitions between targets. This value never authorizes dispatch itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    fallbacks: tuple[ModelTarget, ...] = Field(min_length=1, max_length=7)
    max_total_attempts: StrictInt = Field(default=20, ge=1, le=80)

    @field_validator("fallbacks", mode="before")
    @classmethod
    def copy_fallbacks(cls, value: object) -> tuple[ModelTarget, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("Failover requires between one and seven fallback targets.")
        items = cast("list[object] | tuple[object, ...]", value)
        if not 1 <= len(items) <= 7:
            raise ValueError("Failover requires between one and seven fallback targets.")
        targets = tuple(_copy_failover_model_target(item) for item in items)
        identities = tuple((target.provider_name, target.model) for target in targets)
        if len(set(identities)) != len(identities):
            raise ValueError("Failover targets must be distinct.")
        return targets

    def resolve_targets(self, primary: ModelTarget) -> tuple[ModelTarget, ...]:
        """Copy and validate the ordered plan, including its resolved primary."""

        copied = copy_model_failover_policy(self)
        target = _copy_failover_model_target(primary)
        if target in copied.fallbacks:
            raise ValueError("A fallback must not repeat the primary target.")
        return (target, *copied.fallbacks)


def copy_model_failover_policy(policy: ModelFailoverPolicy) -> ModelFailoverPolicy:
    """Revalidate even frozen values: model_construct/model_copy can bypass validation."""

    if type(policy) is not ModelFailoverPolicy:
        raise TypeError("failover must be an exact ModelFailoverPolicy.")
    return ModelFailoverPolicy(
        fallbacks=policy.fallbacks,
        max_total_attempts=policy.max_total_attempts,
    )


def copy_optional_model_failover_policy(value: object) -> ModelFailoverPolicy | None:
    """Shared ingress for request and reconstructed run-setting representations."""

    if value is None:
        return None
    if type(value) is ModelFailoverPolicy:
        return copy_model_failover_policy(value)
    if type(value) is not dict:
        raise ValueError("failover requires a ModelFailoverPolicy or object.")
    return ModelFailoverPolicy.model_validate(value)


class _FailoverValue(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, revalidate_instances="always"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_raw(cls, value: object) -> object:
        # Never call model_dump on an unvalidated nested model. A mutated field
        # can otherwise escape through Pydantic serialization warnings.
        if type(value) is cls:
            value = {name: getattr(value, name) for name in cls.model_fields}
        if type(value) is not dict:
            raise ValueError("Failover records require an exact object.")
        return value

    def payload(self) -> dict[str, Any]:
        # Revalidate bypassed frozen values before serialization at every handoff.
        copied = type(self).model_validate(self)
        return copy_durable_json_object(copied.model_dump(mode="json"), "model failover")


class ModelFailoverCandidate(_FailoverValue):
    provider_name: str = Field(strict=True, max_length=256)
    model: str = Field(strict=True, max_length=256)
    execution_profile_fingerprint: _Digest
    execution_mode: Literal["synchronous", "background"]

    @field_validator("execution_mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> object:
        if type(value) is not str or value not in {"synchronous", "background"}:
            raise ValueError("Unsupported failover execution mode.")
        return value

    @field_validator("provider_name", "model")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "failover target")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("Failover target exceeds 256 UTF-8 bytes.")
        return value


class ModelFailoverPlan(_FailoverValue):
    """Resolved candidate identities, bound into the configured root profile."""

    schema_version: Literal[1] = 1
    candidates: tuple[ModelFailoverCandidate, ...] = Field(min_length=2, max_length=8)
    max_total_attempts: StrictInt = Field(ge=1, le=80)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported model failover schema.")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value: object) -> tuple[ModelFailoverCandidate, ...]:
        if type(value) not in (list, tuple):
            raise ValueError("A resolved failover plan requires two to eight targets.")
        items = cast("list[object] | tuple[object, ...]", value)
        if not 2 <= len(items) <= 8:
            raise ValueError("A resolved failover plan requires two to eight targets.")
        copied = tuple(ModelFailoverCandidate.model_validate(candidate) for candidate in items)
        identities = {(candidate.provider_name, candidate.model) for candidate in copied}
        if len(identities) != len(copied):
            raise ValueError("Resolved failover targets must be distinct.")
        if len({candidate.execution_mode for candidate in copied}) != 1:
            raise ValueError("Failover candidates must preserve the primary execution mode.")
        return copied

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(self.payload(), "model failover plan")
        ).hexdigest()


class ModelFailoverSelection(_FailoverValue):
    """Durable selection before the first preparation of a new route.

    Fork/admission transactions own creation. This record has no attempt or
    dispatch authority; only exact model-stage preparation can consume it.
    """

    state: Literal["selected"] = "selected"
    schema_version: Literal[1] = 1
    session_id: str = Field(strict=True, max_length=256)
    session_instance_id: str = Field(strict=True, max_length=256)
    execution_profile_fingerprint: _Digest
    plan: ModelFailoverPlan
    candidate_index: StrictInt = Field(ge=0, le=7)
    origin_id: _Digest
    source_run_epoch: _Counter
    source_transcript_cursor: _Counter
    projection_cursor: _Counter

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        return ModelFailoverPlan.validate_version(value)

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> ModelFailoverPlan:
        return ModelFailoverPlan.model_validate(value)

    @field_validator("session_id", "session_instance_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "model failover identity")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("Model failover identity exceeds 256 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.candidate_index >= len(self.plan.candidates):
            raise ValueError("Selected target is outside the configured plan.")
        if self.projection_cursor > self.source_transcript_cursor:
            raise ValueError("Failover projection cannot exceed its source transcript.")
        return self


class ModelFailoverProgress(_FailoverValue):
    """One exact last preparation; retained across terminalization and restart.

    Local/global attempt counts are per logical ordinary model step. Generation
    never resets within this route, including successful steps and run-epoch
    changes. The configured profile and plan never change within a route.
    """

    schema_version: Literal[1] = 1
    session_id: str = Field(strict=True, max_length=256)
    session_instance_id: str = Field(strict=True, max_length=256)
    interaction_id: str = Field(strict=True, max_length=256)
    execution_profile_fingerprint: _Digest
    plan: ModelFailoverPlan
    generation: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    candidate_index: StrictInt = Field(ge=0, le=7)
    logical_step_id: str = Field(strict=True, max_length=256)
    stage_id: str = Field(strict=True, max_length=256)
    request_fingerprint: _Digest
    source_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    source_transcript_cursor: _Counter
    projection_cursor: _Counter
    dispatch_ordinal: _Counter
    attempts_used: StrictInt = Field(ge=1, le=80)
    candidate_attempt: StrictInt = Field(ge=1, le=80)
    provider_effect_observed: StrictBool

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        return ModelFailoverPlan.validate_version(value)

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> ModelFailoverPlan:
        return ModelFailoverPlan.model_validate(value)

    @field_validator(
        "session_id", "session_instance_id", "interaction_id", "logical_step_id", "stage_id"
    )
    @classmethod
    def validate_identity(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "model failover identity")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("Model failover identity exceeds 256 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.candidate_index >= len(self.plan.candidates):
            raise ValueError("Selected target is outside the configured plan.")
        if not self.candidate_attempt <= self.attempts_used <= self.plan.max_total_attempts:
            raise ValueError("Failover attempts exceed the configured plan.")
        if self.projection_cursor > self.source_transcript_cursor:
            raise ValueError("Failover projection cannot exceed its source transcript.")
        return self

    @property
    def route_id(self) -> str:
        value = self.payload()
        identity = {
            name: value[name]
            for name in (
                "session_id",
                "session_instance_id",
                "interaction_id",
                "execution_profile_fingerprint",
                "plan",
            )
        }
        return sha256(canonical_durable_json_bytes(identity, "model failover identity")).hexdigest()


def copy_model_failover_state(
    value: object,
) -> ModelFailoverSelection | ModelFailoverProgress:
    """Validate either complete record shape; no missing-field selection inference."""

    if type(value) is ModelFailoverSelection:
        return ModelFailoverSelection.model_validate(value)
    if type(value) is dict:
        state = cast("dict[object, object]", value).get("state")
        if type(state) is str and state == "selected":
            return ModelFailoverSelection.model_validate(value)
    return ModelFailoverProgress.model_validate(value)


def validate_model_failover_successor(
    previous: ModelFailoverProgress,
    successor: ModelFailoverProgress,
    *,
    transition: Literal["retry", "fallback", "next_step", "reprepare"],
) -> None:
    """Validate exact progression; the caller must additionally prove its cause.

    A fallback requires authenticated live failure evidence; next_step requires
    a published predecessor, or exact terminal/abandonment evidence when starting
    a distinct admitted interaction. None is inferred from these values.
    Store comparisons must compare the entire expected previous record before
    using this structural check, including on read-only replay/readback.
    """

    previous = ModelFailoverProgress.model_validate(previous)
    successor = ModelFailoverProgress.model_validate(successor)
    # A new interaction may inherit the selected target only as a new logical
    # step. The stage transaction must additionally prove the predecessor's
    # publication and the new admitted interaction; terminal/abandonment
    # dispositions additionally require a later epoch. Queue delivery can bind a
    # new interaction in the same epoch, but still needs published completion.
    # Keep interaction in route_id
    # and in the exact CAS tuple; retries/re-preparation cannot cross it.
    continues_selected_target = (
        transition == "next_step"
        and successor.interaction_id != previous.interaction_id
        and successor.source_run_epoch >= previous.source_run_epoch
        and successor.session_id == previous.session_id
        and successor.session_instance_id == previous.session_instance_id
        and successor.execution_profile_fingerprint == previous.execution_profile_fingerprint
        and successor.plan == previous.plan
    )
    if previous.route_id != successor.route_id and not continues_selected_target:
        raise ValueError("Model failover route authority changed.")
    if (
        successor.generation != previous.generation + 1
        or successor.stage_id == previous.stage_id
        or successor.source_run_epoch < previous.source_run_epoch
        or successor.source_transcript_cursor < previous.source_transcript_cursor
        or successor.projection_cursor < previous.projection_cursor
    ):
        raise ValueError("Model failover successor has stale preparation identity.")
    if transition == "reprepare":
        valid = (
            successor.logical_step_id == previous.logical_step_id
            and successor.candidate_index == previous.candidate_index
            and successor.dispatch_ordinal == previous.dispatch_ordinal + 1
            and successor.attempts_used == previous.attempts_used
            and successor.candidate_attempt == previous.candidate_attempt
            and successor.source_transcript_cursor == previous.source_transcript_cursor
            and successor.projection_cursor == previous.projection_cursor
            and successor.provider_effect_observed == previous.provider_effect_observed
        )
    elif transition == "next_step":
        valid = (
            successor.logical_step_id != previous.logical_step_id
            and successor.candidate_index == previous.candidate_index
            and successor.attempts_used == successor.candidate_attempt == 1
            and successor.projection_cursor == previous.projection_cursor
            and successor.provider_effect_observed is False
        )
    elif transition in {"retry", "fallback"}:
        valid = (
            successor.logical_step_id == previous.logical_step_id
            and successor.dispatch_ordinal == previous.dispatch_ordinal + 1
            and successor.attempts_used == previous.attempts_used + 1
            and successor.source_transcript_cursor == previous.source_transcript_cursor
            and (not previous.provider_effect_observed or successor.provider_effect_observed)
            and (
                (
                    transition == "retry"
                    and successor.candidate_index == previous.candidate_index
                    and successor.candidate_attempt == previous.candidate_attempt + 1
                    and successor.projection_cursor == previous.projection_cursor
                )
                or (
                    transition == "fallback"
                    and previous.plan.candidates[previous.candidate_index].execution_mode
                    == "synchronous"
                    and successor.candidate_index == previous.candidate_index + 1
                    and successor.candidate_attempt == 1
                    and successor.projection_cursor == successor.source_transcript_cursor
                    and successor.provider_effect_observed is False
                )
            )
        )
    else:
        raise ValueError("Unknown model failover transition.")
    if not valid:
        raise ValueError("Model failover successor conflicts with its transition.")
