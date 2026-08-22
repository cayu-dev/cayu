from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    FrozenJsonDict,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_finite,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.recall import (
    RecallCandidate,
    RecallEngine,
    RecallResult,
    RecallSituation,
    RecallSourceDiagnostic,
)
from cayu.retrieval import FusedChannelMatch, RetrievalCandidateIdentity

AUTOMATIC_RECALL_POLICY_VERSION = "cayu.automatic_recall_policy.v1"
AUTOMATIC_RECALL_CONTRIBUTION_VERSION = "cayu.automatic_recall_contribution.v1"
MEMORY_FOCUS_VERSION = "cayu.memory_focus.v1"
RECALL_OFFER_VERSION = "cayu.recall_offer.v1"

_MAX_ADMISSION_CANDIDATES = 100
_MAX_ADMISSION_ITEMS = 50
_MAX_ADMISSION_BYTES = 1_000_000
_MAX_CALIBRATION_NAME_BYTES = 256
_MAX_CANDIDATE_TEXT_BYTES = 128_000

_RecallOfferReason: TypeAlias = Literal[
    "calibrated_plausible_match",
    "duplicate_strong_reference",
    "strong_match_not_focused",
    "strong_match_offered_by_mode",
]


class AutomaticRecallMode(StrEnum):
    OFF = "off"
    OFFER = "offer"
    STRONG_MATCHES = "strong_matches"
    OFFER_AND_STRONG_MATCHES = "offer_and_strong_matches"

    @property
    def injects_strong_matches(self) -> bool:
        return self in {
            AutomaticRecallMode.STRONG_MATCHES,
            AutomaticRecallMode.OFFER_AND_STRONG_MATCHES,
        }

    @property
    def emits_offers(self) -> bool:
        return self in {
            AutomaticRecallMode.OFFER,
            AutomaticRecallMode.OFFER_AND_STRONG_MATCHES,
        }


class AutomaticRecallPolicy(BaseModel):
    """Versioned deterministic admission configuration for one fusion calibration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    policy_version: Literal["cayu.automatic_recall_policy.v1"] = AUTOMATIC_RECALL_POLICY_VERSION
    calibration_version: str
    fusion_strategy_version: str
    fusion_configuration_version: str
    mode: AutomaticRecallMode = AutomaticRecallMode.OFFER_AND_STRONG_MATCHES
    minimum_inject_score: float
    minimum_offer_score: float
    max_evaluated_candidates: int = 30
    max_injected_items: int = 5
    max_offered_items: int = 10
    max_candidate_text_bytes: int = 16_000
    max_focus_bytes: int = 64_000
    max_offer_bytes: int = 16_000
    max_total_bytes: int = 96_000

    @field_validator(
        "calibration_version",
        "fusion_strategy_version",
        "fusion_configuration_version",
    )
    @classmethod
    def validate_version_name(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > _MAX_CALIBRATION_NAME_BYTES:
            raise ValueError(
                f"`{info.field_name}` must be at most {_MAX_CALIBRATION_NAME_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("minimum_inject_score", "minimum_offer_score", mode="before")
    @classmethod
    def validate_score(cls, value, info) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        return require_finite(float(value), info.field_name)

    @field_validator(
        "max_evaluated_candidates",
        "max_injected_items",
        "max_offered_items",
        mode="before",
    )
    @classmethod
    def validate_count_bound(cls, value, info) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"`{info.field_name}` must be a positive integer.")
        maximum = (
            _MAX_ADMISSION_CANDIDATES
            if info.field_name == "max_evaluated_candidates"
            else _MAX_ADMISSION_ITEMS
        )
        if value > maximum:
            raise ValueError(f"`{info.field_name}` must be at most {maximum}.")
        return value

    @field_validator(
        "max_candidate_text_bytes",
        "max_focus_bytes",
        "max_offer_bytes",
        "max_total_bytes",
        mode="before",
    )
    @classmethod
    def validate_byte_bound(cls, value, info) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"`{info.field_name}` must be a positive integer.")
        maximum = (
            _MAX_CANDIDATE_TEXT_BYTES
            if info.field_name == "max_candidate_text_bytes"
            else _MAX_ADMISSION_BYTES
        )
        if value > maximum:
            raise ValueError(f"`{info.field_name}` must be at most {maximum}.")
        return value

    @model_validator(mode="after")
    def validate_policy(self) -> AutomaticRecallPolicy:
        if self.minimum_offer_score > self.minimum_inject_score:
            raise ValueError("`minimum_offer_score` cannot exceed `minimum_inject_score`.")
        if self.max_injected_items > self.max_evaluated_candidates:
            raise ValueError("`max_injected_items` cannot exceed `max_evaluated_candidates`.")
        if self.max_offered_items > self.max_evaluated_candidates:
            raise ValueError("`max_offered_items` cannot exceed `max_evaluated_candidates`.")
        if self.max_focus_bytes > self.max_total_bytes:
            raise ValueError("`max_focus_bytes` cannot exceed `max_total_bytes`.")
        if self.max_offer_bytes > self.max_total_bytes:
            raise ValueError("`max_offer_bytes` cannot exceed `max_total_bytes`.")
        canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "automatic recall policy",
        )
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        payload = self.model_dump(mode="python", round_trip=True)
        if update is not None:
            payload.update(update)
        copied = type(self).model_validate(payload)
        fields_set = set(self.model_fields_set)
        if update is not None:
            fields_set.update(update)
        object.__setattr__(copied, "__pydantic_fields_set__", fields_set)
        return copied

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "automatic recall policy",
            )
        ).hexdigest()


class MemoryFocusItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    candidate: RecallCandidate
    fused_rank: int
    selection_reason: Literal["calibrated_strong_match"] = "calibrated_strong_match"

    @field_validator("candidate", mode="before")
    @classmethod
    def copy_candidate(cls, value) -> RecallCandidate:
        return RecallCandidate.model_validate(
            value.model_dump(mode="python") if type(value) is RecallCandidate else value
        )

    @field_validator("fused_rank", mode="before")
    @classmethod
    def validate_rank(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("`fused_rank` must be a positive integer.")
        return value


class MemoryFocus(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    schema_version: Literal["cayu.memory_focus.v1"] = MEMORY_FOCUS_VERSION
    situation_sha256: str
    policy_sha256: str
    calibration_version: str
    items: tuple[MemoryFocusItem, ...]
    sources: tuple[RecallSourceDiagnostic, ...]
    continuations: Mapping[str, str] = Field(default_factory=dict)
    eligible_item_count: int
    omitted_item_count: int
    recall_truncated: bool
    truncated: bool

    @field_validator("situation_sha256", "policy_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("calibration_version")
    @classmethod
    def validate_calibration_version(cls, value: str) -> str:
        return require_clean_nonblank(value, "calibration_version")

    @field_validator("items", mode="before")
    @classmethod
    def copy_items(cls, value) -> tuple[MemoryFocusItem, ...]:
        return tuple(
            MemoryFocusItem.model_validate(
                item.model_dump(mode="python") if type(item) is MemoryFocusItem else item
            )
            for item in value
        )

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value) -> tuple[RecallSourceDiagnostic, ...]:
        return tuple(
            RecallSourceDiagnostic.model_validate(
                item.model_dump(mode="python") if type(item) is RecallSourceDiagnostic else item
            )
            for item in value
        )

    @field_validator("continuations", mode="before")
    @classmethod
    def copy_continuations(cls, value) -> dict[str, str]:
        copied = copy_durable_json_object(value, "memory focus continuations")
        if any(
            type(channel) is not str or type(cursor) is not str
            for channel, cursor in copied.items()
        ):
            raise ValueError("Memory focus continuation channels and cursors must be strings.")
        return {channel: copied[channel] for channel in sorted(copied)}

    @field_validator("continuations")
    @classmethod
    def freeze_continuations(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenJsonDict(value)

    @field_serializer("continuations")
    def serialize_continuations(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("eligible_item_count", "omitted_item_count", mode="before")
    @classmethod
    def validate_count(cls, value, info) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"`{info.field_name}` must be a non-negative integer.")
        return value

    @field_validator("recall_truncated", "truncated", mode="before")
    @classmethod
    def validate_bool(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_focus(self) -> MemoryFocus:
        if not self.items:
            raise ValueError("Memory focus must contain at least one item.")
        ranks = [item.fused_rank for item in self.items]
        identities = [item.candidate.record.identity.sort_key() for item in self.items]
        if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
            raise ValueError("Memory focus items must retain unique fused-rank order.")
        if len(identities) != len(set(identities)):
            raise ValueError("Memory focus cannot repeat a candidate identity.")
        if self.eligible_item_count != len(self.items) + self.omitted_item_count:
            raise ValueError("Memory focus item counts are inconsistent.")
        if self.truncated != (self.recall_truncated or self.omitted_item_count > 0):
            raise ValueError("Memory focus truncation is inconsistent.")
        return self


class RecallOfferItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: RetrievalCandidateIdentity
    representation: str
    content_hash: str
    locator: Mapping[str, Any]
    fused_rank: int
    score: float
    matches: tuple[FusedChannelMatch, ...]
    reason: _RecallOfferReason

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value) -> RetrievalCandidateIdentity:
        return RetrievalCandidateIdentity.model_validate(
            value.model_dump(mode="python") if type(value) is RetrievalCandidateIdentity else value
        )

    @field_validator("representation")
    @classmethod
    def validate_representation(cls, value: str) -> str:
        return require_clean_nonblank(value, "representation")

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        value = require_clean_nonblank(value, "content_hash")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("`content_hash` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("locator", mode="before")
    @classmethod
    def copy_locator(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "recall offer locator")

    @field_validator("locator")
    @classmethod
    def freeze_locator(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return FrozenJsonDict(value)

    @field_serializer("locator")
    def serialize_locator(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("fused_rank", mode="before")
    @classmethod
    def validate_rank(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("`fused_rank` must be a positive integer.")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`score` must be a number.")
        return require_finite(float(value), "score")

    @field_validator("matches", mode="before")
    @classmethod
    def copy_matches(cls, value) -> tuple[FusedChannelMatch, ...]:
        matches = tuple(
            FusedChannelMatch.model_validate(
                match.model_dump(mode="python") if type(match) is FusedChannelMatch else match
            )
            for match in value
        )
        channels = [match.channel for match in matches]
        if not matches or len(channels) != len(set(channels)):
            raise ValueError("`matches` must contain unique channel provenance.")
        return tuple(sorted(matches, key=lambda match: match.channel))

    @model_validator(mode="after")
    def validate_match_evidence(self) -> RecallOfferItem:
        if any(match.content_hash != self.content_hash for match in self.matches):
            raise ValueError("Recall offer channel provenance conflicts with its content hash.")
        return self


class RecallOffer(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    schema_version: Literal["cayu.recall_offer.v1"] = RECALL_OFFER_VERSION
    ticket: str
    situation_sha256: str
    policy_sha256: str
    calibration_version: str
    items: tuple[RecallOfferItem, ...]
    continuations: Mapping[str, str] = Field(default_factory=dict)
    eligible_item_count: int
    omitted_item_count: int
    recall_truncated: bool
    truncated: bool

    @field_validator("ticket", "situation_sha256", "policy_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("calibration_version")
    @classmethod
    def validate_calibration_version(cls, value: str) -> str:
        return require_clean_nonblank(value, "calibration_version")

    @field_validator("items", mode="before")
    @classmethod
    def copy_items(cls, value) -> tuple[RecallOfferItem, ...]:
        return tuple(
            RecallOfferItem.model_validate(
                item.model_dump(mode="python") if type(item) is RecallOfferItem else item
            )
            for item in value
        )

    @field_validator("continuations", mode="before")
    @classmethod
    def copy_continuations(cls, value) -> dict[str, str]:
        copied = copy_durable_json_object(value, "recall offer continuations")
        if any(
            type(channel) is not str or type(cursor) is not str
            for channel, cursor in copied.items()
        ):
            raise ValueError("Recall offer continuation channels and cursors must be strings.")
        return {channel: copied[channel] for channel in sorted(copied)}

    @field_validator("continuations")
    @classmethod
    def freeze_continuations(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenJsonDict(value)

    @field_serializer("continuations")
    def serialize_continuations(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("eligible_item_count", "omitted_item_count", mode="before")
    @classmethod
    def validate_count(cls, value, info) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"`{info.field_name}` must be a non-negative integer.")
        return value

    @field_validator("recall_truncated", "truncated", mode="before")
    @classmethod
    def validate_bool(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_offer(self) -> RecallOffer:
        if not self.items:
            raise ValueError("Recall offer must contain at least one item.")
        ranks = [item.fused_rank for item in self.items]
        identities = [item.identity.sort_key() for item in self.items]
        if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
            raise ValueError("Recall offer items must retain unique fused-rank order.")
        if len(identities) != len(set(identities)):
            raise ValueError("Recall offer cannot repeat a candidate identity.")
        if self.eligible_item_count != len(self.items) + self.omitted_item_count:
            raise ValueError("Recall offer item counts are inconsistent.")
        if self.truncated != (self.recall_truncated or self.omitted_item_count > 0):
            raise ValueError("Recall offer truncation is inconsistent.")
        expected_ticket = _recall_offer_ticket(
            items=self.items,
            policy_sha256=self.policy_sha256,
            situation_sha256=self.situation_sha256,
            calibration_version=self.calibration_version,
            continuations=self.continuations,
            eligible_item_count=self.eligible_item_count,
            omitted_item_count=self.omitted_item_count,
            recall_truncated=self.recall_truncated,
            truncated=self.truncated,
        )
        if self.ticket != expected_ticket:
            raise ValueError("Recall offer ticket does not commit to its complete contents.")
        return self


class AutomaticRecallDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    recall_performed: bool
    recall_candidate_count: int
    evaluated_candidate_count: int
    strong_candidate_count: int
    plausible_candidate_count: int
    injected_count: int
    offered_count: int
    silent_count: int
    duplicate_content_omitted: int
    oversized_candidate_omitted: int
    focus_bound_omitted: int
    offer_bound_omitted: int
    unevaluated_count: int
    recall_truncated: bool
    admission_truncated: bool

    @field_validator("recall_performed", "recall_truncated", "admission_truncated", mode="before")
    @classmethod
    def validate_bool(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @field_validator(
        "recall_candidate_count",
        "evaluated_candidate_count",
        "strong_candidate_count",
        "plausible_candidate_count",
        "injected_count",
        "offered_count",
        "silent_count",
        "duplicate_content_omitted",
        "oversized_candidate_omitted",
        "focus_bound_omitted",
        "offer_bound_omitted",
        "unevaluated_count",
        mode="before",
    )
    @classmethod
    def validate_count(cls, value, info) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"`{info.field_name}` must be a non-negative integer.")
        return value


class AutomaticRecallContribution(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    schema_version: Literal["cayu.automatic_recall_contribution.v1"] = (
        AUTOMATIC_RECALL_CONTRIBUTION_VERSION
    )
    situation_sha256: str
    policy_sha256: str
    mode: AutomaticRecallMode
    focus: MemoryFocus | None = None
    offer: RecallOffer | None = None
    sources: tuple[RecallSourceDiagnostic, ...]
    continuations: Mapping[str, str] = Field(default_factory=dict)
    diagnostics: AutomaticRecallDiagnostics

    @field_validator("situation_sha256", "policy_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("focus", mode="before")
    @classmethod
    def copy_focus(cls, value) -> MemoryFocus | None:
        if value is None:
            return None
        return MemoryFocus.model_validate(
            value.model_dump(mode="python") if type(value) is MemoryFocus else value
        )

    @field_validator("offer", mode="before")
    @classmethod
    def copy_offer(cls, value) -> RecallOffer | None:
        if value is None:
            return None
        return RecallOffer.model_validate(
            value.model_dump(mode="python") if type(value) is RecallOffer else value
        )

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value) -> tuple[RecallSourceDiagnostic, ...]:
        return tuple(
            RecallSourceDiagnostic.model_validate(
                item.model_dump(mode="python") if type(item) is RecallSourceDiagnostic else item
            )
            for item in value
        )

    @field_validator("continuations", mode="before")
    @classmethod
    def copy_continuations(cls, value) -> dict[str, str]:
        copied = copy_durable_json_object(value, "automatic recall continuations")
        if any(
            type(channel) is not str or type(cursor) is not str
            for channel, cursor in copied.items()
        ):
            raise ValueError("Automatic recall continuation channels and cursors must be strings.")
        return {channel: copied[channel] for channel in sorted(copied)}

    @field_validator("continuations")
    @classmethod
    def freeze_continuations(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenJsonDict(value)

    @field_serializer("continuations")
    def serialize_continuations(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("diagnostics", mode="before")
    @classmethod
    def copy_diagnostics(cls, value) -> AutomaticRecallDiagnostics:
        return AutomaticRecallDiagnostics.model_validate(
            value.model_dump(mode="python") if type(value) is AutomaticRecallDiagnostics else value
        )

    @model_validator(mode="after")
    def validate_contribution(self) -> AutomaticRecallContribution:
        if self.mode is AutomaticRecallMode.OFF and (
            self.focus is not None or self.offer is not None
        ):
            raise ValueError("Disabled automatic recall cannot emit focus or offer output.")
        if not self.mode.injects_strong_matches and self.focus is not None:
            raise ValueError("The selected automatic recall mode does not emit memory focus.")
        if not self.mode.emits_offers and self.offer is not None:
            raise ValueError("The selected automatic recall mode does not emit recall offers.")
        source_names = [source.source for source in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Automatic recall cannot repeat source diagnostics.")
        if (self.mode is AutomaticRecallMode.OFF) != (not self.diagnostics.recall_performed):
            raise ValueError("Automatic recall mode and operation diagnostics conflict.")
        if not self.diagnostics.recall_performed and (
            self.sources
            or self.continuations
            or any(
                getattr(self.diagnostics, field_name)
                for field_name in (
                    "recall_candidate_count",
                    "evaluated_candidate_count",
                    "strong_candidate_count",
                    "plausible_candidate_count",
                    "injected_count",
                    "offered_count",
                    "silent_count",
                    "duplicate_content_omitted",
                    "oversized_candidate_omitted",
                    "focus_bound_omitted",
                    "offer_bound_omitted",
                    "unevaluated_count",
                    "recall_truncated",
                    "admission_truncated",
                )
            )
        ):
            raise ValueError("A disabled automatic recall contribution must be empty.")
        if self.diagnostics.injected_count != (0 if self.focus is None else len(self.focus.items)):
            raise ValueError("Automatic recall injected diagnostics conflict with focus output.")
        if self.diagnostics.offered_count != (0 if self.offer is None else len(self.offer.items)):
            raise ValueError("Automatic recall offer diagnostics conflict with offer output.")
        if (
            self.diagnostics.injected_count
            + self.diagnostics.offered_count
            + self.diagnostics.silent_count
            != self.diagnostics.recall_candidate_count
        ):
            raise ValueError("Automatic recall disposition counts are inconsistent.")
        if self.diagnostics.evaluated_candidate_count > self.diagnostics.recall_candidate_count:
            raise ValueError("Automatic recall evaluated more candidates than recall returned.")
        if self.diagnostics.unevaluated_count != (
            self.diagnostics.recall_candidate_count - self.diagnostics.evaluated_candidate_count
        ):
            raise ValueError("Automatic recall unevaluated diagnostics are inconsistent.")
        if (
            self.diagnostics.strong_candidate_count + self.diagnostics.plausible_candidate_count
            > self.diagnostics.evaluated_candidate_count
        ):
            raise ValueError("Automatic recall calibrated candidate counts are inconsistent.")
        if self.focus is not None and (
            self.focus.situation_sha256 != self.situation_sha256
            or self.focus.policy_sha256 != self.policy_sha256
            or self.focus.sources != self.sources
            or self.focus.continuations != self.continuations
            or self.focus.recall_truncated != self.diagnostics.recall_truncated
        ):
            raise ValueError("Memory focus provenance conflicts with its contribution.")
        if self.offer is not None and (
            self.offer.situation_sha256 != self.situation_sha256
            or self.offer.policy_sha256 != self.policy_sha256
            or self.offer.continuations != self.continuations
            or self.offer.recall_truncated != self.diagnostics.recall_truncated
        ):
            raise ValueError("Recall offer provenance conflicts with its contribution.")
        if (
            self.focus is not None
            and self.offer is not None
            and self.focus.calibration_version != self.offer.calibration_version
        ):
            raise ValueError("Memory focus and recall offer calibration versions conflict.")
        focused_identities = (
            set()
            if self.focus is None
            else {item.candidate.record.identity.sort_key() for item in self.focus.items}
        )
        offered_identities = (
            set() if self.offer is None else {item.identity.sort_key() for item in self.offer.items}
        )
        if focused_identities & offered_identities:
            raise ValueError("One candidate identity cannot be both focused and offered.")
        expected_admission_truncated = any(
            (
                self.diagnostics.duplicate_content_omitted,
                self.diagnostics.oversized_candidate_omitted,
                self.diagnostics.focus_bound_omitted,
                self.diagnostics.offer_bound_omitted,
                self.diagnostics.unevaluated_count,
            )
        )
        if self.diagnostics.admission_truncated != expected_admission_truncated:
            raise ValueError("Automatic recall admission truncation diagnostics are inconsistent.")
        return self


class AutomaticRecallContributor:
    """Run bounded recall and apply one immutable admission policy."""

    def __init__(self, engine: RecallEngine, policy: AutomaticRecallPolicy) -> None:
        if not isinstance(engine, RecallEngine):
            raise TypeError("engine must be a RecallEngine.")
        if type(policy) is not AutomaticRecallPolicy:
            raise TypeError("policy must be an AutomaticRecallPolicy.")
        self._engine = engine
        self._policy = AutomaticRecallPolicy.model_validate(policy.model_dump(mode="python"))

    @property
    def policy(self) -> AutomaticRecallPolicy:
        return AutomaticRecallPolicy.model_validate(self._policy.model_dump(mode="python"))

    async def contribute(self, situation: RecallSituation) -> AutomaticRecallContribution:
        if type(situation) is not RecallSituation:
            raise TypeError("situation must be a RecallSituation.")
        situation = RecallSituation.model_validate(situation.model_dump(mode="python"))
        if self._policy.mode is AutomaticRecallMode.OFF:
            return _disabled_contribution(situation, self._policy)
        result = await self._engine.recall(situation)
        return admit_recall(result, self._policy)


def admit_recall(
    result: RecallResult,
    policy: AutomaticRecallPolicy,
) -> AutomaticRecallContribution:
    """Apply deterministic bounded admission without provider-message coupling."""

    if type(result) is not RecallResult:
        raise TypeError("result must be a RecallResult.")
    if type(policy) is not AutomaticRecallPolicy:
        raise TypeError("policy must be an AutomaticRecallPolicy.")
    result = RecallResult.model_validate(result.model_dump(mode="python"))
    policy = AutomaticRecallPolicy.model_validate(policy.model_dump(mode="python"))
    if policy.mode is AutomaticRecallMode.OFF:
        return _disabled_contribution_from_hash(result.situation_sha256, policy)
    if result.fusion.strategy_version != policy.fusion_strategy_version:
        raise ValueError("Recall fusion strategy is not covered by this admission calibration.")
    if result.fusion.configuration_version != policy.fusion_configuration_version:
        raise ValueError(
            "Recall fusion configuration is not covered by this admission calibration."
        )

    evaluated = tuple(result.candidates[: policy.max_evaluated_candidates])
    unevaluated_count = len(result.candidates) - len(evaluated)
    strong: list[tuple[int, RecallCandidate]] = []
    plausible: list[tuple[int, RecallCandidate]] = []
    oversized_count = 0
    for fused_rank, candidate in enumerate(evaluated, start=1):
        if len(candidate.record.text.encode("utf-8")) > policy.max_candidate_text_bytes:
            oversized_count += 1
            continue
        if candidate.fused.score >= policy.minimum_inject_score:
            strong.append((fused_rank, candidate))
        elif candidate.fused.score >= policy.minimum_offer_score:
            plausible.append((fused_rank, candidate))

    policy_sha256 = policy.fingerprint()
    focus_items: list[MemoryFocusItem] = []
    duplicate_count = 0
    focus_bound_count = 0
    offer_pool: list[tuple[int, RecallCandidate, _RecallOfferReason]] = [
        (rank, candidate, "calibrated_plausible_match") for rank, candidate in plausible
    ]
    selected_content_hashes: set[str] = set()
    focus_selection_ranks: list[int] = []
    if policy.mode.injects_strong_matches:
        source_type_occurrences: dict[str, int] = {}
        diversity_order: list[tuple[int, int, RecallCandidate]] = []
        for fused_rank, candidate in strong:
            source_type = _candidate_source_type(candidate)
            occurrence = source_type_occurrences.get(source_type, 0)
            source_type_occurrences[source_type] = occurrence + 1
            diversity_order.append((occurrence, fused_rank, candidate))
        diversity_order.sort(key=lambda item: (item[0], item[1]))
        for _, fused_rank, candidate in diversity_order:
            if candidate.record.content_hash in selected_content_hashes:
                duplicate_count += 1
                offer_pool.append((fused_rank, candidate, "duplicate_strong_reference"))
                continue
            if len(focus_items) >= policy.max_injected_items:
                focus_bound_count += 1
                offer_pool.append((fused_rank, candidate, "strong_match_not_focused"))
                continue
            tentative_items = [
                *focus_items,
                MemoryFocusItem(candidate=candidate, fused_rank=fused_rank),
            ]
            tentative_items.sort(key=lambda item: item.fused_rank)
            tentative = _memory_focus(
                result=result,
                policy=policy,
                policy_sha256=policy_sha256,
                items=tentative_items,
                eligible_item_count=len(strong),
            )
            if _serialized_bytes(tentative, "memory focus") > policy.max_focus_bytes:
                focus_bound_count += 1
                offer_pool.append((fused_rank, candidate, "strong_match_not_focused"))
                continue
            focus_items = tentative_items
            focus_selection_ranks.append(fused_rank)
            selected_content_hashes.add(candidate.record.content_hash)
    else:
        offer_pool.extend(
            (rank, candidate, "strong_match_offered_by_mode") for rank, candidate in strong
        )

    offer_items: list[RecallOfferItem] = []
    offer_bound_count = 0
    offer_eligible = sorted(offer_pool, key=lambda item: item[0])
    if policy.mode.emits_offers:
        for fused_rank, candidate, reason in offer_eligible:
            if len(offer_items) >= policy.max_offered_items:
                offer_bound_count += 1
                continue
            tentative_items = [
                *offer_items,
                _recall_offer_item(candidate, fused_rank=fused_rank, reason=reason),
            ]
            tentative = _recall_offer(
                result=result,
                policy=policy,
                policy_sha256=policy_sha256,
                items=tentative_items,
                eligible_item_count=len(offer_eligible),
            )
            if _serialized_bytes(tentative, "recall offer") > policy.max_offer_bytes:
                offer_bound_count += 1
                continue
            offer_items = tentative_items

    def build_contribution() -> AutomaticRecallContribution:
        focus = (
            None
            if not focus_items
            else _memory_focus(
                result=result,
                policy=policy,
                policy_sha256=policy_sha256,
                items=focus_items,
                eligible_item_count=len(strong),
            )
        )
        offer = (
            None
            if not offer_items
            else _recall_offer(
                result=result,
                policy=policy,
                policy_sha256=policy_sha256,
                items=offer_items,
                eligible_item_count=len(offer_eligible),
            )
        )
        diagnostics = AutomaticRecallDiagnostics(
            recall_performed=True,
            recall_candidate_count=len(result.candidates),
            evaluated_candidate_count=len(evaluated),
            strong_candidate_count=len(strong),
            plausible_candidate_count=len(plausible),
            injected_count=len(focus_items),
            offered_count=len(offer_items),
            silent_count=len(result.candidates) - len(focus_items) - len(offer_items),
            duplicate_content_omitted=duplicate_count,
            oversized_candidate_omitted=oversized_count,
            focus_bound_omitted=focus_bound_count,
            offer_bound_omitted=offer_bound_count,
            unevaluated_count=unevaluated_count,
            recall_truncated=result.truncated,
            admission_truncated=bool(
                oversized_count
                or duplicate_count
                or focus_bound_count
                or offer_bound_count
                or unevaluated_count
            ),
        )
        return AutomaticRecallContribution(
            situation_sha256=result.situation_sha256,
            policy_sha256=policy_sha256,
            mode=policy.mode,
            focus=focus,
            offer=offer,
            sources=result.sources,
            continuations=result.continuations,
            diagnostics=diagnostics,
        )

    contribution = build_contribution()
    while _serialized_bytes(contribution, "automatic recall contribution") > (
        policy.max_total_bytes
    ):
        if offer_items:
            offer_items.pop()
            offer_bound_count += 1
        elif focus_items:
            removed_rank = focus_selection_ranks.pop()
            focus_items = [item for item in focus_items if item.fused_rank != removed_rank]
            focus_bound_count += 1
        else:
            raise ValueError(
                "Automatic recall metadata alone exceeds the configured total byte bound."
            )
        contribution = build_contribution()
    return contribution


def _disabled_contribution(
    situation: RecallSituation,
    policy: AutomaticRecallPolicy,
) -> AutomaticRecallContribution:
    return _disabled_contribution_from_hash(situation.fingerprint(), policy)


def _disabled_contribution_from_hash(
    situation_sha256: str,
    policy: AutomaticRecallPolicy,
) -> AutomaticRecallContribution:
    return AutomaticRecallContribution(
        situation_sha256=situation_sha256,
        policy_sha256=policy.fingerprint(),
        mode=policy.mode,
        sources=(),
        diagnostics=AutomaticRecallDiagnostics(
            recall_performed=False,
            recall_candidate_count=0,
            evaluated_candidate_count=0,
            strong_candidate_count=0,
            plausible_candidate_count=0,
            injected_count=0,
            offered_count=0,
            silent_count=0,
            duplicate_content_omitted=0,
            oversized_candidate_omitted=0,
            focus_bound_omitted=0,
            offer_bound_omitted=0,
            unevaluated_count=0,
            recall_truncated=False,
            admission_truncated=False,
        ),
    )


def _memory_focus(
    *,
    result: RecallResult,
    policy: AutomaticRecallPolicy,
    policy_sha256: str,
    items: Sequence[MemoryFocusItem],
    eligible_item_count: int,
) -> MemoryFocus:
    omitted = eligible_item_count - len(items)
    return MemoryFocus(
        situation_sha256=result.situation_sha256,
        policy_sha256=policy_sha256,
        calibration_version=policy.calibration_version,
        items=tuple(items),
        sources=result.sources,
        continuations=result.continuations,
        eligible_item_count=eligible_item_count,
        omitted_item_count=omitted,
        recall_truncated=result.truncated,
        truncated=result.truncated or omitted > 0,
    )


def _recall_offer_item(
    candidate: RecallCandidate,
    *,
    fused_rank: int,
    reason: _RecallOfferReason,
) -> RecallOfferItem:
    return RecallOfferItem(
        identity=candidate.record.identity,
        representation=candidate.record.representation,
        content_hash=candidate.record.content_hash,
        locator=candidate.record.locator,
        fused_rank=fused_rank,
        score=candidate.fused.score,
        matches=candidate.fused.matches,
        reason=reason,
    )


def _candidate_source_type(candidate: RecallCandidate) -> str:
    record_type = candidate.record.identity.record_type
    if record_type in {"knowledge_entry", "knowledge_chunk"}:
        return "knowledge"
    if record_type == "transcript_message":
        return "transcript"
    return record_type


def _recall_offer(
    *,
    result: RecallResult,
    policy: AutomaticRecallPolicy,
    policy_sha256: str,
    items: Sequence[RecallOfferItem],
    eligible_item_count: int,
) -> RecallOffer:
    omitted = eligible_item_count - len(items)
    ticket = _recall_offer_ticket(
        items=items,
        policy_sha256=policy_sha256,
        situation_sha256=result.situation_sha256,
        calibration_version=policy.calibration_version,
        continuations=result.continuations,
        eligible_item_count=eligible_item_count,
        omitted_item_count=omitted,
        recall_truncated=result.truncated,
        truncated=result.truncated or omitted > 0,
    )
    return RecallOffer(
        ticket=ticket,
        situation_sha256=result.situation_sha256,
        policy_sha256=policy_sha256,
        calibration_version=policy.calibration_version,
        items=tuple(items),
        continuations=result.continuations,
        eligible_item_count=eligible_item_count,
        omitted_item_count=omitted,
        recall_truncated=result.truncated,
        truncated=result.truncated or omitted > 0,
    )


def _recall_offer_ticket(
    *,
    items: Sequence[RecallOfferItem],
    policy_sha256: str,
    situation_sha256: str,
    calibration_version: str,
    continuations: Mapping[str, str],
    eligible_item_count: int,
    omitted_item_count: int,
    recall_truncated: bool,
    truncated: bool,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            {
                "schema_version": RECALL_OFFER_VERSION,
                "items": [item.model_dump(mode="json") for item in items],
                "policy_sha256": policy_sha256,
                "situation_sha256": situation_sha256,
                "calibration_version": calibration_version,
                "continuations": dict(continuations),
                "eligible_item_count": eligible_item_count,
                "omitted_item_count": omitted_item_count,
                "recall_truncated": recall_truncated,
                "truncated": truncated,
            },
            "recall offer ticket",
        )
    ).hexdigest()


def _serialized_bytes(value: BaseModel, field_name: str) -> int:
    return len(canonical_durable_json_bytes(value.model_dump(mode="json"), field_name))


__all__ = [
    "AUTOMATIC_RECALL_CONTRIBUTION_VERSION",
    "AUTOMATIC_RECALL_POLICY_VERSION",
    "MEMORY_FOCUS_VERSION",
    "RECALL_OFFER_VERSION",
    "AutomaticRecallContribution",
    "AutomaticRecallContributor",
    "AutomaticRecallDiagnostics",
    "AutomaticRecallMode",
    "AutomaticRecallPolicy",
    "MemoryFocus",
    "MemoryFocusItem",
    "RecallOffer",
    "RecallOfferItem",
    "admit_recall",
]
