"""Bounded suite-centred read contracts and shared projection primitives.

Selectors locate immutable records; they never confer target authority. Stores
own pagination and atomic progress observations, while result presentation and
comparison retain their existing semantic owners.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from typing import Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from cayu.evals.execution_comparison import (
    CorpusComparisonCompatibility,
    CorpusExecutionRegression,
    EvalStructuredJudgeComparisonV1,
    EvalToolJsonAssertionComparisonV1,
)
from cayu.evals.execution_profiles import EvalExecutionCandidateIdentityV1
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.result_presentation import EvalCasePresentationV2
from cayu.evals.store import (
    EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS,
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CLAIM_TARGETS,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalBaselineRecord,
    EvalRunRecord,
    EvalRunStatus,
    EvalStoreResultTooLarge,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1


class EvalViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class EvalSuiteViewIdentity(EvalViewModel):
    source: Literal["authored", "corpus"]
    target_key: StrictStr = Field(min_length=1, max_length=128)
    revision: StrictStr = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    suite_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.source == "corpus") != (self.suite_id is not None):
            raise ValueError("Only corpus suite selectors carry suite_id.")
        return self

    @property
    def id(self) -> str:
        return encode_view_token(["suite", self.model_dump(mode="json")])


def encode_view_token(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_view_token(token: str) -> object:
    if type(token) is not str or not 1 <= len(token) <= EVAL_STORE_MAX_CURSOR_BYTES:
        raise ValueError("Invalid eval view selector.")
    try:
        raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid eval view selector.") from exc
    if encode_view_token(value) != token:
        raise ValueError("Noncanonical eval view selector.")
    return value


def decode_suite_view(token: str) -> EvalSuiteViewIdentity:
    value = decode_view_token(token)
    if not isinstance(value, list) or len(value) != 2 or value[0] != "suite":
        raise ValueError("Invalid suite selector.")
    return EvalSuiteViewIdentity.model_validate(value[1])


def run_view_id(source: str, revision: str) -> str:
    return encode_view_token([source, revision])


def decode_run_view(token: str) -> tuple[str, str]:
    value = decode_view_token(token)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not str
        or value[0] not in {"run", "result"}
        or type(value[1]) is not str
        or not 1 <= len(value[1]) <= 256
    ):
        raise ValueError("Invalid run selector.")
    return value[0], value[1]


class EvalViewQuery(EvalViewModel):
    target_key: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    target_keys: tuple[StrictStr, ...] | None = Field(
        default=None, max_length=EVAL_STORE_MAX_CLAIM_TARGETS
    )
    suite: EvalSuiteViewIdentity | None = None
    cursor: StrictStr | None = Field(default=None, max_length=EVAL_STORE_MAX_CURSOR_BYTES)
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE, ge=1, le=EVAL_STORE_MAX_PAGE_SIZE
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES, ge=1024, le=EVAL_STORE_MAX_PAGE_BYTES
    )

    @model_validator(mode="after")
    def validate_target(self):
        if self.target_keys is not None:
            if self.target_keys != tuple(sorted(set(self.target_keys))) or any(
                not 1 <= len(key) <= 128 for key in self.target_keys
            ):
                raise ValueError("Target filters must be bounded, unique and sorted.")
            if self.target_key is not None and self.target_key not in self.target_keys:
                raise ValueError("Target is outside the permitted query scope.")
        if self.suite is not None and self.suite.target_key != self.target_key:
            raise ValueError("Suite query target mismatch.")
        return self

    def scope(self) -> object:
        allowed = (
            None
            if self.target_keys is None
            else hashlib.sha256(
                json.dumps(self.target_keys, separators=(",", ":")).encode()
            ).hexdigest()
        )
        return [
            self.target_key,
            None if self.suite is None else self.suite.model_dump(mode="json"),
            allowed,
        ]

    def boundary(self) -> tuple[datetime | None, str] | None:
        if self.cursor is None:
            return None
        value = decode_view_token(self.cursor)
        if (
            not isinstance(value, list)
            or len(value) != 4
            or value[0] != "page"
            or value[1] != self.scope()
            or type(value[3]) is not str
        ):
            raise ValueError("Eval view cursor scope mismatch.")
        if value[2] is not None and type(value[2]) is not str:
            raise ValueError("Invalid eval view cursor timestamp.")
        timestamp = None if value[2] is None else datetime.fromisoformat(value[2])
        if (self.suite is not None) != (timestamp is not None):
            raise ValueError("Invalid eval view cursor ordering.")
        if timestamp is not None and timestamp.utcoffset() is None:
            raise ValueError("Eval view cursor requires an aware timestamp.")
        return (None if timestamp is None else timestamp.astimezone(UTC)), value[3]


class EvalViewIndexEntry(EvalViewModel):
    target_key: StrictStr = Field(min_length=1, max_length=128)
    source: Literal["authored", "corpus", "run", "result"]
    revision: StrictStr
    suite_id: StrictStr | None = None
    created_at: datetime

    @property
    def key(self) -> str:
        return f"{self.source}:{self.revision}:{self.suite_id or ''}"


class EvalViewIndexPage(EvalViewModel):
    items: tuple[EvalViewIndexEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: str | None = None
    has_more: bool = False


def validate_view_index(page: EvalViewIndexPage, query: EvalViewQuery) -> EvalViewIndexPage:
    """Check a custom adapter page before hydration or output expansion."""
    if type(page) is not EvalViewIndexPage:
        raise ValueError("Invalid eval index page type.")
    page = EvalViewIndexPage.model_validate(page.model_dump(mode="python", warnings="none"))
    if (
        len(page.items) > query.limit
        or len(page.model_dump_json().encode()) > query.max_result_bytes
    ):
        raise ValueError("Eval index page exceeds query bounds.")
    if page.has_more != (page.next_cursor is not None) or (page.has_more and not page.items):
        raise ValueError("Invalid eval index continuation.")
    if any(
        (query.target_key is not None and item.target_key != query.target_key)
        or (query.target_keys is not None and item.target_key not in query.target_keys)
        for item in page.items
    ):
        raise ValueError("Eval index target scope mismatch.")
    expected = sorted(page.items, key=lambda item: item.key)
    if query.suite is not None:
        expected.sort(key=lambda item: item.created_at, reverse=True)
    if tuple(expected) != page.items or len({item.key for item in page.items}) != len(page.items):
        raise ValueError("Invalid eval index ordering.")
    if page.next_cursor is not None:
        last = page.items[-1]
        expected_cursor = encode_view_token(
            ["page", query.scope(), last.created_at.isoformat() if query.suite else None, last.key]
        )
        if page.next_cursor != expected_cursor:
            raise ValueError("Invalid eval index cursor binding.")
    return page


def bounded_view_index(items: list[EvalViewIndexEntry], query: EvalViewQuery) -> EvalViewIndexPage:
    selected: list[EvalViewIndexEntry] = []
    for item in items[: query.limit]:
        candidate = (*selected, item)
        cursor = encode_view_token(
            ["page", query.scope(), item.created_at.isoformat() if query.suite else None, item.key]
        )
        page = EvalViewIndexPage(items=candidate, next_cursor=cursor, has_more=True)
        if len(page.model_dump_json().encode()) > query.max_result_bytes:
            if not selected:
                raise EvalStoreResultTooLarge(query.max_result_bytes)
            break
        selected.append(item)
    more = len(items) > len(selected)
    return EvalViewIndexPage(
        items=tuple(selected),
        has_more=more,
        next_cursor=(
            encode_view_token(
                [
                    "page",
                    query.scope(),
                    selected[-1].created_at.isoformat() if query.suite else None,
                    selected[-1].key,
                ]
            )
            if more
            else None
        ),
    )


class EvalTrialViewOutcome(EvalViewModel):
    """Non-secret terminal evidence needed by the existing trial-policy reducer."""

    case_id: StrictStr = Field(min_length=1, max_length=128)
    trial_number: StrictInt = Field(ge=1)
    status: Literal["passed", "failed", "error", "unavailable"]
    score: StrictFloat | None = Field(ge=0.0, le=1.0)
    diagnostic_code: EvalTrialDiagnosticCode


class EvalRunViewSnapshot(EvalViewModel):
    """Atomic lifecycle and terminal outcomes; no claims, errors or raw outputs."""

    run: EvalRunRecord
    terminal_trials: tuple[EvalTrialViewOutcome, ...] = Field(
        max_length=EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS
    )

    @model_validator(mode="after")
    def validate_unique_slots(self):
        keys = [(item.case_id, item.trial_number) for item in self.terminal_trials]
        if len(keys) != len(set(keys)):
            raise ValueError("Terminal trial slots must be unique.")
        return self

    @property
    def completed_trials_by_case(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.terminal_trials:
            counts[item.case_id] = counts.get(item.case_id, 0) + 1
        return counts


# HTTP read DTOs deliberately retain unavailable and incompatible states. They
# do not turn SDK/HTTP provenance into invented CI or dashboard attribution.

EvalRunCaseFilter = Literal["regressed", "improved", "failing", "flaky"]

T = TypeVar("T")


class EvalViewPage(EvalViewModel, Generic[T]):
    items: tuple[T, ...]
    next_cursor: str | None = None
    has_more: bool = False
    total: int | None = None


class EvalViewImprovement(CorpusExecutionRegression):
    """A comparator-proven improvement, oriented from compared baseline to current."""

    @model_validator(mode="after")
    def validate_shape(self) -> EvalViewImprovement:
        reverse = self.model_dump(mode="python")
        for field in ("status", "score", "reliability"):
            reverse[f"baseline_{field}"], reverse[f"current_{field}"] = (
                reverse[f"current_{field}"],
                reverse[f"baseline_{field}"],
            )
        # Preserve the comparator's strict dimension and direction semantics.
        CorpusExecutionRegression.model_validate(reverse)
        return self


class EvalViewComparison(EvalViewModel):
    state: Literal["pending", "unavailable", "no_baseline", "compared", "incompatible"]
    verdict: Literal["improvement", "regression", "tradeoff", "tie"] | None = None
    compared_with: str | None = None
    unavailable_reasons: tuple[str, ...] = ()
    compatibility: CorpusComparisonCompatibility | None = None
    improved_cases: int | None = None
    regressed_cases: int | None = None
    regressions: tuple[CorpusExecutionRegression, ...] = ()
    improvements: tuple[EvalViewImprovement, ...] = ()


class EvalViewProgress(EvalViewModel):
    total_cases: int
    total_trials: int
    completed_cases: int | None
    completed_trials: int | None
    state: Literal["queued", "checkpointed", "published", "unavailable"]


class EvalViewCaseCounts(EvalViewModel):
    """Disjoint settled outcomes plus a separately labelled variability count."""

    total: int
    passed: int
    failed: int
    flaky: int = Field(
        title="Flaky variability (overlapping)",
        description="Cases with both passed and nonpassed trials. This overlaps the outcome counts; never add it to passed, failed, error, unavailable or total.",
    )
    error: int
    unavailable: int


class EvalViewUsage(EvalViewModel):
    scope: Literal["candidate_execution"] = "candidate_execution"
    state: Literal["complete", "partial", "unavailable"]
    tokens: str | None
    observed_tokens: str | None
    cost: str | None = None
    currency: str | None = None
    cost_unavailable_reason: Literal["aggregate_not_retained"] = "aggregate_not_retained"


class EvalRunPlacement(EvalViewModel):
    destination: EvalSuiteViewIdentity
    operation_id: str
    case_id: str
    case_revision: str
    execution_case_count: Literal[1] = 1


class EvalRunViewMetadata(EvalViewModel):
    placement: EvalRunPlacement | None = None
    id: str
    native_run_id: str | None
    authored_suite_revision: str | None
    result_revision: str | None
    suite: EvalSuiteViewIdentity
    suite_name: str
    specialist: str
    status: EvalRunStatus
    origin: str
    actor: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    cases: EvalViewCaseCounts | None
    score: float | None
    judge_score: float | None
    judge_score_state: Literal["available", "not_used", "unavailable"]
    usage: EvalViewUsage
    candidate: EvalExecutionCandidateIdentityV1 | None
    candidate_unavailable_reason: Literal["not_retained"] | None
    baseline: EvalBaselineRecord | None
    is_baseline: bool


class EvalRunViewSummary(EvalRunViewMetadata):
    comparison: EvalViewComparison
    baseline_result: EvalRunViewMetadata | None


class EvalSuiteViewSummary(EvalViewModel):
    id: str
    identity: EvalSuiteViewIdentity
    source: Literal["authored", "corpus"]
    name: str
    description: str | None
    specialist: str
    case_count: int
    trials: EvalSuiteTrialPolicyV1
    baseline: EvalBaselineRecord | None
    baseline_result: EvalRunViewMetadata | None
    baseline_scope: Literal["suite", "execution_group"] = Field(
        description="Authored definitions use execution_group: their baseline is null and each history run exposes the baseline for its exact corpus/suite group. Corpus suites use suite.",
    )
    last_run: EvalRunViewSummary | None


class EvalCaseView(EvalViewModel):
    id: str
    name: str
    input: JsonValue
    checks: tuple[JsonValue, ...]
    source_session_id: None = None
    last: EvalCasePresentationV2 | None = None


class EvalCaseResultView(EvalViewModel):
    case: EvalCaseView
    presentation: EvalCasePresentationV2
    judge_score: float | None
    judge_score_state: Literal["available", "not_used", "unavailable"]
    evidence: JsonValue
    compared: EvalCasePresentationV2 | None
    compared_evidence: JsonValue
    structured_judgments: tuple[EvalStructuredJudgeComparisonV1, ...] = ()
    tool_json_assertions: tuple[EvalToolJsonAssertionComparisonV1, ...] = ()
    change: Literal["improved", "regressed", "tradeoff", "unchanged", "uncompared"]
    regressions: tuple[CorpusExecutionRegression, ...] = ()
    improvements: tuple[EvalViewImprovement, ...] = ()


class EvalSuiteViewDetail(EvalViewModel):
    suite: EvalSuiteViewSummary
    cases: EvalViewPage[EvalCaseView]
    runs: EvalViewPage[EvalRunViewSummary]


class EvalRunViewDetail(EvalViewModel):
    run: EvalRunViewSummary
    compared_run: EvalRunViewMetadata | None
    comparison: EvalViewComparison
    progress: EvalViewProgress
    results: EvalViewPage[EvalCaseResultView]


class EvalCaseHistoryQuery(EvalViewModel):
    suite: EvalSuiteViewIdentity
    case_ids: tuple[StrictStr, ...] = Field(min_length=1, max_length=EVAL_STORE_MAX_PAGE_SIZE)

    @model_validator(mode="after")
    def validate_cases(self):
        if self.case_ids != tuple(sorted(set(self.case_ids))):
            raise ValueError("Case history identities must be unique and sorted.")
        if any(not 1 <= len(value) <= 128 for value in self.case_ids):
            raise ValueError("Invalid case history identity.")
        return self


class EvalCaseResultReference(EvalViewModel):
    case_id: StrictStr
    source: Literal["run", "result"]
    revision: StrictStr
