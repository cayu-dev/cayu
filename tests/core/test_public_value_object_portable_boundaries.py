from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests._session_provenance import fixture_session_invocation

from cayu import (
    AgentSpec,
    BeforeStopDecision,
    BoundWorkspace,
    ContextPressureOverhead,
    ContextRequest,
    Environment,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EvalAssertionResult,
    EvalCase,
    EvalCaseComparison,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalRunComparison,
    EvalStatus,
    EvalSuite,
    EvalTrialResult,
    LocalWorkspace,
    Message,
    ProxyAuthorizationResult,
    ResolvedSecret,
    RunRequest,
    SecretEnv,
    SecretRef,
    Session,
    StepRunOptions,
    SubagentSpec,
    SyncBindingContext,
    TextEmbedding,
    TextEmbeddingRequest,
    TextEmbeddingResult,
    TextEmbeddingUsage,
    ToolContext,
    Trajectory,
    WorkflowSpec,
    WorkspaceInstructions,
    WorkspaceInstructionsConfig,
    WorkspaceSnapshot,
    extract_durable_value_error,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, MIN_DURABLE_JSON_INTEGER
from cayu.egress import VirtualCredentialGrant

_SENSITIVE_CANARY = "private-runtime-value-canary"
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _PublicValueFieldCase:
    constructor: str
    field: str
    capture: Callable[[Any], Any]

    @property
    def id(self) -> str:
        return f"{self.constructor}.{self.field}"


@dataclass(frozen=True)
class _PublicTextFieldCase:
    constructor: str
    field: str
    policy: str
    capture: Callable[[str], str | None]

    @property
    def id(self) -> str:
        return f"{self.constructor}.{self.field}"


@dataclass(frozen=True)
class _PublicTextCollectionCase:
    constructor: str
    field: str
    capture: Callable[[tuple[str, ...]], tuple[str, ...]]

    @property
    def id(self) -> str:
        return f"{self.constructor}.{self.field}"


def _session() -> Session:
    return Session(
        id="sess_portable_value",
        agent_name="agent",
        provider_name="fake",
        model="model",
        invocation=fixture_session_invocation("sess_portable_value"),
    )


def _run_request() -> RunRequest:
    return RunRequest(
        agent_name="agent",
        messages=[Message.text("user", "test")],
    )


def _eval_case(metadata: dict[str, Any] | None = None) -> EvalCase:
    return EvalCase(
        id="case",
        request=_run_request(),
        metadata={} if metadata is None else metadata,
    )


def _eval_case_result(metadata: dict[str, Any]) -> EvalCaseResult:
    trial = EvalTrialResult(
        trial_number=1,
        status=EvalStatus.UNAVAILABLE,
        unavailable_reason="evidence unavailable",
        started_at=_NOW,
        completed_at=_NOW,
    )
    return EvalCaseResult.from_trials(
        case_id="case",
        trials=[trial],
        started_at=_NOW,
        completed_at=_NOW,
        metadata=metadata,
    )


def _eval_run(metadata: dict[str, Any]) -> EvalRun:
    case = _eval_case_result({})
    return EvalRun(
        run_id="run",
        suite_id="suite",
        status=EvalStatus.UNAVAILABLE,
        score=None,
        cases=(case,),
        started_at=_NOW,
        completed_at=_NOW,
        metadata=metadata,
    )


def _environment() -> Environment:
    return Environment(EnvironmentSpec(name="environment"))


def _eval_run_comparison(**overrides: Any) -> EvalRunComparison:
    values: dict[str, Any] = {
        "baseline_run_id": "baseline-run",
        "current_run_id": "current-run",
        "baseline_suite_id": "baseline-suite",
        "current_suite_id": "current-suite",
        "baseline_status": EvalStatus.PASSED,
        "current_status": EvalStatus.PASSED,
    }
    values.update(overrides)
    return EvalRunComparison(**values)


def _tool_context_text(field_name: str, value: str) -> str:
    context = ToolContext.model_validate({"session_id": "session", field_name: value})
    captured = getattr(context, field_name)
    if type(captured) is not str:
        raise AssertionError("ToolContext text case did not retain a string.")
    return captured


_PUBLIC_TEXT_CASES = (
    _PublicTextFieldCase(
        "AgentSpec",
        "name",
        "clean_nonblank",
        lambda value: AgentSpec(name=value, model="model").name,
    ),
    _PublicTextFieldCase(
        "AgentSpec",
        "model",
        "clean_nonblank",
        lambda value: AgentSpec(name="agent", model=value).model,
    ),
    _PublicTextFieldCase(
        "AgentSpec",
        "provider_name",
        "clean_nonblank",
        lambda value: AgentSpec(name="agent", model="model", provider_name=value).provider_name,
    ),
    _PublicTextFieldCase(
        "AgentSpec",
        "system_prompt",
        "text",
        lambda value: AgentSpec(name="agent", model="model", system_prompt=value).system_prompt,
    ),
    _PublicTextFieldCase(
        "AgentSpec",
        "workflow_tool_names[1]",
        "clean_nonblank",
        lambda value: AgentSpec(
            name="agent",
            model="model",
            workflow_tool_names=("first", value, "last"),
        ).workflow_tool_names[1],
    ),
    _PublicTextFieldCase(
        "WorkflowSpec",
        "name",
        "clean_nonblank",
        lambda value: WorkflowSpec(name=value).name,
    ),
    _PublicTextFieldCase(
        "EnvironmentSpec",
        "name",
        "clean_nonblank",
        lambda value: EnvironmentSpec(name=value).name,
    ),
    _PublicTextFieldCase(
        "EvalAssertionResult",
        "name",
        "clean_nonblank",
        lambda value: (
            EvalAssertionResult(
                name=value,
                outcome=EvalOutcome.UNAVAILABLE,
            ).name
        ),
    ),
    _PublicTextFieldCase(
        "EvalCase",
        "id",
        "clean_nonblank",
        lambda value: EvalCase(id=value, request=_run_request()).id,
    ),
    _PublicTextFieldCase(
        "EvalSuite",
        "id",
        "clean_nonblank",
        lambda value: EvalSuite(id=value, cases=[_eval_case()]).id,
    ),
    _PublicTextFieldCase(
        "EvalCaseComparison",
        "case_id",
        "clean_nonblank",
        lambda value: EvalCaseComparison(case_id=value).case_id,
    ),
    _PublicTextFieldCase(
        "EvalRunComparison",
        "baseline_run_id",
        "clean_nonblank",
        lambda value: _eval_run_comparison(baseline_run_id=value).baseline_run_id,
    ),
    _PublicTextFieldCase(
        "EvalRunComparison",
        "current_run_id",
        "clean_nonblank",
        lambda value: _eval_run_comparison(current_run_id=value).current_run_id,
    ),
    _PublicTextFieldCase(
        "EvalRunComparison",
        "baseline_suite_id",
        "clean_nonblank",
        lambda value: _eval_run_comparison(baseline_suite_id=value).baseline_suite_id,
    ),
    _PublicTextFieldCase(
        "EvalRunComparison",
        "current_suite_id",
        "clean_nonblank",
        lambda value: _eval_run_comparison(current_suite_id=value).current_suite_id,
    ),
    _PublicTextFieldCase(
        "EvalCaseComparison",
        "regressions[1]",
        "text",
        lambda value: EvalCaseComparison(
            case_id="case",
            regressions=("first", value, "last"),
        ).regressions[1],
    ),
    _PublicTextFieldCase(
        "EvalRunComparison",
        "regressions[1]",
        "text",
        lambda value: _eval_run_comparison(regressions=("first", value, "last")).regressions[1],
    ),
    _PublicTextFieldCase(
        "ToolContext",
        "session_id",
        "clean_nonblank",
        lambda value: ToolContext(session_id=value).session_id,
    ),
    *(
        _PublicTextFieldCase(
            "ToolContext",
            field_name,
            "clean_nonblank",
            lambda value, field_name=field_name: _tool_context_text(field_name, value),
        )
        for field_name in (
            "agent_name",
            "environment_name",
            "causal_budget_id",
            "workspace_id",
            "artifact_store_id",
            "idempotency_key",
        )
    ),
    _PublicTextFieldCase(
        "WorkspaceInstructions",
        "content",
        "nonblank",
        lambda value: WorkspaceInstructions(content=value).content,
    ),
    _PublicTextFieldCase(
        "WorkspaceInstructions",
        "sources[1]",
        "nonblank",
        lambda value: WorkspaceInstructions(
            content="instructions",
            sources=("first", value, "last"),
        ).sources[1],
    ),
    _PublicTextFieldCase(
        "WorkspaceInstructionsConfig",
        "paths[1]",
        "nonblank",
        lambda value: WorkspaceInstructionsConfig(paths=("first.md", value, "last.md")).paths[1],
    ),
)

_EXPECTED_PUBLIC_TEXT_CASE_IDS = (
    "AgentSpec.name",
    "AgentSpec.model",
    "AgentSpec.provider_name",
    "AgentSpec.system_prompt",
    "AgentSpec.workflow_tool_names[1]",
    "WorkflowSpec.name",
    "EnvironmentSpec.name",
    "EvalAssertionResult.name",
    "EvalCase.id",
    "EvalSuite.id",
    "EvalCaseComparison.case_id",
    "EvalRunComparison.baseline_run_id",
    "EvalRunComparison.current_run_id",
    "EvalRunComparison.baseline_suite_id",
    "EvalRunComparison.current_suite_id",
    "EvalCaseComparison.regressions[1]",
    "EvalRunComparison.regressions[1]",
    "ToolContext.session_id",
    "ToolContext.agent_name",
    "ToolContext.environment_name",
    "ToolContext.causal_budget_id",
    "ToolContext.workspace_id",
    "ToolContext.artifact_store_id",
    "ToolContext.idempotency_key",
    "WorkspaceInstructions.content",
    "WorkspaceInstructions.sources[1]",
    "WorkspaceInstructionsConfig.paths[1]",
)

_PUBLIC_TEXT_COLLECTION_CASES = (
    _PublicTextCollectionCase(
        "AgentSpec",
        "workflow_tool_names",
        lambda values: (
            AgentSpec(
                name="agent",
                model="model",
                workflow_tool_names=values,
            ).workflow_tool_names
        ),
    ),
    _PublicTextCollectionCase(
        "EvalCaseComparison",
        "regressions",
        lambda values: EvalCaseComparison(case_id="case", regressions=values).regressions,
    ),
    _PublicTextCollectionCase(
        "EvalRunComparison",
        "regressions",
        lambda values: _eval_run_comparison(regressions=values).regressions,
    ),
    _PublicTextCollectionCase(
        "WorkspaceInstructions",
        "sources",
        lambda values: (
            WorkspaceInstructions(
                content="instructions",
                sources=values,
            ).sources
        ),
    ),
    _PublicTextCollectionCase(
        "WorkspaceInstructionsConfig",
        "paths",
        lambda values: WorkspaceInstructionsConfig(paths=values).paths,
    ),
)


_CASES = (
    _PublicValueFieldCase(
        "AgentSpec",
        "metadata",
        lambda value: AgentSpec(name="agent", model="model", metadata={"probe": value}).metadata[
            "probe"
        ],
    ),
    _PublicValueFieldCase(
        "AgentSpec",
        "provider_options",
        lambda value: AgentSpec(
            name="agent", model="model", provider_options={"probe": value}
        ).provider_options["probe"],
    ),
    _PublicValueFieldCase(
        "WorkflowSpec",
        "metadata",
        lambda value: WorkflowSpec(name="workflow", metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "SubagentSpec",
        "metadata",
        lambda value: SubagentSpec(agent_name="agent", metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EnvironmentSpec",
        "metadata",
        lambda value: EnvironmentSpec(name="environment", metadata={"probe": value}).metadata[
            "probe"
        ],
    ),
    _PublicValueFieldCase(
        "BeforeStopDecision",
        "metadata",
        lambda value: BeforeStopDecision(metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "StepRunOptions",
        "metadata",
        lambda value: StepRunOptions(metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "ContextRequest",
        "metadata",
        lambda value: ContextRequest(
            session=_session(),
            agent=AgentSpec(name="agent", model="model"),
            messages=[Message.text("user", "test")],
            step=1,
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "Trajectory",
        "metadata",
        lambda value: Trajectory(metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "WorkspaceSnapshot",
        "metadata",
        lambda value: WorkspaceSnapshot(
            snapshot_id="snapshot",
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "SyncBindingContext",
        "metadata",
        lambda value: SyncBindingContext(
            source_workspace=LocalWorkspace(Path.cwd()),
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "BoundWorkspace",
        "metadata",
        lambda value: BoundWorkspace(metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "ToolContext",
        "metadata",
        lambda value: ToolContext(
            session_id="session",
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "ContextPressureOverhead",
        "tools",
        lambda value: ContextPressureOverhead(tools=[{"probe": value}]).tools[0]["probe"],
    ),
    _PublicValueFieldCase(
        "ContextPressureOverhead",
        "request_options",
        lambda value: ContextPressureOverhead(request_options={"probe": value}).request_options[
            "probe"
        ],
    ),
    _PublicValueFieldCase(
        "EvalAssertionResult",
        "metadata",
        lambda value: EvalAssertionResult(
            name="assertion",
            outcome=EvalOutcome.UNAVAILABLE,
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EvalCase",
        "metadata",
        lambda value: _eval_case({"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EvalSuite",
        "metadata",
        lambda value: EvalSuite(
            id="suite",
            cases=[_eval_case()],
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EvalCaseResult",
        "metadata",
        lambda value: _eval_case_result({"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EvalRun",
        "metadata",
        lambda value: _eval_run({"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "TextEmbeddingRequest",
        "options",
        lambda value: TextEmbeddingRequest(
            model="embedding-model",
            texts=["text"],
            options={"probe": value},
        ).options["probe"],
    ),
    _PublicValueFieldCase(
        "TextEmbeddingResult",
        "metadata",
        lambda value: TextEmbeddingResult(
            model="embedding-model",
            embeddings=[TextEmbedding(index=0, vector=[0.1])],
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "TextEmbeddingUsage",
        "metadata",
        lambda value: TextEmbeddingUsage(metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "SecretRef",
        "metadata",
        lambda value: SecretRef(name="secret", metadata={"probe": value}).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "SecretEnv",
        "metadata",
        lambda value: SecretEnv(
            name="SECRET",
            ref=SecretRef(name="secret"),
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "ResolvedSecret",
        "metadata",
        lambda value: ResolvedSecret(
            name="secret",
            value=SecretStr("value"),
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EnvironmentFactoryRequest",
        "metadata",
        lambda value: EnvironmentFactoryRequest(
            session_id="session",
            agent_name="agent",
            environment_name="environment",
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EnvironmentFactoryRequest",
        "reconnect_metadata",
        lambda value: EnvironmentFactoryRequest(
            session_id="session",
            agent_name="agent",
            environment_name="environment",
            reconnect_metadata={"probe": value},
        ).reconnect_metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EnvironmentFactoryResult",
        "metadata",
        lambda value: EnvironmentFactoryResult(
            environment=_environment(),
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "EnvironmentFactoryResult",
        "reconnect_metadata",
        lambda value: EnvironmentFactoryResult(
            environment=_environment(),
            reconnect_metadata={"probe": value},
        ).reconnect_metadata["probe"],
    ),
    _PublicValueFieldCase(
        "ProxyAuthorizationResult",
        "metadata",
        lambda value: ProxyAuthorizationResult(
            allowed=True,
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _PublicValueFieldCase(
        "VirtualCredentialGrant",
        "metadata",
        lambda value: VirtualCredentialGrant(
            grant_id="grant",
            session_id="session",
            env_name="SECRET",
            presented_value="cayu_virtual_test_value",
            secret=SecretRef(name="secret"),
            destination="api.example.com",
            credential_kind="stripe_bearer",
            created_at=_NOW,
            metadata={"probe": value},
        ).metadata["probe"],
    ),
)

_EXPECTED_CONSTRUCTORS = {
    "AgentSpec",
    "WorkflowSpec",
    "SubagentSpec",
    "EnvironmentSpec",
    "BeforeStopDecision",
    "StepRunOptions",
    "ContextRequest",
    "Trajectory",
    "WorkspaceSnapshot",
    "SyncBindingContext",
    "BoundWorkspace",
    "ToolContext",
    "ContextPressureOverhead",
    "EvalAssertionResult",
    "EvalCase",
    "EvalSuite",
    "EvalCaseResult",
    "EvalRun",
    "TextEmbeddingRequest",
    "TextEmbeddingResult",
    "TextEmbeddingUsage",
    "SecretRef",
    "SecretEnv",
    "ResolvedSecret",
    "EnvironmentFactoryRequest",
    "EnvironmentFactoryResult",
    "ProxyAuthorizationResult",
    "VirtualCredentialGrant",
}


def _too_deep_value() -> Any:
    value: Any = "leaf"
    for _ in range(129):
        value = [value]
    return value


def test_public_value_object_conformance_covers_every_issue_constructor() -> None:
    assert {case.constructor for case in _CASES} == _EXPECTED_CONSTRUCTORS


def test_environment_factory_request_preserves_existing_positional_field_order() -> None:
    request = EnvironmentFactoryRequest(
        "session",
        "agent",
        "environment",
        "parent-session",
    )

    assert request.parent_session_id == "parent-session"
    assert request.execution_profile_fingerprint is None


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        (MAX_DURABLE_JSON_INTEGER + 1, "integer_out_of_range"),
        (MIN_DURABLE_JSON_INTEGER - 1, "integer_out_of_range"),
        (float(2**63), "integral_float_out_of_range"),
        (-(float(2**63) + 2048.0), "integral_float_out_of_range"),
        (float("nan"), "non_finite_number"),
        (f"{_SENSITIVE_CANARY}\x00", "nul_character"),
        (f"{_SENSITIVE_CANARY}\ud800", "unicode_surrogate"),
        ({f"{_SENSITIVE_CANARY}\x00": "value"}, "nul_character"),
        ({f"{_SENSITIVE_CANARY}\ud800": "value"}, "unicode_surrogate"),
        (_too_deep_value(), "nesting_too_deep"),
    ],
)
def test_public_value_object_fields_reject_nonportable_values(
    case: _PublicValueFieldCase,
    value: Any,
    expected_code: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        case.capture(value)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    assert _SENSITIVE_CANARY not in str(raised.value)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_public_value_object_fields_normalize_and_copy_portable_values(
    case: _PublicValueFieldCase,
) -> None:
    value: dict[str, Any] = {
        "bounds": [MIN_DURABLE_JSON_INTEGER, MAX_DURABLE_JSON_INTEGER],
        "integral": 42.0,
        "negative_zero": -0.0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "boolean": True,
        "null": None,
        "nested": [{"status": "original"}],
    }

    captured = case.capture(value)

    assert captured == {
        "bounds": [MIN_DURABLE_JSON_INTEGER, MAX_DURABLE_JSON_INTEGER],
        "integral": 42,
        "negative_zero": 0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "boolean": True,
        "null": None,
        "nested": [{"status": "original"}],
    }
    assert type(captured["integral"]) is int
    assert type(captured["negative_zero"]) is int
    assert type(captured["fractional"]) is float

    value["nested"][0]["status"] = "mutated"
    assert captured["nested"][0]["status"] == "original"


def test_public_text_conformance_covers_every_issue_boundary() -> None:
    assert tuple(case.id for case in _PUBLIC_TEXT_CASES) == _EXPECTED_PUBLIC_TEXT_CASE_IDS
    assert len(_PUBLIC_TEXT_CASES) == 27


@pytest.mark.parametrize("case", _PUBLIC_TEXT_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize(
    ("invalid_text", "expected_code"),
    [
        ("invalid\x00text", "nul_character"),
        ("invalid\ud800text", "unicode_surrogate"),
    ],
)
def test_public_text_fields_reject_nonportable_text(
    case: _PublicTextFieldCase,
    invalid_text: str,
    expected_code: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        case.capture(invalid_text)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code


@pytest.mark.parametrize("case", _PUBLIC_TEXT_CASES, ids=lambda case: case.id)
def test_public_text_fields_preserve_ordinary_unicode(case: _PublicTextFieldCase) -> None:
    value = "Zażółć_日本語_😀"

    assert case.capture(value) == value


@pytest.mark.parametrize("case", _PUBLIC_TEXT_CASES, ids=lambda case: case.id)
def test_public_text_fields_preserve_existing_whitespace_semantics(
    case: _PublicTextFieldCase,
) -> None:
    if case.policy == "clean_nonblank":
        for value in ("", "   ", " padded "):
            with pytest.raises(ValueError):
                case.capture(value)
        return

    if case.policy == "nonblank":
        for value in ("", "   "):
            with pytest.raises(ValueError):
                case.capture(value)
        assert case.capture(" padded ") == " padded "
        return

    assert case.policy == "text"
    assert case.capture("") == ""
    assert case.capture("   ") == "   "
    assert case.capture(" padded ") == " padded "


def test_public_text_optional_fields_still_accept_none() -> None:
    agent = AgentSpec(
        name="agent",
        model="model",
        provider_name=None,
        system_prompt=None,
    )
    context = ToolContext(session_id="session")

    assert agent.provider_name is None
    assert agent.system_prompt is None
    assert context.agent_name is None
    assert context.environment_name is None
    assert context.causal_budget_id is None
    assert context.workspace_id is None
    assert context.artifact_store_id is None
    assert context.idempotency_key is None


@pytest.mark.parametrize(
    "case",
    _PUBLIC_TEXT_COLLECTION_CASES,
    ids=lambda case: case.id,
)
@pytest.mark.parametrize("invalid_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_text", ["invalid\x00text", "invalid\ud800text"])
def test_public_text_collections_validate_every_item(
    case: _PublicTextCollectionCase,
    invalid_index: int,
    invalid_text: str,
) -> None:
    values = ["first", "middle", "last"]
    values[invalid_index] = invalid_text

    with pytest.raises(ValueError) as raised:
        case.capture(tuple(values))

    assert extract_durable_value_error(raised.value) is not None
