from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

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
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
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


def _session() -> Session:
    return Session(
        id="sess_portable_value",
        agent_name="agent",
        provider_name="fake",
        model="model",
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
