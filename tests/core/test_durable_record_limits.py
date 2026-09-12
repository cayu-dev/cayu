from __future__ import annotations

import json
import warnings
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from cayu._validation import (
    DURABLE_METADATA_LIMITS,
    SESSION_LABEL_MAX_ENTRIES,
    SESSION_METADATA_LIMITS,
    DurableValueError,
    copy_durable_metadata,
    copy_label_map,
    copy_session_metadata,
    extract_durable_value_error,
)
from cayu.core.agents import AgentSpec
from cayu.core.messages import Message
from cayu.core.workflows import WorkflowSpec
from cayu.embeddings import TextEmbeddingResult, TextEmbeddingUsage
from cayu.evals.models import EvalAssertionResult
from cayu.evals.runner import EvalCase
from cayu.proxies.base import ProxyAuthorizationResult
from cayu.runtime.approvals import ToolApprovalRequest
from cayu.runtime.dispatch import DispatchRequest
from cayu.runtime.loop_policies import BeforeStopDecision
from cayu.runtime.sessions import (
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    replace_session_user_metadata,
)
from cayu.runtime.tasks import TaskCreate
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.runtime.user_input import UserInputResponse
from cayu.storage.knowledge_indexer import KnowledgeIndexRequest
from cayu.storage.memory import KnowledgeChunk, KnowledgeEntry
from cayu.tools.subagents import SubagentSpec
from cayu.vaults.base import ResolvedSecret, SecretEnv, SecretRef


def _metadata_bytes(size: int, *, character: str = "x") -> dict[str, Any]:
    overhead = len(json.dumps({"value": ""}, separators=(",", ":")).encode())
    count, remainder = divmod(size - overhead, len(character.encode("utf-8")))
    return {"value": character * count + "x" * remainder}


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (AgentSpec, {"name": "agent", "model": "model"}),
        (WorkflowSpec, {"name": "workflow"}),
        (BeforeStopDecision, {}),
        (SubagentSpec, {"agent_name": "child"}),
        (SecretRef, {"name": "credential"}),
        (SecretEnv, {"name": "TOKEN", "ref": SecretRef(name="credential")}),
        (ResolvedSecret, {"name": "credential", "value": "credential-value"}),
        (KnowledgeEntry, {"id": "entry", "text": "knowledge"}),
        (
            KnowledgeChunk,
            {"id": "chunk", "entry_id": "entry", "text": "knowledge", "chunk_index": 0},
        ),
        (KnowledgeIndexRequest, {"text": "knowledge"}),
        (TextEmbeddingUsage, {}),
        (TextEmbeddingResult, {"model": "model", "embeddings": [{"index": 0, "vector": [1.0]}]}),
        (ProxyAuthorizationResult, {"allowed": True}),
        (EvalAssertionResult, {"name": "assertion", "outcome": "error"}),
        (
            EvalCase,
            {
                "id": "case",
                "request": RunRequest(agent_name="agent", messages=[Message.text("user", "run")]),
            },
        ),
        (RunRequest, {"agent_name": "agent", "messages": [Message.text("user", "hi")]}),
        (ResumeRequest, {"session_id": "session", "messages": [Message.text("user", "hi")]}),
        (DispatchRequest, {"session_id": "session", "messages": [Message.text("user", "hi")]}),
        (InterruptSessionRequest, {"session_id": "session"}),
        (IncompleteSessionRecoveryRequest, {"session_id": "session"}),
        (IncompleteSessionsRecoveryRequest, {"statuses": ["running"]}),
        (TaskCreate, {"type": "test"}),
        (
            ToolRoundRecoveryRequest,
            {
                "session_id": "session",
                "round_id": "round",
                "tool_call_id": "call",
                "outcome": "failed",
                "message": "recovered",
            },
        ),
        (UserInputResponse, {"session_id": "session", "input_id": "input", "answer": "ok"}),
        (
            ToolApprovalRequest,
            {
                "session_id": "session",
                "approval_id": "approval",
                "tool_round_id": "round",
                "tool_call_id": "call",
                "decision": "approve",
            },
        ),
    ],
)
def test_public_metadata_models_share_exact_limit(
    model: type[BaseModel], fields: dict[str, Any], capsys, caplog
) -> None:
    limit = DURABLE_METADATA_LIMITS.max_bytes
    exact = _metadata_bytes(limit, character="€")
    accepted = model.model_validate({**fields, "metadata": exact})
    assert accepted.model_dump(mode="python")["metadata"] == exact
    canary = "private-metadata-canary"
    oversized = {**exact, canary: canary}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as caught:
            model.model_validate({**fields, "metadata": oversized})
    error = extract_durable_value_error(caught.value)
    assert error is not None
    assert error.dimension == "bytes"
    assert error.limit == limit
    assert error.observed_lower_bound > limit
    assert canary not in str(caught.value)
    assert canary not in repr(caught.value)
    assert not captured
    output = capsys.readouterr()
    assert canary not in output.out + output.err + caplog.text


@pytest.mark.parametrize("field", ["metadata", "chunk_metadata"])
def test_index_request_metadata_limit_precedes_indexing(field) -> None:
    exact = _metadata_bytes(DURABLE_METADATA_LIMITS.max_bytes)
    request = KnowledgeIndexRequest(text="knowledge", **{field: exact})
    assert request.model_dump(mode="python")[field] == exact
    with pytest.raises(ValidationError) as caught:
        KnowledgeIndexRequest(text="knowledge", **{field: {"value": exact["value"] + "x"}})
    error = extract_durable_value_error(caught.value)
    assert error is not None
    assert error.field_name == field
    assert error.limit == DURABLE_METADATA_LIMITS.max_bytes


def test_complete_metadata_merge_counts_retained_runtime_authority() -> None:
    user = _metadata_bytes(DURABLE_METADATA_LIMITS.max_bytes)
    # Use a runtime-owned prefix, not a guessed absence of a user marker.
    from cayu.runtime.sessions import SESSION_RUNTIME_METADATA_PREFIX

    key = SESSION_RUNTIME_METADATA_PREFIX + "limit_probe"
    merged = {**user, key: ""}
    overhead = len(json.dumps(merged, ensure_ascii=False, separators=(",", ":")).encode())
    merged[key] = "x" * (SESSION_METADATA_LIMITS.max_bytes - overhead)
    current = copy_session_metadata(merged)
    assert replace_session_user_metadata(current, user) == current
    with pytest.raises(DurableValueError) as caught:
        replace_session_user_metadata(current, {"value": user["value"] + "x"})
    assert caught.value.field_name == "session.metadata"
    assert caught.value.limit == SESSION_METADATA_LIMITS.max_bytes
    assert current == merged


@pytest.mark.parametrize("field", ["metadata", "reconnect_metadata"])
def test_factory_result_metadata_uses_shared_limit(field) -> None:
    from cayu.environments import Environment, EnvironmentFactoryResult, EnvironmentSpec

    environment = Environment(EnvironmentSpec(name="local"))
    exact = _metadata_bytes(DURABLE_METADATA_LIMITS.max_bytes, character="€")
    accepted = EnvironmentFactoryResult(environment=environment, **{field: exact})
    assert getattr(accepted, field) == exact
    with pytest.raises(DurableValueError) as caught:
        EnvironmentFactoryResult(
            environment=environment, **{field: {"value": exact["value"] + "x"}}
        )
    assert caught.value.limit == DURABLE_METADATA_LIMITS.max_bytes
    exact["value"] = "changed"
    assert getattr(accepted, field)["value"] != "changed"


def test_metadata_node_and_depth_limits_are_independent_of_bytes() -> None:
    assert copy_durable_metadata({"items": [None] * (50_000 - 2)})
    with pytest.raises(DurableValueError) as caught:
        copy_durable_metadata({"items": [None] * (50_000 - 1)})
    assert caught.value.dimension == "nodes"
    assert caught.value.limit == 50_000
    nested: Any = None
    for _ in range(63):
        nested = [nested]
    assert copy_durable_metadata({"value": nested})
    with pytest.raises(DurableValueError) as caught:
        copy_durable_metadata({"value": [nested]})
    assert caught.value.dimension == "depth"
    assert caught.value.limit == 64


@pytest.mark.parametrize("kind", ["model_completion", "tool_round"])
def test_reconstructed_request_metadata_obeys_shared_byte_limit(kind) -> None:
    from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
    from cayu.runtime._tool_round_recovery import PendingToolRound

    model = ModelCompletionRecoveryContext if kind == "model_completion" else PendingToolRound
    fields = (
        {}
        if kind == "model_completion"
        else {
            "tool_round_id": "tround_" + "1" * 32,
            "model_step_id": "mstep_" + "2" * 32,
            "model_attempt_id": "matt_" + "3" * 32,
            "agent_name": "agent",
            "tool_calls": [{"tool_call_id": "call", "tool_name": "tool"}],
        }
    )
    maximum_metadata_bytes = DURABLE_METADATA_LIMITS.max_bytes
    if kind == "model_completion":
        from cayu._validation import canonical_durable_json_bytes

        empty = model.model_validate({**fields, "request_metadata": {}})
        envelope_bytes = (
            len(canonical_durable_json_bytes(empty.model_dump(mode="json"), "context")) - 2
        )
        maximum_metadata_bytes -= envelope_bytes
    metadata = _metadata_bytes(maximum_metadata_bytes, character="€")
    accepted = model.model_validate({**fields, "request_metadata": metadata})
    assert model.model_validate_json(accepted.model_dump_json()) == accepted
    persisted = accepted.model_dump(mode="json")
    persisted["request_metadata"]["value"] += "x"
    with pytest.raises(ValidationError) as caught:
        model.model_validate_json(json.dumps(persisted, ensure_ascii=False))
    error = extract_durable_value_error(caught.value)
    assert error is not None and error.limit == DURABLE_METADATA_LIMITS.max_bytes
    assert error.field_name == (
        "model_completion_recovery_context" if kind == "model_completion" else "request_metadata"
    )


def test_model_recovery_preserves_stricter_metadata_entry_limit() -> None:
    from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext

    assert ModelCompletionRecoveryContext(
        request_metadata={str(index): None for index in range(256)}
    )
    with pytest.raises(ValidationError, match="more than 256 entries"):
        ModelCompletionRecoveryContext(request_metadata={str(index): None for index in range(257)})


@pytest.mark.parametrize("kind", ["execution_profile", "tool_capability"])
def test_runtime_metadata_helpers_recheck_complete_replacement(kind) -> None:
    from cayu.runtime.execution_profiles import (
        EXECUTION_PROFILE_METADATA_KEY,
        build_execution_profile_identity,
        execution_profile_metadata_after_adoption,
        execution_profile_session_metadata,
    )
    from cayu.runtime.tool_exposure import (
        ToolCapabilityCeiling,
        session_metadata_with_tool_capability_ceiling,
    )

    if kind == "execution_profile":
        profile = build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version="test",
            provider_name="provider",
            model="model",
            durable_system_prompt=None,
            direct_tools=[],
            tool_catalogue_revision="sha256:" + "c" * 64,
        )
        current = {EXECUTION_PROFILE_METADATA_KEY: execution_profile_session_metadata(profile)}

        def update(value):
            return execution_profile_metadata_after_adoption(value, profile)
    else:
        current = {}

        def update(value):
            return session_metadata_with_tool_capability_ceiling(value, ToolCapabilityCeiling())

    proposed = update({**current, "runtime_padding": ""})
    overhead = len(json.dumps(proposed, ensure_ascii=False, separators=(",", ":")).encode())
    current["runtime_padding"] = "x" * (SESSION_METADATA_LIMITS.max_bytes - overhead)
    accepted = update(current)
    assert (
        len(json.dumps(accepted, ensure_ascii=False, separators=(",", ":")).encode()) == 1024 * 1024
    )
    oversized = {**current, "runtime_padding": current["runtime_padding"] + "x"}
    before = dict(oversized)
    with pytest.raises(DurableValueError) as caught:
        update(oversized)
    assert caught.value.limit == SESSION_METADATA_LIMITS.max_bytes
    assert oversized == before


def test_label_map_aggregate_bytes_do_not_replace_character_limits() -> None:
    labels = {f"label-{index}": "😀" * 512 for index in range(200)}
    with pytest.raises(DurableValueError) as caught:
        copy_label_map(labels, "labels")
    assert caught.value.dimension == "bytes"
    assert caught.value.limit == 128 * 1024


def test_label_map_accepts_exact_entry_ceiling() -> None:
    labels = {f"label-{index}": "value" for index in range(SESSION_LABEL_MAX_ENTRIES)}
    assert len(copy_label_map(labels, "labels")) == SESSION_LABEL_MAX_ENTRIES


def test_limit_diagnostic_preserves_numeric_evidence_without_rejected_values() -> None:
    from cayu.runtime._diagnostics import exception_diagnostic

    with pytest.raises(DurableValueError) as caught:
        copy_durable_metadata(_metadata_bytes(DURABLE_METADATA_LIMITS.max_bytes + 1))
    payload = exception_diagnostic(caught.value).payload_fields()
    assert payload["durable_value_error_limit"] == DURABLE_METADATA_LIMITS.max_bytes
    assert payload["durable_value_error_observed_lower_bound"] > DURABLE_METADATA_LIMITS.max_bytes
    assert "x" * 32 not in json.dumps(payload)


@pytest.mark.parametrize("value", [True, -1, 2**64, "private-bound"])
def test_limit_diagnostic_rejects_mutated_numeric_authority(value) -> None:
    from cayu.runtime._diagnostics import exception_diagnostic

    error = DurableValueError("json_value_too_large", "metadata", limit=16, observed_lower_bound=17)
    error.limit = value
    payload = exception_diagnostic(error).payload_fields()
    assert "durable_value_error_limit" not in payload
    assert "durable_value_error_observed_lower_bound" not in payload
    assert "private-bound" not in json.dumps(payload)
