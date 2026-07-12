from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from jsonschema import Draft202012Validator

from cayu import Event, EventType, ScriptedModelProvider
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.providers import ModelStreamEvent, build_openai_payload
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage import KnowledgeEntry, KnowledgeHit, KnowledgeQuery, KnowledgeSearchResult

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_source(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_live_checks = _load_source("cayu_test_live_checks", EXAMPLES_DIR / "_live_checks.py")


def _load_example(name: str) -> ModuleType:
    previous = sys.modules.get("_live_checks")
    sys.modules["_live_checks"] = _live_checks
    try:
        return _load_source(f"cayu_test_{name}", EXAMPLES_DIR / f"{name}.py")
    finally:
        if previous is None:
            sys.modules.pop("_live_checks", None)
        else:
            sys.modules["_live_checks"] = previous


artifact_file_live = _load_example("artifact_file_live")
bedrock_provider_live = _load_example("bedrock_provider_live")
context_counting_live = _load_example("context_counting_live")
knowledge_embedding_live = _load_example("knowledge_embedding_live")
real_spend_budget_live = _load_example("real_spend_budget_live")
structured_output_live = _load_example("structured_output_live")

DEMO_ONLY_EXAMPLES = (
    "context_pressure_calibration_live",
    "knowledge_recall_live",
    "knowledge_recall_many_live",
    "subagent_live",
    "subagent_parallel_live",
)


def test_live_example_imports_do_not_modify_process_module_search_path() -> None:
    assert str(EXAMPLES_DIR) not in sys.path


@pytest.mark.parametrize("example_name", DEMO_ONLY_EXAMPLES)
def test_demo_only_live_example_imports_and_exposes_main_without_sys_path_mutation(
    example_name: str,
) -> None:
    example = _load_example(example_name)

    assert callable(example.main)
    assert str(EXAMPLES_DIR) not in sys.path


@pytest.mark.parametrize(
    ("input_count", "token_counting"),
    [(9, "validated"), (None, "unsupported")],
)
def test_bedrock_live_contract_requires_structured_output_usage_and_completion(
    input_count: int | None,
    token_counting: str,
) -> None:
    evidence = bedrock_provider_live._validate_events(
        [
            _event(
                EventType.MODEL_COMPLETED,
                payload={
                    "usage_metrics": {
                        "provider_name": "bedrock",
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    }
                },
            ),
            _event(
                EventType.STRUCTURED_OUTPUT_VALIDATED,
                payload={"output": {"answer": "bedrock-live"}},
            ),
            _event(EventType.SESSION_COMPLETED),
        ],
        model="anthropic.claude-test",
        input_count=input_count,
    )

    assert evidence == {
        "provider": "bedrock",
        "model": "anthropic.claude-test",
        "input_count": input_count,
        "token_counting": token_counting,
        "total_tokens": 12,
        "structured_output": "validated",
    }


def _event(
    event_type: EventType,
    *,
    payload: dict | None = None,
    tool_name: str | None = None,
) -> Event:
    return Event(
        type=event_type,
        session_id="live-contract-session",
        tool_name=tool_name,
        payload={} if payload is None else payload,
    )


def _context_contract_events() -> list[Event]:
    return [
        _event(
            EventType.CONTEXT_COUNTED,
            payload={
                "provider": "openai",
                "model": "test-model",
                "observation_id": "count-1",
                "count": {
                    "method": "official",
                    "confidence": "high",
                    "input_tokens": 12,
                },
            },
        ),
        _event(
            EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "requested_model": "test-model",
                    "model": "test-model-2026-01-01",
                    "input_tokens": 15,
                    "output_tokens": 2,
                    "total_tokens": 17,
                }
            },
        ),
        _event(
            EventType.CONTEXT_COUNT_RECONCILED,
            payload={
                "observation_id": "count-1",
                "actual_input_tokens": 15,
                "delta_tokens": 3,
                "reconciled": True,
            },
        ),
        _event(EventType.SESSION_COMPLETED),
    ]


def test_context_counting_contract_requires_successful_terminal_and_usage() -> None:
    context_counting_live._validate_runtime_events(_context_contract_events())


def test_context_counting_contract_rejects_missing_session_completion() -> None:
    with pytest.raises(RuntimeError, match="session.completed"):
        context_counting_live._validate_runtime_events(_context_contract_events()[:-1])


def test_context_counting_contract_rejects_failure_terminal() -> None:
    events = [
        *_context_contract_events()[:-1],
        _event(EventType.MODEL_ERROR, payload={"error": "provider failed"}),
        _event(EventType.SESSION_FAILED, payload={"error": "provider failed"}),
    ]

    with pytest.raises(RuntimeError, match="model.error"):
        context_counting_live._validate_runtime_events(events)


def _artifact_contract_events(artifact_id: str = "artifact-1") -> list[Event]:
    return [
        _event(
            EventType.TOOL_CALL_COMPLETED,
            tool_name="read_file",
            payload={
                "result": {
                    "content": "attached image",
                    "artifacts": [
                        {
                            "type": "cayu.file_attachment.v1",
                            "artifact_id": artifact_id,
                            "kind": "image",
                            "filename": "red-dot.png",
                            "content_type": "image/png",
                            "size_bytes": 69,
                            "metadata": {"source_artifact_id": artifact_id},
                        }
                    ],
                }
            },
        ),
        _event(
            EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "requested_model": "test-model",
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "total_tokens": 24,
                }
            },
        ),
        _event(EventType.SESSION_COMPLETED),
    ]


def test_artifact_file_contract_requires_read_usage_and_completion() -> None:
    artifact_file_live._validate_runtime_events(
        _artifact_contract_events(),
        artifact_id="artifact-1",
    )


def test_artifact_file_contract_allows_recovered_tool_error_before_success() -> None:
    artifact_file_live._validate_runtime_events(
        [
            _event(
                EventType.TOOL_CALL_FAILED,
                tool_name="read_file",
                payload={"result": {"content": "invalid arguments", "is_error": True}},
            ),
            *_artifact_contract_events(),
        ],
        artifact_id="artifact-1",
    )


def test_artifact_file_contract_rejects_wrong_artifact() -> None:
    with pytest.raises(RuntimeError, match="artifact-1"):
        artifact_file_live._validate_runtime_events(
            _artifact_contract_events("artifact-other"),
            artifact_id="artifact-1",
        )


def test_artifact_file_contract_rejects_non_attachment_artifact_reference() -> None:
    events = _artifact_contract_events()
    result_artifact = events[0].payload["result"]["artifacts"][0]
    result_artifact["type"] = "cayu.artifact_reference.v1"

    with pytest.raises(RuntimeError, match="file attachment"):
        artifact_file_live._validate_runtime_events(
            events,
            artifact_id="artifact-1",
        )


def test_artifact_file_contract_fails_when_file_readers_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_file_live, "_has_file_reader_dependencies", lambda: False)

    with pytest.raises(SystemExit, match="optional file readers"):
        asyncio.run(artifact_file_live.main())


def test_artifact_file_contract_fails_when_provider_configuration_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_file_live, "_has_file_reader_dependencies", lambda: True)
    monkeypatch.setattr(
        artifact_file_live,
        "_provider_config",
        lambda: (_ for _ in ()).throw(RuntimeError("provider configuration missing")),
    )

    with pytest.raises(SystemExit, match="provider configuration missing"):
        asyncio.run(artifact_file_live.main())


def test_structured_output_contract_validates_expected_invoice_and_usage() -> None:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_structured_output",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": structured_output_live.EXPECTED_OUTPUT},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "model": "structured-live-model-2026-01-01",
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 12,
                        "total_tokens": 42,
                    },
                }
            ),
        ],
        name="structured-live-test",
    )

    evidence = asyncio.run(
        structured_output_live._run_contract(
            provider=provider,
            provider_name="structured-live-test",
            model="structured-live-model",
            strategy=structured_output_live.StructuredOutputStrategy.TOOL,
        )
    )

    assert evidence == {
        "provider": "structured-live-test",
        "model": "structured-live-model",
        "resolved_model": "structured-live-model-2026-01-01",
        "strategy": "tool",
        "invoice_number": "INV-1042",
        "invoice_status": "paid",
        "total_tokens": 42,
    }
    prompt = provider.requests[0].messages[-1].content[0].text
    assert "Use the exact line-item description 'Managed hosting'." in prompt


def test_structured_output_schema_rejects_paraphrased_line_item_description() -> None:
    output = copy.deepcopy(structured_output_live.EXPECTED_OUTPUT)
    output["invoice"]["line_items"][0]["description"] = "Managed hosting for 125.50 USD"

    errors = list(Draft202012Validator(structured_output_live.INVOICE_SCHEMA).iter_errors(output))

    assert len(errors) == 1
    assert list(errors[0].path) == ["invoice", "line_items", 0, "description"]


def test_structured_output_contract_rejects_semantically_wrong_valid_output() -> None:
    events = [
        _event(
            EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "structured-live-test",
                    "requested_model": "structured-live-model",
                    "input_tokens": 30,
                    "output_tokens": 12,
                    "total_tokens": 42,
                }
            },
        ),
        _event(
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            payload={
                "output": {
                    **structured_output_live.EXPECTED_OUTPUT,
                    "invoice": {
                        **structured_output_live.EXPECTED_OUTPUT["invoice"],
                        "status": "unpaid",
                    },
                }
            },
        ),
        _event(EventType.SESSION_COMPLETED),
    ]

    with pytest.raises(RuntimeError, match="validated invoice output"):
        structured_output_live._validate_runtime_events(
            events,
            provider_name="structured-live-test",
            model="structured-live-model",
            strategy=structured_output_live.StructuredOutputStrategy.TOOL,
        )


def test_structured_output_contract_rejects_unexpected_resolved_model() -> None:
    events = [
        _event(
            EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": "structured-live-test",
                    "requested_model": "structured-live-model",
                    "model": "structured-live-model-mini",
                    "input_tokens": 30,
                    "output_tokens": 12,
                    "total_tokens": 42,
                }
            },
        ),
        _event(
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            payload={"output": structured_output_live.EXPECTED_OUTPUT},
        ),
        _event(EventType.SESSION_COMPLETED),
    ]

    with pytest.raises(RuntimeError, match="resolved model"):
        structured_output_live._validate_runtime_events(
            events,
            provider_name="structured-live-test",
            model="structured-live-model",
            strategy=structured_output_live.StructuredOutputStrategy.TOOL,
        )


@pytest.mark.parametrize(
    ("requested", "resolved"),
    [
        ("gpt-5.5", "gpt-5.5"),
        ("gpt-5.5", "gpt-5.5-2026-01-01"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6-20250514"),
    ],
)
def test_structured_output_contract_accepts_exact_or_date_versioned_model(
    requested: str,
    resolved: str,
) -> None:
    assert structured_output_live._resolved_model_matches(requested, resolved) is True


def test_structured_output_contract_fails_when_provider_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_PROVIDER", raising=False)

    with pytest.raises(SystemExit, match="Set OPENAI_API_KEY"):
        asyncio.run(structured_output_live.main())


class _KeywordEmbeddingProvider(TextEmbeddingProvider):
    name = "embedding-live-test"

    def __init__(self) -> None:
        self.calls: list[TextEmbeddingRequest] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(request)
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=_embedding_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


def _embedding_vector(text: str) -> list[float]:
    folded = text.casefold()
    return [
        1.0
        if any(
            term in folded for term in ("auth", "broker", "credential", "github", "proxy", "git")
        )
        else 0.0,
        1.0 if any(term in folded for term in ("invoice", "payment", "refund")) else 0.0,
        1.0 if any(term in folded for term in ("sendgrid", "email")) else 0.0,
    ]


def test_knowledge_embedding_contract_requires_expected_semantic_top_hit() -> None:
    provider = _KeywordEmbeddingProvider()

    evidence = asyncio.run(
        knowledge_embedding_live._run_contract(
            provider=provider,
            embedding_model="embedding-live-model",
            dimensions=None,
        )
    )

    assert evidence["provider"] == "embedding-live-test"
    assert evidence["embedding_model"] == "embedding-live-model"
    assert evidence["top_entry_id"] == "remote_git_credentials"
    assert evidence["score_kind"] == "inmemory_semantic"
    assert evidence["score_normalized"] == 1.0
    assert evidence["hit_count"] > 0
    assert provider.calls
    assert {request.model for request in provider.calls} == {"embedding-live-model"}
    assert provider.calls[-1].texts == [knowledge_embedding_live.QUERY]


def test_knowledge_embedding_contract_rejects_wrong_top_hit() -> None:
    result = KnowledgeSearchResult(
        query=KnowledgeQuery(
            text=knowledge_embedding_live.QUERY,
            limit=5,
            max_bytes=4_000,
        ),
        hits=[
            KnowledgeHit(
                entry=KnowledgeEntry(id="wrong", text="Wrong entry."),
                score=1.0,
                score_kind="inmemory_semantic",
                score_normalized=1.0,
                reason="semantic entry match",
                rank=1,
            )
        ],
        limit=5,
        max_bytes=4_000,
        total_hits_known=1,
    )

    with pytest.raises(RuntimeError, match="remote_git_credentials"):
        knowledge_embedding_live._validate_search_result(result)


def test_knowledge_embedding_validator_trusts_store_threshold_for_returned_hit() -> None:
    result = KnowledgeSearchResult(
        query=KnowledgeQuery(
            text=knowledge_embedding_live.QUERY,
            limit=5,
            max_bytes=4_000,
        ),
        hits=[
            KnowledgeHit(
                entry=KnowledgeEntry(
                    id="remote_git_credentials",
                    text="Use a brokered Git HTTP proxy.",
                ),
                score=1.0,
                score_kind="inmemory_semantic",
                score_normalized=None,
                reason="semantic entry match",
                rank=1,
            )
        ],
        limit=5,
        max_bytes=4_000,
        total_hits_known=1,
    )

    top_hit = knowledge_embedding_live._validate_search_result(result)

    assert top_hit.entry.id == "remote_git_credentials"


def test_knowledge_embedding_contract_fails_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_PROVIDER", raising=False)

    with pytest.raises(SystemExit, match="Set OPENAI_API_KEY"):
        asyncio.run(knowledge_embedding_live.main())


def test_real_spend_budget_contract_allows_one_call_then_rejects_next_reservation() -> None:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("bounded response"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                    },
                }
            ),
        ],
        name="budget-live-test",
    )

    evidence = asyncio.run(
        real_spend_budget_live._run_contract(
            provider=provider,
            provider_name="budget-live-test",
            model="budget-live-model",
            provider_options=real_spend_budget_live.OPENAI_PROVIDER_OPTIONS,
        )
    )

    request_payload = build_openai_payload(provider.requests[0])
    assert request_payload["max_output_tokens"] == real_spend_budget_live.MAX_OUTPUT_TOKENS

    assert evidence == {
        "provider": "budget-live-test",
        "model": "budget-live-model",
        "currency": "USD",
        "max_estimated_cost": "0.00104",
        "reserved_amount": "0.001040",
        "actual_estimated_cost": "0.000025",
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
        "provider_calls_attempted": 1,
        "provider_calls_completed": 1,
        "enforcement": "second_reservation_rejected_before_provider",
    }


def test_real_spend_budget_contract_fails_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="Set OPENAI_API_KEY"):
        asyncio.run(real_spend_budget_live.main())


def test_real_spend_budget_contract_rejects_incorrect_reconciliation() -> None:
    first_events = [
        _event(EventType.BUDGET_RESERVED, payload={"actual": "0.001040"}),
        _event(EventType.MODEL_STARTED),
        _event(
            EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                }
            },
        ),
        _event(EventType.BUDGET_RECONCILED, payload={"actual_amount": "0.000099"}),
        _event(EventType.SESSION_COMPLETED),
    ]
    second_events = [
        _event(EventType.BUDGET_RESERVATION_FAILED),
        _event(EventType.SESSION_INTERRUPTED),
    ]

    with pytest.raises(RuntimeError, match="reconciled amount"):
        real_spend_budget_live._validate_contract(
            first_events,
            second_events,
            provider_name="budget-live-test",
            model="budget-live-model",
        )
