from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from cayu import Event, EventType, ScriptedModelProvider
from cayu.providers import ModelStreamEvent, build_openai_payload

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
context_counting_live = _load_example("context_counting_live")
real_spend_budget_live = _load_example("real_spend_budget_live")


def test_live_example_imports_do_not_modify_process_module_search_path() -> None:
    assert str(EXAMPLES_DIR) not in sys.path


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
