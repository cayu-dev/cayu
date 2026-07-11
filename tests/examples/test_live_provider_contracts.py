from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from cayu import Event, EventType

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import artifact_file_live  # noqa: E402
import context_counting_live  # noqa: E402


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
