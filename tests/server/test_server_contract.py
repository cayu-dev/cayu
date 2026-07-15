from __future__ import annotations

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.core.events import EVENT_ID_MAX_CHARS, Event, EventType
from cayu.server import create_server
from cayu.server.contracts import SSE_LAST_EVENT_ID_FORMAT
from cayu.server.sse import (
    SSE_ERROR_SESSION_ID_MAX_BYTES,
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_ERROR_TYPE_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SSE_REPLAY_START_MARKER_FORMAT,
    SseEventFrameTooLargeError,
    error_to_sse_message,
    event_to_sse_data,
    event_to_sse_message,
    parse_last_event_id,
)

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi-contract-summary.json"
_STREAMING_ROUTES = {
    "/api/run",
    "/api/resume",
    "/api/sessions/{session_id}/compact",
    "/api/sessions/{session_id}/interrupt",
    "/api/tool-approvals/resolve",
    "/api/tool-approvals/recover",
    "/api/tool-rounds/recover",
    "/api/user-input/resolve",
    "/api/user-input/recover",
}


def _client() -> TestClient:
    return TestClient(create_server(CayuApp(), dev=True))


def _normalize_schema_node(value):
    if isinstance(value, dict):
        return {key: _normalize_schema_node(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_schema_node(item) for item in value]
    return value


def _openapi_content_contract(content: dict) -> dict:
    return {
        media_type: _normalize_schema_node(media.get("schema", {}))
        for media_type, media in sorted(content.items())
    }


def _openapi_request_contract(request_body: dict) -> dict:
    result = {
        "content": _openapi_content_contract(request_body.get("content", {})),
    }
    if "required" in request_body:
        result["required"] = request_body["required"]
    return result


def _openapi_response_contract(responses: dict) -> dict:
    return {
        status: {
            "content": _openapi_content_contract(response.get("content", {})),
        }
        for status, response in sorted(responses.items())
    }


def _openapi_parameter_contract(parameters: list[dict]) -> list[dict]:
    return [
        {
            "name": parameter.get("name"),
            "in": parameter.get("in"),
            "required": parameter.get("required", False),
            "schema": _normalize_schema_node(parameter.get("schema", {})),
        }
        for parameter in sorted(
            parameters,
            key=lambda parameter: (parameter.get("in", ""), parameter.get("name", "")),
        )
    ]


def _openapi_contract_summary(schema: dict) -> dict:
    summary = {
        "info": schema["info"],
        "paths": {},
        "components": _normalize_schema_node(schema.get("components", {}).get("schemas", {})),
    }
    for path, path_item in schema["paths"].items():
        summary["paths"][path] = {}
        for method, operation in sorted(path_item.items()):
            if method not in {"delete", "get", "patch", "post", "put"}:
                continue
            summary["paths"][path][method] = {
                "operation_id": operation.get("operationId"),
                "parameters": _openapi_parameter_contract(operation.get("parameters", [])),
                "request_body": _openapi_request_contract(operation.get("requestBody", {})),
                "responses": _openapi_response_contract(operation.get("responses", {})),
            }
    return summary


def test_openapi_contract_summary_matches_snapshot() -> None:
    schema = _client().get("/openapi.json").json()
    expected = json.loads(_SNAPSHOT_PATH.read_text())

    assert _openapi_contract_summary(schema) == expected


def test_contract_endpoint_declares_versioning_sse_and_client_generation() -> None:
    response = _client().get("/api/contract")

    assert response.status_code == 200
    body = response.json()
    assert body["api_prefix"] == "/api"
    assert body["contract_version"] == "1"
    assert body["versioning"]["breaking_change_requires"] == [
        "openapi_snapshot_update",
        "client_regeneration",
        "migration_note",
    ]
    assert body["sse"]["content_type"] == "text/event-stream"
    assert body["sse"]["event_id_format"] == SSE_LAST_EVENT_ID_FORMAT
    assert body["sse"]["max_event_id_chars"] == EVENT_ID_MAX_CHARS
    assert body["sse"]["mutation_id_header"] == "Cayu-Mutation-ID"
    assert body["sse"]["mutation_acceptance_event_type"] == "server.mutation.accepted"
    assert body["sse"]["replay_start_marker_format"] == SSE_REPLAY_START_MARKER_FORMAT
    assert body["sse"]["unknown_event_marker_behavior"] == "reject"
    assert body["sse"]["event_data_schema"] == "SseEventEnvelope"
    assert body["sse"]["error_data_schema"] == "SseErrorEnvelope"
    assert body["sse"]["max_event_data_bytes"] == SSE_EVENT_DATA_MAX_BYTES
    assert body["sse"]["max_error_text_bytes"] == SSE_ERROR_TEXT_MAX_BYTES
    assert body["client_generation"] == {
        "openapi_url": "/openapi.json",
        "supported_targets": ["typescript", "python"],
        "source_of_truth": "openapi",
    }


def test_custom_api_path_updates_contract_and_openapi_paths() -> None:
    client = TestClient(create_server(CayuApp(), dev=True, api_path="/cayu/api"))

    response = client.get("/cayu/api/contract")

    assert response.status_code == 200
    assert response.json()["api_prefix"] == "/cayu/api"
    assert client.get("/api/contract").status_code == 404

    schema = client.get("/openapi.json").json()
    assert "/cayu/api/run" in schema["paths"]
    assert "/api/run" not in schema["paths"]


def test_streaming_routes_document_sse_response_contract() -> None:
    schema = _client().get("/openapi.json").json()
    components = schema["components"]["schemas"]

    assert "SseEventEnvelope" in components
    assert "SseErrorEnvelope" in components
    for path in _STREAMING_ROUTES:
        operation = schema["paths"][path]["post"]
        response = operation["responses"]["200"]
        assert sorted(response["content"]) == ["text/event-stream"]
        description = response["content"]["text/event-stream"]["schema"]["description"]
        assert "SseEventEnvelope" in description
        assert "SseErrorEnvelope" in description
        for status_code in ("404", "409", "500"):
            assert operation["responses"][status_code]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ApiErrorResponse"
            }


def test_artifact_routes_document_typed_errors_and_content_response() -> None:
    schema = _client().get("/openapi.json").json()
    operation = schema["paths"]["/api/artifacts/{artifact_id}/content"]["get"]

    artifact_store_parameter = next(
        parameter
        for parameter in operation["parameters"]
        if parameter["name"] == "artifact_store_id"
    )
    assert artifact_store_parameter["required"] is True
    assert artifact_store_parameter["schema"]["minLength"] == 1

    success = operation["responses"]["200"]
    assert success["content"]["application/octet-stream"]["schema"] == {
        "type": "string",
        "format": "binary",
    }
    assert set(success["headers"]) == {
        "Cache-Control",
        "Content-Disposition",
        "X-Content-Type-Options",
        "X-Cayu-Artifact-Id",
        "X-Cayu-Artifact-Store-Id",
    }
    for status_code in ("404", "409", "413", "500", "503"):
        assert operation["responses"][status_code]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ApiErrorResponse"
        }

    for path in ("/api/artifacts", "/api/artifacts/{artifact_id}"):
        responses = schema["paths"][path]["get"]["responses"]
        for status_code in ("404", "409", "500", "503"):
            assert responses[status_code]["content"]["application/json"]["schema"] == {
                "$ref": "#/components/schemas/ApiErrorResponse"
            }


def test_sse_serialization_matches_contract_envelope() -> None:
    event = Event(
        id="event_1",
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="session_1",
        agent_name="assistant",
        environment_name="production",
        workflow_name="review",
        tool_name="read_file",
        payload={"path": "README.md"},
    )

    data = json.loads(event_to_sse_data(event))

    assert data["id"] == "event_1"
    assert data["type"] == "tool.call.completed"
    assert data["session_id"] == "session_1"
    assert data["agent_name"] == "assistant"
    assert data["environment_name"] == "production"
    assert data["workflow_name"] == "review"
    assert data["tool_name"] == "read_file"
    assert data["payload"] == {"path": "README.md"}
    assert isinstance(data["timestamp"], str)
    assert event_to_sse_message(event)["id"] == "session_1:event_1"


def test_sse_replay_markers_distinguish_events_from_explicit_start() -> None:
    assert parse_last_event_id("session_1:event_1") == ("session_1", "event_1")
    assert parse_last_event_id("session_1:") == ("session_1", None)
    assert parse_last_event_id(
        "tenant:session_1:event_1",
        expected_session_id="tenant:session_1",
    ) == ("tenant:session_1", "event_1")
    assert parse_last_event_id(
        "tenant:session_1:",
        expected_session_id="tenant:session_1",
    ) == ("tenant:session_1", None)
    assert parse_last_event_id(":event_1") is None
    assert parse_last_event_id(" session_1:event_1") is None
    assert parse_last_event_id("session_1:event_1\n") is None
    assert parse_last_event_id(f"session_1:{'e' * (EVENT_ID_MAX_CHARS + 1)}") is None


def test_sse_event_frame_limit_rejects_before_serializing_durable_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = Event(
        id="event_large",
        type="custom.large",
        session_id="session_large",
        payload={"value": "x" * SSE_EVENT_DATA_MAX_BYTES},
    )

    def fail_if_serialized(*args: object, **kwargs: object) -> str:
        pytest.fail("oversized SSE payload reached json.dumps")

    monkeypatch.setattr("cayu.server.sse.json.dumps", fail_if_serialized)
    with pytest.raises(SseEventFrameTooLargeError) as captured:
        event_to_sse_message(event)

    assert captured.value.session_id == "session_large"
    assert captured.value.actual_bytes is None
    assert captured.value.max_bytes == SSE_EVENT_DATA_MAX_BYTES
    assert len(event.payload["value"]) == SSE_EVENT_DATA_MAX_BYTES


def test_sse_event_frame_preflight_matches_compact_utf8_encoding() -> None:
    event = Event(
        id="event_utf8",
        type="custom.utf8",
        session_id="session_utf8",
        payload={"value": 'é"\n😀'},
    )
    data = event_to_sse_data(event)
    data_bytes = len(data.encode("utf-8"))

    assert event_to_sse_message(event, max_data_bytes=data_bytes)["data"] == data
    with pytest.raises(SseEventFrameTooLargeError):
        event_to_sse_message(event, max_data_bytes=data_bytes - 1)


def test_sse_event_frame_preflight_counts_ascii_del_escape() -> None:
    event = Event(
        id="event_del",
        type="custom.utf8",
        session_id="session_del",
        payload={"value": "\x7f"},
    )
    data = event_to_sse_data(event)
    data_bytes = len(data.encode("utf-8"))

    assert "\\u007f" in data
    assert event_to_sse_message(event, max_data_bytes=data_bytes)["data"] == data
    with pytest.raises(SseEventFrameTooLargeError) as captured:
        event_to_sse_message(event, max_data_bytes=data_bytes - 1)

    assert captured.value.actual_bytes is None


def test_sse_event_frame_handles_lone_unicode_surrogates_safely() -> None:
    event = Event(
        id="event_surrogate",
        type="custom.utf8",
        session_id="session_surrogate",
        payload={"value": "\ud800"},
    )

    message = event_to_sse_message(event)

    assert json.loads(message["data"])["payload"] == {"value": "\ud800"}


def test_sse_error_frame_is_classified_and_utf8_bounded() -> None:
    message = error_to_sse_message(
        RuntimeError("raw secret must not be used"),
        kind="observer",
        code="observer_lagged",
        retryable=True,
        session_id="session_1",
        error_text="é" * SSE_ERROR_TEXT_MAX_BYTES,
    )
    data = json.loads(message["data"])

    assert message["event"] == "error"
    assert data["type"] == "stream.error"
    assert data["kind"] == "observer"
    assert data["code"] == "observer_lagged"
    assert data["retryable"] is True
    assert data["session_id"] == "session_1"
    assert data["error_type"] == "RuntimeError"
    assert "raw secret" not in data["error"]
    assert data["error"].endswith("... [truncated]")
    assert len(data["error"].encode("utf-8")) <= SSE_ERROR_TEXT_MAX_BYTES


def test_sse_error_frame_bounds_auxiliary_identity_fields() -> None:
    oversized_error_type = type("X" * (SSE_ERROR_TYPE_MAX_BYTES + 100), (RuntimeError,), {})
    message = error_to_sse_message(
        oversized_error_type("raw-secret"),
        kind="runtime",
        code="runtime_failed",
        retryable=False,
        session_id="s" * (SSE_ERROR_SESSION_ID_MAX_BYTES + 1),
    )
    data = json.loads(message["data"])

    assert len(data["error_type"].encode("utf-8")) <= SSE_ERROR_TYPE_MAX_BYTES
    assert data["error_type"].endswith("... [truncated]")
    assert data["session_id"] is None
    assert "raw-secret" not in data["error"]


def test_sse_error_frame_handles_lone_unicode_surrogates_safely() -> None:
    message = error_to_sse_message(
        RuntimeError("failed"),
        kind="runtime",
        code="runtime_failed",
        retryable=False,
        session_id="session_surrogate",
        error_text="bad \ud800 value",
    )

    assert json.loads(message["data"])["error"] == "bad � value"
