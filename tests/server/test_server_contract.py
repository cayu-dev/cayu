from __future__ import annotations

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.core.events import Event, EventType
from cayu.server import create_server
from cayu.server.contracts import SSE_LAST_EVENT_ID_FORMAT
from cayu.server.sse import event_to_sse_data, event_to_sse_message

_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi-contract-summary.json"
_STREAMING_ROUTES = {
    "/api/run",
    "/api/resume",
    "/api/sessions/{session_id}/interrupt",
    "/api/tool-approvals/resolve",
    "/api/tool-approvals/recover",
    "/api/user-input/resolve",
    "/api/user-input/recover",
}


def _client() -> TestClient:
    return TestClient(create_server(CayuApp()))


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
    assert body["sse"]["event_data_schema"] == "SseEventEnvelope"
    assert body["sse"]["error_data_schema"] == "SseErrorEnvelope"
    assert body["client_generation"] == {
        "openapi_url": "/openapi.json",
        "supported_targets": ["typescript", "python"],
        "source_of_truth": "openapi",
    }


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
