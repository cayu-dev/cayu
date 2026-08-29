from __future__ import annotations

# ruff: noqa: E402
import json
from hashlib import sha256
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    LocalArtifactStore,
    default_price_book,
)
from cayu.core.events import EVENT_ID_MAX_CHARS, Event, EventType
from cayu.runtime import InMemorySessionStore
from cayu.server import (
    AuthContext,
    DashboardConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
)
from cayu.server.contracts import (
    SERVER_CONTRACT_VERSION,
    SSE_LAST_EVENT_ID_FORMAT,
    CapabilityOperation,
    EvalsOperationReadiness,
)
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


class _TestKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("access_scope", KnowledgeAccessScope.privileged())
        super().__init__(*args, **kwargs)


_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "openapi-contract-summary.json"
_STREAMING_ROUTES = {
    "/api/run",
    "/api/resume",
    "/api/provider-operations/resolve",
    "/api/sessions/{session_id}/compact",
    "/api/sessions/{session_id}/messages",
    "/api/sessions/{session_id}/interrupt",
    "/api/tool-approvals/resolve",
    "/api/tool-approvals/recover",
    "/api/tool-rounds/recover",
    "/api/user-input/resolve",
    "/api/user-input/recover",
}


class _UncalledEnvironmentFactory(EnvironmentFactory):
    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        del request
        raise AssertionError("Capability discovery must not materialize environment factories.")


def _client() -> TestClient:
    return TestClient(create_server(CayuApp(), config=ServerConfig.local_development()))


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
    assert body["contract_version"] == SERVER_CONTRACT_VERSION
    assert body["versioning"]["contract_version"] == SERVER_CONTRACT_VERSION
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
    assert body["capabilities"]["configured_store_roles"] == ["session"]
    assert body["capabilities"]["actor"] is None
    assert body["capabilities"]["surfaces"] == {
        "dashboard": {
            "configured": True,
            "read": {"enabled": True, "unavailable_reason": None},
            "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
        },
        "workflow": {
            "configured": True,
            "read": {"enabled": True, "unavailable_reason": None},
            "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
        },
        "tasks": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "not_configured"},
        },
        "reviewed_knowledge": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "not_configured"},
        },
        "artifacts": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
        },
        "usage": {
            "configured": True,
            "read": {"enabled": True, "unavailable_reason": None},
            "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
        },
        "pricing": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
        },
        "evaluation_promotion": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "not_configured"},
        },
        "evals": {
            "configured": False,
            "read": {"enabled": False, "unavailable_reason": "not_configured"},
            "mutate": {"enabled": False, "unavailable_reason": "not_configured"},
        },
    }
    assert body["capabilities"]["mutations"] == {
        "session_execution": {"enabled": True, "unavailable_reason": None},
        "session_interruption": {"enabled": True, "unavailable_reason": None},
        "provider_operation_resolution": {"enabled": True, "unavailable_reason": None},
        "pending_action_resolution": {"enabled": True, "unavailable_reason": None},
        "session_annotations": {"enabled": True, "unavailable_reason": None},
        "task_lifecycle": {"enabled": False, "unavailable_reason": "not_configured"},
        "knowledge_review": {"enabled": False, "unavailable_reason": "not_configured"},
    }
    assert body["capabilities"]["evals_readiness"] == {
        "captured_evaluation": {
            "state": "gated",
            "reason_code": "eval_target_not_configured",
        },
        "catalog_read": {"state": "gated", "reason_code": "eval_store_not_configured"},
        "catalog_write": {"state": "gated", "reason_code": "eval_store_not_configured"},
        "captured_result_persistence": {
            "state": "gated",
            "reason_code": "eval_store_not_configured",
        },
        "scenario_conversion": {
            "state": "gated",
            "reason_code": "eval_target_not_configured",
        },
        "fresh_launch": {"state": "gated", "reason_code": "eval_target_not_configured"},
        "cancellation": {"state": "gated", "reason_code": "eval_store_not_configured"},
        "comparison": {"state": "gated", "reason_code": "eval_store_not_configured"},
        "reports": {"state": "gated", "reason_code": "eval_store_not_configured"},
    }


def test_contract_reports_configured_optional_capabilities_and_redacted_actor(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="contract-artifacts")
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        knowledge_store=_TestKnowledgeStore(),
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="contract-environment"),
            artifact_store=artifact_store,
        )
    )

    auth_calls = 0

    def authenticate(_request) -> AuthContext:
        nonlocal auth_calls
        auth_calls += 1
        return AuthContext(
            subject="operator-a",
            tenant="tenant-a",
            claims={"credential": "must-not-appear"},
        )

    client = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                authenticate,
                dashboard=DashboardConfig(
                    runtime_config={
                        "priceBook": default_price_book(),
                        "privateToken": "dashboard-secret-must-not-appear",
                    }
                ),
            ),
        )
    )

    response = client.get("/api/contract")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert auth_calls == 1
    capabilities = response.json()["capabilities"]
    assert capabilities["configured_store_roles"] == [
        "session",
        "task",
        "knowledge",
        "artifact",
    ]
    assert capabilities["actor"] == {"subject": "operator-a", "tenant": "tenant-a"}
    assert "must-not-appear" not in response.text
    for name, surface in capabilities["surfaces"].items():
        if name in {"evaluation_promotion", "evals"}:
            assert surface == {
                "configured": False,
                "read": {"enabled": False, "unavailable_reason": "not_configured"},
                "mutate": {"enabled": False, "unavailable_reason": "not_configured"},
            }
            continue
        assert surface["configured"] is True
        assert surface["read"] == {"enabled": True, "unavailable_reason": None}
    assert capabilities["surfaces"]["artifacts"]["mutate"] == {
        "enabled": False,
        "unavailable_reason": "unsupported",
    }
    assert capabilities["surfaces"]["usage"]["mutate"] == {
        "enabled": False,
        "unavailable_reason": "unsupported",
    }
    assert capabilities["surfaces"]["pricing"]["mutate"] == {
        "enabled": False,
        "unavailable_reason": "unsupported",
    }
    assert capabilities["mutations"]["task_lifecycle"]["enabled"] is True
    assert capabilities["mutations"]["knowledge_review"]["enabled"] is True


def test_system_diagnostics_reports_bounded_protected_runtime_state(tmp_path) -> None:
    raw_store_id = str(tmp_path / "private-artifact-path")
    artifact_store = LocalArtifactStore(
        tmp_path / "artifacts",
        store_id=raw_store_id,
    )
    app = CayuApp(
        task_store=InMemoryTaskStore(),
        knowledge_store=_TestKnowledgeStore(),
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="diagnostic-environment"),
            artifact_store=artifact_store,
        )
    )
    price_book = default_price_book()
    auth_calls = 0

    def authenticate(_request) -> AuthContext:
        nonlocal auth_calls
        auth_calls += 1
        return AuthContext(
            subject="operator-a",
            tenant="tenant-a",
            claims={"credential": "diagnostic-secret-must-not-appear"},
        )

    client = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                authenticate,
                deployment_name="production-eu",
                dashboard=DashboardConfig(
                    runtime_config={
                        "priceBook": price_book,
                        "privateToken": "dashboard-secret-must-not-appear",
                    }
                ),
            ),
        )
    )

    response = client.get("/api/system/diagnostics")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert auth_calls == 1
    body = response.json()
    assert body["deployment"] == {
        "name": "production-eu",
        "name_status": "available",
        "api_access": "authenticated",
        "dashboard_access": "authenticated",
        "dashboard_enabled": True,
        "docs_enabled": False,
    }
    assert body["versions"]["server_contract"] == SERVER_CONTRACT_VERSION
    assert body["versions"]["cayu"] == body["capabilities"]["cayu_version"]
    assert body["capabilities"]["actor"] == {
        "subject": "operator-a",
        "tenant": "tenant-a",
    }
    assert body["artifact_stores"] == {
        "registrations": [
            {
                "fingerprint": f"sha256:{sha256(raw_store_id.encode()).hexdigest()}",
                "store_contract_operations": ["list", "read", "write", "delete"],
            }
        ],
        "total_count": 1,
        "truncated": False,
    }
    assert body["pricing_catalog"] == {
        "configured": True,
        "metadata_status": "available",
        "price_book_version": price_book.price_book_version,
        "generated_at": price_book.generated_at,
    }
    assert raw_store_id not in response.text
    assert "diagnostic-secret-must-not-appear" not in response.text
    assert "dashboard-secret-must-not-appear" not in response.text


def test_system_diagnostics_bounds_dynamic_artifact_registrations(tmp_path) -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=ServerConfig.local_development()))

    before = client.get("/api/system/diagnostics").json()
    assert before["artifact_stores"] == {
        "registrations": [],
        "total_count": 0,
        "truncated": False,
    }
    assert before["capabilities"]["surfaces"]["artifacts"]["configured"] is False

    raw_store_ids = []
    for index in range(65):
        store_id = str(tmp_path / f"artifact-store-{index}")
        raw_store_ids.append(store_id)
        app.register_environment(
            Environment(
                EnvironmentSpec(name=f"diagnostic-environment-{index}"),
                artifact_store=LocalArtifactStore(
                    tmp_path / f"artifacts-{index}",
                    store_id=store_id,
                ),
            )
        )

    response = client.get("/api/system/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["artifact_stores"]["total_count"] == 65
    assert len(body["artifact_stores"]["registrations"]) == 64
    assert body["artifact_stores"]["truncated"] is True
    assert body["capabilities"]["surfaces"]["artifacts"]["configured"] is True
    assert body["artifact_stores"]["registrations"][0]["fingerprint"] == (
        f"sha256:{sha256(raw_store_ids[0].encode()).hexdigest()}"
    )
    assert all(store_id not in response.text for store_id in raw_store_ids)


def test_system_diagnostics_omits_oversized_optional_provenance() -> None:
    price_book = default_price_book().model_dump(mode="json")
    price_book["price_book_version"] = "v" * 257
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                deployment_name="d" * 129,
                dashboard=DashboardConfig(runtime_config={"priceBook": price_book}),
            ),
        )
    )

    response = client.get("/api/system/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["deployment"]["name"] is None
    assert body["deployment"]["name_status"] == "omitted"
    assert body["pricing_catalog"] == {
        "configured": True,
        "metadata_status": "omitted",
        "price_book_version": None,
        "generated_at": None,
    }
    assert "d" * 129 not in response.text
    assert "v" * 257 not in response.text


@pytest.mark.parametrize("registration_kind", ["concrete", "factory"])
def test_contract_reflects_artifact_store_registered_after_server_construction(
    registration_kind: str,
    tmp_path,
) -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=ServerConfig.local_development()))
    artifact_store = LocalArtifactStore(
        tmp_path / registration_kind,
        store_id=f"late-{registration_kind}-artifacts",
    )

    before = client.get("/api/contract").json()["capabilities"]
    assert before["surfaces"]["artifacts"]["configured"] is False
    assert before["configured_store_roles"] == ["session"]

    if registration_kind == "concrete":
        app.register_environment(
            Environment(
                EnvironmentSpec(name="late-concrete-environment"),
                artifact_store=artifact_store,
            )
        )
    else:
        app.register_environment_factory(
            EnvironmentSpec(name="late-factory-environment"),
            _UncalledEnvironmentFactory(),
            artifact_store=artifact_store,
        )

    after = client.get("/api/contract").json()["capabilities"]
    assert after["surfaces"]["artifacts"] == {
        "configured": True,
        "read": {"enabled": True, "unavailable_reason": None},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    assert after["configured_store_roles"] == ["session", "artifact"]


@pytest.mark.parametrize(
    "configured_feature", ["tasks", "reviewed_knowledge", "artifacts", "pricing"]
)
def test_contract_keeps_optional_capability_combinations_independent(
    configured_feature: str,
    tmp_path,
) -> None:
    app = CayuApp(
        task_store=InMemoryTaskStore() if configured_feature == "tasks" else None,
        knowledge_store=(
            _TestKnowledgeStore() if configured_feature == "reviewed_knowledge" else None
        ),
    )
    if configured_feature == "artifacts":
        app.register_environment(
            Environment(
                EnvironmentSpec(name="artifact-environment"),
                artifact_store=LocalArtifactStore(
                    tmp_path / "artifacts",
                    store_id="independent-artifacts",
                ),
            )
        )
    dashboard = DashboardConfig(
        runtime_config=(
            {"priceBook": default_price_book()} if configured_feature == "pricing" else {}
        )
    )

    capabilities = (
        TestClient(
            create_server(
                app,
                config=ServerConfig.local_development(dashboard=dashboard),
            )
        )
        .get("/api/contract")
        .json()["capabilities"]
    )

    optional_surface_names = {"tasks", "reviewed_knowledge", "artifacts", "pricing"}
    configured_surfaces = {
        name
        for name, capability in capabilities["surfaces"].items()
        if name in optional_surface_names and capability["configured"]
    }
    assert configured_surfaces == {configured_feature}
    expected_roles = {
        "tasks": ["session", "task"],
        "reviewed_knowledge": ["session", "knowledge"],
        "artifacts": ["session", "artifact"],
        "pricing": ["session"],
    }
    assert capabilities["configured_store_roles"] == expected_roles[configured_feature]
    assert capabilities["surfaces"]["usage"] == {
        "configured": True,
        "read": {"enabled": True, "unavailable_reason": None},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    assert capabilities["mutations"]["task_lifecycle"]["enabled"] is (configured_feature == "tasks")
    assert capabilities["mutations"]["knowledge_review"]["enabled"] is (
        configured_feature == "reviewed_knowledge"
    )


def test_contract_reports_disabled_dashboard_without_inventing_pricing_availability() -> None:
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(
                    enabled=False,
                    runtime_config={"priceBook": default_price_book()},
                )
            ),
        )
    )

    surfaces = client.get("/api/contract").json()["capabilities"]["surfaces"]

    assert surfaces["dashboard"] == {
        "configured": False,
        "read": {"enabled": False, "unavailable_reason": "not_configured"},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    assert surfaces["pricing"]["configured"] is False
    diagnostics = client.get("/api/system/diagnostics").json()
    assert diagnostics["deployment"]["dashboard_enabled"] is False
    assert diagnostics["deployment"]["dashboard_access"] is None
    assert diagnostics["pricing_catalog"] == {
        "configured": False,
        "metadata_status": "not_configured",
        "price_book_version": None,
        "generated_at": None,
    }


def test_contract_rejects_an_invalid_session_store_capability_declaration() -> None:
    class InvalidCapabilityStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_usage_aggregates = "yes"  # type: ignore[assignment]

    with pytest.raises(
        TypeError,
        match="session_usage_aggregates_supported must be a bool",
    ):
        create_server(
            CayuApp(session_store=InvalidCapabilityStore()),
            config=ServerConfig.local_development(),
        )


def test_contract_rejects_an_invalid_session_topology_capability_declaration() -> None:
    class InvalidCapabilityStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_session_topology = "yes"  # type: ignore[assignment]

    with pytest.raises(
        TypeError,
        match="session_topology_supported must be a bool",
    ):
        create_server(
            CayuApp(session_store=InvalidCapabilityStore()),
            config=ServerConfig.local_development(),
        )


def test_capability_operation_rejects_inconsistent_availability() -> None:
    with pytest.raises(ValueError, match="cannot have an unavailable reason"):
        CapabilityOperation(enabled=True, unavailable_reason="unsupported")
    with pytest.raises(ValueError, match="require an unavailable reason"):
        CapabilityOperation(enabled=False)


def test_evals_operation_readiness_rejects_inconsistent_reason_codes() -> None:
    with pytest.raises(ValueError, match="Ready Evals operations cannot have a reason code"):
        EvalsOperationReadiness(
            state="ready",
            reason_code="eval_store_not_configured",
        )
    with pytest.raises(ValueError, match="Unavailable Evals operations require a reason code"):
        EvalsOperationReadiness(state="gated", reason_code=None)
    with pytest.raises(ValueError, match="Gated Evals operations require a gated reason code"):
        EvalsOperationReadiness(
            state="gated",
            reason_code="scenario_v2_not_available",
        )
    with pytest.raises(
        ValueError,
        match="Unsupported Evals operations require an unsupported reason code",
    ):
        EvalsOperationReadiness(
            state="unsupported",
            reason_code="eval_target_not_configured",
        )

    assert (
        EvalsOperationReadiness(
            state="unsupported",
            reason_code="session_lineage_not_supported",
        ).reason_code
        == "session_lineage_not_supported"
    )


def test_openapi_declares_auth_tenant_as_provenance_only() -> None:
    schema = _client().get("/openapi.json").json()
    operation = schema["paths"]["/api/contract"]["get"]
    auth_context_schema = operation["x-cayu-auth-context"]

    assert auth_context_schema["title"] == "AuthContext"
    assert auth_context_schema["additionalProperties"] is False
    assert auth_context_schema["required"] == ["subject"]
    auth_context_description = " ".join(auth_context_schema["description"].split())
    assert "operator-action provenance" in auth_context_description
    assert "does not scope sessions" in auth_context_description

    tenant_description = auth_context_schema["properties"]["tenant"]["description"]
    assert (
        tenant_description == AuthContext.model_json_schema()["properties"]["tenant"]["description"]
    )
    for warning in (
        "provenance only",
        "not a storage partition",
        "authorization rule",
        "row-level filter",
        "tenant-isolation primitive",
        "does not scope Cayu data",
    ):
        assert warning in tenant_description

    operation_description = operation["description"]
    assert "AuthContext.tenant is actor provenance only" in operation_description
    assert "does not filter or isolate Cayu data" in operation_description


def test_custom_api_path_updates_contract_and_openapi_paths() -> None:
    client = TestClient(
        create_server(
            CayuApp(),
            config=ServerConfig.local_development(api=ServerApiConfig(path="/cayu/api")),
        )
    )

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
        interaction_id="interaction_1",
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
    assert data["interaction_id"] == "interaction_1"
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
    assert parse_last_event_id(
        "legacy-[REDACTED_SECRET]:session:cayu_event_1",
        expected_session_id="legacy-private:session",
        public_session_id="legacy-[REDACTED_SECRET]:session",
    ) == ("legacy-[REDACTED_SECRET]:session", "cayu_event_1")
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


def test_sse_event_frame_handles_legacy_lone_unicode_surrogates_safely() -> None:
    # New Event construction rejects this value. Keep the transport defensive
    # for legacy rows and validation-bypassed custom integrations.
    event = Event.model_construct(
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
