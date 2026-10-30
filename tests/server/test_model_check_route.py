"""The protected model check runs `cayu providers check` on the served app."""

from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.budgets import ModelPrice, PriceBook, default_price_book
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelProviderError, ModelStreamEvent
from cayu.runtime._model_check import (
    _MODEL_CHECK_PROMPT,
    MODEL_CHECK_MAX_OUTPUT_TOKENS,
    MODEL_CHECK_TIMEOUT_SECONDS,
    ModelCheckResult,
)
from cayu.server import AuthenticatedAccess, DashboardConfig, ServerConfig, create_server
from cayu.server._model_check_routes import MODEL_CHECK_MIN_INTERVAL_SECONDS, ModelCheckAdmission
from cayu.server.auth import AuthContext
from cayu.server.proxy_identity import ControlPlaneRoles
from cayu.vaults.redaction import SecretRedactor

_PATH = "/api/diagnostics/model-check"
_ANSWER = "ROUTE-PROBE-ANSWER"
_CREDENTIAL = "sk-live-ROUTE-PROBE-CREDENTIAL"
_RAW_PROVIDER_TEXT = f"Incorrect API key provided: {_CREDENTIAL}"


def _require_bounded(request, provider_name: str) -> None:
    assert request.tools == [] and request.hosted_tools == ()
    assert request.options == {provider_name: {"max_output_tokens": MODEL_CHECK_MAX_OUTPUT_TOKENS}}
    assert [message.role.value for message in request.messages] == ["user"]


class RaisingProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        raise ModelProviderError(
            _RAW_PROVIDER_TEXT,
            provider=self.name,
            status_code=401,
            error_type="invalid_request_error",
            error_code="invalid_api_key",
            retryable=False,
        )
        yield


class SetupRequiredProvider(ScriptedModelProvider):
    def preflight_model_target(self, *, model):
        raise RuntimeError(f"OPENAI_API_KEY={_CREDENTIAL} was rejected")


def _app() -> tuple[CayuApp, list[str]]:
    """An app whose providers record each request's model; the answering one is the default."""

    requests: list[str] = []

    def answer(request):
        requests.append(request.model)
        _require_bounded(request, "scripted")
        return [
            ModelStreamEvent.text_delta(_ANSWER),
            ModelStreamEvent.completed(
                {"finish_reason": "stop", "usage": {"input_tokens": 9, "output_tokens": 2}}
            ),
        ]

    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor([_CREDENTIAL]))
    app.register_provider(
        ScriptedModelProvider(response_factory=answer), model_patterns=["probe-*"], default=True
    )
    app.register_provider(RaisingProvider([], name="failing"))
    app.register_provider(SetupRequiredProvider([], name="unconfigured"))
    return app, requests


def _assert_private_text_absent(text: str) -> None:
    for private in (_MODEL_CHECK_PROMPT, _ANSWER, _RAW_PROVIDER_TEXT, _CREDENTIAL):
        assert private not in text


@pytest.mark.parametrize("configured_price_book", [True, False])
def test_route_reports_the_check_with_the_servers_price_book(configured_price_book: bool) -> None:
    app, requests = _app()
    price_book = PriceBook(
        price_book_version="served-prices",
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="probe-model",
                match="exact",
                input_per_million=Decimal("1000"),
                output_per_million=Decimal("2000"),
            ),
        ),
    )
    config = ServerConfig.local_development(
        dashboard=DashboardConfig(
            runtime_config={"priceBook": price_book} if configured_price_book else {}
        )
    )
    with TestClient(create_server(app, config=config)) as client:
        response = client.post(_PATH, json={"model": "probe-model"})

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    _assert_private_text_absent(response.text)
    report = response.json()
    assert requests == ["probe-model"]
    assert report["ok"] is True
    assert report["error"] is None
    assert report["route"] == {
        "provider": "scripted",
        "model": "probe-model",
        "resolution": "model_pattern",
    }
    assert report["limits"] == {
        "max_output_tokens": MODEL_CHECK_MAX_OUTPUT_TOKENS,
        "timeout_seconds": MODEL_CHECK_TIMEOUT_SECONDS,
    }
    assert type(report["latency_ms"]) is int
    assert report["answer"] == {"finish_reason": "stop", "text_received": True}
    assert report["usage"]["input_tokens"] == 9
    if configured_price_book:
        assert report["pricing"] == {
            "priced": True,
            "price_book": {
                "source": "project",
                "version": "served-prices",
                "generated_at": "unspecified",
            },
            "currency": "USD",
            "estimated_cost": "0.013",
        }
    else:
        bundled = default_price_book()
        assert report["pricing"]["priced"] is False
        assert report["pricing"]["price_book"] == {
            "source": "bundled",
            "version": bundled.price_book_version,
            "generated_at": bundled.generated_at,
        }
    # The report is the local command's: it round-trips through the same record.
    assert ModelCheckResult.model_validate(report).model_dump(mode="json") == report


def test_a_provider_failure_is_a_report_without_the_providers_text() -> None:
    app, _ = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(_PATH, json={"model": "probe-model", "provider": "failing"})

    assert response.status_code == 200, response.text
    _assert_private_text_absent(response.text)
    report = response.json()
    assert report["ok"] is False
    assert report["answer"] is None
    assert report["route"]["resolution"] == "explicit"
    assert report["error"] == {
        "category": "authentication_failed",
        "message": "The provider rejected the configured credentials.",
        "status_code": 401,
        "provider_error_type": "invalid_request_error",
        "provider_error_code": "invalid_api_key",
        "retryable": False,
    }


def test_a_preflight_detail_is_redacted_with_the_apps_secrets() -> None:
    app, _ = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(_PATH, json={"model": "probe-model", "provider": "unconfigured"})

    assert response.status_code == 200, response.text
    _assert_private_text_absent(response.text)
    error = response.json()["error"]
    assert error["category"] == "preflight_rejected"
    assert error["detail"] == "OPENAI_API_KEY=[REDACTED_SECRET] was rejected"


def test_only_full_access_callers_may_run_a_check() -> None:
    roles = ControlPlaneRoles()
    roles.add_api_path("/api")

    class Auth:
        control_plane_roles = roles

        def __call__(self, request):
            role = request.headers.get("x-test-role")
            if role is None:
                raise HTTPException(401, "Authentication required")
            context = AuthContext(subject="caller", claims={"role": role})
            roles.require(request, context)
            return context

    app, requests = _app()
    config = ServerConfig(access=AuthenticatedAccess(dependency=Auth()))
    with TestClient(create_server(app, config=config)) as client:
        body = {"model": "probe-model"}
        assert client.post(_PATH, json=body).status_code == 401
        refused = client.post(_PATH, json=body, headers={"x-test-role": "member"})
        assert refused.status_code == 403
        assert "latency_ms" not in refused.text
        assert requests == []
        allowed = client.post(_PATH, json=body, headers={"x-test-role": "developer"})

    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["ok"] is True
    assert requests == ["probe-model"]


def test_requests_that_name_no_route_or_try_to_change_the_check_send_nothing() -> None:
    app, requests = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        unknown = client.post(_PATH, json={"model": "probe-model", "provider": "missing"})
        assert unknown.status_code == 422
        assert unknown.json() == {
            "detail": (
                "Provider not registered: missing. Registered providers: failing, scripted, "
                "unconfigured."
            )
        }
        injected = client.post(_PATH, json={"model": "probe-model", "prompt": "INJECTED-PROMPT"})
        assert injected.status_code == 422
        assert "INJECTED-PROMPT" not in injected.text
        for model in (" probe-model", "x" * 257, ""):
            assert client.post(_PATH, json={"model": model}).status_code == 422
        assert requests == []
        # Refused requests don't use up the rate limit.
        assert client.post(_PATH, json={"model": "probe-model"}).status_code == 200

    assert requests == ["probe-model"]


def test_a_second_check_waits_for_the_interval() -> None:
    app, requests = _app()
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        assert client.post(_PATH, json={"model": "probe-model"}).status_code == 200
        limited = client.post(_PATH, json={"model": "probe-model"})

    assert limited.status_code == 429
    assert 1 <= int(limited.headers["retry-after"]) <= MODEL_CHECK_MIN_INTERVAL_SECONDS
    assert limited.headers["cache-control"] == "private, no-store"
    assert requests == ["probe-model"]


def test_admission_holds_one_check_and_then_the_interval() -> None:
    now = [100.0]
    gate = ModelCheckAdmission(min_interval_seconds=5.0, clock=lambda: now[0])

    assert gate.try_start() is None
    now[0] = 140.0
    assert gate.try_start() == 5.0  # Still running, however long it takes.
    gate.finish()
    now[0] = 142.0
    assert gate.try_start() == pytest.approx(3.0)
    now[0] = 145.0
    assert gate.try_start() is None
