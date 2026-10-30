from __future__ import annotations

import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.budgets import ModelPrice, PriceBook, default_price_book, dump_price_book
from cayu.cli import main
from cayu.evals import ScriptedModelProvider
from cayu.runtime._model_check import (
    _MODEL_CHECK_PROMPT,
    MODEL_CHECK_MAX_OUTPUT_TOKENS,
    check_model,
    resolve_model_check_route,
)

_ANSWER = "PROBE-ANSWER-TEXT"
_RAW_PROVIDER_TEXT = "Incorrect API key provided: sk-live-PROBE-CREDENTIAL"

# Scripted providers in the project assert the request shape, so a malformed
# request surfaces as a failed check instead of passing silently.
_PROJECT = f"""
import asyncio

from cayu import CayuApp
from cayu.budgets import BillingIdentity, PricingContext
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelProviderError, ModelStreamEvent


def _require_bounded(request, provider_name):
    assert request.tools == [] and request.hosted_tools == ()
    assert request.options == {{provider_name: {{"max_output_tokens": {MODEL_CHECK_MAX_OUTPUT_TOKENS}}}}}
    assert [message.role.value for message in request.messages] == ["user"]


def _answer(request):
    _require_bounded(request, "scripted")
    return [
        ModelStreamEvent.text_delta({_ANSWER!r}),
        ModelStreamEvent.completed(
            {{"finish_reason": "stop", "usage": {{"input_tokens": 9, "output_tokens": 2}}}}
        ),
    ]


_RESOLVED_MODELS = {{"alias-to-priced": "resolved-priced", "alias-to-unpriced": "resolved-unpriced"}}
_FINISH_REASONS = {{"finish-error": "malformed_model_output", "finish-length": "max_tokens"}}


def _resolve_alias(request):
    _require_bounded(request, "aliasing")
    return [
        ModelStreamEvent.completed(
            {{"finish_reason": "stop", "model": _RESOLVED_MODELS[request.model]}}
        )
    ]


def _finish(request):
    _require_bounded(request, "finishing")
    return [
        ModelStreamEvent.completed(
            {{
                "finish_reason": _FINISH_REASONS[request.model],
                "usage": {{"input_tokens": 9, "output_tokens": 16}},
            }}
        )
    ]


def _provider_error():
    return ModelProviderError(
        {_RAW_PROVIDER_TEXT!r},
        provider="failing",
        status_code=401,
        error_type="invalid_request_error",
        error_code="invalid_api_key",
        retryable=False,
    )


class ErrorEventProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        error = _provider_error()
        yield ModelStreamEvent.error(str(error), cause=error)


class RaisingProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        raise _provider_error()
        yield


class SelfCancellingProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        raise asyncio.CancelledError
        yield


class FailsAfterCompletionProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        yield ModelStreamEvent.text_delta({_ANSWER!r})
        yield ModelStreamEvent.completed(
            {{
                "finish_reason": "stop",
                "model": "resolved-unpriced",
                "usage": {{"input_tokens": 9, "output_tokens": 2}},
            }}
        )
        raise ModelProviderError(
            {_RAW_PROVIDER_TEXT!r},
            provider=self.name,
            error_type="ProviderStreamCleanupError",
            retryable=False,
        )


class LateEventProvider(ScriptedModelProvider):
    async def stream(self, request):
        _require_bounded(request, self.name)
        yield ModelStreamEvent.completed(
            {{
                "finish_reason": "stop",
                "model": "resolved-unpriced",
                "usage": {{"input_tokens": 9, "output_tokens": 2}},
            }}
        )
        if request.model == "late-error":
            error = _provider_error()
            yield ModelStreamEvent.error(str(error), cause=error)
        else:
            yield ModelStreamEvent.text_delta({_ANSWER!r})


# Bills one of two tiers; only the completion says which one served the request.
class TieredProvider(ScriptedModelProvider):
    async def billing_identity_for_request(self, request):
        return BillingIdentity(
            provider_name=self.name,
            resource_id=request.model,
            pricing_contexts=(
                PricingContext(dimensions={{"tier": "reserved"}}),
                PricingContext(dimensions={{"tier": "default"}}),
            ),
        )

    def billing_identity_for_completion(self, identity, payload):
        return BillingIdentity(
            provider_name=identity.provider_name,
            resource_id=identity.resource_id,
            pricing_contexts=(PricingContext(dimensions={{"tier": payload["service_tier"]}}),),
        )

    async def stream(self, request):
        _require_bounded(request, self.name)
        yield ModelStreamEvent.completed(
            {{
                "finish_reason": "stop",
                # An unknown tier widens the request's contexts, which accounting rejects.
                "service_tier": "default" if request.model == "probe-model" else "premium",
                "usage": {{"input_tokens": 9, "output_tokens": 2}},
            }}
        )


class SetupRequiredProvider(ScriptedModelProvider):
    def preflight_model_target(self, *, model):
        raise RuntimeError("provider 'openai' is selected but OPENAI_API_KEY is not set")


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_provider(ErrorEventProvider([], name="failing"), default=True)
    app.register_provider(RaisingProvider([], name="raising"))
    app.register_provider(SetupRequiredProvider([], name="unconfigured"))
    app.register_provider(SelfCancellingProvider([], name="cancelling"))
    app.register_provider(FailsAfterCompletionProvider([], name="closing"))
    app.register_provider(LateEventProvider([], name="late"), model_patterns=["late-*"])
    app.register_provider(TieredProvider([], name="tiered"))
    app.register_provider(
        ScriptedModelProvider(response_factory=_resolve_alias, name="aliasing"),
        model_patterns=["alias-*"],
    )
    app.register_provider(
        ScriptedModelProvider(response_factory=_finish, name="finishing"),
        model_patterns=["finish-*"],
    )
    app.register_provider(
        ScriptedModelProvider(response_factory=_answer), model_patterns=["probe-*"]
    )
    return app
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "probe_app:build_app"\n', encoding="utf-8"
    )
    (tmp_path / "probe_app.py").write_text(_PROJECT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("probe_app", None)
    return tmp_path


def _configure_project_price_book(project: Path, *prices: ModelPrice) -> None:
    price_book = PriceBook(price_book_version="probe-prices", prices=prices)
    (project / "prices.json").write_text(dump_price_book(price_book), encoding="utf-8")
    with (project / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('\n[tool.cayu.serve]\nprice_book = "prices.json"\n')


def _price(provider_name: str, model: str) -> ModelPrice:
    return ModelPrice.fixed(
        provider_name=provider_name,
        model=model,
        match="exact",
        input_per_million=Decimal("1000"),
        output_per_million=Decimal("2000"),
    )


def _assert_private_text_absent(output: str) -> None:
    for private in (_MODEL_CHECK_PROMPT, _ANSWER, _RAW_PROVIDER_TEXT, "PROBE-CREDENTIAL"):
        assert private not in output


@pytest.mark.parametrize("project_price_book", [True, False])
def test_check_reports_answer_latency_usage_and_route_pricing(
    project: Path, capsys: pytest.CaptureFixture[str], project_price_book: bool
) -> None:
    if project_price_book:
        _configure_project_price_book(project, _price("scripted", "probe-model"))

    assert main(["providers", "check", "probe-model"]) == 0

    output = capsys.readouterr().out
    _assert_private_text_absent(output)
    result = json.loads(output)
    assert result["ok"] is True
    assert result["error"] is None
    assert result["route"] == {
        "provider": "scripted",
        "model": "probe-model",
        "resolution": "model_pattern",
    }
    assert result["limits"]["max_output_tokens"] == MODEL_CHECK_MAX_OUTPUT_TOKENS
    assert type(result["latency_ms"]) is int and result["latency_ms"] >= 0
    assert result["answer"] == {"finish_reason": "stop", "text_received": True}
    assert result["usage"]["input_tokens"] == 9
    assert result["usage"]["output_tokens"] == 2
    pricing = result["pricing"]
    if project_price_book:
        assert pricing == {
            "priced": True,
            "price_book": {
                "source": "project",
                "version": "probe-prices",
                "generated_at": "unspecified",
            },
            "currency": "USD",
            "estimated_cost": "0.013",
        }
    else:
        bundled = default_price_book()
        assert pricing == {
            "priced": False,
            "price_book": {
                "source": "bundled",
                "version": bundled.price_book_version,
                "generated_at": bundled.generated_at,
            },
            "unpriced_reason": "no matching model pricing",
        }


@pytest.mark.parametrize(
    ("model", "provider_model", "priced"),
    [
        ("alias-to-priced", "resolved-priced", True),
        ("alias-to-unpriced", "resolved-unpriced", False),
    ],
)
def test_route_pricing_follows_the_provider_reported_model(
    project: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    provider_model: str,
    priced: bool,
) -> None:
    _configure_project_price_book(
        project,
        _price("aliasing", "resolved-priced"),
        _price("aliasing", "alias-to-unpriced"),
    )

    assert main(["providers", "check", model]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["route"]["provider_model"] == provider_model
    assert result["usage"] is None
    assert result["pricing"]["priced"] is priced
    assert "estimated_cost" not in result["pricing"]


def test_an_error_finish_fails_and_a_token_limit_finish_answers(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["providers", "check", "finish-error"]) == 1

    failed = json.loads(capsys.readouterr().out)
    assert failed["ok"] is False
    assert failed["answer"] is None
    assert failed["usage"]["output_tokens"] == 16
    assert failed["error"] == {
        "category": "generation_failed",
        "message": "The provider completed the request, but the model's generation failed.",
    }

    assert main(["providers", "check", "finish-length"]) == 0

    answered = json.loads(capsys.readouterr().out)
    assert answered["ok"] is True
    assert answered["answer"] == {"finish_reason": "length", "text_received": False}


def test_a_failure_after_completion_fails_but_keeps_usage_and_the_reported_model(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The requested alias is priced, the model the provider reports is not.
    _configure_project_price_book(project, _price("closing", "alias-to-unpriced"))

    assert main(["providers", "check", "alias-to-unpriced", "--provider", "closing"]) == 1

    output = capsys.readouterr().out
    _assert_private_text_absent(output)
    result = json.loads(output)
    assert result["ok"] is False
    assert result["answer"] is None
    assert result["route"]["provider_model"] == "resolved-unpriced"
    assert result["usage"]["input_tokens"] == 9
    assert result["usage"]["output_tokens"] == 2
    assert result["pricing"]["priced"] is False
    assert result["error"] == {
        "category": "provider_error",
        "message": (
            "The provider failed without an HTTP status, for example on a connection error."
        ),
        "provider_error_type": "ProviderStreamCleanupError",
        "retryable": False,
    }


@pytest.mark.parametrize("model", ["late-error", "late-text"])
def test_an_event_after_completion_fails_but_keeps_usage_and_the_reported_model(
    project: Path, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    _configure_project_price_book(project, _price("late", model))

    assert main(["providers", "check", model]) == 1

    output = capsys.readouterr().out
    _assert_private_text_absent(output)
    result = json.loads(output)
    assert result["ok"] is False
    assert result["answer"] is None
    assert result["route"]["provider_model"] == "resolved-unpriced"
    assert result["usage"]["input_tokens"] == 9
    assert result["pricing"]["priced"] is False
    assert result["error"] == {
        "category": "event_after_completion",
        "message": "The provider sent another stream event after completing the request.",
    }


def test_the_completion_narrows_the_billing_identity_before_pricing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only the default tier has a rate; the request alone could be either tier.
    _configure_project_price_book(
        project,
        ModelPrice.fixed(
            provider_name="tiered",
            model="probe-model",
            match="exact",
            pricing_context={"tier": ("default",)},
            input_per_million=Decimal("1000"),
            output_per_million=Decimal("2000"),
        ),
    )

    assert main(["providers", "check", "probe-model", "--provider", "tiered"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["pricing"]["priced"] is True
    assert result["pricing"]["estimated_cost"] == "0.013"


def test_a_completion_billing_identity_failure_fails_and_keeps_usage(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["providers", "check", "probe-premium", "--provider", "tiered"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["usage"]["input_tokens"] == 9
    assert result["pricing"]["priced"] is False
    assert result["error"] == {
        "category": "billing_identity_failed",
        "message": "The provider could not resolve the request's billing identity.",
    }


def test_provider_self_cancellation_is_a_provider_failure(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["providers", "check", "probe-model", "--provider", "cancelling"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["error"] == {
        "category": "provider_error",
        "message": (
            "The provider failed without an HTTP status, for example on a connection error."
        ),
        "provider_error_type": "ProviderStreamCancellationError",
        "provider_error_code": "provider_stream_cancelled_itself",
        "retryable": False,
    }


@pytest.mark.parametrize("provider", ["failing", "raising"])
def test_provider_error_is_categorized_without_its_raw_text(
    project: Path, capsys: pytest.CaptureFixture[str], provider: str
) -> None:
    assert main(["providers", "check", "probe-model", "--provider", provider]) == 1

    output = capsys.readouterr().out
    _assert_private_text_absent(output)
    result = json.loads(output)
    assert result["ok"] is False
    assert result["answer"] is None
    assert result["route"]["resolution"] == "explicit"
    assert type(result["latency_ms"]) is int
    assert result["error"] == {
        "category": "authentication_failed",
        "message": "The provider rejected the configured credentials.",
        "status_code": 401,
        "provider_error_type": "invalid_request_error",
        "provider_error_code": "invalid_api_key",
        "retryable": False,
    }

    assert main(["providers", "check", "probe-model", "--provider", provider, "--table"]) == 1
    human = capsys.readouterr().out
    _assert_private_text_absent(human)
    assert "authentication_failed HTTP 401" in human


def test_preflight_rejection_reports_the_setup_detail_without_a_request(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["providers", "check", "probe-model", "--provider", "unconfigured"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["latency_ms"] is None
    assert result["error"] == {
        "category": "preflight_rejected",
        "message": "The provider refused this model before sending a request.",
        "detail": "provider 'openai' is selected but OPENAI_API_KEY is not set",
    }


def test_unknown_provider_is_reported_without_a_request(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["providers", "check", "probe-model", "--provider", "missing"]) == 2

    assert json.loads(capsys.readouterr().out) == {
        "schema_version": "1",
        "error": {
            "code": "MODEL_CHECK_NOT_RUN",
            "message": (
                "Provider not registered: missing. Registered providers: "
                "aliasing, cancelling, closing, failing, finishing, late, raising, scripted, "
                "tiered, unconfigured."
            ),
        },
    }


def test_doctor_model_spelling_runs_the_same_check(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--model", "probe-model", "--provider", "scripted"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["route"]["resolution"] == "explicit"
    assert not list(project.glob("*.zip"))

    assert main(["doctor", "--bundle", "bundle.zip", "--provider", "scripted"]) == 2
    assert main(["doctor", "--model", "probe-model", "--session", "session-1"]) == 2
    assert not (project / "bundle.zip").exists()
    with pytest.raises(SystemExit) as raised:
        main(["doctor", "--bundle", "bundle.zip", "--model", "probe-model"])
    assert raised.value.code == 2


def test_check_reports_a_provider_that_does_not_answer_before_the_deadline() -> None:
    class SilentProvider(ScriptedModelProvider):
        async def stream(self, request):
            await asyncio.sleep(30)
            yield

    app = CayuApp(enable_logging=False)
    app.register_provider(SilentProvider([], name="silent"), default=True)
    registered, resolution = resolve_model_check_route(app, model="probe-model", provider_name=None)

    async def check():
        try:
            return await check_model(
                registered,
                model="probe-model",
                resolution=resolution,
                pricing=default_price_book(),
                price_book_source="bundled",
                redactor=app._secret_redactor,
                timeout_seconds=0.2,
            )
        finally:
            await app.aclose()

    result = asyncio.run(check())

    assert result.ok is False
    assert resolution == "default"
    assert result.error is not None
    assert result.error.category == "timeout"
    assert result.latency_ms is not None and result.latency_ms < 10_000


_SERVICE_PROJECT = """
from fastapi import HTTPException, Request

from cayu import AgentSpec, CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ServiceIdentityStoreKind,
    create_agent_service,
)
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore


class Store:
    category = ServiceIdentityStoreKind.DURABLE

    async def reserve(self, **kwargs):
        raise AssertionError

    async def find(self, **kwargs):
        raise AssertionError

    async def find_by_session_id(self, **kwargs):
        raise AssertionError

    async def claim_execution(self, **kwargs):
        raise AssertionError

    async def heartbeat_execution(self, **kwargs):
        raise AssertionError

    async def release_execution(self, **kwargs):
        raise AssertionError

    async def record_result_receipt(self, **kwargs):
        raise AssertionError

    async def record_recovery_status(self, **kwargs):
        raise AssertionError

    async def finish(self, **kwargs):
        raise AssertionError


class PlaceholderProvider(ScriptedModelProvider):
    def preflight_model_target(self, *, model):
        raise RuntimeError("the plain factory's placeholder provider was checked")


async def product_auth(request: Request):
    raise HTTPException(status_code=401)


def _answer(request):
    return [ModelStreamEvent.completed({"finish_reason": "stop"})]


def build_app(provider=None):
    app = CayuApp(
        session_store=SQLiteSessionStore("runtime.db"),
        task_store=SQLiteTaskStore("runtime.db"),
        enable_logging=False,
    )
    app.register_provider(provider or PlaceholderProvider([], name="placeholder"), default=True)
    app.register_agent(AgentSpec(name="agent", model="probe-model"))
    return app


def build_service(*, mode):
    return create_agent_service(
        build_app(ScriptedModelProvider(response_factory=_answer, name="production")),
        agent_name="agent",
        mode=mode,
        product_access=AuthenticatedProductAccess(dependency=product_auth),
        operator_access=AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="secret-password")
        ),
        product_store=Store(),
    )
"""


def test_check_uses_the_maintained_service_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "service_app:build_app"\n'
        'service_factory = "service_app:build_service"\n',
        encoding="utf-8",
    )
    (tmp_path / "service_app.py").write_text(_SERVICE_PROJECT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("service_app", None)

    assert main(["providers", "check", "probe-model"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["route"] == {
        "provider": "production",
        "model": "probe-model",
        "resolution": "default",
    }
