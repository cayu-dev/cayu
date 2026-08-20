from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    EvalCaseSpec,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PromotionCandidateV1,
    RunRequest,
    default_price_book,
    eval_corpus_from_json,
)
from cayu.runtime import InMemorySessionStore
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvaluationPromotionConfig,
    OpenAccess,
    ServerConfig,
    create_server,
)
from cayu.server.contracts import MAX_EVALUATION_PROMOTION_REQUEST_BYTES
from cayu.vaults import SecretRedactor

_SESSION_ID = "server-promotion-session"
_AUTH_HEADERS = {"Authorization": "Bearer valid"}


class _PromotionProvider(ModelProvider):
    name = "promotion-provider"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("captured answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        )


class _NoTerminalEvidenceSessionStore(InMemorySessionStore):
    supports_terminal_session_evidence = False


class _NoSessionLineageStore(InMemorySessionStore):
    supports_session_lineage = False


def _authenticate(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != "Bearer valid":
        raise HTTPException(status_code=401, detail="unauthorized")
    return AuthContext(subject="eval-operator")


async def _seed_app(*, secret: str | None = None) -> CayuApp:
    app = CayuApp(
        enable_logging=False,
        secret_redactor=None if secret is None else SecretRedactor(secret),
    )
    app.register_provider(_PromotionProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="promotion-model"))
    async for _ in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=_SESSION_ID,
            messages=[Message.text("user", "promote this completed run")],
        )
    ):
        pass
    return app


def _promotion_config() -> EvaluationPromotionConfig:
    return EvaluationPromotionConfig(
        target_key="support.regressions",
        source_agent_name="assistant",
        application_release_id="release-2026-08-06",
    )


def _client(app: CayuApp, *, with_pricing: bool = False) -> TestClient:
    return TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                _authenticate,
                dashboard=(
                    DashboardConfig(runtime_config={"priceBook": default_price_book()})
                    if with_pricing
                    else DashboardConfig(enabled=False)
                ),
                evaluation_promotion=_promotion_config(),
            ),
        )
    )


def _draft(candidate: dict) -> dict:
    suite = candidate["suite"]
    case = candidate["case"]
    return {
        "expected_baseline_revision": candidate["revision"],
        "suite": {
            "id": suite["id"],
            "name": suite["name"],
            "description": suite["description"],
            "trial_request": suite["trial_request"],
        },
        "case": {
            "id": case["id"],
            "suite_id": case["suite_id"],
            "name": case["name"],
            "description": case["description"],
            "input": case["input"],
            "assertions": case["assertions"],
        },
    }


def test_promotion_routes_are_absent_without_complete_authenticated_configuration() -> None:
    app = asyncio.run(_seed_app())
    disabled = TestClient(
        create_server(
            app,
            config=ServerConfig.protected(
                _authenticate,
                dashboard=DashboardConfig(enabled=False),
            ),
        )
    )

    assert (
        disabled.post(
            f"/api/evals/promotion/sessions/{_SESSION_ID}/preview",
            headers=_AUTH_HEADERS,
            json={},
        ).status_code
        == 404
    )
    with pytest.raises(ValidationError, match="authenticated API access"):
        ServerConfig(
            access=OpenAccess(),
            dashboard=DashboardConfig(enabled=False),
            evaluation_promotion=_promotion_config(),
        )
    with pytest.raises(ValidationError, match="source_agent_name"):
        EvaluationPromotionConfig(
            target_key="support.regressions",
            application_release_id="release",
        )


def test_promotion_wiring_requires_registered_redaction_safe_identity() -> None:
    unregistered = CayuApp(enable_logging=False)
    with pytest.raises(ValueError, match="source_agent_name is not registered"):
        create_server(
            unregistered,
            config=ServerConfig.protected(
                _authenticate,
                dashboard=DashboardConfig(enabled=False),
                evaluation_promotion=_promotion_config(),
            ),
        )

    secret = "release-secret-ABCDEFGHIJKLMNOP"
    app = asyncio.run(_seed_app(secret=secret))
    unsafe = EvaluationPromotionConfig(
        target_key="support.regressions",
        source_agent_name="assistant",
        application_release_id=secret,
    )
    with pytest.raises(ValueError, match="contains a workload secret"):
        create_server(
            app,
            config=ServerConfig.protected(
                _authenticate,
                dashboard=DashboardConfig(enabled=False),
                evaluation_promotion=unsafe,
            ),
        )


def test_promotion_capability_and_routes_require_authentication() -> None:
    client = _client(asyncio.run(_seed_app()))

    assert client.get("/api/contract").status_code == 401
    assert (
        client.post(f"/api/evals/promotion/sessions/{_SESSION_ID}/preview", json={}).status_code
        == 401
    )
    capabilities = client.get("/api/contract", headers=_AUTH_HEADERS).json()["capabilities"]
    surface = capabilities["surfaces"]["evaluation_promotion"]
    assert surface == {
        "configured": True,
        "read": {"enabled": True, "unavailable_reason": None},
        "mutate": {"enabled": True, "unavailable_reason": None},
    }
    assert capabilities["evals_readiness"]["captured_evaluation"] == {
        "state": "ready",
        "reason_code": None,
    }
    assert capabilities["evals_readiness"]["catalog_read"] == {
        "state": "gated",
        "reason_code": "eval_store_not_configured",
    }


@pytest.mark.parametrize(
    ("store_type", "reason_code"),
    [
        (_NoTerminalEvidenceSessionStore, "terminal_evidence_not_supported"),
        (_NoSessionLineageStore, "session_lineage_not_supported"),
    ],
)
def test_captured_evaluation_readiness_identifies_the_missing_store_capability(
    store_type: type[InMemorySessionStore],
    reason_code: str,
) -> None:
    app = CayuApp(session_store=store_type(), enable_logging=False)
    app.register_provider(_PromotionProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="promotion-model"))

    capabilities = _client(app).get("/api/contract", headers=_AUTH_HEADERS).json()["capabilities"]

    assert capabilities["surfaces"]["evaluation_promotion"] == {
        "configured": True,
        "read": {"enabled": False, "unavailable_reason": "unsupported"},
        "mutate": {"enabled": False, "unavailable_reason": "unsupported"},
    }
    assert capabilities["evals_readiness"]["captured_evaluation"] == {
        "state": "unsupported",
        "reason_code": reason_code,
    }


def test_preview_edits_and_export_are_stateless_revalidated_and_identity_free() -> None:
    app = asyncio.run(_seed_app())
    client = _client(app)
    preview_url = f"/api/evals/promotion/sessions/{_SESSION_ID}/preview"
    export_url = f"/api/evals/promotion/sessions/{_SESSION_ID}/export"

    initial = client.post(preview_url, headers=_AUTH_HEADERS, json={})
    assert initial.status_code == 200
    initial_body = initial.json()
    assert initial_body["baseline_revision"] == initial_body["candidate"]["revision"]
    assert initial_body["captured_score"]["status"] == "passed"
    assert initial_body["captured_score"]["score"] == 1.0
    assert _SESSION_ID not in initial.text

    draft = _draft(initial_body["candidate"])
    draft["suite"]["name"] = "Production support regressions"
    draft["case"]["name"] = "Answer a customer request"
    draft["case"]["input"]["messages"][0]["text"] = "answer this regression request"
    edited = client.post(preview_url, headers=_AUTH_HEADERS, json={"draft": draft})
    assert edited.status_code == 200
    edited_body = edited.json()
    assert edited_body["baseline_revision"] == initial_body["candidate"]["revision"]
    assert edited_body["candidate"]["revision"] != initial_body["candidate"]["revision"]
    assert edited_body["candidate"]["case"]["name"] == "Answer a customer request"
    assert (
        edited_body["captured_score"]["candidate_revision"] == edited_body["candidate"]["revision"]
    )

    exported = client.post(
        export_url,
        headers=_AUTH_HEADERS,
        json={
            "expected_candidate_revision": edited_body["candidate"]["revision"],
            "candidate": edited_body["candidate"],
        },
    )
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/json")
    assert exported.headers["content-disposition"] == (
        'attachment; filename="support.regressions.eval.json"'
    )
    corpus = eval_corpus_from_json(exported.content.decode("utf-8"))
    assert corpus.target_key == "support.regressions"
    assert corpus.cases[0].name == "Answer a customer request"
    assert _SESSION_ID.encode() not in exported.content

    stale_revision = client.post(
        export_url,
        headers=_AUTH_HEADERS,
        json={
            "expected_candidate_revision": initial_body["candidate"]["revision"],
            "candidate": edited_body["candidate"],
        },
    )
    assert stale_revision.status_code == 409
    assert stale_revision.json()["detail"]["code"] == "preview_stale"

    app.register_agent(AgentSpec(name="new-release-agent", model="promotion-model"))
    changed_manifest = client.post(
        export_url,
        headers=_AUTH_HEADERS,
        json={
            "expected_candidate_revision": edited_body["candidate"]["revision"],
            "candidate": edited_body["candidate"],
        },
    )
    assert changed_manifest.status_code == 409
    assert changed_manifest.json()["detail"]["code"] == "preview_stale"

    stale_preview = client.post(preview_url, headers=_AUTH_HEADERS, json={"draft": draft})
    assert stale_preview.status_code == 409
    assert stale_preview.json()["detail"]["code"] == "preview_stale"


def test_preview_rejects_stale_or_secret_drafts_and_bounds_bodies_before_parsing() -> None:
    secret = "draft-secret-ABCDEFGHIJKLMNOP"
    client = _client(asyncio.run(_seed_app(secret=secret)))
    preview_url = f"/api/evals/promotion/sessions/{_SESSION_ID}/preview"
    candidate = client.post(preview_url, headers=_AUTH_HEADERS, json={}).json()["candidate"]

    stale = _draft(candidate)
    stale["expected_baseline_revision"] = f"sha256:{'0' * 64}"
    stale_response = client.post(preview_url, headers=_AUTH_HEADERS, json={"draft": stale})
    assert stale_response.status_code == 409
    assert stale_response.json()["detail"]["code"] == "preview_stale"

    unsafe = _draft(candidate)
    unsafe["case"]["description"] = f"Do not export {secret}"
    unsafe_response = client.post(preview_url, headers=_AUTH_HEADERS, json={"draft": unsafe})
    assert unsafe_response.status_code == 400
    assert unsafe_response.json()["detail"]["code"] == "draft_rejected"
    assert secret not in unsafe_response.text

    baseline = PromotionCandidateV1.model_validate_json(json.dumps(candidate))
    unsafe_case = EvalCaseSpec.create(
        id=baseline.case.id,
        suite_id=baseline.case.suite_id,
        name=baseline.case.name,
        description=f"Do not export {secret}",
        source=baseline.case.source,
        input=baseline.case.input,
        assertions=baseline.case.assertions,
    )
    unsafe_candidate = PromotionCandidateV1.create(
        target_key=baseline.target_key,
        source=baseline.source,
        evidence_policy=baseline.evidence_policy,
        pricing_profile=baseline.pricing_profile,
        evidence=baseline.evidence,
        suite=baseline.suite,
        case=unsafe_case,
    )
    unsafe_export = client.post(
        f"/api/evals/promotion/sessions/{_SESSION_ID}/export",
        headers=_AUTH_HEADERS,
        json={
            "expected_candidate_revision": unsafe_candidate.revision,
            "candidate": unsafe_candidate.model_dump(mode="json"),
        },
    )
    assert unsafe_export.status_code == 400
    assert unsafe_export.json()["detail"]["code"] == "candidate_rejected"
    assert secret not in unsafe_export.text

    oversized = (
        b'{"draft":null,"padding":"' + (b"x" * MAX_EVALUATION_PROMOTION_REQUEST_BYTES) + b'"}'
    )
    oversized_response = client.post(
        preview_url,
        headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
        content=oversized,
    )
    assert oversized_response.status_code == 413
    assert oversized_response.json() == {
        "detail": "Evaluation promotion request exceeds the server byte limit."
    }


def test_preview_returns_only_candidates_that_the_unchanged_export_route_accepts() -> None:
    unpriced_client = _client(asyncio.run(_seed_app()))
    preview_url = f"/api/evals/promotion/sessions/{_SESSION_ID}/preview"
    export_url = f"/api/evals/promotion/sessions/{_SESSION_ID}/export"
    unpriced_candidate = unpriced_client.post(
        preview_url,
        headers=_AUTH_HEADERS,
        json={},
    ).json()["candidate"]
    unpriced_draft = _draft(unpriced_candidate)
    unpriced_draft["case"]["assertions"] = [
        {
            "id": "cost",
            "kind": "max_estimated_cost",
            "maximum": "1",
            "currency": "USD",
        }
    ]

    missing_pricing = unpriced_client.post(
        preview_url,
        headers=_AUTH_HEADERS,
        json={"draft": unpriced_draft},
    )

    assert missing_pricing.status_code == 400
    assert missing_pricing.json()["detail"]["code"] == "draft_rejected"
    assert "pricing profile" in missing_pricing.json()["detail"]["message"]

    priced_client = _client(asyncio.run(_seed_app()), with_pricing=True)
    priced_candidate = priced_client.post(
        preview_url,
        headers=_AUTH_HEADERS,
        json={},
    ).json()["candidate"]
    priced_draft = _draft(priced_candidate)
    priced_draft["case"]["assertions"] = unpriced_draft["case"]["assertions"]

    previewed = priced_client.post(
        preview_url,
        headers=_AUTH_HEADERS,
        json={"draft": priced_draft},
    )

    assert previewed.status_code == 200
    previewed_candidate = previewed.json()["candidate"]
    exported = priced_client.post(
        export_url,
        headers=_AUTH_HEADERS,
        json={
            "expected_candidate_revision": previewed_candidate["revision"],
            "candidate": previewed_candidate,
        },
    )
    assert exported.status_code == 200
    assert eval_corpus_from_json(exported.text).pricing_profile is not None

    unsupported_currency = _draft(priced_candidate)
    unsupported_currency["case"]["assertions"] = [
        {
            "id": "cost",
            "kind": "max_estimated_cost",
            "maximum": "1",
            "currency": "ZZZ",
        }
    ]
    rejected_currency = priced_client.post(
        preview_url,
        headers=_AUTH_HEADERS,
        json={"draft": unsupported_currency},
    )
    assert rejected_currency.status_code == 400
    assert rejected_currency.json()["detail"]["code"] == "draft_rejected"
    assert "pricing profile" in rejected_currency.json()["detail"]["message"]
