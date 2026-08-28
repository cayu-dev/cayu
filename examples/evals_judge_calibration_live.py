"""Credential-gated fixed-evidence calibration through explicit judge authority.

OpenAI:
    CAYU_PROVIDER=openai uv run python examples/evals_judge_calibration_live.py

Anthropic:
    CAYU_PROVIDER=anthropic uv run python examples/evals_judge_calibration_live.py

The candidate provider is a deterministic sentinel and must receive no request.
Only the explicitly registered real-provider ``ModelJudgeTarget`` may execute.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from _live_checks import require, require_equal
from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    CorpusTarget,
    EvalJudgeCalibrationCriterionLabelV1,
    EvalJudgeCalibrationDraftV1,
    EvalJudgeEvidenceSelectionV1,
    ModelJudgeTarget,
    ModelProvider,
    OpenAIProvider,
    RunRequest,
    ScriptedModelProvider,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricCriterionV1,
    StructuredRubricV1,
    model_judge_profile,
)
from cayu.server import BasicAuth, EvalsConfig, ServerConfig, create_server
from cayu.storage.evals_sqlite import SQLiteEvalStore

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
AUTH_USERNAME = "calibration-operator"
AUTH_PASSWORD = "calibration-live-contract"


def main() -> None:
    provider_name = _provider_name()
    _require_api_key(provider_name)
    model = _model(provider_name)
    with tempfile.TemporaryDirectory(prefix="cayu-evals-judge-calibration-live-") as temporary:
        evidence = _run_contract(
            judge_provider=_provider(provider_name),
            provider_name=provider_name,
            model=model,
            store_path=Path(temporary) / "evals.sqlite",
        )
    print(EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True))


def _run_contract(
    *,
    judge_provider: ModelProvider,
    provider_name: str,
    model: str,
    store_path: Path,
) -> dict[str, object]:
    candidate_provider = ScriptedModelProvider((), name="candidate-calibration-sentinel")
    candidate_app = CayuApp(enable_logging=False)
    candidate_app.register_provider(candidate_provider, default=True)
    candidate_app.register_agent(AgentSpec(name="candidate", model="candidate-sentinel-model"))

    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(judge_provider, default=True)
    judge_app.register_agent(AgentSpec(name="judge", model=model))
    judge = ModelJudgeTarget(
        key="explicit-live-judge",
        label="Explicit live calibration judge",
        app=judge_app,
        agent_name="judge",
        timeout_seconds=180,
        max_input_tokens=8_192,
        max_output_tokens=1_024,
        max_total_tokens=9_216,
    )
    profile = model_judge_profile(judge)
    target = CorpusTarget(
        key="judge-calibration-live",
        app=candidate_app,
        request_base=RunRequest(agent_name="candidate", messages=[]),
        application_release_id="judge-calibration-live-v1",
        model_judges=(judge,),
    )
    draft = EvalJudgeCalibrationDraftV1(
        id="known-capital-answer",
        target_key=target.key,
        assertion=StructuredModelJudgeAssertionSpec(
            id="answer-correctness",
            judge_profile_key=profile.key,
            judge_profile_revision=profile.revision,
            rubric=StructuredRubricV1.create(
                id="answer-correctness",
                criteria=(
                    StructuredRubricCriterionV1(
                        id="correctness",
                        name="Correctness",
                        description=(
                            "Score 1 only when the candidate output gives Paris as the capital "
                            "of France; otherwise score 0."
                        ),
                        weight="1",
                    ),
                ),
            ),
            threshold="1",
            evidence=EvalJudgeEvidenceSelectionV1(),
        ),
        evidence_source_id="checked-live-known-answer",
        task="What is the capital of France?",
        final_output="Paris is the capital of France.",
        human_criteria=(
            EvalJudgeCalibrationCriterionLabelV1(
                criterion_id="correctness",
                score="1",
            ),
        ),
        trials=1,
    )
    store = SQLiteEvalStore(store_path)
    server = create_server(
        candidate_app,
        config=ServerConfig.protected(
            BasicAuth(username=AUTH_USERNAME, password=AUTH_PASSWORD),
            evals=EvalsConfig(target=target, store=store),
        ),
    )
    try:
        with TestClient(server) as client:
            preview = client.post(
                "/api/evals/judge-calibrations/preview",
                auth=(AUTH_USERNAME, AUTH_PASSWORD),
                json={"draft": draft.model_dump(mode="json")},
            )
            require_equal(preview.status_code, 200, "live calibration preview must succeed")
            reviewed = preview.json()
            require_equal(reviewed["ready"], True, "live calibration preview must be ready")
            require_equal(
                reviewed["candidate_route_relation"],
                "independent_model",
                "the explicitly configured live judge must be independent of the sentinel candidate",
            )
            require_equal(
                candidate_provider.requests,
                [],
                "calibration preview must not invoke the candidate provider",
            )

            executed = client.post(
                "/api/evals/judge-calibrations",
                auth=(AUTH_USERNAME, AUTH_PASSWORD),
                json={
                    "run_id": "explicit-live-judge-trial",
                    "expected_definition_revision": reviewed["definition"]["revision"],
                    "definition": reviewed["definition"],
                },
            )
            require_equal(executed.status_code, 201, "live judge calibration must succeed")
            report = executed.json()["report"]
            trial = report["trials"][0]
            detail = trial["judgment"]["detail"]
            require_equal(trial["judgment"]["outcome"], "passed", "live judge must pass")
            require_equal(
                detail["diagnostic"],
                "judgment_recorded",
                "live judge output must be valid",
            )
            require_equal(
                detail["criteria"][0]["criterion_id"],
                "correctness",
                "live judge must return the configured criterion",
            )
            total_tokens = int(detail["usage"]["total_tokens"]) if detail["usage"] else 0
            require(
                total_tokens > 0,
                "live judge calibration must retain positive provider usage",
            )
            require_equal(
                candidate_provider.requests,
                [],
                "fixed-evidence calibration must never invoke the candidate provider",
            )
            return {
                "candidate_provider_calls": 0,
                "evidence_revision": report["definition"]["evidence"]["revision"],
                "judge_profile_revision": report["judge_profile"]["revision"],
                "model": model,
                "provider": provider_name,
                "total_tokens": total_tokens,
                "trials": 1,
            }
    finally:
        asyncio.run(store.close())


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise SystemExit("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6-luna")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _provider(provider_name: str) -> ModelProvider:
    if provider_name == "openai":
        return OpenAIProvider()
    return AnthropicProvider()


def _require_api_key(provider_name: str) -> None:
    if provider_name == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
    if provider_name == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")


if __name__ == "__main__":
    main()
