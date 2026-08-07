from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

import cayu.evals.execution as execution_module
import cayu.evals.runner as runner_module
from cayu import (
    AgentSpec,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    StructuredOutputSpec,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
    _content_revision,
    pricing_profile_identity,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES,
    CorpusExecutionLimits,
    CorpusExecutionResult,
    CorpusTarget,
    compile_corpus_suite,
    evaluation_target_identity,
    run_corpus_suite,
)
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    corpus_execution_compatibility,
)
from cayu.evals.execution_reporting import (
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
    load_corpus_execution_result,
    render_corpus_execution_html,
    write_corpus_execution_result,
)
from cayu.evals.result_contract import EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES
from cayu.evals.runner import EvalPlan, run_eval_plan
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.vaults import SecretRedactor


def _source() -> EvaluationSourceIdentityV1:
    return EvaluationSourceIdentityV1(
        application_release_id="captured-release",
        app_manifest_schema_version="7",
        app_manifest_fingerprint="a" * 64,
        evidence_revision="sha256:" + "e" * 64,
    )


def _price_book(*, version: str = "prices-1") -> PriceBook:
    return PriceBook(
        price_book_version=version,
        generated_at="2026-08-06T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="fixture-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        ),
    )


def _corpus(
    *,
    target_key: str = "refund-agent",
    trials: int = 2,
    input_text: str = "Refund order 42.",
    price_book: PriceBook | None = None,
    expected_output: str = "Approved",
    include_root_status: bool = True,
) -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id="refund-regressions",
        name="Refund regressions",
        trial_request=TrialRequestSpec(trials=trials, timeout_seconds=30),
    )
    assertions = []
    if include_root_status:
        assertions.append(RootStatusAssertionSpec(id="completed", expected="completed"))
    assertions.append(FinalOutputEqualsAssertionSpec(id="answer", expected=expected_output))
    if price_book is not None:
        assertions.append(MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="USD"))
    case = EvalCaseSpec.create(
        id="refund-approval",
        suite_id=suite.id,
        name="Refund approval",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text=input_text),)),
        assertions=tuple(assertions),
    )
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing_profile=(None if price_book is None else pricing_profile_identity(price_book)),
        suites=(suite,),
        cases=(case,),
    )


def _target(
    provider: ModelProvider,
    *,
    key: str = "refund-agent",
    price_book: PriceBook | None = None,
    limits: CorpusExecutionLimits | None = None,
    application_release_id: str = "release-2026-08-06",
    secret_redactor: SecretRedactor | None = None,
) -> CorpusTarget:
    app = CayuApp(enable_logging=False, secret_redactor=secret_redactor)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    return CorpusTarget(
        key=key,
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
        bootstrap_messages=(Message.text("system", "Follow the refund policy."),),
        application_release_id=application_release_id,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        price_book=price_book,
        limits=limits or CorpusExecutionLimits(),
    )


def _provider(*, trials: int = 2, output: str = "Approved") -> ScriptedModelProvider:
    batch = (
        ModelStreamEvent.text_delta(output),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        ),
    )
    return ScriptedModelProvider([batch for _ in range(trials)])


def test_compile_corpus_suite_uses_only_trusted_bootstrap_then_corpus_user_input():
    provider = _provider()
    target = _target(provider)
    corpus = _corpus()

    compiled = compile_corpus_suite(corpus, target, "refund-regressions")

    assert compiled.run_contract.corpus_revision == corpus.revision
    assert compiled.trials == 2
    assert compiled.timeout_seconds == 30
    request = compiled.suite.cases[0].request
    assert [(message.role.value, message.content[0].text) for message in request.messages] == [
        ("system", "Follow the refund policy."),
        ("user", "Refund order 42."),
    ]
    assert request.session_id is None
    assert request.parent_session_id is None
    assert request.causal_budget_id is None
    assert request.task_id is None
    assert request.task_worker_id is None


def test_run_corpus_suite_retains_every_trial_and_fresh_target_identity():
    provider = _provider()
    target = _target(provider)
    corpus = _corpus()

    result = asyncio.run(run_corpus_suite(target, corpus, "refund-regressions"))

    assert result.target == evaluation_target_identity(target)
    assert result.schema_version == 1
    assert result.run.schema_version == 2
    assert result.target.application_release_id == "release-2026-08-06"
    assert result.target.app_manifest_fingerprint == target.app.describe().fingerprint
    assert result.run.corpus_revision == corpus.revision
    assert result.run.status == "passed"
    assert result.run.score == 1.0
    assert len(result.run.cases[0].trials) == 2
    assert [trial.status for trial in result.run.cases[0].trials] == ["passed", "passed"]
    assert len(provider.requests) == 2

    encoded = result.model_dump_json()
    assert type(result).model_validate_json(encoded) == result
    forged = result.model_copy(update={"revision": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError, match="revision does not match"):
        type(result).model_validate(forged.model_dump(mode="python"))

    document = result.model_dump(mode="json")
    document["run"]["cases"][0]["trials"][0]["output"] = {
        "schema_version": 1,
        "text": "",
        "evidence_state": "unavailable",
        "preview_truncated": False,
        "retained_chars": 0,
        "retained_bytes": 0,
        "retained_sha256": None,
    }
    document["run"]["revision"] = _content_revision(
        document["run"],
        "published eval run",
    )
    with pytest.raises(ValidationError, match="Scored corpus execution trials"):
        CorpusExecutionResult.model_validate_json(json.dumps(document))


@pytest.mark.parametrize(
    ("key", "application_release_id", "field_name"),
    (
        ("secret-token", "release", "key"),
        ("refund-agent", "release-secret-token", "application_release_id"),
    ),
)
def test_corpus_target_rejects_secret_bearing_public_identity_before_dispatch(
    key,
    application_release_id,
    field_name,
):
    provider = _provider()

    with pytest.raises(
        ValidationError,
        match=rf"{field_name} contains a workload secret",
    ) as exc_info:
        _target(
            provider,
            key=key,
            application_release_id=application_release_id,
            secret_redactor=SecretRedactor("secret-token"),
        )

    assert "secret-token" not in str(exc_info.value)
    assert provider.requests == []


def test_execution_revalidates_forged_secret_bearing_release_before_dispatch():
    provider = _provider()
    target = _target(provider, secret_redactor=SecretRedactor("secret-token"))
    forged = target.model_copy(update={"application_release_id": "secret-token"})

    with pytest.raises(
        ValidationError,
        match="application_release_id contains a workload secret",
    ) as exc_info:
        asyncio.run(run_corpus_suite(forged, _corpus(), "refund-regressions"))

    assert "secret-token" not in str(exc_info.value)
    assert provider.requests == []


def test_corpus_execution_discards_raw_trial_output_before_publication(monkeypatch):
    retained_outputs: list[str] = []
    publish = execution_module._publish_eval_run_with_trial_public_data

    def observe_internal_run(corpus, run, *, trial_public_data_by_case):
        retained_outputs.extend(trial.final_output for case in run.cases for trial in case.trials)
        return publish(
            corpus,
            run,
            trial_public_data_by_case=trial_public_data_by_case,
        )

    monkeypatch.setattr(
        execution_module,
        "_publish_eval_run_with_trial_public_data",
        observe_internal_run,
    )

    result = asyncio.run(run_corpus_suite(_target(_provider()), _corpus(), "refund-regressions"))

    assert result.run.status == "passed"
    assert retained_outputs == ["", ""]
    assert [trial.output.text for trial in result.run.cases[0].trials] == [
        "Approved",
        "Approved",
    ]
    assert [trial.code for trial in result.run.cases[0].trials] == ["passed", "passed"]


def test_eval_plan_corpus_mode_uses_the_shared_execution_service():
    provider = _provider()
    target = _target(provider)
    corpus = _corpus()

    result = asyncio.run(
        run_eval_plan(
            EvalPlan(corpus_target=target),
            corpus=corpus,
            suite_id="refund-regressions",
        )
    )

    assert result.run.status == "passed"
    assert len(provider.requests) == 2


def test_eval_plan_modes_are_mutually_exclusive_and_corpus_settings_are_authoritative():
    provider = _provider()
    target = _target(provider)
    corpus = _corpus()

    with pytest.raises(ValueError, match="exactly one mode"):
        EvalPlan()
    with pytest.raises(ValueError, match="exactly one mode"):
        EvalPlan(app=target.app, corpus_target=target)
    with pytest.raises(ValueError, match="come only from the corpus contract"):
        asyncio.run(
            run_eval_plan(
                EvalPlan(corpus_target=target),
                corpus=corpus,
                suite_id="refund-regressions",
                trials=1,
            )
        )
    assert provider.requests == []


def test_target_identity_rejects_manifest_content_with_a_forged_fingerprint():
    target = _target(_provider())
    manifest = target.app.describe().model_copy(update={"fingerprint": "0" * 64})

    with pytest.raises(ValidationError, match="fingerprint does not match"):
        type(evaluation_target_identity(target))(
            target_key=target.key,
            application_release_id=target.application_release_id,
            app_manifest=manifest,
        )


def test_execution_rejects_an_application_manifest_change_during_the_run(monkeypatch):
    provider = _provider()
    target = _target(provider)
    before = target.app.describe()
    target.app.register_agent(AgentSpec(name="changed-agent", model="fixture-model"))
    after = target.app.describe()
    manifests = iter((before, after))
    monkeypatch.setattr(target.app, "describe", lambda: next(manifests))

    with pytest.raises(RuntimeError, match="manifest changed"):
        asyncio.run(run_corpus_suite(target, _corpus(), "refund-regressions"))

    assert len(provider.requests) == 2


def test_published_execution_json_is_deterministic_bounded_and_loadable(tmp_path):
    result = asyncio.run(run_corpus_suite(_target(_provider()), _corpus(), "refund-regressions"))

    encoded = corpus_execution_result_to_json(result)
    destination = tmp_path / "result.json"
    write_corpus_execution_result(result, destination)

    assert encoded == corpus_execution_result_to_json(result)
    assert corpus_execution_result_from_json(encoded) == result
    assert load_corpus_execution_result(destination) == result
    assert destination.read_text(encoding="utf-8") == encoded

    duplicated = encoded.replace(
        '  "schema_version": 1,',
        '  "schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        corpus_execution_result_from_json(duplicated)

    versionless_run = json.loads(encoded)
    versionless_run["run"].pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version is required"):
        corpus_execution_result_from_json(json.dumps(versionless_run))


def test_published_execution_html_escapes_identity_and_shows_only_redacted_output():
    target = _target(
        _provider(output="Approved secret-token"),
        application_release_id='<script>alert("release")</script>',
        secret_redactor=SecretRedactor("secret-token"),
    )
    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(expected_output="Approved [REDACTED_SECRET]"),
            "refund-regressions",
        )
    )

    report = render_corpus_execution_html(result)

    assert '<script>alert("release")</script>' not in report
    assert "&lt;script&gt;alert(&quot;" in report
    assert "Approved [REDACTED_SECRET]" in report
    assert "secret-token" not in report
    assert "session_id" not in report
    assert result.target.app_manifest_fingerprint in report


def test_corpus_execution_bounds_long_redacted_output_and_explains_unavailability():
    output = "x" * 65_537
    result = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1, output=output)),
            _corpus(trials=1),
            "refund-regressions",
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.status == "unavailable"
    assert trial.code == "assertion_evidence_unavailable"
    assert trial.message == "Required assertion evidence was unavailable."
    assert trial.output.evidence_state == "limit_exceeded"
    assert trial.output.preview_truncated is True
    assert len(trial.output.text.encode("utf-8")) == EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES
    assert trial.output.retained_chars == 65_536
    assert trial.output.retained_sha256 == hashlib.sha256(b"x" * 65_536).hexdigest()
    assert output not in result.model_dump_json()


def test_failed_corpus_trial_retains_the_output_needed_to_diagnose_the_failure():
    result = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            _corpus(trials=1, expected_output="Denied"),
            "refund-regressions",
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.status == "failed"
    assert trial.code == "assertion_failed"
    assert trial.output.evidence_state == "complete"
    assert trial.output.text == "Approved"


class _FailingProvider(ModelProvider):
    name = "failing"

    async def stream(self, request):
        del request
        raise RuntimeError("provider exploded near secret-token")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _EchoProvider(ModelProvider):
    name = "echo"

    async def stream(self, request):
        output = request.messages[-1].content[0].text
        yield ModelStreamEvent.text_delta(output)
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )


def test_corpus_execution_publishes_stable_safe_failure_diagnostics():
    app = CayuApp(enable_logging=False)
    app.register_provider(_FailingProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    target = CorpusTarget(
        key="refund-agent",
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
        application_release_id="release",
    )

    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(trials=1, include_root_status=False),
            "refund-regressions",
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.status == "error"
    assert trial.code == "session_failed"
    assert trial.message == "The trial session failed."
    assert trial.output.evidence_state == "complete"
    assert "secret-token" not in result.model_dump_json()
    report = render_corpus_execution_html(result)
    assert "The trial session failed." in report
    assert "secret-token" not in report


def test_corpus_execution_distinguishes_safe_evidence_preparation_failures(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_prepare_portable_evidence",
        lambda *_args, **_kwargs: (None, RuntimeError("secret evidence failure")),
    )

    result = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            _corpus(trials=1),
            "refund-regressions",
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.status == "error"
    assert trial.code == "evidence_preparation_failed"
    assert trial.message == "Assertion evidence preparation failed."
    assert "secret evidence failure" not in result.model_dump_json()


def test_concurrent_corpus_execution_keeps_output_projection_bound_to_each_case():
    corpus = _corpus(
        trials=1,
        input_text="alpha output",
        expected_output="alpha output",
    )
    first = corpus.cases[0]
    second = EvalCaseSpec.create(
        id="refund-output-beta",
        suite_id=first.suite_id,
        name="Beta output",
        source=first.source,
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="beta output"),)),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputEqualsAssertionSpec(id="answer", expected="beta output"),
        ),
    )
    corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        suites=corpus.suites,
        cases=(first, second),
    )

    result = asyncio.run(
        run_corpus_suite(
            _target(_EchoProvider()),
            corpus,
            "refund-regressions",
            max_concurrency=2,
        )
    )

    assert {case.case_id: case.trials[0].output.text for case in result.run.cases} == {
        "refund-approval": "alpha output",
        "refund-output-beta": "beta output",
    }


def test_comparison_compatibility_permits_a_different_release_and_manifest():
    baseline = asyncio.run(run_corpus_suite(_target(_provider()), _corpus(), "refund-regressions"))
    changed_target = _target(
        _provider(),
        application_release_id="release-2026-08-07",
    )
    changed_target.app.register_agent(AgentSpec(name="diagnostic-change", model="fixture-model"))
    current = CorpusExecutionResult.create(
        target=evaluation_target_identity(changed_target),
        run=baseline.run,
    )

    compatibility = corpus_execution_compatibility(baseline, current)

    assert compatibility.comparable is True
    assert compatibility.reasons == ()
    assert baseline.target.application_release_id != current.target.application_release_id
    assert baseline.target.app_manifest_fingerprint != current.target.app_manifest_fingerprint


def test_comparison_compatibility_reports_changed_corpus_and_case_contracts():
    baseline = asyncio.run(run_corpus_suite(_target(_provider()), _corpus(), "refund-regressions"))
    current = asyncio.run(
        run_corpus_suite(
            _target(_provider()),
            _corpus(input_text="Refund order 43."),
            "refund-regressions",
        )
    )

    compatibility = corpus_execution_compatibility(baseline, current)

    assert compatibility.comparable is False
    assert compatibility.reasons == (
        CorpusComparisonReason.CORPUS_REVISION_MISMATCH,
        CorpusComparisonReason.CASE_CONTRACT_MISMATCH,
    )


@pytest.mark.parametrize(
    "request_base",
    [
        RunRequest(agent_name="agent", messages=[Message.text("user", "not empty")]),
        RunRequest(agent_name="agent", messages=[], session_id="fixed-session"),
        RunRequest(agent_name="agent", messages=[], parent_session_id="parent"),
        RunRequest(agent_name="agent", messages=[], causal_budget_id="budget"),
        RunRequest(agent_name="agent", messages=[], task_id="task"),
    ],
)
def test_corpus_target_rejects_message_or_runtime_identity_bearing_request_base(request_base):
    app = CayuApp(enable_logging=False)

    with pytest.raises(ValidationError, match="no messages|runtime identity fields"):
        CorpusTarget(
            key="refund-agent",
            app=app,
            request_base=request_base,
            application_release_id="release",
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda request: setattr(
                request,
                "structured_output",
                StructuredOutputSpec(json_schema={"type": "string"}),
            ),
            "structured output",
        ),
        (
            lambda request: setattr(
                request,
                "_runtime_generated_authority",
                frozenset({("session_id", "runtime-session")}),
            ),
            "runtime-generated authority",
        ),
        (
            lambda request: setattr(request, "_input_redactions_applied", True),
            "prior input-redaction state",
        ),
    ],
)
def test_corpus_target_rejects_nonfresh_request_state(mutate, match):
    request = RunRequest(agent_name="agent", messages=[])
    mutate(request)

    with pytest.raises(ValidationError, match=match):
        CorpusTarget(
            key="refund-agent",
            app=CayuApp(enable_logging=False),
            request_base=request,
            application_release_id="release",
        )


def test_target_and_execution_limits_fail_before_provider_dispatch():
    provider = _provider()
    target = _target(
        provider,
        limits=CorpusExecutionLimits(max_trials=1, max_total_input_chars=64),
    )

    with pytest.raises(ValueError, match="trial limit"):
        asyncio.run(run_corpus_suite(target, _corpus(trials=2), "refund-regressions"))
    assert provider.requests == []

    with pytest.raises(ValueError, match="target key"):
        asyncio.run(
            run_corpus_suite(target, _corpus(target_key="other-agent"), "refund-regressions")
        )
    assert provider.requests == []


def test_compilation_bounds_bootstrap_amplification_across_the_selected_suite():
    base = _corpus(trials=1)
    first = base.cases[0]
    second = EvalCaseSpec.create(
        id="refund-approval-two",
        suite_id=first.suite_id,
        name="Refund approval two",
        source=first.source,
        input=first.input,
        assertions=first.assertions,
    )
    corpus = EvalCorpusDocument.create(
        target_key=base.target_key,
        evidence_policy=base.evidence_policy,
        suites=base.suites,
        cases=(first, second),
    )
    per_case_chars = len("Follow the refund policy.") + len("Refund order 42.")
    provider = _provider(trials=1)
    target = _target(
        provider,
        limits=CorpusExecutionLimits(max_compiled_input_chars=per_case_chars),
    )

    with pytest.raises(ValueError, match="compiled-input limit"):
        compile_corpus_suite(corpus, target, "refund-regressions")

    assert provider.requests == []


def test_pricing_identity_mismatch_fails_before_provider_dispatch():
    provider = _provider()
    corpus_price_book = _price_book(version="captured-prices")
    target = _target(provider, price_book=_price_book(version="current-prices"))

    with pytest.raises(ValueError, match="pricing profile"):
        asyncio.run(
            run_corpus_suite(
                target,
                _corpus(price_book=corpus_price_book),
                "refund-regressions",
            )
        )
    assert provider.requests == []


def test_corpus_compilation_builds_one_shared_pricing_binding(monkeypatch):
    import cayu.evals.portable_assertions as portable_assertions_module

    price_book = _price_book()
    base = _corpus(trials=1, price_book=price_book)
    base_case = base.cases[0]
    assertions = tuple(
        MaxEstimatedCostAssertionSpec(
            id=f"cost-{index}",
            maximum="1",
            currency="USD",
        )
        for index in range(64)
    )
    case = EvalCaseSpec.create(
        id=base_case.id,
        suite_id=base_case.suite_id,
        name=base_case.name,
        source=base_case.source,
        input=base_case.input,
        assertions=assertions,
    )
    corpus = EvalCorpusDocument.create(
        target_key=base.target_key,
        evidence_policy=base.evidence_policy,
        pricing_profile=base.pricing_profile,
        suites=base.suites,
        cases=(case,),
    )
    target = _target(_provider(trials=1), price_book=price_book)
    identity_calls = 0
    pricing_identity = portable_assertions_module.pricing_profile_identity

    def counted_identity(source):
        nonlocal identity_calls
        identity_calls += 1
        return pricing_identity(source)

    monkeypatch.setattr(
        portable_assertions_module,
        "pricing_profile_identity",
        counted_identity,
    )

    compiled = compile_corpus_suite(corpus, target, base.suites[0].id)

    bindings = tuple(
        object.__getattribute__(assertion, "_pricing_binding")
        for assertion in compiled.suite.cases[0].assertions
    )
    assert identity_calls == 1
    assert len({id(binding) for binding in bindings}) == 1


def test_full_target_validation_reuses_corpus_and_pricing_across_suites(monkeypatch):
    import cayu.evals.portable_assertions as portable_assertions_module

    price_book = _price_book()
    base = _corpus(trials=1, price_book=price_book)
    second_suite = EvalSuiteSpec.create(
        id="second-regressions",
        name="Second regressions",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    base_case = base.cases[0]
    second_case = EvalCaseSpec.create(
        id="second-case",
        suite_id=second_suite.id,
        name="Second case",
        source=base_case.source,
        input=base_case.input,
        assertions=base_case.assertions,
    )
    corpus = EvalCorpusDocument.create(
        target_key=base.target_key,
        evidence_policy=base.evidence_policy,
        pricing_profile=base.pricing_profile,
        suites=(*base.suites, second_suite),
        cases=(*base.cases, second_case),
    )
    target = _target(_provider(trials=1), price_book=price_book)
    identity_calls = 0
    pricing_identity = execution_module.pricing_profile_identity

    def counted_identity(source):
        nonlocal identity_calls
        identity_calls += 1
        return pricing_identity(source)

    monkeypatch.setattr(execution_module, "pricing_profile_identity", counted_identity)
    monkeypatch.setattr(
        portable_assertions_module,
        "pricing_profile_identity",
        counted_identity,
    )

    execution_module._validate_corpus_target_compatibility(corpus, target)

    assert 0 < identity_calls <= 2


def test_pricing_is_required_only_for_the_selected_suite_that_uses_it():
    price_book = _price_book()
    priced = _corpus(trials=1, price_book=price_book)
    plain_suite = EvalSuiteSpec.create(
        id="plain-regressions",
        name="Plain regressions",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    plain_case = EvalCaseSpec.create(
        id="plain-case",
        suite_id=plain_suite.id,
        name="Plain case",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run plain case."),)),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
    )
    corpus = EvalCorpusDocument.create(
        target_key=priced.target_key,
        evidence_policy=priced.evidence_policy,
        pricing_profile=priced.pricing_profile,
        suites=(*priced.suites, plain_suite),
        cases=(*priced.cases, plain_case),
    )
    provider = _provider(trials=1)
    target = _target(provider, price_book=None)

    result = asyncio.run(run_corpus_suite(target, corpus, plain_suite.id))

    assert result.run.status == "passed"
    assert result.run.pricing_profile_fingerprint is None
    with pytest.raises(ValueError, match="pricing profile"):
        compile_corpus_suite(corpus, target, priced.suites[0].id)
    assert len(provider.requests) == 1


def test_corpus_target_defensively_copies_request_bootstrap_policy_and_pricing():
    price_book = _price_book()
    request = RunRequest(agent_name="agent", messages=[], metadata={"safe": ["original"]})
    bootstrap = Message.text("system", "Original")
    app = CayuApp(enable_logging=False)
    target = CorpusTarget(
        key="refund-agent",
        app=app,
        request_base=request,
        bootstrap_messages=(bootstrap,),
        application_release_id="release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        price_book=price_book,
    )

    request.metadata["safe"].append("mutated")
    price_book.price_book_version = "mutated"

    assert target.request_base.metadata == {"safe": ["original"]}
    assert target.bootstrap_messages[0] is not bootstrap
    assert target.price_book is not price_book
    assert target.price_book.price_book_version == "prices-1"


def test_corpus_target_rejects_oversized_trusted_request_base():
    app = CayuApp(enable_logging=False)
    request = RunRequest(
        agent_name="agent",
        messages=[],
        metadata={"oversized": "x" * CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES},
    )

    with pytest.raises(ValidationError, match="request_base exceeds"):
        CorpusTarget(
            key="refund-agent",
            app=app,
            request_base=request,
            application_release_id="release",
        )
