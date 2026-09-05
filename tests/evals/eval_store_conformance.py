from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import pytest

from cayu.evals.calibration import EvalJudgeCalibrationReportV1
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvaluationSourceIdentityV1,
    RunInputSpec,
    _content_revision,
)
from cayu.evals.execution import (
    CorpusExecutionResult,
    EvaluationTargetIdentity,
)
from cayu.evals.models import EvalStatus, EvalTrialResult
from cayu.evals.promotion import CapturedRunScoreV1
from cayu.evals.result_contract import (
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
)
from cayu.evals.scenario import (
    EvalScenarioDocumentV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioQueuedInputEventV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
)
from cayu.evals.store import (
    EvalAuthoredSuiteCatalogQuery,
    EvalAuthoredSuiteReferenceError,
    EvalBaselineConflict,
    EvalBaselineKey,
    EvalBaselineMutationRecord,
    EvalBaselineUpdate,
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalResultQuery,
    EvalRunClaimLost,
    EvalRunCostBudget,
    EvalRunFailureCode,
    EvalRunFailureDiagnostic,
    EvalRunFailureReason,
    EvalRunInvocation,
    EvalRunQuery,
    EvalRunRequest,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalRunTrialCheckpoint,
    EvalScenarioApprovalSubmission,
    EvalScenarioCatalogQuery,
    EvalScenarioRunInvocation,
    EvalScenarioRunProgress,
    EvalScenarioTrialPhase,
    EvalScenarioTrialProgress,
    EvalStore,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogQuery,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV1,
    EvalSuiteDraftV1,
    compile_eval_suite_draft,
)
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.runtime.manifest import AppManifest, ToolManifest, _app_manifest_fingerprint
from cayu.runtime.stop_policy import RunLimits
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


def _scenario(corpus: EvalCorpusDocument, *, text: str) -> EvalScenarioDocumentV2:
    def scenario_input(value: str) -> ScenarioInputV2:
        return ScenarioInputV2.create(
            (
                ScenarioUserMessageV2.create(
                    (ScenarioTextPartV2(text=value),),
                ),
            )
        )

    return EvalScenarioDocumentV2.create(
        id="checkout-regression",
        target_key=corpus.target_key,
        name="Checkout regression",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=scenario_input(text),
            ),
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="approve-payment",
                tool_name="charge-card",
                occurrence=1,
            ),
            ScenarioQueuedInputEventV2(
                sequence=2,
                id="follow-up",
                delivery_mode="next_turn",
                input=scenario_input("Confirm the final amount."),
            ),
        ),
    )


def _corpus_with_input(corpus: EvalCorpusDocument, text: str) -> EvalCorpusDocument:
    original = corpus.cases[0]
    case = EvalCaseSpec.create(
        id=original.id,
        suite_id=original.suite_id,
        name=original.name,
        description=original.description,
        source=original.source,
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text=text),)),
        assertions=original.assertions,
    )
    return EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(case,),
    )


def _corpus_with_target(corpus: EvalCorpusDocument, target_key: str) -> EvalCorpusDocument:
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=corpus.cases,
    )


def _broken_redaction_boundary(_value):
    raise RuntimeError("must not cross the store boundary")


def _result_with_run_update(result: CorpusExecutionResult, **updates) -> CorpusExecutionResult:
    changed = result.run.model_copy(update=updates)
    revision_document = changed.model_dump(mode="json", exclude={"revision"})
    changed = changed.model_copy(
        update={"revision": _content_revision(revision_document, "published eval run")}
    )
    return CorpusExecutionResult.create(target=result.target, run=changed)


def captured_result_for_corpus(
    corpus: EvalCorpusDocument,
    fresh_result: CorpusExecutionResult,
) -> CapturedEvaluationResultV1:
    """Build the exact captured counterpart used by store conformance tests."""

    case = corpus.cases[0]
    source = case.source
    assert source is not None
    assertions = fresh_result.run.cases[0].trials[0].assertions
    score_document = {
        "schema_version": 1,
        "candidate_revision": "sha256:" + "c" * 64,
        "case_id": case.id,
        "case_revision": case.revision,
        "evidence_revision": source.evidence_revision,
        "evidence_policy_revision": corpus.evidence_policy.revision,
        "pricing_profile_fingerprint": (
            None if corpus.pricing_profile is None else corpus.pricing_profile.fingerprint
        ),
        "memory_attribution": fresh_result.run.cases[0]
        .trials[0]
        .memory_attribution.model_dump(mode="json"),
        "status": fresh_result.run.cases[0].status,
        "score": fresh_result.run.cases[0].score,
        "assertions": [item.model_dump(mode="json") for item in assertions],
    }
    score = CapturedRunScoreV1.model_validate_json(
        json.dumps(
            {
                "revision": _content_revision(score_document, "captured run score"),
                **score_document,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return CapturedEvaluationResultV1.create(
        corpus=corpus,
        target=EvalResultTargetIdentityV1(
            target_key=corpus.target_key,
            application_release_id=source.application_release_id,
            app_manifest_schema_version=source.app_manifest_schema_version,
            app_manifest_fingerprint=source.app_manifest_fingerprint,
        ),
        score=score,
    )


def _result_with_secret_manifest_key(
    result: CorpusExecutionResult,
    secret: str,
) -> CorpusExecutionResult:
    manifest = result.target.app_manifest
    agent = manifest.agents[0]
    tool = ToolManifest(
        name="credential-key-probe",
        description="Credential-key publication regression.",
        effect="read",
        parallel_safe=True,
        input_schema={secret: {"type": "string"}},
        policy_coverage="allowed",
        registration_provenance=agent.registration_provenance,
        implementation_provenance=agent.implementation_provenance,
    )
    document = manifest.model_dump(mode="json")
    document["agents"][0]["tools"] = [tool.model_dump(mode="json")]
    document.pop("fingerprint")
    document["fingerprint"] = _app_manifest_fingerprint(document)
    target = EvaluationTargetIdentity(
        target_key=result.target.target_key,
        application_release_id=result.target.application_release_id,
        app_manifest=AppManifest.model_validate(document),
    )
    return CorpusExecutionResult.create(target=target, run=result.run)


def _results_with_conflicting_corpus_contract(
    result: CorpusExecutionResult,
) -> tuple[CorpusExecutionResult, ...]:
    wrong_revision = "sha256:" + "f" * 64
    case = result.run.cases[0]
    trial = case.trials[0]
    assertion = trial.assertions[0]
    contradictory_detail = assertion.detail.model_copy(
        update={"expected": "failed", "actual": "failed"}
    )
    contradictory_assertion = assertion.model_copy(update={"detail": contradictory_detail})
    contradictory_trial = trial.model_copy(
        update={"assertions": (contradictory_assertion, *trial.assertions[1:])}
    )
    contradictory_case = case.model_copy(update={"trials": (contradictory_trial, *case.trials[1:])})
    return (
        _result_with_run_update(result, evidence_policy_revision=wrong_revision),
        _result_with_run_update(result, pricing_profile_fingerprint=wrong_revision),
        _result_with_run_update(
            result,
            cases=(
                case.model_copy(update={"case_revision": wrong_revision}),
                *result.run.cases[1:],
            ),
        ),
        _result_with_run_update(
            result,
            cases=(contradictory_case, *result.run.cases[1:]),
        ),
    )


def _request(
    corpus,
    *,
    suffix: str,
    concurrency: int = 1,
    invocation: EvalRunInvocation | None = None,
) -> EvalRunRequest:
    suite = corpus.suites[0]
    digest_character = {"a": "a", "b": "b", "c": "c"}[suffix]
    return EvalRunRequest(
        run_id=f"conformance-{suffix}",
        idempotency_key="sha256:" + digest_character * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=concurrency,
        invocation=EvalRunInvocation() if invocation is None else invocation,
    )


def _terminal_trial_checkpoint(corpus: EvalCorpusDocument) -> EvalRunTrialCheckpoint:
    now = datetime.now(UTC)
    return EvalRunTrialCheckpoint(
        case_id=corpus.cases[0].id,
        result=EvalTrialResult(
            trial_number=1,
            status=EvalStatus.ERROR,
            error="candidate execution failed",
            started_at=now,
            completed_at=now,
        ),
        public_data=_EvalTrialPublicData(
            diagnostic_code=EvalTrialDiagnosticCode.EXECUTION_FAILED,
            output=EvalTrialOutputPreviewV1.unavailable(),
        ),
    )


async def assert_scenario_progress_conformance(
    store: EvalStore,
    *,
    corpus: EvalCorpusDocument,
) -> None:
    """Exercise fenced progress, approval CAS, and restart checkpoint recovery."""

    scenario = _scenario(corpus, text="Execute a controlled checkout.")
    await store.save_scenario(scenario, redact_json=_NO_SECRETS.redact_json)
    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    binding_revision = "sha256:" + "b" * 64
    suite = corpus.suites[0]
    request = EvalRunRequest(
        run_id="scenario-progress-run",
        idempotency_key="sha256:" + "9" * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=1,
        invocation=EvalRunInvocation(
            scenario=EvalScenarioRunInvocation(
                scenario_revision=scenario.revision,
                binding_revision=binding_revision,
                trials=1,
                timeout_seconds=30,
            )
        ),
    )
    await store.admit_run(request, redact_json=_NO_SECRETS.redact_json)
    first = await store.claim_run(target_key=corpus.target_key, lease_seconds=30)
    assert first is not None
    progress = EvalScenarioRunProgress.create(
        scenario_revision=scenario.revision,
        binding_revision=binding_revision,
        attempt=first.claim.epoch,
        trials=(
            EvalScenarioTrialProgress(
                trial_number=1,
                phase=EvalScenarioTrialPhase.PENDING,
                next_event_sequence=0,
            ),
        ),
    )
    await store.initialize_scenario_progress(first.claim, progress)
    waiting = await store.update_scenario_trial(
        first.claim,
        EvalScenarioTrialProgress(
            trial_number=1,
            phase=EvalScenarioTrialPhase.AWAITING_APPROVAL,
            session_id="scenario-session-1",
            next_event_sequence=2,
            pending_event_id="approve-payment",
            pending_tool_name="charge-card",
        ),
    )
    assert waiting.scenario_progress is not None
    with pytest.raises(EvalRunStateConflict, match="progress changed"):
        await store.submit_scenario_approval(
            request.run_id,
            EvalScenarioApprovalSubmission(
                expected_progress_revision="sha256:" + "0" * 64,
                trial_number=1,
                event_id="approve-payment",
                decision="approve",
                actor_id="reviewer",
            ),
        )
    approved = await store.submit_scenario_approval(
        request.run_id,
        EvalScenarioApprovalSubmission(
            expected_progress_revision=waiting.scenario_progress.revision,
            trial_number=1,
            event_id="approve-payment",
            decision="approve",
            actor_id="reviewer",
        ),
    )
    assert approved.scenario_progress is not None
    assert approved.scenario_progress.trials[0].approval is not None
    stale_revision = approved.scenario_progress.revision

    await store.release_run(first.claim)
    second = await store.claim_run(target_key=corpus.target_key, lease_seconds=30)
    assert second is not None
    assert second.claim.epoch == first.claim.epoch + 1
    resumed = second.run.scenario_progress
    assert resumed is not None
    assert resumed.attempt == second.claim.epoch
    assert resumed.revision != stale_revision
    assert resumed.trials[0].phase is EvalScenarioTrialPhase.AWAITING_APPROVAL
    assert resumed.trials[0].session_id == "scenario-session-1"
    assert resumed.trials[0].approval is not None
    with pytest.raises(EvalRunStateConflict, match="progress"):
        await store.submit_scenario_approval(
            request.run_id,
            EvalScenarioApprovalSubmission(
                expected_progress_revision=stale_revision,
                trial_number=1,
                event_id="approve-payment",
                decision="approve",
                actor_id="reviewer",
            ),
        )
    await store.update_scenario_trial(
        second.claim,
        EvalScenarioTrialProgress(
            trial_number=1,
            phase=EvalScenarioTrialPhase.RUNNING,
            session_id="scenario-session-1",
            next_event_sequence=2,
        ),
    )
    await store.release_run(second.claim)
    third = await store.claim_run(target_key=corpus.target_key, lease_seconds=30)
    assert third is not None
    restarted = third.run.scenario_progress
    assert restarted is not None
    assert restarted.attempt == third.claim.epoch
    assert restarted.trials == (
        EvalScenarioTrialProgress(
            trial_number=1,
            phase=EvalScenarioTrialPhase.PENDING,
            next_event_sequence=0,
        ),
    )
    await store.release_run(third.claim)
    fourth = await store.claim_run(target_key=corpus.target_key, lease_seconds=30)
    assert fourth is not None
    await store.update_scenario_trial(
        fourth.claim,
        EvalScenarioTrialProgress(
            trial_number=1,
            phase=EvalScenarioTrialPhase.AWAITING_RESUME,
            session_id="scenario-session-2",
            next_event_sequence=3,
            pending_event_id="resume-environment",
            pending_input_id="input-1",
            pending_resume_kind="user_input",
        ),
    )
    await store.release_run(fourth.claim)
    fifth = await store.claim_run(target_key=corpus.target_key, lease_seconds=30)
    assert fifth is not None
    resume_progress = fifth.run.scenario_progress
    assert resume_progress is not None
    assert resume_progress.attempt == fifth.claim.epoch
    assert resume_progress.trials[0].phase is EvalScenarioTrialPhase.AWAITING_RESUME
    assert resume_progress.trials[0].session_id == "scenario-session-2"
    assert resume_progress.trials[0].pending_input_id == "input-1"
    await store.release_run(fifth.claim)


async def assert_eval_store_reconstruction_releases_heartbeat_capacity(
    store: EvalStore,
    *,
    corpus: EvalCorpusDocument,
    result: CorpusExecutionResult,
    read_kind: Literal["corpus", "result"],
    parser_owner: object,
    parser_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove CPU reconstruction cannot occupy durable lease-operation capacity."""

    if read_kind not in {"corpus", "result"}:
        raise ValueError("read_kind must be corpus or result.")
    suite = corpus.suites[0]

    def request(run_id: str, idempotency_character: str) -> EvalRunRequest:
        return EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + idempotency_character * 64,
            corpus_revision=corpus.revision,
            target_key=corpus.target_key,
            suite_id=suite.id,
            suite_revision=suite.revision,
            max_concurrency=1,
        )

    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(
        request("capacity-completed-run", "7"),
        redact_json=_NO_SECRETS.redact_json,
    )
    completed_lease = await store.claim_run(target_key=corpus.target_key)
    assert completed_lease is not None
    await store.publish_result(
        completed_lease.claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )

    await store.admit_run(
        request("capacity-active-run", "8"),
        redact_json=_NO_SECRETS.redact_json,
    )
    active_lease = await store.claim_run(
        target_key=corpus.target_key,
        lease_seconds=5,
    )
    assert active_lease is not None

    parser_started = threading.Event()
    release_parser = threading.Event()
    original_parser = getattr(parser_owner, parser_name)

    def blocking_parser(document):
        parser_started.set()
        if not release_parser.wait(timeout=5):
            raise AssertionError("Timed out releasing eval document reconstruction.")
        return original_parser(document)

    monkeypatch.setattr(parser_owner, parser_name, blocking_parser)
    read_task = asyncio.create_task(
        store.load_corpus(corpus.revision)
        if read_kind == "corpus"
        else store.load_result("capacity-completed-run")
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2
        while not parser_started.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert parser_started.is_set()

        heartbeat = await asyncio.wait_for(
            store.heartbeat_run(active_lease.claim, extend_seconds=5),
            timeout=2,
        )
        assert heartbeat.ownership is not None
        assert heartbeat.ownership.epoch == active_lease.claim.epoch

        release_parser.set()
        assert await read_task == (corpus if read_kind == "corpus" else result)
        await store.release_run(active_lease.claim)
    finally:
        release_parser.set()
        await asyncio.gather(read_task, return_exceptions=True)


async def assert_eval_store_conformance(
    store: EvalStore,
    *,
    corpus,
    result: CorpusExecutionResult,
) -> None:
    """Pin backend-neutral catalog, lifecycle, fencing, and result semantics."""

    assert store.scenarios is True
    assert store.suite_authoring is True
    first_scenario = _scenario(corpus, text="Buy the standard plan.")
    first_scenario_entry = await store.save_scenario(
        first_scenario,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert (
        await store.save_scenario(
            first_scenario,
            redact_json=_NO_SECRETS.redact_json,
        )
        == first_scenario_entry
    )
    scenario_bytes = len(
        json.dumps(
            first_scenario.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert first_scenario_entry.document_bytes == scenario_bytes
    assert (
        await store.load_scenario(first_scenario.revision, max_bytes=scenario_bytes)
        == first_scenario
    )
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_scenario(first_scenario.revision, max_bytes=scenario_bytes - 1)

    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    unsafe_scenario = _scenario(corpus, text=secret)
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.save_scenario(
            unsafe_scenario,
            redact_json=SecretRedactor(secret).redact_json,
        )
    assert await store.load_scenario(unsafe_scenario.revision) is None
    with pytest.raises(EvalStorePublicationRejected, match="could not cross"):
        await store.save_scenario(
            first_scenario,
            redact_json=_broken_redaction_boundary,
        )

    second_scenario = _scenario(corpus, text="Buy the premium plan.")
    second_scenario_entry = await store.save_scenario(
        second_scenario,
        redact_json=_NO_SECRETS.redact_json,
    )
    first_page = await store.list_scenarios(
        EvalScenarioCatalogQuery(
            target_key=corpus.target_key,
            scenario_id=first_scenario.id,
            limit=1,
        )
    )
    assert len(first_page.items) == 1
    assert first_page.has_more is True
    assert first_page.next_cursor is not None
    second_page = await store.list_scenarios(
        EvalScenarioCatalogQuery(
            target_key=corpus.target_key,
            scenario_id=first_scenario.id,
            limit=1,
            cursor=first_page.next_cursor,
        )
    )
    assert second_page.has_more is False
    assert {first_page.items[0].revision, second_page.items[0].revision} == {
        first_scenario_entry.revision,
        second_scenario_entry.revision,
    }
    with pytest.raises(ValueError, match="cursor does not match"):
        await store.list_scenarios(
            EvalScenarioCatalogQuery(
                target_key="another-target",
                limit=1,
                cursor=first_page.next_cursor,
            )
        )

    corpus_case = corpus.cases[0]
    assert corpus_case.input is not None
    authored_simple_case = EvalCaseDraftV1(
        id="authored-simple",
        name="Authored simple case",
        source=None,
        stimulus=EvalSimpleInputStimulusV1(input=corpus_case.input),
        assertions=corpus_case.assertions,
    )
    authored_scenario_case = EvalCaseDraftV1(
        id="authored-scenario",
        name="Authored scenario case",
        source=None,
        stimulus=EvalScenarioStimulusV1(
            scenario_id=first_scenario.id,
            scenario_revision=first_scenario.revision,
        ),
        assertions=corpus_case.assertions,
    )
    authored_suite = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="authored-regressions",
            target_key=corpus.target_key,
            name="Authored regressions",
            cases=(authored_simple_case, authored_scenario_case),
        )
    )
    authored_entry = await store.save_authored_suite(
        authored_suite,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert (
        await store.save_authored_suite(
            authored_suite,
            redact_json=_NO_SECRETS.redact_json,
        )
        == authored_entry
    )
    authored_bytes = len(
        json.dumps(
            authored_suite.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert authored_entry.document_bytes == authored_bytes
    assert authored_entry.simple_input_count == 1
    assert authored_entry.scenario_count == 1
    assert (
        await store.load_authored_suite(
            authored_suite.revision,
            max_bytes=authored_bytes,
        )
        == authored_suite
    )
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_authored_suite(
            authored_suite.revision,
            max_bytes=authored_bytes - 1,
        )
    authored_page = await store.list_authored_suites(
        EvalAuthoredSuiteCatalogQuery(
            target_key=corpus.target_key,
            suite_id=authored_suite.suite.id,
        )
    )
    assert authored_page.items == (authored_entry,)

    revised_authored_suite = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id=authored_suite.suite.id,
            target_key=corpus.target_key,
            name="Authored regressions revised",
            cases=(authored_simple_case, authored_scenario_case),
        )
    )
    revised_authored_entry = await store.save_authored_suite(
        revised_authored_suite,
        redact_json=_NO_SECRETS.redact_json,
    )
    first_authored_page = await store.list_authored_suites(
        EvalAuthoredSuiteCatalogQuery(
            target_key=corpus.target_key,
            suite_id=authored_suite.suite.id,
            limit=1,
        )
    )
    assert len(first_authored_page.items) == 1
    assert first_authored_page.has_more is True
    assert first_authored_page.next_cursor is not None
    second_authored_page = await store.list_authored_suites(
        EvalAuthoredSuiteCatalogQuery(
            target_key=corpus.target_key,
            suite_id=authored_suite.suite.id,
            limit=1,
            cursor=first_authored_page.next_cursor,
        )
    )
    assert second_authored_page.has_more is False
    assert {
        first_authored_page.items[0].revision,
        second_authored_page.items[0].revision,
    } == {authored_entry.revision, revised_authored_entry.revision}
    with pytest.raises(ValueError, match="cursor does not match"):
        await store.list_authored_suites(
            EvalAuthoredSuiteCatalogQuery(
                target_key="another-target",
                limit=1,
                cursor=first_authored_page.next_cursor,
            )
        )

    unsafe_authored_suite = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="unsafe-authored-suite",
            target_key=corpus.target_key,
            name="Unsafe authored suite",
            cases=(
                authored_simple_case.model_copy(
                    update={
                        "stimulus": EvalSimpleInputStimulusV1(
                            input=RunInputSpec(messages=(CorpusUserMessageSpec(text=secret),))
                        )
                    }
                ),
            ),
        )
    )
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.save_authored_suite(
            unsafe_authored_suite,
            redact_json=SecretRedactor(secret).redact_json,
        )
    assert await store.load_authored_suite(unsafe_authored_suite.revision) is None
    with pytest.raises(EvalStorePublicationRejected, match="could not cross"):
        await store.save_authored_suite(
            authored_suite,
            redact_json=_broken_redaction_boundary,
        )

    dangling = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="dangling-scenario",
            target_key=corpus.target_key,
            name="Dangling scenario",
            cases=(
                authored_scenario_case.model_copy(
                    update={
                        "stimulus": EvalScenarioStimulusV1(
                            scenario_id=first_scenario.id,
                            scenario_revision="sha256:" + "f" * 64,
                        )
                    }
                ),
            ),
        )
    )
    with pytest.raises(EvalAuthoredSuiteReferenceError, match="unavailable"):
        await store.save_authored_suite(
            dangling,
            redact_json=_NO_SECRETS.redact_json,
        )
    assert await store.load_authored_suite(dangling.revision) is None

    def suite_for_scenario_reference(
        *,
        suite_id: str,
        reference: EvalScenarioStimulusV1,
        source: EvaluationSourceIdentityV1 | None = None,
    ) -> EvalSuiteDocumentV1:
        return compile_eval_suite_draft(
            EvalSuiteDraftV1(
                id=suite_id,
                target_key=corpus.target_key,
                name="Invalid scenario reference",
                cases=(
                    authored_scenario_case.model_copy(
                        update={"source": source, "stimulus": reference}
                    ),
                ),
            )
        )

    wrong_id = suite_for_scenario_reference(
        suite_id="wrong-scenario-id",
        reference=EvalScenarioStimulusV1(
            scenario_id="different-scenario",
            scenario_revision=first_scenario.revision,
        ),
    )
    with pytest.raises(EvalAuthoredSuiteReferenceError, match="ID does not match"):
        await store.save_authored_suite(
            wrong_id,
            redact_json=_NO_SECRETS.redact_json,
        )

    wrong_source = suite_for_scenario_reference(
        suite_id="wrong-scenario-source",
        reference=authored_scenario_case.stimulus,
        source=corpus_case.source,
    )
    with pytest.raises(EvalAuthoredSuiteReferenceError, match="source does not match"):
        await store.save_authored_suite(
            wrong_source,
            redact_json=_NO_SECRETS.redact_json,
        )

    other_target_scenario = EvalScenarioDocumentV2.create(
        id=first_scenario.id,
        target_key="another-target",
        name=first_scenario.name,
        events=first_scenario.events,
    )
    await store.save_scenario(
        other_target_scenario,
        redact_json=_NO_SECRETS.redact_json,
    )
    wrong_target = suite_for_scenario_reference(
        suite_id="wrong-scenario-target",
        reference=EvalScenarioStimulusV1(
            scenario_id=other_target_scenario.id,
            scenario_revision=other_target_scenario.revision,
        ),
    )
    with pytest.raises(EvalAuthoredSuiteReferenceError, match="target does not match"):
        await store.save_authored_suite(
            wrong_target,
            redact_json=_NO_SECRETS.redact_json,
        )

    saved = await store.save_corpus(
        corpus,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert (
        await store.save_corpus(
            corpus,
            redact_json=_NO_SECRETS.redact_json,
        )
        == saved
    )
    corpus_bytes = len(
        json.dumps(
            corpus.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert saved.document_bytes == corpus_bytes
    assert await store.load_corpus(corpus.revision, max_bytes=corpus_bytes) == corpus
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_corpus(corpus.revision, max_bytes=corpus_bytes - 1)

    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    unsafe = _corpus_with_input(corpus, secret)
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.save_corpus(
            unsafe,
            redact_json=SecretRedactor(secret).redact_json,
        )
    assert await store.load_corpus(unsafe.revision) is None
    with pytest.raises(EvalStorePublicationRejected, match="could not cross"):
        await store.save_corpus(
            corpus,
            redact_json=_broken_redaction_boundary,
        )

    corpora = await store.list_corpora(EvalCatalogQuery(limit=1))
    assert corpora.items == (saved,)
    assert corpora.has_more is False
    suites = await store.list_suites(EvalSuiteCatalogQuery(corpus_revision=corpus.revision))
    assert [(item.id, item.revision) for item in suites.items] == [
        (corpus.suites[0].id, corpus.suites[0].revision)
    ]
    cases = await store.list_cases(
        EvalCaseCatalogQuery(
            corpus_revision=corpus.revision,
            suite_id=corpus.suites[0].id,
        )
    )
    assert [(item.id, item.revision) for item in cases.items] == [
        (corpus.cases[0].id, corpus.cases[0].revision)
    ]

    launch_corpus = _corpus_with_target(corpus, "suite-launch-target")
    await store.save_corpus(launch_corpus, redact_json=_NO_SECRETS.redact_json)
    launch_suite = launch_corpus.suites[0]
    launch_revision = "sha256:" + "1" * 64
    suite_revision = "sha256:" + "2" * 64
    selection_revision = "sha256:" + "3" * 64

    def launch_part(part: int, lane: int) -> EvalRunRequest:
        return EvalRunRequest(
            run_id=f"suite-launch-part-{part}",
            idempotency_key="sha256:" + str(part + 3) * 64,
            corpus_revision=launch_corpus.revision,
            target_key=launch_corpus.target_key,
            suite_id=launch_suite.id,
            suite_revision=launch_suite.revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                authored_suite_revision=suite_revision,
                authored_suite_selection_revision=selection_revision,
                authored_suite_launch_revision=launch_revision,
                authored_suite_launch_lane=lane,
            ),
        )

    await store.admit_run(launch_part(1, 0), redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(launch_part(2, 1), redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(launch_part(3, 0), redact_json=_NO_SECRETS.redact_json)
    parallel_claims = await asyncio.gather(
        store.claim_run(target_key=launch_corpus.target_key),
        store.claim_run_for_targets((launch_corpus.target_key,)),
    )
    claimed_launches = tuple(claim for claim in parallel_claims if claim is not None)
    assert len(claimed_launches) == 2
    claimed_by_id = {claim.run.id: claim for claim in claimed_launches}
    assert set(claimed_by_id) == {"suite-launch-part-1", "suite-launch-part-2"}
    assert await store.claim_run(target_key=launch_corpus.target_key) is None
    await store.fail_run(
        claimed_by_id["suite-launch-part-1"].claim,
        EvalRunFailureCode.EXECUTION_FAILED,
    )
    third_launch = await store.claim_run_for_targets((launch_corpus.target_key,))
    assert third_launch is not None
    assert third_launch.run.id == "suite-launch-part-3"
    await store.fail_run(
        claimed_by_id["suite-launch-part-2"].claim,
        EvalRunFailureCode.EXECUTION_FAILED,
    )
    await store.fail_run(third_launch.claim, EvalRunFailureCode.EXECUTION_FAILED)

    invocation = EvalRunInvocation(
        source=SessionExecutionSource.HTTP_RUN,
        origin=InvocationOrigin(
            trust=InvocationOriginTrust.SERVER_VERIFIED,
            subject="eval-operator",
            tenant="tenant-a",
        ),
        max_steps=7,
        limits=RunLimits(max_total_tokens=2_000, max_tool_calls=3, scope="run"),
        cost_budget=EvalRunCostBudget(
            max_estimated_cost=Decimal("1.25"),
            currency="USD",
        ),
    )
    cancel_request = _request(corpus, suffix="a", invocation=invocation)
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.admit_run(
            cancel_request,
            redact_json=SecretRedactor(cancel_request.run_id).redact_json,
        )
    assert await store.load_run(cancel_request.run_id) is None
    admitted = await store.admit_run(
        cancel_request,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert admitted.spec.invocation == invocation
    assert await store.load_run_by_idempotency_key(cancel_request.idempotency_key) == admitted
    assert await store.load_run_by_idempotency_key("sha256:" + "0" * 64) is None
    with pytest.raises(ValueError, match="lowercase sha256 content revision"):
        await store.load_run_by_idempotency_key("not-a-revision")
    assert (
        await store.admit_run(
            cancel_request.model_copy(update={"run_id": "conformance-a-retry"}),
            redact_json=_NO_SECRETS.redact_json,
        )
        == admitted
    )
    claimed = await store.claim_run()
    assert claimed is not None
    assert claimed.run.id == admitted.id
    active_public_record = await store.load_run(admitted.id)
    assert active_public_record is not None
    assert active_public_record == claimed.run
    assert active_public_record.spec.invocation == invocation
    active_public_json = active_public_record.model_dump_json()
    assert "claim_id" not in active_public_json
    assert "idempotency_key" not in active_public_json
    assert "owner_id" not in active_public_json
    observation = await store.load_run_observation(admitted.id)
    assert observation is not None
    assert observation.run_id == admitted.id
    assert observation.status is EvalRunStatus.RUNNING
    assert observation.attempt_count == claimed.claim.epoch
    assert observation.ownership == active_public_record.ownership
    assert "invocation" not in observation.model_dump_json()
    assert await store.load_run_observation("missing-run") is None
    terminal_wait = asyncio.create_task(
        store.wait_for_run_terminal(
            admitted.id,
            timeout_seconds=1.0,
            poll_interval_seconds=0.001,
            max_poll_interval_seconds=0.05,
        )
    )
    claim = claimed.claim
    renewed = await store.heartbeat_run(claim)
    assert renewed.ownership is not None
    renewed_observation = await store.heartbeat_run_observation(claim)
    assert renewed_observation.status is EvalRunStatus.RUNNING
    assert renewed_observation.ownership is not None
    assert renewed_observation.ownership.epoch == claim.epoch
    cancelling = await store.request_cancel(admitted.id)
    assert cancelling.status is EvalRunStatus.CANCELLING
    with pytest.raises(EvalRunStateConflict):
        await store.publish_result(
            claim,
            result,
            redact_json=_NO_SECRETS.redact_json,
        )
    cancelled = await store.finish_cancel(claim)
    assert cancelled.status is EvalRunStatus.CANCELLED
    assert await store.finish_cancel(claim) == cancelled
    terminal_observation = await terminal_wait
    assert terminal_observation is not None
    assert terminal_observation.status is EvalRunStatus.CANCELLED
    assert terminal_observation.terminal is True

    result_request = _request(corpus, suffix="b")
    await store.admit_run(
        result_request,
        redact_json=_NO_SECRETS.redact_json,
    )
    result_claimed = await store.claim_run()
    assert result_claimed is not None
    result_claim = result_claimed.claim
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.publish_result(
            result_claim,
            result,
            redact_json=SecretRedactor(result.target.application_release_id).redact_json,
        )
    secret_key = "workload-secret-key-canary-ABCDEFGHIJKLMNOP"
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.publish_result(
            result_claim,
            _result_with_secret_manifest_key(result, secret_key),
            redact_json=SecretRedactor(secret_key).redact_json,
        )
    for conflicting_result in _results_with_conflicting_corpus_contract(result):
        with pytest.raises(EvalRunStateConflict, match="immutable corpus suite contract"):
            await store.publish_result(
                result_claim,
                conflicting_result,
                redact_json=_NO_SECRETS.redact_json,
            )
    still_running = await store.load_run(result_claim.run_id)
    assert still_running is not None
    assert still_running.status is EvalRunStatus.RUNNING
    completed = await store.publish_result(
        result_claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert completed.status is EvalRunStatus.COMPLETED
    assert (
        await store.publish_result(
            result_claim,
            result,
            redact_json=_NO_SECRETS.redact_json,
        )
        == completed
    )
    result_bytes = len(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert await store.load_result(completed.id, max_bytes=result_bytes) == result
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_result(completed.id, max_bytes=result_bytes - 1)

    failure_request = _request(corpus, suffix="c")
    await store.admit_run(
        failure_request,
        redact_json=_NO_SECRETS.redact_json,
    )
    failure_claimed = await store.claim_run()
    assert failure_claimed is not None
    stale_failure_claim = failure_claimed.claim
    checkpoint = _terminal_trial_checkpoint(corpus)
    checkpoint_secret = "checkpoint-secret-canary-ABCDEFGHIJKLMNOP"
    unsafe_checkpoint = checkpoint.model_copy(
        update={"result": checkpoint.result.model_copy(update={"error": checkpoint_secret})}
    )
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.save_trial_checkpoint(
            stale_failure_claim,
            unsafe_checkpoint,
            redact_json=SecretRedactor(checkpoint_secret).redact_json,
        )
    await store.save_trial_checkpoint(
        stale_failure_claim,
        checkpoint,
        redact_json=_NO_SECRETS.redact_json,
    )
    await store.save_trial_checkpoint(
        stale_failure_claim,
        checkpoint,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert await store.load_trial_checkpoints(stale_failure_claim) == (checkpoint,)
    conflicting_checkpoint = checkpoint.model_copy(
        update={
            "result": checkpoint.result.model_copy(update={"error": "different terminal outcome"})
        }
    )
    with pytest.raises(EvalRunStateConflict, match="another terminal result"):
        await store.save_trial_checkpoint(
            stale_failure_claim,
            conflicting_checkpoint,
            redact_json=_NO_SECRETS.redact_json,
        )
    released = await store.release_run(stale_failure_claim)
    assert released.status is EvalRunStatus.QUEUED
    failure_reclaimed = await store.claim_run()
    assert failure_reclaimed is not None
    with pytest.raises(EvalRunClaimLost):
        await store.heartbeat_run(stale_failure_claim)
    failure_claim = failure_reclaimed.claim
    assert await store.load_trial_checkpoints(failure_claim) == (checkpoint,)
    with pytest.raises(EvalRunClaimLost):
        await store.load_trial_checkpoints(stale_failure_claim)
    diagnostic = EvalRunFailureDiagnostic(reason=EvalRunFailureReason.EXECUTION_FAILED)
    with pytest.raises(EvalRunClaimLost):
        await store.fail_run(
            stale_failure_claim, EvalRunFailureCode.EXECUTION_FAILED, diagnostic=diagnostic
        )
    failed = await store.fail_run(
        failure_claim, EvalRunFailureCode.EXECUTION_FAILED, diagnostic=diagnostic
    )
    assert failed.status is EvalRunStatus.FAILED
    assert failed.failure_diagnostic == diagnostic
    assert (
        await store.fail_run(
            failure_claim, EvalRunFailureCode.EXECUTION_FAILED, diagnostic=diagnostic
        )
        == failed
    )
    with pytest.raises(EvalRunStateConflict):
        await store.fail_run(
            failure_claim,
            EvalRunFailureCode.EXECUTION_FAILED,
            diagnostic=EvalRunFailureDiagnostic(
                reason=EvalRunFailureReason.RESULT_PUBLICATION_FAILED
            ),
        )
    with pytest.raises(EvalRunStateConflict):
        await store.fail_run(failure_claim, EvalRunFailureCode.EXECUTION_FAILED)
    assert await store.load_run(failure_claim.run_id) == failed

    terminal = await store.list_runs(EvalRunQuery(limit=3))
    assert {item.status for item in terminal.items} == {
        EvalRunStatus.CANCELLED,
        EvalRunStatus.COMPLETED,
        EvalRunStatus.FAILED,
    }
    public_records = terminal.model_dump_json()
    assert "trajectory" not in public_records
    assert "exception" not in public_records
    assert "credential" not in public_records
    assert "request-" not in public_records
    assert "idempotency_key" not in public_records
    assert "claim_id" not in public_records

    other_corpus = _corpus_with_target(corpus, "other-target")
    await store.save_corpus(other_corpus, redact_json=_NO_SECRETS.redact_json)
    other_suite = other_corpus.suites[0]
    other_request = EvalRunRequest(
        run_id="target-scope-other",
        idempotency_key="sha256:" + "d" * 64,
        corpus_revision=other_corpus.revision,
        target_key=other_corpus.target_key,
        suite_id=other_suite.id,
        suite_revision=other_suite.revision,
        max_concurrency=1,
    )
    main_suite = corpus.suites[0]
    main_request = EvalRunRequest(
        run_id="target-scope-main",
        idempotency_key="sha256:" + "e" * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=main_suite.id,
        suite_revision=main_suite.revision,
        max_concurrency=1,
    )
    await store.admit_run(other_request, redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(main_request, redact_json=_NO_SECRETS.redact_json)

    other_claimed = await store.claim_run_for_targets((corpus.target_key, other_corpus.target_key))
    assert other_claimed is not None
    assert other_claimed.run.id == other_request.run_id
    await store.release_run(other_claimed.claim)
    await store.request_cancel(other_request.run_id)
    main_claimed = await store.claim_run(target_key=corpus.target_key)
    assert main_claimed is not None
    assert main_claimed.run.id == main_request.run_id
    await store.release_run(main_claimed.claim)
    await store.request_cancel(main_request.run_id)

    with pytest.raises(ValueError, match="cannot be empty"):
        await store.claim_run_for_targets(())
    with pytest.raises(ValueError, match="must be unique"):
        await store.claim_run_for_targets((corpus.target_key, corpus.target_key))

    target_page = await store.list_runs(EvalRunQuery(target_key=corpus.target_key, limit=1))
    assert target_page.items
    assert all(item.spec.target_key == corpus.target_key for item in target_page.items)
    assert target_page.next_cursor is not None
    with pytest.raises(ValueError, match="cursor does not match this query"):
        await store.list_runs(
            EvalRunQuery(
                target_key=other_corpus.target_key,
                limit=1,
                cursor=target_page.next_cursor,
            )
        )


async def assert_judge_calibration_store_conformance(
    store: EvalStore,
    *,
    report: EvalJudgeCalibrationReportV1,
) -> None:
    """Pin immutable calibration persistence and restart lookup for each backend."""

    assert store.judge_calibrations is True
    assert await store.load_judge_calibration(report.revision) is None
    assert await store.load_judge_calibration_by_run_id(report.run_id) is None

    stored = await store.save_judge_calibration(
        report,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert stored == report
    assert (
        await store.save_judge_calibration(
            report,
            redact_json=_NO_SECRETS.redact_json,
        )
        == report
    )
    assert await store.load_judge_calibration(report.revision) == report
    assert await store.load_judge_calibration_by_run_id(report.run_id) == report

    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_judge_calibration(report.revision, max_bytes=1)
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_judge_calibration_by_run_id(report.run_id, max_bytes=1)


async def assert_captured_eval_store_conformance(
    store: EvalStore,
    *,
    corpus: EvalCorpusDocument,
    result: CorpusExecutionResult,
) -> tuple[CapturedEvaluationResultV1, EvalBaselineKey, EvalBaselineMutationRecord]:
    """Pin captured persistence and audited baseline semantics for every backend."""

    assert store.captured_results is True
    captured = captured_result_for_corpus(corpus, result)
    saved = await store.save_captured_result(
        corpus,
        captured,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert saved.origin is EvalResultOrigin.CAPTURED_SESSION
    assert saved.revision == captured.revision
    assert (
        await store.save_captured_result(
            corpus,
            captured,
            redact_json=_NO_SECRETS.redact_json,
        )
        == saved
    )
    assert await store.load_corpus(corpus.revision) == corpus
    loaded_captured = await store.load_result_by_revision(captured.revision)
    assert loaded_captured == captured
    assert type(loaded_captured) is CapturedEvaluationResultV1
    assert loaded_captured.score.memory_attribution == captured.score.memory_attribution
    assert await store.load_result_record(captured.revision) == saved
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_result_by_revision(
            captured.revision,
            max_bytes=saved.document_bytes - 1,
        )

    fresh_record = await store.load_result_record(result.revision)
    assert fresh_record is not None
    assert fresh_record.origin is EvalResultOrigin.FRESH_EXECUTION
    loaded_fresh = await store.load_result_by_revision(result.revision)
    assert loaded_fresh == result
    assert type(loaded_fresh) is CorpusExecutionResult
    assert loaded_fresh.run.cases[0].trials[0].memory_attribution == (
        result.run.cases[0].trials[0].memory_attribution
    )

    first_result_page = await store.list_results(
        EvalResultQuery(target_key=corpus.target_key, limit=1)
    )
    assert len(first_result_page.items) == 1
    assert first_result_page.next_cursor is not None
    second_result_page = await store.list_results(
        EvalResultQuery(
            target_key=corpus.target_key,
            limit=1,
            cursor=first_result_page.next_cursor,
        )
    )
    assert len(second_result_page.items) == 1
    assert {
        first_result_page.items[0].revision,
        second_result_page.items[0].revision,
    } == {captured.revision, result.revision}
    captured_page = await store.list_results(
        EvalResultQuery(
            target_key=corpus.target_key,
            origin=EvalResultOrigin.CAPTURED_SESSION,
        )
    )
    assert [item.revision for item in captured_page.items] == [captured.revision]
    with pytest.raises(ValueError, match="cursor does not match this query"):
        await store.list_results(
            EvalResultQuery(
                target_key=corpus.target_key,
                origin=EvalResultOrigin.CAPTURED_SESSION,
                cursor=first_result_page.next_cursor,
            )
        )

    key = EvalBaselineKey(
        target_key=corpus.target_key,
        corpus_revision=corpus.revision,
        suite_id=corpus.suites[0].id,
    )
    first_update = EvalBaselineUpdate(
        key=key,
        result_revision=captured.revision,
        expected_generation=0,
        operation_id="sha256:" + "4" * 64,
        actor_id="operator-initial",
    )
    first = await store.set_baseline(
        first_update,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert first.resulting_generation == 1
    assert (
        await store.set_baseline(
            first_update,
            redact_json=_NO_SECRETS.redact_json,
        )
        == first
    )
    assert await store.load_baseline_mutation(first.operation_id) == first

    contenders = (
        EvalBaselineUpdate(
            key=key,
            result_revision=result.revision,
            expected_generation=1,
            operation_id="sha256:" + "5" * 64,
            actor_id="operator-a",
        ),
        EvalBaselineUpdate(
            key=key,
            result_revision=captured.revision,
            expected_generation=1,
            operation_id="sha256:" + "6" * 64,
            actor_id="operator-b",
        ),
    )
    outcomes = await asyncio.gather(
        *(store.set_baseline(update, redact_json=_NO_SECRETS.redact_json) for update in contenders),
        return_exceptions=True,
    )
    committed = [item for item in outcomes if isinstance(item, EvalBaselineMutationRecord)]
    conflicts = [item for item in outcomes if isinstance(item, EvalBaselineConflict)]
    assert len(committed) == 1
    assert len(conflicts) == 1
    final_mutation = committed[0]
    assert final_mutation.resulting_generation == 2
    final_baseline = await store.load_baseline(key)
    assert final_baseline is not None
    assert final_baseline.generation == 2
    assert final_baseline.result_revision == final_mutation.selected_result_revision
    assert final_baseline.updated_by == final_mutation.actor_id

    winning_update = next(
        item for item in contenders if item.operation_id == final_mutation.operation_id
    )
    assert (
        await store.set_baseline(
            winning_update,
            redact_json=_NO_SECRETS.redact_json,
        )
        == final_mutation
    )
    with pytest.raises(EvalBaselineConflict, match="another mutation"):
        await store.set_baseline(
            winning_update.model_copy(update={"actor_id": "different-operator"}),
            redact_json=_NO_SECRETS.redact_json,
        )
    with pytest.raises(EvalBaselineConflict, match="requested scope"):
        await store.set_baseline(
            EvalBaselineUpdate(
                key=key.model_copy(update={"target_key": "different-target"}),
                result_revision=captured.revision,
                expected_generation=0,
                operation_id="sha256:" + "7" * 64,
                actor_id="operator-scope-check",
            ),
            redact_json=_NO_SECRETS.redact_json,
        )
    with pytest.raises(KeyError, match="Eval result not found"):
        await store.set_baseline(
            EvalBaselineUpdate(
                key=key,
                result_revision="sha256:" + "f" * 64,
                expected_generation=2,
                operation_id="sha256:" + "8" * 64,
                actor_id="operator-missing-result",
            ),
            redact_json=_NO_SECRETS.redact_json,
        )

    captured_case = EvalCaseSpec.create(
        id=corpus.cases[0].id,
        suite_id=corpus.cases[0].suite_id,
        name=corpus.cases[0].name,
        description=corpus.cases[0].description,
        source=corpus.cases[0].source,
        input=None,
        assertions=corpus.cases[0].assertions,
    )
    captured_only_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(captured_case,),
    )
    captured_only_result = captured_result_for_corpus(captured_only_corpus, result)
    captured_only_record = await store.save_captured_result(
        captured_only_corpus,
        captured_only_result,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert captured_only_record.corpus_revision == captured_only_corpus.revision
    captured_cases = await store.list_cases(
        EvalCaseCatalogQuery(
            corpus_revision=captured_only_corpus.revision,
            suite_id=captured_only_corpus.suites[0].id,
        )
    )
    assert captured_cases.items[0].message_count == 0
    return captured, key, final_mutation
