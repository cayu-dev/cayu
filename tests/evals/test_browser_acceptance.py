from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cayu.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.core.events import Event, EventType
from cayu.evals import (
    BROWSER_ACCEPTANCE_FIXTURE_REVISION,
    BrowserAcceptanceAccessState,
    BrowserAcceptanceAgentReportState,
    BrowserAcceptanceCaseCategory,
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceCompletionState,
    BrowserAcceptanceDiagnosticState,
    BrowserAcceptanceDiagnosticV1,
    BrowserAcceptanceInfrastructureState,
    BrowserAcceptanceLimitsV1,
    BrowserAcceptanceManifestV1,
    BrowserAcceptanceMode,
    BrowserAcceptanceOperationEvidenceV1,
    BrowserAcceptanceOperationState,
    BrowserAcceptanceRuntimeIdentityV1,
    BrowserAcceptanceSemanticOracle,
    BrowserAcceptanceSemanticState,
    BrowserAcceptanceState,
    BrowserAcceptanceTrialReceiptV1,
    BrowserAcceptanceUsageV1,
    BrowserAllocationDisposition,
    EvalStatus,
    EvalTrialResult,
    Trajectory,
    browser_acceptance_report_from_json,
    browser_acceptance_report_to_json,
    build_browser_acceptance_report,
    build_browser_acceptance_retry_report,
    render_browser_acceptance_html,
    write_browser_acceptance_report,
)
from cayu.evals import browser_acceptance as acceptance_module
from cayu.evals.browser_acceptance import (
    _request_summaries_from_trajectory,
    _semantic_state,
)
from cayu.evals.browser_acceptance_manifests import (
    deterministic_browser_acceptance_manifest,
    live_authenticated_browser_acceptance_manifest,
    live_public_browser_acceptance_manifest,
)


def _case(
    case_id: str = "navigation",
    *,
    expected_state: BrowserAcceptanceState = BrowserAcceptanceState.PASSED,
) -> BrowserAcceptanceCaseV1:
    unsupported = expected_state is BrowserAcceptanceState.UNSUPPORTED
    return BrowserAcceptanceCaseV1.build(
        case_id=case_id,
        category=(
            BrowserAcceptanceCaseCategory.CAPABILITY
            if unsupported
            else BrowserAcceptanceCaseCategory.SUCCESS
        ),
        expected_state=expected_state,
        semantic_oracle=(
            BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED
            if unsupported
            else BrowserAcceptanceSemanticOracle.OBSERVATION
        ),
        semantic_success_required=not unsupported,
        required=True,
        fixture_route=None if unsupported else "navigation",
        operations=("history_back",) if unsupported else ("navigate",),
        screenshot_checkpoints=(),
        oracle_parameters=(
            {"operation": "history_back"} if unsupported else {"required_operations": ["navigate"]}
        ),
    )


def _manifest(*cases: BrowserAcceptanceCaseV1) -> BrowserAcceptanceManifestV1:
    retained = tuple(sorted(cases or (_case(),), key=lambda item: item.case_id))
    return BrowserAcceptanceManifestV1.build(
        corpus_revision="sha256:" + "1" * 64,
        suite_id="browser-deterministic-v1",
        mode=BrowserAcceptanceMode.DETERMINISTIC,
        enabled=True,
        trial_count=1,
        allowed_origins=("https://docs.browser.test",),
        limits=BrowserAcceptanceLimitsV1(
            max_destinations=2,
            max_browser_operations=32,
            max_model_steps=16,
            max_wall_time_ms=30_000,
            max_artifact_bytes=1 << 20,
            max_concurrency=1,
        ),
        cases=retained,
    )


def _runtime_identity() -> BrowserAcceptanceRuntimeIdentityV1:
    build = RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.WHEEL_RECORD,
        artifact_kind=RuntimeBuildArtifactKind.WHEEL,
        artifact_digest="2" * 64,
    )
    return BrowserAcceptanceRuntimeIdentityV1.build(
        runtime_build_provenance=build,
        browser_protocol="cayu.browser-session.v2",
        browser_worker_version="6",
        playwright_version="1.62.0",
        chromium_identity="chromium-fixture",
        runner_fingerprint="3" * 64,
        workload_fingerprint="4" * 64,
        egress_fingerprint="5" * 64,
        artifact_store_fingerprint="6" * 64,
        execution_profile_fingerprint="7" * 64,
        execution_suite_fingerprint="8" * 64,
        provider_name="scripted",
        model="scripted-browser-v1",
        platform_system="linux",
        platform_machine="x86_64",
        python_version="3.14.0",
    )


def _run_identity(
    manifest: BrowserAcceptanceManifestV1,
    runtime: BrowserAcceptanceRuntimeIdentityV1,
) -> str:
    from cayu.evals.corpus import _content_revision

    return _content_revision(
        {
            "manifest_revision": manifest.revision,
            "runtime_identity": runtime.model_dump(
                mode="json", exclude={"revision", "chromium_identity"}
            ),
            "mode": manifest.mode.value,
        },
        "browser acceptance run identity",
    )


def _row(
    case: BrowserAcceptanceCaseV1,
    run_identity_revision: str,
    *,
    observed_state: BrowserAcceptanceState | None = None,
    diagnostic_state: BrowserAcceptanceDiagnosticState = BrowserAcceptanceDiagnosticState.CAPTURED,
    attempt_number: int = 1,
    access_state: BrowserAcceptanceAccessState | None = None,
) -> BrowserAcceptanceTrialReceiptV1:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    observed = case.expected_state if observed_state is None else observed_state
    semantic = (
        BrowserAcceptanceSemanticState.PASSED
        if case.expected_state is BrowserAcceptanceState.PASSED
        else BrowserAcceptanceSemanticState.NOT_APPLICABLE
    )
    return BrowserAcceptanceTrialReceiptV1.build(
        run_identity_revision=run_identity_revision,
        case_id=case.case_id,
        case_revision=case.revision,
        trial_number=1,
        attempt_number=attempt_number,
        expected_state=case.expected_state,
        observed_state=observed,
        semantic_state=semantic,
        infrastructure_state=BrowserAcceptanceInfrastructureState.AVAILABLE,
        completion_state=BrowserAcceptanceCompletionState.COMPLETE,
        agent_report_state=BrowserAcceptanceAgentReportState.CLAIMED_SUCCESS,
        started_at=started,
        completed_at=started + timedelta(milliseconds=25),
        elapsed_ms=25,
        usage=BrowserAcceptanceUsageV1(
            model_steps=2,
            browser_operations=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
        error_categories={},
        truncation_categories={},
        diagnostic=BrowserAcceptanceDiagnosticV1(
            state=diagnostic_state,
            error_code=(
                "diagnostic_projection_failed"
                if diagnostic_state is BrowserAcceptanceDiagnosticState.UNAVAILABLE
                else None
            ),
            operations=(
                (
                    BrowserAcceptanceOperationEvidenceV1(
                        sequence=1,
                        invocation_revision="sha256:" + "9" * 64,
                        operation="navigate",
                        state=BrowserAcceptanceOperationState.TERMINAL,
                        allocation_disposition=BrowserAllocationDisposition.LIVE,
                        access_state=access_state,
                        artifacts=(),
                    ),
                )
                if access_state is not None
                else ()
            ),
        ),
    )


def test_browser_acceptance_report_is_complete_content_bound_and_portable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    supported = _case()
    unsupported = _case("history-back", expected_state=BrowserAcceptanceState.UNSUPPORTED)
    manifest = _manifest(supported, unsupported)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    rows = tuple(_row(case, run_identity) for case in manifest.cases)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    report = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started + timedelta(seconds=1),
        rows=rows,
    )

    assert report.aggregate.overall_status.value == "passed"
    assert report.aggregate.state_counts[BrowserAcceptanceState.PASSED] == 1
    assert report.aggregate.state_counts[BrowserAcceptanceState.UNSUPPORTED] == 1
    assert all(count == 0 for count in report.aggregate.access_state_counts.values())
    assert report.aggregate.total_tokens == 30
    assert report.cases[1].variability.value == "not_applicable"
    assert browser_acceptance_report_from_json(browser_acceptance_report_to_json(report)) == report
    assert "Browser acceptance" in render_browser_acceptance_html(report)
    original_publish = acceptance_module._publish_browser_acceptance_report_file
    publication_count = 0

    def interrupt_between_representations(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal publication_count
        publication_count += 1
        if publication_count == 2:
            raise KeyboardInterrupt
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        acceptance_module,
        "_publish_browser_acceptance_report_file",
        interrupt_between_representations,
    )
    with pytest.raises(KeyboardInterrupt):
        write_browser_acceptance_report(report, tmp_path)
    stem = report.revision.replace(":", "-")
    json_path = tmp_path / f"{stem}.json"
    html_path = tmp_path / f"{stem}.html"
    assert json_path.exists()
    assert not html_path.exists()

    monkeypatch.setattr(
        acceptance_module,
        "_publish_browser_acceptance_report_file",
        original_publish,
    )
    assert write_browser_acceptance_report(report, tmp_path) == (json_path, html_path)
    assert html_path.exists()
    assert write_browser_acceptance_report(report, tmp_path) == (json_path, html_path)

    html_path.unlink()
    assert write_browser_acceptance_report(report, tmp_path) == (json_path, html_path)
    assert html_path.exists()

    json_path.write_text("conflicting report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts with immutable durable output"):
        write_browser_acceptance_report(report, tmp_path)


def test_browser_acceptance_aggregate_counts_closed_access_states() -> None:
    first = _case("available")
    second = _case("blocked")
    third = _case("unknown")
    manifest = _manifest(first, second, third)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    report = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=(
            _row(first, run_identity, access_state=BrowserAcceptanceAccessState.AVAILABLE),
            _row(second, run_identity, access_state=BrowserAcceptanceAccessState.BLOCKED),
            _row(third, run_identity, access_state=BrowserAcceptanceAccessState.UNKNOWN),
        ),
    )

    assert report.aggregate.access_state_counts == {
        BrowserAcceptanceAccessState.AVAILABLE: 1,
        BrowserAcceptanceAccessState.BLOCKED: 1,
        BrowserAcceptanceAccessState.UNKNOWN: 1,
    }


def test_browser_acceptance_report_rejects_missing_required_row() -> None:
    first = _case()
    second = _case("second")
    manifest = _manifest(first, second)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="omits or reorders"):
        build_browser_acceptance_report(
            manifest=manifest,
            runtime_identity=runtime,
            started_at=started,
            completed_at=started,
            rows=(_row(first, run_identity),),
        )


def test_browser_acceptance_campaign_aggregate_enforces_split_operation_limit() -> None:
    first = _case("first")
    second = _case("second")
    source = _manifest(first, second)
    manifest = BrowserAcceptanceManifestV1.build(
        corpus_revision=source.corpus_revision,
        suite_id=source.suite_id,
        mode=source.mode,
        enabled=source.enabled,
        trial_count=source.trial_count,
        allowed_origins=source.allowed_origins,
        limits=source.limits.model_copy(update={"max_browser_operations": 1}),
        cases=source.cases,
    )
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)

    report = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=tuple(_row(case, run_identity) for case in manifest.cases),
    )

    assert all(row.conformance.value == "passed" for row in report.rows)
    assert report.aggregate.total_browser_operations == 2
    assert report.aggregate.limit_violations == ("browser_operations",)
    assert report.aggregate.overall_status.value == "failed"


def test_browser_acceptance_diagnostic_failure_preserves_case_outcome_but_fails_run() -> None:
    case = _case()
    manifest = _manifest(case)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    row = _row(
        case,
        run_identity,
        diagnostic_state=BrowserAcceptanceDiagnosticState.UNAVAILABLE,
    )

    report = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=(row,),
    )

    assert row.observed_state is BrowserAcceptanceState.PASSED
    assert row.semantic_state is BrowserAcceptanceSemanticState.PASSED
    assert row.conformance.value == "incomplete"
    assert report.aggregate.overall_status.value == "incomplete"


def test_browser_acceptance_retry_preserves_prior_attempt_and_uses_latest_outcome() -> None:
    case = _case()
    manifest = _manifest(case)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    first = _row(
        case,
        run_identity,
        observed_state=BrowserAcceptanceState.FAILED,
    )
    initial = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=(first,),
    )
    replacement = _row(case, run_identity, attempt_number=2)

    retried = build_browser_acceptance_retry_report(
        initial,
        retry_rows=(replacement,),
        completed_at=started + timedelta(seconds=1),
    )

    assert retried.source_report_revision == initial.revision
    assert retried.prior_rows == (first,)
    assert retried.rows == (replacement,)
    assert retried.aggregate.overall_status.value == "passed"
    assert (
        browser_acceptance_report_from_json(browser_acceptance_report_to_json(retried)) == retried
    )


def test_browser_acceptance_retry_rejects_changed_known_chromium_identity() -> None:
    case = _case()
    manifest = _manifest(case)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    initial = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=(_row(case, run_identity),),
    )
    replacement = _row(case, run_identity, attempt_number=2)
    changed_runtime = BrowserAcceptanceRuntimeIdentityV1.build(
        **{
            field_name: getattr(runtime, field_name)
            for field_name in BrowserAcceptanceRuntimeIdentityV1.model_fields
            if field_name not in {"revision", "chromium_identity"}
        },
        chromium_identity="different-chromium",
    )

    with pytest.raises(ValueError, match="changed known Chromium"):
        build_browser_acceptance_retry_report(
            initial,
            retry_rows=(replacement,),
            completed_at=started + timedelta(seconds=1),
            runtime_identity=changed_runtime,
        )


def test_browser_acceptance_case_revision_rejects_redigested_semantic_contradiction() -> None:
    case = _case("history-back", expected_state=BrowserAcceptanceState.UNSUPPORTED)
    document = case.model_dump(mode="json")
    document["semantic_success_required"] = True
    from cayu.evals.corpus import _content_revision

    document["revision"] = _content_revision(document, "browser acceptance case")
    with pytest.raises(ValidationError, match="Unsupported cases"):
        BrowserAcceptanceCaseV1.model_validate(document)


def test_browser_acceptance_html_escapes_untrusted_case_identity() -> None:
    case = _case("hostile-<script>")
    manifest = _manifest(case)
    runtime = _runtime_identity()
    run_identity = _run_identity(manifest, runtime)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    report = build_browser_acceptance_report(
        manifest=manifest,
        runtime_identity=runtime,
        started_at=started,
        completed_at=started,
        rows=(_row(case, run_identity),),
    )

    rendered = render_browser_acceptance_html(report)
    assert "<script>" not in rendered
    assert "hostile-&lt;script&gt;" in rendered


def test_checked_browser_manifests_cover_required_categories_and_modes() -> None:
    from cayu.evals.corpus import _content_revision

    deterministic = deterministic_browser_acceptance_manifest()
    live = live_public_browser_acceptance_manifest()
    authenticated = live_authenticated_browser_acceptance_manifest()

    assert {case.category for case in deterministic.cases} >= set(BrowserAcceptanceCaseCategory)
    assert {case.expected_state for case in deterministic.cases} >= {
        BrowserAcceptanceState.PASSED,
        BrowserAcceptanceState.FAILED,
        BrowserAcceptanceState.REFUSED,
        BrowserAcceptanceState.AMBIGUOUS,
        BrowserAcceptanceState.UNSUPPORTED,
    }
    deterministic_ids = {case.case_id for case in deterministic.cases}
    assert deterministic.corpus_revision == _content_revision(
        {
            "suite_id": deterministic.suite_id,
            "fixture_revision": BROWSER_ACCEPTANCE_FIXTURE_REVISION,
            "cases": [case.revision for case in deterministic.cases],
        },
        "deterministic browser acceptance corpus",
    )
    for required_fragment in (
        "artifact-upload",
        "artifact-trace",
        "artifact-video",
        "cancellation",
        "conflicting-operation",
        "detached-control",
        "hidden-control",
        "history-back",
        "history-forward",
        "multiple",
        "occluded-control",
        "oversized-accessible-name",
        "oversized-download",
        "oversized-response-body",
        "process-loss-acknowledgement",
        "process-loss-artifact-publication",
        "process-loss-dispatched",
        "process-loss-guest-terminal",
        "process-loss-intent",
        "reload",
        "visual-canvas",
    ):
        assert any(required_fragment in case_id for case_id in deterministic_ids)
    assert live.mode is BrowserAcceptanceMode.LIVE_PUBLIC
    assert live.trial_count > 1
    assert live.limits.max_model_steps == (
        max(len(case.operations) + 1 for case in live.cases) * len(live.cases) * live.trial_count
    )
    assert live.limits.max_artifact_bytes == (
        8 * 1024 * 1024 * sum(len(case.operations) * live.trial_count for case in live.cases)
    )
    assert live.limits.max_input_tokens == 96_000
    assert live.limits.max_output_tokens is not None
    assert live.limits.max_estimated_cost is not None
    assert authenticated.mode is BrowserAcceptanceMode.LIVE_AUTHENTICATED
    assert authenticated.enabled is False
    assert all(
        case.expected_state is BrowserAcceptanceState.UNSUPPORTED for case in authenticated.cases
    )
    assert {
        case.case_id.removeprefix("revision-stale-ref-after-")
        for case in deterministic.cases
        if case.case_id.startswith("revision-stale-ref-after-")
    } == {"click", "download", "fill", "press", "screenshot", "select", "wait"}


def test_redirect_oracle_requires_the_final_observed_destination() -> None:
    from cayu.evals.corpus import _content_revision

    case = next(
        item
        for item in deterministic_browser_acceptance_manifest().cases
        if item.case_id == "navigation-redirect"
    )
    operation = BrowserAcceptanceOperationEvidenceV1(
        sequence=1,
        invocation_revision="sha256:" + "9" * 64,
        operation="navigate",
        state=BrowserAcceptanceOperationState.TERMINAL,
        allocation_disposition=BrowserAllocationDisposition.LIVE,
        target_revision=_content_revision(
            {"url": "https://docs.browser.test/redirect"},
            "browser acceptance operation target",
        ),
        observed_target_revision=_content_revision(
            {"url": "https://docs.browser.test/redirect"},
            "browser acceptance observed target",
        ),
    )
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state=BrowserAcceptanceDiagnosticState.CAPTURED,
        fixture_route_observed=True,
        fixture_route_request_count=1,
        operations=(operation,),
    )

    for observed_url in (
        "https://docs.browser.test/redirect",
        "https://docs.browser.test/forms",
    ):
        rejected = diagnostic.model_copy(
            update={
                "operations": (
                    operation.model_copy(
                        update={
                            "observed_target_revision": _content_revision(
                                {"url": observed_url},
                                "browser acceptance observed target",
                            )
                        }
                    ),
                )
            }
        )
        assert (
            _semantic_state(case, rejected, public_operations=frozenset({"navigate"}))
            is BrowserAcceptanceSemanticState.FAILED
        )
    followed = diagnostic.model_copy(
        update={
            "operations": (
                operation.model_copy(
                    update={
                        "observed_target_revision": _content_revision(
                            {"url": "https://docs.browser.test/basic"},
                            "browser acceptance observed target",
                        )
                    }
                ),
            )
        }
    )
    assert (
        _semantic_state(case, followed, public_operations=frozenset({"navigate"}))
        is BrowserAcceptanceSemanticState.PASSED
    )


def test_challenge_oracle_requires_positive_blocked_access_evidence() -> None:
    from cayu.evals.corpus import _content_revision

    case = next(
        item
        for item in deterministic_browser_acceptance_manifest().cases
        if item.case_id == "adversarial-challenge-page"
    )
    operation = BrowserAcceptanceOperationEvidenceV1(
        sequence=1,
        invocation_revision="sha256:" + "9" * 64,
        operation="navigate",
        state=BrowserAcceptanceOperationState.TERMINAL,
        allocation_disposition=BrowserAllocationDisposition.LIVE,
        target_revision=_content_revision(
            {"url": "https://docs.browser.test/challenge"},
            "browser acceptance operation target",
        ),
        access_state=BrowserAcceptanceAccessState.AVAILABLE,
    )
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state=BrowserAcceptanceDiagnosticState.CAPTURED,
        fixture_route_observed=True,
        fixture_route_request_count=1,
        operations=(operation,),
    )

    assert (
        _semantic_state(case, diagnostic, public_operations=frozenset({"navigate"}))
        is BrowserAcceptanceSemanticState.FAILED
    )
    blocked = diagnostic.model_copy(
        update={
            "operations": (
                operation.model_copy(update={"access_state": BrowserAcceptanceAccessState.BLOCKED}),
            )
        }
    )
    assert (
        _semantic_state(case, blocked, public_operations=frozenset({"navigate"}))
        is BrowserAcceptanceSemanticState.PASSED
    )


def test_exact_terminal_replay_requires_one_observed_fixture_dispatch() -> None:
    from cayu.evals.corpus import _content_revision

    case = next(
        item
        for item in deterministic_browser_acceptance_manifest().cases
        if item.case_id == "recovery-exact-terminal-replay"
    )
    target_revision = _content_revision(
        {"url": "https://docs.browser.test/basic"},
        "browser acceptance operation target",
    )
    operations = tuple(
        BrowserAcceptanceOperationEvidenceV1(
            sequence=sequence,
            invocation_revision="sha256:" + "9" * 64,
            operation="navigate",
            state=BrowserAcceptanceOperationState.TERMINAL,
            allocation_disposition=BrowserAllocationDisposition.LIVE,
            target_revision=target_revision,
        )
        for sequence in (1, 2)
    )
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state=BrowserAcceptanceDiagnosticState.CAPTURED,
        fixture_route_observed=True,
        fixture_route_request_count=1,
        browser_dispatches=1,
        operations=operations,
    )

    assert (
        _semantic_state(case, diagnostic, public_operations=frozenset({"navigate"}))
        is BrowserAcceptanceSemanticState.PASSED
    )
    assert (
        _semantic_state(
            case,
            diagnostic.model_copy(update={"browser_dispatches": 2}),
            public_operations=frozenset({"navigate"}),
        )
        is BrowserAcceptanceSemanticState.FAILED
    )
    assert (
        _semantic_state(
            case,
            diagnostic.model_copy(update={"fixture_route_request_count": 2}),
            public_operations=frozenset({"navigate"}),
        )
        is BrowserAcceptanceSemanticState.FAILED
    )


def test_browser_acceptance_request_summary_hashes_destination_and_route() -> None:
    destination = "private.browser.test"
    path = "/account?token=BROWSER_REQUEST_SECRET_CANARY"
    trial = EvalTrialResult.model_construct(
        trial_number=1,
        status=EvalStatus.PASSED,
        trajectory=Trajectory(
            events=(
                Event(
                    type=EventType.EGRESS_REQUEST_AUTHORIZED,
                    session_id="browser-request-summary",
                    payload={
                        "method": "GET",
                        "destination": destination,
                        "path": path,
                        "status_code": 200,
                    },
                ),
            )
        ),
    )

    summaries, truncated = _request_summaries_from_trajectory(trial)

    assert truncated is False
    assert len(summaries) == 1
    encoded = summaries[0].model_dump_json()
    assert summaries[0].outcome == "authorized"
    assert destination not in encoded
    assert path not in encoded
    assert "BROWSER_REQUEST_SECRET_CANARY" not in encoded
