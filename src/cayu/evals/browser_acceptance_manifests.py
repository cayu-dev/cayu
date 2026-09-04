"""Pinned case manifests for Cayu's browser acceptance profiles."""

from __future__ import annotations

from cayu.evals.browser_acceptance import (
    BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION,
    BrowserAcceptanceCaseCategory,
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceFaultScenario,
    BrowserAcceptanceLimitsV1,
    BrowserAcceptanceManifestV1,
    BrowserAcceptanceMode,
    BrowserAcceptanceSemanticOracle,
    BrowserAcceptanceState,
)
from cayu.evals.browser_acceptance_fixture import BROWSER_ACCEPTANCE_FIXTURE_REVISION
from cayu.evals.corpus import _content_revision

DETERMINISTIC_BROWSER_ACCEPTANCE_SUITE_ID = "browser-acceptance-deterministic-v1"
LIVE_PUBLIC_BROWSER_ACCEPTANCE_SUITE_ID = "browser-acceptance-live-public-v1"
LIVE_AUTHENTICATED_BROWSER_ACCEPTANCE_SUITE_ID = "browser-acceptance-live-authenticated-v1"


def _case(
    case_id: str,
    *,
    category: BrowserAcceptanceCaseCategory,
    state: BrowserAcceptanceState = BrowserAcceptanceState.PASSED,
    operations: tuple[str, ...] = ("navigate",),
    route: str | None = None,
    oracle: BrowserAcceptanceSemanticOracle = BrowserAcceptanceSemanticOracle.OBSERVATION,
    parameters: dict[str, object] | None = None,
    checkpoints: tuple[str, ...] = (),
    fault_scenario: BrowserAcceptanceFaultScenario | None = None,
) -> BrowserAcceptanceCaseV1:
    return BrowserAcceptanceCaseV1.build(
        case_id=case_id,
        category=category,
        expected_state=state,
        semantic_oracle=oracle,
        semantic_success_required=state is BrowserAcceptanceState.PASSED,
        fault_scenario=fault_scenario,
        required=True,
        fixture_route=route,
        operations=operations,
        screenshot_checkpoints=checkpoints,
        oracle_parameters=(
            {"required_operations": list(operations)} if parameters is None else parameters
        ),
    )


def _unsupported(case_id: str, operation: str) -> BrowserAcceptanceCaseV1:
    return _case(
        case_id,
        category=BrowserAcceptanceCaseCategory.CAPABILITY,
        state=BrowserAcceptanceState.UNSUPPORTED,
        operations=(operation,),
        oracle=BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED,
        parameters={"operation": operation},
    )


def deterministic_browser_acceptance_manifest() -> BrowserAcceptanceManifestV1:
    """Return the immutable credential-free v1 conformance manifest."""

    cases = tuple(
        sorted(
            (
                _case(
                    "access-broker-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    route="/denied-subresource",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "fetch_failed"},
                ),
                _case(
                    "access-main-document-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    route="https://blocked.browser.test/private",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "fetch_failed"},
                ),
                _case(
                    "access-redirect-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    route="/redirect-denied",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "fetch_failed"},
                ),
                _case(
                    "action-delayed-element",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "wait", "click"),
                    route="/delayed",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "wait", "click"],
                        "expected_effects": {"delayed-clicked": 1},
                    },
                ),
                _case(
                    "action-disabled-control",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/forms",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "actionability_failed"},
                ),
                _case(
                    "action-duplicate-labels",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click"),
                    route="/duplicate-labels",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "expected_effects": {"duplicate-first": 1},
                    },
                ),
                _case(
                    "action-form-controls",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "fill", "select", "press", "click"),
                    route="/forms",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "fill", "select", "press", "click"],
                        "expected_effects": {"form-saved": 1},
                    },
                ),
                _case(
                    "action-form-validation",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click"),
                    route="/forms",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "expected_effects": {},
                    },
                ),
                _case(
                    "action-hidden-control",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/hidden",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "actionability_failed"},
                ),
                _case(
                    "action-detached-control",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/detached",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "stale_observation"},
                ),
                _case(
                    "action-occluded-control",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/occluded",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "actionability_failed"},
                ),
                _case(
                    "action-replaced-element",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/replaced",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "stale_observation"},
                ),
                _case(
                    "action-readonly-control",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "fill"),
                    route="/forms",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "actionability_failed"},
                ),
                _case(
                    "adversarial-challenge-page",
                    category=BrowserAcceptanceCaseCategory.ADVERSARIAL,
                    route="/challenge",
                    parameters={
                        "required_operations": ["navigate"],
                        "forbidden_operations": ["click", "download"],
                        "access_state": "blocked",
                    },
                ),
                _case(
                    "adversarial-hostile-page-fields",
                    category=BrowserAcceptanceCaseCategory.ADVERSARIAL,
                    route="/hostile",
                    parameters={
                        "required_operations": ["navigate"],
                        "forbidden_operations": ["click", "download"],
                    },
                ),
                _case(
                    "ambiguity-acknowledgement-loss",
                    category=BrowserAcceptanceCaseCategory.AMBIGUITY,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS,
                ),
                _case(
                    "artifact-bounded-download",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "download"),
                    route="/download",
                    oracle=BrowserAcceptanceSemanticOracle.ARTIFACT,
                    parameters={"kind": "download"},
                ),
                _case(
                    "artifact-screenshot",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "screenshot"),
                    route="/long-page",
                    oracle=BrowserAcceptanceSemanticOracle.ARTIFACT,
                    parameters={"kind": "screenshot"},
                    checkpoints=("after-navigation",),
                ),
                _unsupported("artifact-upload", "upload"),
                _unsupported("artifact-trace", "trace"),
                _unsupported("artifact-video", "video"),
                _case(
                    "cancellation-after-dispatched-marker",
                    category=BrowserAcceptanceCaseCategory.CANCELLATION,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={"expected_browser_dispatches": 1},
                    fault_scenario=BrowserAcceptanceFaultScenario.CANCEL_AFTER_DISPATCHED,
                ),
                _case(
                    "cancellation-after-final-receipt",
                    category=BrowserAcceptanceCaseCategory.CANCELLATION,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={"expected_browser_dispatches": 1},
                    fault_scenario=BrowserAcceptanceFaultScenario.CANCEL_AFTER_TERMINAL,
                ),
                _case(
                    "cancellation-during-artifact-publication",
                    category=BrowserAcceptanceCaseCategory.CANCELLATION,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    operations=("navigate", "screenshot"),
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.CANCEL_AFTER_ARTIFACT,
                ),
                _case(
                    "cancellation-during-guest-effect",
                    category=BrowserAcceptanceCaseCategory.CANCELLATION,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.CANCEL_BEFORE_TERMINAL,
                ),
                _case(
                    "cancellation-during-intent-publication",
                    category=BrowserAcceptanceCaseCategory.CANCELLATION,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={"expected_browser_dispatches": 1},
                    fault_scenario=BrowserAcceptanceFaultScenario.CANCEL_AFTER_INTENT,
                ),
                _case(
                    "crash-after-effect",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_AFTER_EFFECT,
                ),
                _case(
                    "crash-before-dispatch",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    state=BrowserAcceptanceState.FAILED,
                    operations=("navigate", "wait"),
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "required_operations": ["navigate", "wait"],
                        "error": "browser_crash",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_BEFORE_DISPATCH,
                ),
                _case(
                    "crash-during-cleanup",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    operations=("navigate", "close"),
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "required_operations": ["navigate", "close"],
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_DURING_CLEANUP,
                ),
                _case(
                    "crash-during-execution",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    operations=("navigate", "wait"),
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "required_operations": ["navigate", "wait"],
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_DURING_EXECUTION,
                ),
                _case(
                    "iframe-cross-origin",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "fill", "click"),
                    route="/cross-origin-frame",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "fill", "click"],
                        "expected_effects": {"frame-applied": 1},
                    },
                ),
                _case(
                    "iframe-same-origin",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "fill", "click"),
                    route="/same-origin-frame",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "fill", "click"],
                        "expected_effects": {"frame-applied": 1},
                    },
                ),
                _case(
                    "limit-long-observation-truncation",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    route="/long-page",
                    parameters={
                        "required_operations": ["navigate"],
                        "required_truncation": ["snapshot"],
                    },
                ),
                _case(
                    "limit-normal-capacity-keeps-cleanup",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=(
                        "navigate",
                        "observe",
                        "observe",
                        "observe",
                        "observe",
                        "observe",
                        "observe",
                        "observe",
                        "observe",
                        "close",
                    ),
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "required_operations": [
                            "navigate",
                            "observe",
                            "observe",
                            "observe",
                            "observe",
                            "observe",
                            "observe",
                            "observe",
                            "observe",
                            "close",
                        ],
                        "error": "resource_exhausted",
                        "allocation_disposition": "retired",
                    },
                ),
                _case(
                    "limit-oversized-artifact",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.FAILED,
                    operations=("navigate", "screenshot"),
                    route="/oversized",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "oversized_artifact"},
                ),
                _case(
                    "limit-oversized-download",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.FAILED,
                    operations=("navigate", "download"),
                    route="/download-oversized",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "oversized_artifact"},
                ),
                _case(
                    "limit-oversized-dom-and-names",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.FAILED,
                    route="/oversized-dom",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "oversized_snapshot"},
                ),
                _case(
                    "limit-oversized-accessible-name",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.FAILED,
                    route="/oversized-name",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "oversized_snapshot"},
                ),
                _case(
                    "limit-oversized-response-body",
                    category=BrowserAcceptanceCaseCategory.LIMIT,
                    state=BrowserAcceptanceState.FAILED,
                    route="/oversized-response",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "oversized_snapshot"},
                ),
                _unsupported("navigation-history-back", "go_back"),
                _unsupported("navigation-history-forward", "go_forward"),
                _unsupported("navigation-reload", "reload"),
                _case(
                    "navigation-redirect",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    route="/redirect",
                    parameters={
                        "required_operations": ["navigate"],
                        "expected_observed_target": "https://docs.browser.test/basic",
                    },
                ),
                _case(
                    "navigation-scroll-dependent-control",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click"),
                    route="/long-page",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "expected_effects": {"bottom-clicked": 1},
                    },
                ),
                _unsupported("page-multiple", "list_pages"),
                _case(
                    "page-popup-refused",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/popup",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "actionability_failed"},
                ),
                _case(
                    "recovery-conflicting-operation-id",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "navigate"),
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "required_operations": ["navigate", "navigate"],
                        "error": "operation_conflict",
                        "expected_browser_dispatches": 1,
                    },
                ),
                _case(
                    "recovery-exact-terminal-replay",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate", "navigate"),
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "expected_route_requests": 1,
                        "expected_browser_dispatches": 1,
                    },
                ),
                _case(
                    "recovery-process-loss-acknowledgement",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate",),
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={"expected_browser_dispatches": 1},
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_TERMINAL,
                ),
                _case(
                    "recovery-process-loss-artifact-publication",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate", "screenshot"),
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "required_operations": ["navigate", "screenshot"],
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_ARTIFACT,
                ),
                _case(
                    "recovery-process-loss-dispatched",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 0,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_DISPATCHED,
                ),
                _case(
                    "recovery-process-loss-guest-terminal",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate",),
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_BEFORE_TERMINAL,
                ),
                _case(
                    "recovery-process-loss-intent",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    state=BrowserAcceptanceState.REFUSED,
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "error": "operation_not_dispatched",
                        "expected_browser_dispatches": 0,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_INTENT,
                ),
                *(
                    _case(
                        f"revision-stale-ref-after-{operation}",
                        category=BrowserAcceptanceCaseCategory.REFUSAL,
                        state=BrowserAcceptanceState.REFUSED,
                        operations=(
                            "navigate",
                            operation,
                            "download" if operation == "download" else "click",
                        ),
                        route="/download" if operation == "download" else "/forms",
                        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                        parameters={"error": "stale_observation"},
                    )
                    for operation in (
                        "click",
                        "download",
                        "fill",
                        "press",
                        "screenshot",
                        "select",
                        "wait",
                    )
                ),
                _unsupported("visual-canvas-control", "visual_click"),
                _unsupported("visual-inaccessible-control", "visual_click"),
                _unsupported("visual-positioned-control", "visual_click"),
            ),
            key=lambda item: item.case_id,
        )
    )
    corpus_revision = _content_revision(
        {
            "suite_id": DETERMINISTIC_BROWSER_ACCEPTANCE_SUITE_ID,
            "fixture_revision": BROWSER_ACCEPTANCE_FIXTURE_REVISION,
            "cases": [case.revision for case in cases],
        },
        "deterministic browser acceptance corpus",
    )
    return BrowserAcceptanceManifestV1.build(
        corpus_revision=corpus_revision,
        suite_id=DETERMINISTIC_BROWSER_ACCEPTANCE_SUITE_ID,
        mode=BrowserAcceptanceMode.DETERMINISTIC,
        enabled=True,
        trial_count=1,
        allowed_origins=(
            "https://docs.browser.test",
            "https://static.browser.test",
        ),
        limits=BrowserAcceptanceLimitsV1(
            max_destinations=2,
            max_browser_operations=sum(
                len(case.operations)
                for case in cases
                if case.expected_state is not BrowserAcceptanceState.UNSUPPORTED
            ),
            max_model_steps=sum(
                len(case.operations) + 1
                for case in cases
                if case.expected_state is not BrowserAcceptanceState.UNSUPPORTED
            ),
            max_wall_time_ms=300_000,
            max_artifact_bytes=(
                BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION
                * sum(
                    len(case.operations)
                    for case in cases
                    if case.expected_state is not BrowserAcceptanceState.UNSUPPORTED
                )
            ),
            max_concurrency=1,
        ),
        cases=cases,
    )


def live_public_browser_acceptance_manifest() -> BrowserAcceptanceManifestV1:
    """Return the separately invoked, finite public-web v1 manifest."""

    cases = (
        _case(
            "live-iana-navigation",
            category=BrowserAcceptanceCaseCategory.SUCCESS,
            route="https://www.iana.org/domains/reserved",
        ),
        _case(
            "live-iana-screenshot",
            category=BrowserAcceptanceCaseCategory.SUCCESS,
            operations=("navigate", "screenshot"),
            route="https://www.iana.org/domains/reserved",
            oracle=BrowserAcceptanceSemanticOracle.ARTIFACT,
            parameters={"kind": "screenshot"},
            checkpoints=("terminal",),
        ),
    )
    return BrowserAcceptanceManifestV1.build(
        corpus_revision=_content_revision(
            {
                "suite_id": LIVE_PUBLIC_BROWSER_ACCEPTANCE_SUITE_ID,
                "cases": [case.revision for case in cases],
            },
            "live public browser acceptance corpus",
        ),
        suite_id=LIVE_PUBLIC_BROWSER_ACCEPTANCE_SUITE_ID,
        mode=BrowserAcceptanceMode.LIVE_PUBLIC,
        enabled=True,
        trial_count=3,
        allowed_origins=("https://www.iana.org",),
        limits=BrowserAcceptanceLimitsV1(
            max_destinations=1,
            max_browser_operations=16,
            max_model_steps=(max(len(case.operations) + 1 for case in cases) * len(cases) * 3),
            max_wall_time_ms=180_000,
            max_artifact_bytes=(
                BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION
                * sum(len(case.operations) * 3 for case in cases)
            ),
            max_concurrency=1,
            max_input_tokens=96_000,
            max_output_tokens=4_000,
            max_estimated_cost="1.00 USD",
        ),
        cases=cases,
    )


def live_authenticated_browser_acceptance_manifest() -> BrowserAcceptanceManifestV1:
    """Return the disabled v1 authenticated-suite capability declaration."""

    case = _unsupported("authenticated-profile-restoration", "restore_profile")
    return BrowserAcceptanceManifestV1.build(
        corpus_revision=_content_revision(
            {
                "suite_id": LIVE_AUTHENTICATED_BROWSER_ACCEPTANCE_SUITE_ID,
                "cases": [case.revision],
            },
            "live authenticated browser acceptance corpus",
        ),
        suite_id=LIVE_AUTHENTICATED_BROWSER_ACCEPTANCE_SUITE_ID,
        mode=BrowserAcceptanceMode.LIVE_AUTHENTICATED,
        enabled=False,
        trial_count=1,
        allowed_origins=("https://disabled.invalid",),
        limits=BrowserAcceptanceLimitsV1(
            max_destinations=1,
            max_browser_operations=1,
            max_model_steps=1,
            max_wall_time_ms=1_000,
            max_artifact_bytes=1,
            max_concurrency=1,
        ),
        cases=(case,),
    )


__all__ = [
    "DETERMINISTIC_BROWSER_ACCEPTANCE_SUITE_ID",
    "LIVE_AUTHENTICATED_BROWSER_ACCEPTANCE_SUITE_ID",
    "LIVE_PUBLIC_BROWSER_ACCEPTANCE_SUITE_ID",
    "deterministic_browser_acceptance_manifest",
    "live_authenticated_browser_acceptance_manifest",
    "live_public_browser_acceptance_manifest",
]
