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
DETERMINISTIC_BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION = 4 * 1024 * 1024
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


def _stale_ref_case(operation: str) -> BrowserAcceptanceCaseV1:
    operation_flow, route = {
        "back": (("navigate", "click", "back", "click"), "/history-start"),
        "forward": (
            ("navigate", "click", "back", "forward", "click"),
            "/history-start",
        ),
        "hover": (("navigate", "hover", "click"), "/hover"),
        "reload": (("navigate", "reload", "click"), "/reload"),
        "scroll": (("navigate", "scroll", "click"), "/forms"),
        "upload": (("navigate", "upload", "click"), "/upload"),
    }.get(
        operation,
        (
            ("navigate", operation, "download" if operation == "download" else "click"),
            "/download" if operation == "download" else "/forms",
        ),
    )
    return _case(
        f"revision-stale-ref-after-{operation}",
        category=BrowserAcceptanceCaseCategory.REFUSAL,
        state=BrowserAcceptanceState.REFUSED,
        operations=operation_flow,
        route=route,
        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
        parameters={"error": "stale_observation"},
    )


def deterministic_browser_acceptance_manifest() -> BrowserAcceptanceManifestV1:
    """Return the immutable credential-free v1 conformance manifest."""

    cases = tuple(
        sorted(
            (
                _case(
                    "access-broker-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    route="/denied-subresource",
                    parameters={
                        "required_operations": ["navigate"],
                        "required_requests": [{"path": "/private/denied.js", "outcome": "denied"}],
                    },
                ),
                _case(
                    "access-main-document-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    route="https://blocked.browser.test/private",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "error": "fetch_failed",
                        "required_requests": [
                            {
                                "method": "CONNECT",
                                "destination": "blocked.browser.test",
                                "path": "/",
                                "outcome": "denied",
                            }
                        ],
                    },
                ),
                _case(
                    "access-redirect-denial",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    route="/redirect-denied",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "error": "fetch_failed",
                        "required_requests": [
                            {
                                "method": "GET",
                                "destination": "docs.browser.test",
                                "path": "/redirect-denied",
                                "outcome": "authorized",
                            },
                            {
                                "method": "CONNECT",
                                "destination": "blocked.browser.test",
                                "path": "/",
                                "outcome": "denied",
                            },
                        ],
                    },
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
                    parameters={"error": "actionability_failed"},
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
                    parameters={"error": "actionability_failed"},
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
                    state=BrowserAcceptanceState.REFUSED,
                    route="/challenge",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "required_operations": ["navigate"],
                        "forbidden_operations": ["click", "download"],
                        "error": "access_blocked",
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
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
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
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.ARTIFACT,
                    parameters={"kind": "screenshot"},
                    checkpoints=("after-navigation",),
                ),
                _case(
                    "artifact-upload",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "upload"),
                    route="/upload",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "upload"],
                        "expected_effects": {"upload-selected": 1},
                    },
                ),
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
                    route="/long-observation",
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
                _case(
                    "navigation-history-back",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "back", "click"),
                    route="/history-start",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "click", "back", "click"],
                        "expected_effects": {"history-back-confirmed": 1},
                    },
                ),
                _case(
                    "navigation-forward-unavailable",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "forward"),
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "history_unavailable"},
                ),
                _case(
                    "navigation-reload-dialog-refused",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "reload"),
                    route="/reload-dialog",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "error": "unsafe_reload",
                        "expected_effects": {},
                    },
                ),
                *(
                    _case(
                        f"artifact-upload-{suffix}",
                        category=BrowserAcceptanceCaseCategory.REFUSAL,
                        state=BrowserAcceptanceState.REFUSED,
                        operations=("navigate", "upload"),
                        route="/upload",
                        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                        parameters={"error": error, "expected_effects": {}},
                    )
                    for suffix, error in (
                        ("missing", "artifact_unavailable"),
                        ("wrong-session", "artifact_refused"),
                        ("too-large", "upload_too_large"),
                        ("incompatible-target", "incompatible_upload_target"),
                    )
                ),
                *(
                    _case(
                        f"artifact-upload-{phase}",
                        category=BrowserAcceptanceCaseCategory.CRASH,
                        state=state,
                        operations=("navigate", "upload"),
                        route="/upload",
                        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                        parameters={
                            "error": error,
                            "expected_browser_dispatches": 2,
                            "expected_effects": effects,
                            "allocation_disposition": disposition,
                        },
                        fault_scenario=scenario,
                    )
                    for phase, scenario, effects, disposition, error, state in (
                        (
                            "disconnection",
                            BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_DISCONNECTION,
                            {"upload-selected": 1},
                            "uncertain",
                            "browser_crash",
                            BrowserAcceptanceState.FAILED,
                        ),
                        (
                            "acknowledgement-loss",
                            BrowserAcceptanceFaultScenario.BROWSER_UPLOAD_ACKNOWLEDGEMENT_LOSS,
                            {"upload-selected": 1},
                            "uncertain",
                            "outcome_ambiguous",
                            BrowserAcceptanceState.AMBIGUOUS,
                        ),
                    )
                ),
                *(
                    _case(
                        f"action-hover-{suffix}",
                        category=BrowserAcceptanceCaseCategory.REFUSAL,
                        state=BrowserAcceptanceState.REFUSED,
                        operations=("navigate", "hover"),
                        route=route,
                        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                        parameters={"error": error, "expected_effects": {}},
                    )
                    for suffix, route, error in (
                        ("detached", "/detached", "actionability_failed"),
                        ("occluded", "/occluded", "actionability_failed"),
                    )
                ),
                _case(
                    "navigation-history-forward",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "back", "forward", "click"),
                    route="/history-start",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": [
                            "navigate",
                            "click",
                            "back",
                            "forward",
                            "click",
                        ],
                        "expected_effects": {"history-forward-confirmed": 1},
                    },
                ),
                _case(
                    "navigation-reload",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "reload", "click"),
                    route="/reload",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "reload", "click"],
                        "expected_effects": {"reload-confirmed": 1},
                    },
                ),
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
                    operations=("navigate", "scroll", "click"),
                    route="/scroll",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "scroll", "click"],
                        "expected_effects": {"bottom-clicked": 1},
                    },
                ),
                _case(
                    "navigation-scroll-over-limit",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.FAILED,
                    operations=("navigate", "scroll"),
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "invalid_arguments", "expected_effects": {}},
                ),
                _case(
                    "action-strict-hover",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "hover"),
                    route="/hover",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "hover"],
                        "expected_effects": {"hover-observed": 1},
                    },
                ),
                _case(
                    "page-about-blank-popup-transition",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "list_pages"),
                    route="/popup-about-blank",
                ),
                _case(
                    "page-active-page-crash",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    operations=("navigate", "click", "list_pages"),
                    route="/popup",
                    parameters={
                        "required_operations": ["navigate", "click", "list_pages"],
                        "expected_browser_dispatches": 3,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_ACTIVE_PAGE_CRASH,
                ),
                _case(
                    "page-allocation-loss",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    state=BrowserAcceptanceState.FAILED,
                    operations=("navigate", "wait"),
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={
                        "required_operations": ["navigate", "wait"],
                        "error": "allocation_lost",
                        "allocation_disposition": "retired",
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_ALLOCATION_LOSS,
                ),
                _case(
                    "page-background-page-crash",
                    category=BrowserAcceptanceCaseCategory.CRASH,
                    operations=("navigate", "click", "list_pages"),
                    route="/popup",
                    parameters={
                        "required_operations": ["navigate", "click", "list_pages"],
                        "expected_browser_dispatches": 3,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.BROWSER_BACKGROUND_PAGE_CRASH,
                ),
                _case(
                    "page-complete-cleanup",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "close"),
                    route="/popup",
                ),
                _case(
                    "page-cross-origin-popup",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "list_pages", "switch_page"),
                    route="/popup-cross-origin",
                ),
                _case(
                    "page-cross-page-stale-ref",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click", "switch_page", "click"),
                    route="/popup",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "unknown_element"},
                ),
                _case(
                    "page-popup-burst",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/popup-burst",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "resource_exhausted"},
                ),
                _case(
                    "page-popup-redirect-pivot",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "click"),
                    route="/popup-redirect",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "policy_denied"},
                ),
                _case(
                    "page-multiple-popup-tab-switch-close",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=(
                        "navigate",
                        "click",
                        "list_pages",
                        "switch_page",
                        "close_page",
                        "list_pages",
                        "close",
                    ),
                    route="/popup",
                ),
                _case(
                    "page-popup-opener-navigation",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click", "list_pages"),
                    route="/popup-opener-navigation",
                ),
                _case(
                    "page-popup-exact-replay",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate", "click"),
                    route="/popup",
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS,
                ),
                _case(
                    "page-popup-process-loss-ambiguity",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    operations=("navigate", "click"),
                    route="/popup",
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "error": "outcome_ambiguous",
                        "expected_browser_dispatches": 2,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_BEFORE_TERMINAL,
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
                    operations=("navigate",),
                    route="/basic",
                    oracle=BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
                    parameters={
                        "expected_route_requests": 1,
                        "expected_browser_dispatches": 1,
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS,
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
                    _stale_ref_case(operation)
                    for operation in (
                        "back",
                        "click",
                        "download",
                        "fill",
                        "forward",
                        "hover",
                        "press",
                        "reload",
                        "screenshot",
                        "scroll",
                        "select",
                        "upload",
                        "wait",
                    )
                ),
                *(
                    _case(
                        case_id,
                        category=BrowserAcceptanceCaseCategory.SUCCESS,
                        operations=("navigate", "observe_visual", "click_visual_target"),
                        route=route,
                        oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                        parameters={
                            "required_operations": [
                                "navigate",
                                "observe_visual",
                                "click_visual_target",
                            ],
                            "expected_effects": {"visual-activated": 1},
                        },
                    )
                    for case_id, route in (
                        ("visual-canvas-control", "/visual-canvas"),
                        ("visual-inaccessible-control", "/visual-image"),
                        ("visual-positioned-control", "/visual-positioned-canvas"),
                        ("visual-hostile-pixels-labels", "/visual-hostile"),
                        ("visual-popup-outcome", "/visual-popup"),
                    )
                ),
                _case(
                    "visual-semantic-preference",
                    category=BrowserAcceptanceCaseCategory.SUCCESS,
                    operations=("navigate", "click"),
                    route="/visual-semantic",
                    oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                    parameters={
                        "required_operations": ["navigate", "click"],
                        "expected_effects": {"visual-activated": 1},
                    },
                ),
                _case(
                    "visual-opaque-host-refusal",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "observe_visual", "click_visual_target"),
                    route="/visual-positioned",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "unsupported_visual_surface"},
                ),
                _case(
                    "visual-stale-screenshot",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "observe_visual", "wait", "click_visual_target"),
                    route="/visual-canvas",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "stale_observation"},
                ),
                _case(
                    "visual-secret-refusal",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "observe_visual"),
                    route="/visual-canvas",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "policy_denied", "expected_browser_dispatches": 1},
                    fault_scenario=BrowserAcceptanceFaultScenario.SECRET_BEFORE_CAPTURE,
                ),
                *(
                    _case(
                        case_id,
                        category=BrowserAcceptanceCaseCategory.RECOVERY,
                        operations=("navigate", "observe_visual", "click_visual_target"),
                        route="/visual-popup",
                        oracle=BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT,
                        parameters={
                            "required_operations": [
                                "navigate",
                                "observe_visual",
                                "click_visual_target",
                            ],
                            "expected_effects": {"visual-activated": 1},
                            "expected_browser_dispatches": 3,
                        },
                        fault_scenario=fault,
                    )
                    for case_id, fault in (
                        (
                            "visual-terminal-acknowledgement-loss",
                            BrowserAcceptanceFaultScenario.ACKNOWLEDGEMENT_LOSS,
                        ),
                    )
                ),
                _case(
                    "visual-process-terminal-replay",
                    category=BrowserAcceptanceCaseCategory.RECOVERY,
                    state=BrowserAcceptanceState.AMBIGUOUS,
                    operations=("navigate", "observe_visual", "click_visual_target"),
                    route="/visual-popup",
                    oracle=BrowserAcceptanceSemanticOracle.QUARANTINED_RECOVERY,
                    parameters={
                        "quarantined_tool_calls": 1,
                        "expected_browser_dispatches": 3,
                        "expected_effects": {"visual-activated": 1},
                    },
                    fault_scenario=BrowserAcceptanceFaultScenario.PROCESS_AFTER_TERMINAL,
                ),
                *(
                    _case(
                        case_id,
                        category=BrowserAcceptanceCaseCategory.REFUSAL,
                        state=BrowserAcceptanceState.REFUSED,
                        operations=("navigate", "observe_visual", "click_visual_target"),
                        route=route,
                        oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                        parameters={"error": error},
                    )
                    for case_id, route, error in (
                        ("visual-virtualized-movement", "/visual-moved", "visual_evidence_expired"),
                        ("visual-sticky-overlay", "/visual-overlay", "visual_evidence_expired"),
                        (
                            "visual-viewport-scroll-change",
                            "/visual-scroll",
                            "visual_viewport_mismatch",
                        ),
                    )
                ),
                _case(
                    "visual-cross-origin-frame",
                    category=BrowserAcceptanceCaseCategory.REFUSAL,
                    state=BrowserAcceptanceState.REFUSED,
                    operations=("navigate", "observe_visual"),
                    route="/cross-origin-frame",
                    oracle=BrowserAcceptanceSemanticOracle.STABLE_ERROR,
                    parameters={"error": "unsupported_visual_surface"},
                ),
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
            max_wall_time_ms=900_000,
            max_artifact_bytes=(
                DETERMINISTIC_BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION
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
