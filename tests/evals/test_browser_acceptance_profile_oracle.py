"""A canonical restoration pass requires positive checkpoint and browser evidence."""

import pytest

from cayu.evals.browser_acceptance import (
    BrowserAcceptanceDiagnosticV1,
    BrowserAcceptanceProfileEvidenceV1,
    _semantic_state,
)
from cayu.evals.browser_acceptance_manifests import deterministic_browser_acceptance_manifest
from cayu.evals.corpus import _content_revision


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"generation_after": 1},
        {"active_writer": True},
        {"status": "revoked"},
        {"status": "outcome_unknown"},
        {"cookie_count": 0},
        {"checkpoint_receipt_revision": None},
        {"restore_receipt_revision": None},
    ],
)
def test_profile_oracle_requires_settled_store_evidence_after_json_reconstruction(change):
    case = next(
        case
        for case in deterministic_browser_acceptance_manifest().cases
        if case.case_id == "profile-cookie-restoration"
    )
    profile = BrowserAcceptanceProfileEvidenceV1(
        **{
            "store_kind": "sqlite",
            "status": "available",
            "authority_fingerprint": "1" * 64,
            "generation_before": 0,
            "generation_after": 2,
            "active_writer": False,
            "cookie_count": 1,
            "checkpoint_receipt_revision": "sha256:" + "2" * 64,
            "restore_receipt_revision": "sha256:" + "3" * 64,
            **(change or {}),
        }
    )
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state="captured",
        profile=profile,
        fixture_route_observed=True,
        fixture_route_request_count=1,
        fixture_authenticated_request_count=1,
        operations=tuple(
            {
                "sequence": index + 1,
                "invocation_revision": "sha256:" + str(index + 1) * 64,
                "operation": operation,
                "state": "terminal",
                "allocation_disposition": "retired" if operation == "close" else "live",
                "browser_session_revision": "sha256:" + ("a" if index < 2 else "b") * 64,
                "target_revision": _content_revision(
                    {"url": "https://docs.browser.test/auth/login"},
                    "browser acceptance operation target",
                )
                if index == 0
                else None,
            }
            for index, operation in enumerate(case.operations)
        ),
    )
    reconstructed = BrowserAcceptanceDiagnosticV1.model_validate_json(diagnostic.model_dump_json())
    assert _semantic_state(case, reconstructed, public_operations=frozenset()).value == (
        "passed" if change is None else "failed"
    )
    if change is None:
        absent = reconstructed.model_copy(update={"profile": None})
        assert _semantic_state(case, absent, public_operations=frozenset()).value == "failed"
        one_browser = reconstructed.model_copy(
            update={
                "operations": tuple(
                    item.model_copy(update={"browser_session_revision": "sha256:" + "a" * 64})
                    for item in reconstructed.operations
                )
            }
        )
        assert _semantic_state(case, one_browser, public_operations=frozenset()).value == "failed"
