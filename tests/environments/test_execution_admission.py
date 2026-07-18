from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cayu import (
    EXECUTION_CAPABILITY_EVIDENCE_SCHEMA,
    ExecutionAdmissionDecision,
    ExecutionAdmissionRefusal,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionEvidenceOverride,
    ExecutionRequirements,
    LocalRunner,
    evaluate_execution_admission,
)


def test_untrusted_requirements_cover_every_execution_security_dimension() -> None:
    requirements = ExecutionRequirements.untrusted()

    assert requirements.model_dump(mode="json") == {
        "code_trust": "untrusted",
        "real_secret_visibility": "non_possession",
        "network_access": "deny_by_default",
        "guest_privilege": "contained",
        "host_filesystem": "isolated",
        "cancellation": "confirmed",
        "cleanup": "confirmed",
        "durability": "ephemeral",
        "minimum_evidence": "available",
        "evidence_overrides": [],
    }


def test_capability_evidence_overrides_must_be_unique_and_required() -> None:
    override = ExecutionEvidenceOverride(
        capability="deny_by_default_network",
        minimum_evidence="live_verified",
    )
    requirements = ExecutionRequirements.untrusted(evidence_overrides=(override,))

    assert requirements.minimum_evidence_for("deny_by_default_network") == "live_verified"
    assert requirements.minimum_evidence_for("confirmed_cleanup") == "available"
    with pytest.raises(ValueError, match="unique capabilities"):
        ExecutionRequirements.untrusted(evidence_overrides=(override, override))
    with pytest.raises(ValueError, match="required capabilities"):
        ExecutionRequirements.trusted(evidence_overrides=(override,))


@pytest.mark.parametrize(
    ("status", "refusals"),
    [
        (
            "admitted",
            (
                ExecutionAdmissionRefusal(
                    code="missing_capability",
                    capability="deny_by_default_network",
                    required_state="live_verified",
                    observed_state="missing",
                ),
            ),
        ),
        ("refused", ()),
    ],
)
def test_admission_decision_rejects_status_refusal_contradictions(
    status: str,
    refusals: tuple[ExecutionAdmissionRefusal, ...],
) -> None:
    with pytest.raises(ValueError, match="admission status"):
        ExecutionAdmissionDecision(
            status=status,
            stage="pre_exposure",
            candidate="hosted-runner",
            requirements=ExecutionRequirements.untrusted(),
            refusals=refusals,
        )


def test_execution_evidence_distinguishes_support_availability_and_live_proof() -> None:
    observed_at = datetime(2026, 7, 16, tzinfo=UTC)
    evidence = ExecutionCapabilityEvidence(
        subject="hosted_microvm",
        claims=(
            ExecutionCapabilityClaim(
                capability="untrusted_code_isolation",
                state="declared",
                proof_source="integration_declaration",
                observation="supported",
            ),
            ExecutionCapabilityClaim(
                capability="host_filesystem_isolation",
                state="available",
                proof_source="integration_validation",
                observation="available",
            ),
            ExecutionCapabilityClaim(
                capability="deny_by_default_network",
                state="live_verified",
                proof_source="runtime_preflight",
                observation="denied",
                observed_at=observed_at,
                valid_until=observed_at + timedelta(hours=1),
            ),
            ExecutionCapabilityClaim(
                capability="metadata_isolation",
                state="unverified",
                proof_source="operator_opt_out",
                observation="not_probed",
                reason_code="metadata_boundary_unverified",
                remediation_code="enable_metadata_preflight",
            ),
            ExecutionCapabilityClaim(
                capability="reconnect",
                state="unsupported",
                proof_source="integration_declaration",
                observation="unavailable",
                reason_code="reconnect_unsupported",
                remediation_code="use_reconnectable_integration",
            ),
        ),
    )

    assert evidence.schema_version == EXECUTION_CAPABILITY_EVIDENCE_SCHEMA
    assert [claim.state for claim in evidence.claims] == [
        "declared",
        "available",
        "live_verified",
        "unverified",
        "unsupported",
    ]
    assert evidence.to_metadata()["subject"] == "hosted_microvm"


def test_explicit_candidate_is_admitted_when_every_requirement_has_available_evidence() -> None:
    capabilities = (
        "untrusted_code_isolation",
        "real_credential_non_possession",
        "deny_by_default_network",
        "guest_privilege_containment",
        "host_filesystem_isolation",
        "confirmed_cancellation",
        "confirmed_cleanup",
    )
    evidence = ExecutionCapabilityEvidence(
        subject="configured_microvm",
        claims=tuple(
            ExecutionCapabilityClaim(
                capability=capability,
                state="available",
                proof_source="integration_validation",
                observation="available",
            )
            for capability in capabilities
        ),
    )

    decision = evaluate_execution_admission(
        candidate="configured_microvm",
        requirements=ExecutionRequirements.untrusted(),
        evidence=evidence,
    )

    assert decision.status == "admitted"
    assert decision.candidate == "configured_microvm"
    assert decision.refusals == ()


def test_available_evidence_rejects_runtime_observations() -> None:
    decision = evaluate_execution_admission(
        candidate="configured_microvm",
        requirements=ExecutionRequirements(
            network_access="deny_by_default",
            minimum_evidence="available",
        ),
        evidence=ExecutionCapabilityEvidence(
            subject="configured_microvm",
            claims=(
                ExecutionCapabilityClaim(
                    capability="deny_by_default_network",
                    state="available",
                    proof_source="process_preflight",
                    observation="reachable",
                ),
            ),
        ),
    )

    assert decision.status == "refused"
    assert decision.refusals[0].code == "contradictory_evidence"


def test_admission_reports_every_unmet_requirement_without_trusting_candidate_name() -> None:
    evidence = ExecutionCapabilityEvidence(
        subject="microsandbox",
        claims=(
            ExecutionCapabilityClaim(
                capability="untrusted_code_isolation",
                state="unsupported",
                proof_source="integration_declaration",
                observation="unavailable",
                reason_code="isolation_boundary_unsupported",
                remediation_code="select_isolated_execution",
            ),
            ExecutionCapabilityClaim(
                capability="real_credential_non_possession",
                state="available",
                proof_source="integration_validation",
                observation="available",
            ),
            ExecutionCapabilityClaim(
                capability="deny_by_default_network",
                state="unverified",
                proof_source="operator_opt_out",
                observation="not_probed",
                reason_code="network_boundary_unverified",
                remediation_code="enable_network_preflight",
            ),
            ExecutionCapabilityClaim(
                capability="confirmed_cleanup",
                state="declared",
                proof_source="integration_declaration",
                observation="supported",
            ),
        ),
    )

    decision = evaluate_execution_admission(
        candidate="microsandbox",
        requirements=ExecutionRequirements.untrusted(),
        evidence=evidence,
    )

    assert decision.status == "refused"
    assert {refusal.capability: refusal.code for refusal in decision.refusals} == {
        "untrusted_code_isolation": "unsupported_capability",
        "deny_by_default_network": "unverified_capability",
        "guest_privilege_containment": "missing_capability",
        "host_filesystem_isolation": "missing_capability",
        "confirmed_cancellation": "missing_capability",
        "confirmed_cleanup": "insufficient_evidence",
    }


def test_local_runner_exposes_explicit_negative_isolation_evidence(tmp_path) -> None:
    evidence = LocalRunner(tmp_path).execution_capability_evidence()

    decision = evaluate_execution_admission(
        candidate="local",
        requirements=ExecutionRequirements.untrusted(),
        evidence=evidence,
    )

    isolation = next(
        refusal for refusal in decision.refusals if refusal.capability == "untrusted_code_isolation"
    )
    assert isolation.code == "unsupported_capability"
    assert isolation.reason_code == "local_process_isolation_unsupported"


def test_live_evidence_fails_closed_when_expired_or_from_an_unknown_schema() -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    requirements = ExecutionRequirements(
        network_access="deny_by_default",
        minimum_evidence="live_verified",
    )
    evidence = ExecutionCapabilityEvidence(
        subject="verified_microvm",
        claims=(
            ExecutionCapabilityClaim.live_verified(
                "deny_by_default_network",
                observation="denied",
                observed_at=now - timedelta(hours=2),
                valid_until=now - timedelta(hours=1),
            ),
        ),
    )

    stale = evaluate_execution_admission(
        candidate="verified_microvm",
        requirements=requirements,
        evidence=evidence,
        now=now,
    )
    unknown_schema = evaluate_execution_admission(
        candidate="verified_microvm",
        requirements=requirements,
        evidence={
            **evidence.model_dump(mode="python", by_alias=True),
            "schema": "cayu.execution_capabilities.v2",
        },
        now=now,
    )

    assert stale.refusals[0].code == "stale_evidence"
    assert stale.refusals[0].observed_state == "stale"
    assert unknown_schema.refusals[0].code == "malformed_evidence"


def test_live_evidence_rejects_contradictory_future_and_overlong_observations() -> None:
    now = datetime(2026, 7, 16, 12, tzinfo=UTC)
    requirements = ExecutionRequirements(
        network_access="deny_by_default",
        minimum_evidence="live_verified",
    )

    contradictory = evaluate_execution_admission(
        candidate="verified_microvm",
        requirements=requirements,
        evidence=ExecutionCapabilityEvidence(
            subject="verified_microvm",
            claims=(
                ExecutionCapabilityClaim.live_verified(
                    "deny_by_default_network",
                    observation="reachable",
                    observed_at=now - timedelta(minutes=1),
                    valid_until=now + timedelta(minutes=4),
                ),
            ),
        ),
        now=now,
    )
    future = evaluate_execution_admission(
        candidate="verified_microvm",
        requirements=requirements,
        evidence=ExecutionCapabilityEvidence(
            subject="verified_microvm",
            claims=(
                ExecutionCapabilityClaim.live_verified(
                    "deny_by_default_network",
                    observation="denied",
                    observed_at=now + timedelta(days=1),
                    valid_until=now + timedelta(days=1, minutes=5),
                ),
            ),
        ),
        now=now,
    )
    overlong = evaluate_execution_admission(
        candidate="verified_microvm",
        requirements=requirements,
        evidence=ExecutionCapabilityEvidence(
            subject="verified_microvm",
            claims=(
                ExecutionCapabilityClaim.live_verified(
                    "deny_by_default_network",
                    observation="denied",
                    observed_at=now - timedelta(minutes=1),
                    valid_until=now + timedelta(days=1),
                ),
            ),
        ),
        now=now,
    )

    assert contradictory.refusals[0].code == "contradictory_evidence"
    assert future.refusals[0].code == "future_evidence"
    assert overlong.refusals[0].code == "overlong_evidence"


def test_unusable_evidence_names_every_affected_hard_requirement() -> None:
    requirements = ExecutionRequirements.untrusted()
    malformed = evaluate_execution_admission(
        candidate="configured_microvm",
        requirements=requirements,
        evidence={"schema": "cayu.execution_capabilities.v2"},
    )
    mismatch = evaluate_execution_admission(
        candidate="configured_microvm",
        requirements=requirements,
        evidence=ExecutionCapabilityEvidence(
            subject="different_candidate",
            unclaimed_reason_code="capabilities_unclaimed",
        ),
    )

    expected = set(requirements.required_capabilities())
    assert {item.capability for item in malformed.refusals} == expected
    assert {item.capability for item in mismatch.refusals} == expected
    assert {item.code for item in malformed.refusals} == {"malformed_evidence"}
    assert {item.code for item in mismatch.refusals} == {"evidence_candidate_mismatch"}
