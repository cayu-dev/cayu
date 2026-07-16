from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from cayu.egress import (
    EgressCapabilityClaim,
    EgressCapabilityDetail,
    EgressCapabilityEvidence,
    SandboxEgressAdapter,
)
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter
from cayu.egress.proxy_exposure import ProxyExposure
from cayu.runners import LocalRunner


def test_capability_evidence_has_stable_versioned_json_projection() -> None:
    evidence = EgressCapabilityEvidence(
        adapter="lambda_microvm",
        claims=(
            EgressCapabilityClaim(
                capability="metadata_isolation",
                state="verified",
                proof_source="agent_preflight",
                observation="denied",
            ),
            EgressCapabilityClaim(
                capability="direct_public_egress",
                state="verified",
                proof_source="agent_preflight",
                observation="denied",
            ),
        ),
    )

    assert evidence.to_metadata() == {
        "schema": "cayu.egress_capabilities.v1",
        "adapter": "lambda_microvm",
        "claims": [
            {
                "capability": "direct_public_egress",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": "denied",
            },
            {
                "capability": "metadata_isolation",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": "denied",
            },
        ],
    }
    assert evidence.state_for("metadata_isolation") == "verified"
    assert evidence.state_for("proxy_reachability") == "unclaimed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability", "metadata isolation"),
        ("capability", "sk_test_fixture_secret"),
        ("proof_source", "agent preflight"),
        ("observation", "sk_test_fixture_secret"),
    ],
)
def test_capability_claim_rejects_unbounded_or_secret_bearing_fields(
    field: str,
    value: str,
) -> None:
    values = {
        "capability": "metadata_isolation",
        "state": "verified",
        "proof_source": "agent_preflight",
        "observation": "denied",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(values)


def test_capability_evidence_rejects_duplicate_claims() -> None:
    claim = EgressCapabilityClaim(
        capability="metadata_isolation",
        state="verified",
        proof_source="agent_preflight",
        observation="denied",
    )

    with pytest.raises(ValidationError, match="unique"):
        EgressCapabilityEvidence(adapter="lambda_microvm", claims=(claim, claim))


def test_capability_evidence_rejects_secret_bearing_adapter_identity() -> None:
    with pytest.raises(ValidationError):
        EgressCapabilityEvidence.unclaimed("sk_test_fixture_secret")


def test_capability_evidence_rejects_unknown_schema_and_configuration_state() -> None:
    with pytest.raises(ValidationError):
        EgressCapabilityEvidence.model_validate(
            {
                "schema": "cayu.egress_capabilities.v2",
                "adapter": "lambda_microvm",
                "claims": [],
                "unclaimed_reason_code": "adapter_capabilities_unclaimed",
            }
        )
    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(
            {
                "capability": "metadata_isolation",
                "state": "required",
                "proof_source": "operator_configuration",
                "observation": "configured",
            }
        )
    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(
            {
                "capability": "metadata_isolation",
                "state": "verified",
                "proof_source": "operator_configuration",
                "observation": "denied",
            }
        )
    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(
            {
                "capability": "metadata_isolation",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": "required",
            }
        )


@pytest.mark.parametrize(
    ("state", "proof_source", "observation"),
    [
        ("verified", "agent_preflight", "unavailable"),
        ("verified", "external_live_verification", "not_probed"),
        ("unverified", "operator_opt_out", "denied"),
        ("unsupported", "adapter_declaration", "reachable"),
        ("unsupported", "adapter_declaration", "supported"),
    ],
)
def test_capability_claim_rejects_contradictory_proof_matrix(
    state: str,
    proof_source: str,
    observation: str,
) -> None:
    values = {
        "capability": "metadata_isolation",
        "state": state,
        "proof_source": proof_source,
        "observation": observation,
    }
    if state != "verified":
        values["reason_code"] = "capability_unsupported"
    if state == "unsupported":
        values["remediation_code"] = "use_supported_configuration"

    with pytest.raises(ValidationError, match="proof_source.*observation"):
        EgressCapabilityClaim.model_validate(values)


def test_non_verified_capability_requires_bounded_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason_code"):
        EgressCapabilityClaim(
            capability="metadata_isolation",
            state="unverified",
            proof_source="operator_opt_out",
            observation="not_probed",
        )


def test_capability_claim_rejects_non_json_proof_values() -> None:
    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(
            {
                "capability": "metadata_isolation",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": object(),
            }
        )


@pytest.mark.parametrize("field", ["reason_code", "remediation_code"])
def test_capability_claim_rejects_secret_bearing_safe_references(field: str) -> None:
    values = {
        "capability": "metadata_isolation",
        "state": "unsupported",
        "proof_source": "adapter_declaration",
        "observation": "unavailable",
        "reason_code": "capability_unsupported",
        "remediation_code": "use_supported_configuration",
    }
    values[field] = "sk_test_fixture_secret"

    with pytest.raises(ValidationError):
        EgressCapabilityClaim.model_validate(values)


def test_capability_claim_preserves_bounded_adapter_specific_details() -> None:
    evidence = EgressCapabilityEvidence(
        adapter="example",
        claims=(
            EgressCapabilityClaim(
                capability="metadata_isolation",
                state="verified",
                proof_source="agent_preflight",
                observation="denied",
                adapter_details=(
                    EgressCapabilityDetail(name="network_namespace_count", value=1),
                    EgressCapabilityDetail(name="relay_only", value=True),
                    EgressCapabilityDetail(name="agent_namespace", value=True),
                ),
            ),
        ),
    )

    assert evidence.to_metadata()["claims"][0]["adapter_details"] == [  # type: ignore[index]
        {"name": "agent_namespace", "value": True},
        {"name": "network_namespace_count", "value": 1},
        {"name": "relay_only", "value": True},
    ]


@pytest.mark.parametrize(
    "value",
    ["sk_test_fixture_secret", "sk-proj-fixture", "gho_fixture", "rk_live_fixture"],
)
def test_capability_detail_rejects_string_values(value: str) -> None:
    with pytest.raises(ValidationError):
        EgressCapabilityDetail(name="topology", value=value)


@pytest.mark.parametrize("value", [-9_007_199_254_740_991, 9_007_199_254_740_991])
def test_capability_detail_accepts_safe_json_integer_boundaries(value: int) -> None:
    assert EgressCapabilityDetail(name="count", value=value).value == value


@pytest.mark.parametrize("value", [-9_007_199_254_740_992, 9_007_199_254_740_992])
def test_capability_detail_rejects_integers_outside_safe_json_range(value: int) -> None:
    with pytest.raises(ValidationError):
        EgressCapabilityDetail(name="count", value=value)


def test_capability_evidence_bounds_claim_count() -> None:
    claims = tuple(
        EgressCapabilityClaim(
            capability=f"capability_{index}",
            state="verified",
            proof_source="agent_preflight",
            observation="supported",
        )
        for index in range(65)
    )

    evidence = EgressCapabilityEvidence(adapter="example", claims=claims[:64])

    assert len(evidence.claims) == 64
    with pytest.raises(ValidationError):
        EgressCapabilityEvidence(adapter="example", claims=claims)


def test_unsupported_capability_is_distinct_and_actionable() -> None:
    evidence = EgressCapabilityEvidence(
        adapter="example",
        claims=(
            EgressCapabilityClaim(
                capability="metadata_isolation",
                state="unsupported",
                proof_source="adapter_declaration",
                observation="unavailable",
                reason_code="capability_unsupported",
                remediation_code="use_supported_configuration",
            ),
        ),
    )

    assert evidence.state_for("metadata_isolation") == "unsupported"


@pytest.mark.parametrize(
    ("adapter", "adapter_name"),
    [
        (DockerEgressAdapter(), "docker"),
        (E2BEgressAdapter(exposure=cast("ProxyExposure", object())), "e2b"),
        (MicrosandboxEgressAdapter(), "microsandbox"),
    ],
)
def test_builtin_adapters_without_runtime_claims_publish_explicit_unclaimed_evidence(
    adapter: SandboxEgressAdapter,
    adapter_name: str,
    tmp_path: Path,
) -> None:
    runner = LocalRunner(tmp_path, inherit_env=False)
    evidence = adapter.capability_evidence(runner)

    assert evidence.to_metadata() == {
        "schema": "cayu.egress_capabilities.v1",
        "adapter": adapter_name,
        "claims": [],
        "unclaimed_reason_code": "adapter_capabilities_unclaimed",
    }
