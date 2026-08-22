from __future__ import annotations

import pytest

from cayu.egress import (
    BrowserEgressPolicy,
    EgressAuthorityBindingIdentity,
    EgressAuthorityChangeKind,
    EgressAuthorityCutoverStrategy,
    HttpEgressPolicy,
    build_egress_authority_identity,
    compare_egress_authority,
)
from cayu.runtime.egress import VirtualCredentialSpec, VirtualEgressEnvironmentFactory
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfilePolicyRequest,
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_with_egress_authority,
)
from cayu.vaults import SecretRef, StaticVault


def _bounded_authority_from_policies(policies: dict[str, HttpEgressPolicy]):
    first_name = next(iter(policies))
    first_host = next(iter(policies[first_name].allowed_hosts))
    return build_egress_authority_identity(
        policies=policies,
        bindings=(
            EgressAuthorityBindingIdentity(
                destination=first_host,
                policy_name=first_name,
                credential_kind="opaque_bearer",
                credential_authority_fingerprint="1" * 64,
            ),
        ),
        generation=1,
        authority_source="trusted-app",
        authority_scope="session",
        policy_version="bounded-test",
        runner_kind="docker",
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def _authority(
    *,
    generation: int = 1,
    endpoints: tuple[tuple[str, str], ...] = (("GET", "/v1/items"),),
    destination: str = "api.example.com",
    credential_authority_fingerprint: str = "1" * 64,
):
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=(destination,),
        allowed_endpoints=endpoints,
    )
    return build_egress_authority_identity(
        policies={policy.name: policy},
        bindings=(
            EgressAuthorityBindingIdentity(
                destination=destination,
                policy_name=policy.name,
                credential_kind="opaque_bearer",
                credential_authority_fingerprint=credential_authority_fingerprint,
            ),
        ),
        generation=generation,
        authority_source="trusted-app",
        authority_scope="session",
        policy_version="2026-08-21",
        runner_kind="docker",
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def _profile(authority=None):
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model="fake-model",
        durable_system_prompt="system",
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'0' * 64}",
        egress_authority=authority,
    )


def test_authority_comparison_is_typed_and_conservative() -> None:
    original = _authority()
    next_generation = _authority(generation=2)
    wider = _authority(
        generation=2,
        endpoints=(("GET", "/v1/items"), ("POST", "/v1/items")),
    )
    narrower = _authority(generation=2, endpoints=(("GET", "/v1/items"),))
    broad_original = _authority(
        endpoints=(("GET", "/v1/items"), ("POST", "/v1/items")),
    )
    unrelated = _authority(generation=2, destination="other.example.com")
    different_credential = _authority(
        generation=2,
        credential_authority_fingerprint="2" * 64,
    )

    assert compare_egress_authority(original, next_generation) is (
        EgressAuthorityChangeKind.UNCHANGED
    )
    assert compare_egress_authority(original, wider) is EgressAuthorityChangeKind.WIDER
    assert compare_egress_authority(broad_original, narrower) is (
        EgressAuthorityChangeKind.NARROWER
    )
    assert compare_egress_authority(original, unrelated) is (EgressAuthorityChangeKind.INCOMPARABLE)
    assert compare_egress_authority(original, different_credential) is (
        EgressAuthorityChangeKind.INCOMPARABLE
    )
    assert compare_egress_authority(original, wider, refused=True) is (
        EgressAuthorityChangeKind.REFUSED
    )


def test_profile_binds_typed_egress_generation_and_policy_request_comparison() -> None:
    expected_authority = _authority()
    target_authority = _authority(
        generation=2,
        endpoints=(("GET", "/v1/items"), ("POST", "/v1/items")),
    )
    expected = _profile(expected_authority)
    candidate = execution_profile_with_egress_authority(expected, target_authority)
    changed = changed_execution_profile_components(expected, candidate)

    assert changed == (ExecutionProfileComponentClass.EGRESS_AUTHORITY,)
    assert candidate.egress_authority == target_authority
    request = ExecutionProfilePolicyRequest(
        session_id="session-1",
        expected_profile=expected,
        candidate_profile=candidate,
        changed_component_classes=changed,
        authority_review_required=True,
        source_provider_name="fake",
        source_model="fake-model",
        target_provider_name="fake",
        target_model="fake-model",
    )
    assert request.egress_authority_change is EgressAuthorityChangeKind.WIDER


def test_authority_projection_is_secret_free_and_bounded() -> None:
    authority = _authority()
    dumped = authority.model_dump_json()

    assert "opaque_bearer" in dumped
    assert "api.example.com" in dumped
    assert "secret" not in dumped.lower()
    assert "credential_material" not in dumped


@pytest.mark.parametrize(
    ("policies", "message"),
    (
        (
            {
                f"policy-{policy_index}": HttpEgressPolicy(
                    name=f"policy-{policy_index}",
                    allowed_hosts=(f"host-{policy_index}.example.com",),
                    allowed_endpoints=tuple(
                        ("GET", f"/operation-{operation_index}") for operation_index in range(33)
                    ),
                )
                for policy_index in range(128)
            },
            "aggregate operations",
        ),
        (
            {
                f"policy-{policy_index}": HttpEgressPolicy(
                    name=f"policy-{policy_index}",
                    allowed_hosts=tuple(
                        f"host-{policy_index}-{host_index}.example.com" for host_index in range(9)
                    ),
                    allowed_endpoints=(("GET", "/"),),
                )
                for policy_index in range(128)
            },
            "aggregate destinations",
        ),
        (
            {
                f"policy-{policy_index}": HttpEgressPolicy(
                    name=f"policy-{policy_index}",
                    allowed_hosts=(f"host-{policy_index}.example.com",),
                    allowed_endpoints=(("GET", "/"),),
                    denied_prefixes=tuple(
                        f"/denied-{policy_index}-{prefix_index}" for prefix_index in range(9)
                    ),
                )
                for policy_index in range(128)
            },
            "aggregate denied path prefixes",
        ),
    ),
)
def test_authority_rejects_split_aggregate_limits(
    policies: dict[str, HttpEgressPolicy],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _bounded_authority_from_policies(policies)


def test_authority_rejects_aggregate_canonical_byte_limit() -> None:
    policy = HttpEgressPolicy(
        name="large-policy",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=tuple(
            ("GET", "/" + str(index).zfill(4) + "x" * 2030) for index in range(512)
        ),
    )

    with pytest.raises(ValueError, match="canonical byte limit"):
        _bounded_authority_from_policies({policy.name: policy})


def test_authority_comparison_fails_closed_before_quadratic_maximum() -> None:
    policies = {
        f"policy-{policy_index}": HttpEgressPolicy(
            name=f"policy-{policy_index}",
            allowed_hosts=(f"host-{policy_index}.example.com",),
            allowed_endpoints=tuple(
                ("GET", f"/operation-{operation_index}") for operation_index in range(512)
            ),
        )
        for policy_index in range(8)
    }
    bindings = tuple(
        EgressAuthorityBindingIdentity(
            destination=f"host-{policy_index}.example.com",
            policy_name=f"policy-{policy_index}",
            credential_kind="opaque_bearer",
            credential_authority_fingerprint="1" * 64,
        )
        for policy_index in range(8)
    )

    def build(generation: int):
        return build_egress_authority_identity(
            policies=policies,
            bindings=bindings,
            generation=generation,
            authority_source="trusted-app",
            authority_scope="session",
            policy_version=f"v{generation}",
            runner_kind="docker",
            cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
        )

    assert compare_egress_authority(build(1), build(2)) is (EgressAuthorityChangeKind.INCOMPARABLE)


def test_authority_comparison_bounds_denied_prefix_cross_product() -> None:
    denied_prefixes = tuple(f"/denied-{index}" for index in range(1024))
    policy = BrowserEgressPolicy(
        name="browser",
        allowed_hosts=("browser.example.com",),
        allowed_path_prefixes=("/",),
        denied_prefixes=denied_prefixes,
    )

    def build(generation: int):
        return build_egress_authority_identity(
            policies={policy.name: policy},
            bindings=(
                EgressAuthorityBindingIdentity(
                    destination="browser.example.com",
                    policy_name=policy.name,
                    credential_kind="opaque_bearer",
                    credential_authority_fingerprint="1" * 64,
                ),
            ),
            generation=generation,
            authority_source="trusted-app",
            authority_scope="session",
            policy_version=f"v{generation}",
            runner_kind="docker",
            cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
        )

    assert compare_egress_authority(build(1), build(2)) is (EgressAuthorityChangeKind.INCOMPARABLE)


def test_factory_commits_secret_ref_identity_without_publishing_it() -> None:
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=(("GET", "/v1/items"),),
    )

    def factory(secret_name: str) -> VirtualEgressEnvironmentFactory:
        return VirtualEgressEnvironmentFactory(
            policies={policy.name: policy},
            credentials=(
                VirtualCredentialSpec(
                    env_name="PROVIDER_TOKEN",
                    secret=SecretRef(name=secret_name),
                    destination="api.example.com",
                    policy_name=policy.name,
                    credential_kind="opaque_bearer",
                ),
            ),
            resolver=StaticVault({secret_name: "sk_test_material"}),
            runner_kind="docker",
        )

    first = factory("provider-test-secret")
    second = factory("provider-rotated-secret")
    first_authority = first.egress_authority_identity
    second_authority = second.egress_authority_identity

    assert first_authority.bindings[0].credential_authority_fingerprint != (
        second_authority.bindings[0].credential_authority_fingerprint
    )
    assert first_authority.fingerprint != second_authority.fingerprint
    assert "provider-test-secret" not in first_authority.model_dump_json()
    assert "sk_test_material" not in first_authority.model_dump_json()
