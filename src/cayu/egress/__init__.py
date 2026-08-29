"""Virtual egress credentials for explicitly selected runners.

The runner receives only a virtual credential while a trusted broker outside
the runner swaps in the real vault secret and enforces per-request egress
policy. Egress enforcement provides credential non-possession; isolation
strength comes from the selected runner. See ``docs/virtual-egress.md``.
"""

from cayu.credentials import CredentialMode
from cayu.egress.adapter import (
    EgressAdapterRegistry,
    EgressAuthorityCutoverRequest,
    EgressAuthorityCutoverResult,
    EgressAuthorityRenewalRequest,
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    UnsupportedEgressAdapter,
    VirtualEgressRunnerRequest,
)
from cayu.egress.authority import (
    EGRESS_AUTHORITY_SCHEMA_VERSION,
    EgressAuthorityBindingIdentity,
    EgressAuthorityChangeKind,
    EgressAuthorityCutoverReceipt,
    EgressAuthorityCutoverStrategy,
    EgressAuthorityIdentity,
    EgressAuthorityOperation,
    EgressAuthorityPolicyIdentity,
    EgressAuthorityTransitionState,
    build_egress_authority_cutover_receipt,
    build_egress_authority_identity,
    compare_egress_authority,
)
from cayu.egress.broker import (
    CapturedRequest,
    CapturedResponse,
    EgressDecision,
    EgressUpstream,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpxUpstream,
    TransparentEgressBroker,
)
from cayu.egress.capabilities import (
    EGRESS_CAPABILITY_EVIDENCE_SCHEMA,
    EgressCapabilityClaim,
    EgressCapabilityDetail,
    EgressCapabilityEvidence,
    EgressCapabilityState,
)
from cayu.egress.credential_kinds import CredentialKind
from cayu.egress.destinations import ApprovedEgressDestination, EgressProtocol
from cayu.egress.errors import (
    EgressAuthorityCutoverError,
    EgressAuthorityCutoverNeedsAttention,
    EgressError,
    EgressReconnectConflictError,
    EgressReconnectError,
    EgressReconnectNotFoundError,
    InvalidEgressReconnectMetadataError,
    UnsupportedEgressAuthorityCutoverError,
    UnsupportedEgressCapabilityError,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
    VirtualCredentialError,
)
from cayu.egress.grants import (
    VirtualCredentialGrant,
    VirtualCredentialLease,
    VirtualCredentialRegistry,
)
from cayu.egress.policy import (
    BrowserEgressPolicy,
    EgressPolicy,
    EgressRequest,
    HttpEgressPolicy,
)
from cayu.egress.proxy_exposure import VpcTaskProxyExposure

__all__ = [
    "EGRESS_AUTHORITY_SCHEMA_VERSION",
    "EGRESS_CAPABILITY_EVIDENCE_SCHEMA",
    "ApprovedEgressDestination",
    "BrowserEgressPolicy",
    "CapturedRequest",
    "CapturedResponse",
    "CredentialKind",
    "CredentialMode",
    "EgressAdapterRegistry",
    "EgressAuthorityBindingIdentity",
    "EgressAuthorityChangeKind",
    "EgressAuthorityCutoverError",
    "EgressAuthorityCutoverNeedsAttention",
    "EgressAuthorityCutoverReceipt",
    "EgressAuthorityCutoverRequest",
    "EgressAuthorityCutoverResult",
    "EgressAuthorityCutoverStrategy",
    "EgressAuthorityIdentity",
    "EgressAuthorityOperation",
    "EgressAuthorityPolicyIdentity",
    "EgressAuthorityRenewalRequest",
    "EgressAuthorityTransitionState",
    "EgressBinding",
    "EgressCapabilityClaim",
    "EgressCapabilityDetail",
    "EgressCapabilityEvidence",
    "EgressCapabilityState",
    "EgressDecision",
    "EgressError",
    "EgressPolicy",
    "EgressProtocol",
    "EgressReconnectConflictError",
    "EgressReconnectError",
    "EgressReconnectNotFoundError",
    "EgressRequest",
    "EgressUpstream",
    "EgressUpstreamLimits",
    "EgressUpstreamOperation",
    "HttpEgressPolicy",
    "HttpxUpstream",
    "InvalidEgressReconnectMetadataError",
    "RunnerFinalizationResult",
    "SandboxEgressAdapter",
    "TransparentEgressBroker",
    "UnsupportedEgressAdapter",
    "UnsupportedEgressAuthorityCutoverError",
    "UnsupportedEgressCapabilityError",
    "UnsupportedEgressError",
    "UnsupportedEgressReconnectError",
    "VirtualCredentialError",
    "VirtualCredentialGrant",
    "VirtualCredentialLease",
    "VirtualCredentialRegistry",
    "VirtualEgressRunnerRequest",
    "VpcTaskProxyExposure",
    "build_egress_authority_cutover_receipt",
    "build_egress_authority_identity",
    "compare_egress_authority",
]
