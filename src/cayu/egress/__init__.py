"""Virtual egress credentials.

A secure sandbox credential path: the sandbox receives only a virtual
credential while a trusted broker outside the sandbox swaps in the real vault
secret and enforces per-request egress policy. See ``docs/virtual-egress.md``.
"""

from cayu.credentials import CredentialMode
from cayu.egress.adapter import (
    EgressAdapterRegistry,
    EgressBinding,
    SandboxEgressAdapter,
    UnsupportedEgressAdapter,
    VirtualEgressRunnerRequest,
)
from cayu.egress.broker import (
    CapturedRequest,
    CapturedResponse,
    EgressDecision,
    EgressUpstream,
    HttpxUpstream,
    TransparentEgressBroker,
)
from cayu.egress.credential_kinds import CredentialKind
from cayu.egress.destinations import ApprovedEgressDestination, EgressProtocol
from cayu.egress.errors import (
    EgressError,
    UnsupportedEgressCapabilityError,
    UnsupportedEgressError,
    VirtualCredentialError,
)
from cayu.egress.grants import (
    VirtualCredentialGrant,
    VirtualCredentialLease,
    VirtualCredentialRegistry,
)
from cayu.egress.policy import (
    EgressPolicy,
    EgressRequest,
    HttpEgressPolicy,
)
from cayu.egress.proxy_exposure import VpcTaskProxyExposure

__all__ = [
    "ApprovedEgressDestination",
    "CapturedRequest",
    "CapturedResponse",
    "CredentialKind",
    "CredentialMode",
    "EgressAdapterRegistry",
    "EgressBinding",
    "EgressDecision",
    "EgressError",
    "EgressPolicy",
    "EgressProtocol",
    "EgressRequest",
    "EgressUpstream",
    "HttpEgressPolicy",
    "HttpxUpstream",
    "SandboxEgressAdapter",
    "TransparentEgressBroker",
    "UnsupportedEgressAdapter",
    "UnsupportedEgressCapabilityError",
    "UnsupportedEgressError",
    "VirtualCredentialError",
    "VirtualCredentialGrant",
    "VirtualCredentialLease",
    "VirtualCredentialRegistry",
    "VirtualEgressRunnerRequest",
    "VpcTaskProxyExposure",
]
