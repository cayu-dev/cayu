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
from cayu.egress.errors import (
    EgressError,
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

__all__ = [
    "CapturedRequest",
    "CapturedResponse",
    "CredentialKind",
    "CredentialMode",
    "EgressAdapterRegistry",
    "EgressBinding",
    "EgressDecision",
    "EgressError",
    "EgressPolicy",
    "EgressRequest",
    "EgressUpstream",
    "HttpEgressPolicy",
    "HttpxUpstream",
    "SandboxEgressAdapter",
    "TransparentEgressBroker",
    "UnsupportedEgressAdapter",
    "UnsupportedEgressError",
    "VirtualCredentialError",
    "VirtualCredentialGrant",
    "VirtualCredentialLease",
    "VirtualCredentialRegistry",
    "VirtualEgressRunnerRequest",
]
