"""Trusted registration for request-owner operations, not agent execution."""

from dataclasses import dataclass

from cayu.collaboration.mandates import MandateResolver, ResourceSelectorOwner


@dataclass(frozen=True)
class RequestRegistration:
    mandates: MandateResolver
    max_ttl_ms: int
    resource_owners: tuple[ResourceSelectorOwner, ...] = ()
