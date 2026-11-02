"""Register an AWS Lambda MicroVM environment with CayuApp.

The factory is the library's ``cayu.environments.LambdaMicroVMEnvironmentFactory``.
CayuApp creates each MicroVM through the recoverable-allocation seam: a durable
intent and a ``RunMicrovm`` client token, so a create whose acknowledgement was
lost is adopted, not repeated. A resumed session reconnects to the same MicroVM
with fresh endpoint credentials, and a terminal outcome terminates it with
positive readback.

Sharing one MicroVM across the steps of a workflow run (``allocation_scope="parent"``
or ``"causal_budget"``) is refused for now: the shared path does not yet create
allocations through ``create_recoverable``. See "Sharing one allocation across
workflow steps and subagents" in ``docs/environment-factories.md``.

This factory doesn't mediate the MicroVM's network. For proxied, audited egress
with virtual credentials use ``SandboxCodingEnvironmentFactory`` with
``LambdaMicroVMEgressAdapter`` instead.
"""

from __future__ import annotations

from typing import Any

from cayu import CayuApp
from cayu.environments import (
    AllocationSharingScope,
    EnvironmentSpec,
    LambdaMicroVMEnvironmentFactory,
    LambdaMicroVMLifecycleBinding,
)

__all__ = [
    "LambdaMicroVMEnvironmentFactory",
    "LambdaMicroVMLifecycleBinding",
    "register_lambda_microvm_environment",
]


def register_lambda_microvm_environment(
    app: CayuApp,
    *,
    name: str,
    image_identifier: str,
    region_name: str,
    allocation_scope: AllocationSharingScope | str = AllocationSharingScope.SESSION,
    default: bool = True,
    **factory_options: Any,
) -> LambdaMicroVMEnvironmentFactory:
    """Register one Lambda MicroVM environment and return its factory."""

    factory = LambdaMicroVMEnvironmentFactory(
        image_identifier=image_identifier,
        region_name=region_name,
        **factory_options,
    )
    app.register_environment_factory(
        EnvironmentSpec(name=name),
        factory,
        default=default,
        allocation_scope=allocation_scope,
    )
    return factory
