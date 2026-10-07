"""Runtime-derived verifier-profile components for provider-backed verifiers.

A provider verifier's decisions depend on the provider adapter, the model and
the request limits as much as on the application's prompt. Those are runtime
facts, so the runtime adds them to the immutable verifier profile itself rather
than trusting the application to declare them.
"""

from __future__ import annotations

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu._version import package_version
from cayu.execution_profiles import ExecutionProfileIdentityStrength
from cayu.runtime._execution_profile_admission import resolve_provider_adapter_component
from cayu.runtime._runtime_records import RegisteredProvider
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tasks.completion_verifier_profiles import (
    CompletionVerifierProfileComponentDeclaration,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.verification.completion_verifiers import CompletionVerifierUnavailable
from cayu.verification.provider_completion_verifiers import (
    PROVIDER_COMPLETION_VERIFIER_DECISION_CONTRACT_VERSION,
    ProviderCompletionVerifierTarget,
)

PROVIDER_ADAPTER_COMPONENT_ID = "runtime.provider-adapter"
PROVIDER_TARGET_COMPONENT_ID = "runtime.provider-target"
RUNTIME_COMPONENT_PREFIX = "runtime."

_STABLE_STRENGTHS = frozenset(
    {
        ExecutionProfileIdentityStrength.APPLICATION_VERSIONED,
        ExecutionProfileIdentityStrength.STRUCTURAL,
    }
)


def provider_verifier_target_fingerprint(target: ProviderCompletionVerifierTarget) -> str:
    return sha256(
        canonical_durable_json_bytes(
            {
                "decision_contract_version": (
                    PROVIDER_COMPLETION_VERIFIER_DECISION_CONTRACT_VERSION
                ),
                "target": target.model_dump(mode="json", warnings=False),
            },
            "provider_completion_verifier_target",
        )
    ).hexdigest()


def provider_verifier_runtime_components(
    *,
    target: ProviderCompletionVerifierTarget,
    registered_provider: RegisteredProvider,
    process_identity: str,
    redactor: SecretRedactor,
) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
    """Return the provider-adapter and target components, or fail closed.

    A process-local provider identity cannot be compared after restart, so a
    durable verifier profile cannot be bound to it.
    """

    component = resolve_provider_adapter_component(
        registered_provider=registered_provider,
        runtime_version=package_version(),
        process_identity=process_identity,
        redactor=redactor,
    )
    if component.strength not in _STABLE_STRENGTHS or component.fingerprint is None:
        raise CompletionVerifierUnavailable(
            "Provider-backed completion verifiers require a provider with a stable "
            "execution-profile identity."
        )
    return (
        CompletionVerifierProfileComponentDeclaration(
            component_id=PROVIDER_ADAPTER_COMPONENT_ID,
            identity=ExecutionProfileBehaviorIdentity(
                name="cayu.provider-adapter",
                behavior_version=component.fingerprint,
                implementation_version=component.strength.value,
            ),
        ),
        CompletionVerifierProfileComponentDeclaration(
            component_id=PROVIDER_TARGET_COMPONENT_ID,
            identity=ExecutionProfileBehaviorIdentity(
                name="cayu.provider-verifier-target",
                behavior_version=provider_verifier_target_fingerprint(target),
                implementation_version="1",
            ),
        ),
    )


__all__ = [
    "PROVIDER_ADAPTER_COMPONENT_ID",
    "PROVIDER_TARGET_COMPONENT_ID",
    "RUNTIME_COMPONENT_PREFIX",
    "provider_verifier_runtime_components",
    "provider_verifier_target_fingerprint",
]
