"""Canonical lazy exports for verified-work adapters and workers."""

EXPORTS = {
    "COMPLETION_RESULT_RESOLUTION_MAX_SECONDS": (
        "cayu.verification.completion_result_resolvers",
        "COMPLETION_RESULT_RESOLUTION_MAX_SECONDS",
    ),
    "CompletionResultResolutionRequest": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultResolutionRequest",
    ),
    "CompletionResultResolver": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultResolver",
    ),
    "CompletionResultResolverExecutionError": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultResolverExecutionError",
    ),
    "CompletionResultResolverRequest": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultResolverRequest",
    ),
    "CompletionResultResolverUnavailable": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultResolverUnavailable",
    ),
    "CompletionResultUnavailable": (
        "cayu.verification.completion_result_resolvers",
        "CompletionResultUnavailable",
    ),
    "CompletionVerifierExecutionError": (
        "cayu.verification.completion_verifiers",
        "CompletionVerifierExecutionError",
    ),
    "CompletionVerifierExecutionRequest": (
        "cayu.verification.completion_verifiers",
        "CompletionVerifierExecutionRequest",
    ),
    "CompletionVerifierRequest": (
        "cayu.verification.completion_verifiers",
        "CompletionVerifierRequest",
    ),
    "CompletionVerifierUnavailable": (
        "cayu.verification.completion_verifiers",
        "CompletionVerifierUnavailable",
    ),
    "DeterministicCompletionVerifier": (
        "cayu.verification.completion_verifiers",
        "DeterministicCompletionVerifier",
    ),
    "VerifiedTaskHandler": ("cayu.verification.verified_task_worker", "VerifiedTaskHandler"),
    "VerifiedTaskHandlerReport": (
        "cayu.verification.verified_task_worker",
        "VerifiedTaskHandlerReport",
    ),
    "VerifiedTaskPreparationContext": (
        "cayu.verification.verified_task_worker",
        "VerifiedTaskPreparationContext",
    ),
    "VerifiedTaskProposalContext": (
        "cayu.verification.verified_task_worker",
        "VerifiedTaskProposalContext",
    ),
    "VerifiedTaskWorker": ("cayu.verification.verified_task_worker", "VerifiedTaskWorker"),
    "VerifiedTaskWorkerDraining": (
        "cayu.verification.verified_task_worker",
        "VerifiedTaskWorkerDraining",
    ),
    "CompletionVerifierUsageSummary": (
        "cayu.verification.provider_completion_verifiers",
        "CompletionVerifierUsageSummary",
    ),
    "ProviderCompletionVerifier": (
        "cayu.verification.provider_completion_verifiers",
        "ProviderCompletionVerifier",
    ),
    "ProviderCompletionVerifierBudgetExhausted": (
        "cayu.verification.provider_completion_verifiers",
        "ProviderCompletionVerifierBudgetExhausted",
    ),
    "ProviderCompletionVerifierDecodingError": (
        "cayu.verification.provider_completion_verifiers",
        "ProviderCompletionVerifierDecodingError",
    ),
    "ProviderCompletionVerifierDispatchError": (
        "cayu.verification.provider_completion_verifiers",
        "ProviderCompletionVerifierDispatchError",
    ),
    "ProviderCompletionVerifierTarget": (
        "cayu.verification.provider_completion_verifiers",
        "ProviderCompletionVerifierTarget",
    ),
    "summarize_completion_verifier_dispatches": (
        "cayu.verification.provider_completion_verifiers",
        "summarize_completion_verifier_dispatches",
    ),
}

PUBLIC_NAMES = [
    "COMPLETION_RESULT_RESOLUTION_MAX_SECONDS",
    "CompletionResultResolutionRequest",
    "CompletionResultResolver",
    "CompletionResultResolverExecutionError",
    "CompletionResultResolverRequest",
    "CompletionResultResolverUnavailable",
    "CompletionResultUnavailable",
    "CompletionVerifierExecutionError",
    "CompletionVerifierExecutionRequest",
    "CompletionVerifierRequest",
    "CompletionVerifierUnavailable",
    "DeterministicCompletionVerifier",
    "VerifiedTaskHandler",
    "VerifiedTaskHandlerReport",
    "VerifiedTaskPreparationContext",
    "VerifiedTaskProposalContext",
    "VerifiedTaskWorker",
    "VerifiedTaskWorkerDraining",
    "CompletionVerifierUsageSummary",
    "ProviderCompletionVerifier",
    "ProviderCompletionVerifierBudgetExhausted",
    "ProviderCompletionVerifierDecodingError",
    "ProviderCompletionVerifierDispatchError",
    "ProviderCompletionVerifierTarget",
    "summarize_completion_verifier_dispatches",
]
