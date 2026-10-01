"""Static declarations for the lazy verification API."""

from cayu.verification.completion_result_resolvers import (
    COMPLETION_RESULT_RESOLUTION_MAX_SECONDS as COMPLETION_RESULT_RESOLUTION_MAX_SECONDS,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultResolutionRequest as CompletionResultResolutionRequest,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultResolver as CompletionResultResolver,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultResolverExecutionError as CompletionResultResolverExecutionError,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultResolverRequest as CompletionResultResolverRequest,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultResolverUnavailable as CompletionResultResolverUnavailable,
)
from cayu.verification.completion_result_resolvers import (
    CompletionResultUnavailable as CompletionResultUnavailable,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError as CompletionVerifierExecutionError,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionRequest as CompletionVerifierExecutionRequest,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierRequest as CompletionVerifierRequest,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierUnavailable as CompletionVerifierUnavailable,
)
from cayu.verification.completion_verifiers import (
    DeterministicCompletionVerifier as DeterministicCompletionVerifier,
)
from cayu.verification.verified_task_worker import VerifiedTaskHandler as VerifiedTaskHandler
from cayu.verification.verified_task_worker import (
    VerifiedTaskHandlerReport as VerifiedTaskHandlerReport,
)
from cayu.verification.verified_task_worker import (
    VerifiedTaskPreparationContext as VerifiedTaskPreparationContext,
)
from cayu.verification.verified_task_worker import (
    VerifiedTaskProposalContext as VerifiedTaskProposalContext,
)
from cayu.verification.verified_task_worker import VerifiedTaskWorker as VerifiedTaskWorker
from cayu.verification.verified_task_worker import (
    VerifiedTaskWorkerDraining as VerifiedTaskWorkerDraining,
)
