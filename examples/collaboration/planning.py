"""Explicit, non-dispatching host planning with registered deterministic policy.

Register the returned policy in RequestRegistration.planning_policies before
constructing the application. The host supplies an authenticated mandate and a
complete RequestPlanningRequest; neither a policy nor a request is authority.
See docs/collaboration-requests.md for exact identity and recovery requirements.
"""

from cayu import (
    CayuApp,
    ConfiguredRequestPlanningPolicy,
    ForkRecipientPreparation,
    FreshRecipientPreparation,
    MandateAccessContext,
    RequestPlanningDecline,
    RequestPlanningFork,
    RequestPlanningFresh,
    RequestPlanningLimits,
    RequestPlanningRecord,
    RequestPlanningRequest,
    RequestPlanningResource,
)
from cayu.collaboration._contracts import ObjectRef


def decline_policy(
    reference: ObjectRef, limits: RequestPlanningLimits
) -> ConfiguredRequestPlanningPolicy:
    """An immutable total policy; construction neither authorizes nor publishes."""
    return ConfiguredRequestPlanningPolicy(
        reference=reference,
        limits=limits,
        rules=(),
        default=RequestPlanningDecline(reason="unsupported_request"),
    )


def fresh_policy(
    reference: ObjectRef,
    limits: RequestPlanningLimits,
    preparation: FreshRecipientPreparation,
    *,
    resources: tuple[RequestPlanningResource, ...] = (),
) -> ConfiguredRequestPlanningPolicy:
    """Freeze a proposal from app.prepare_recipient_creation, not permission.

    Register this immutable policy before making its exact planning request.
    The driver must preserve the native creation key and full proposal across
    lost acknowledgement; do not rerun preflight with different defaults during
    recovery. plan_once hands off to the real native creation/admission owners.
    """
    return ConfiguredRequestPlanningPolicy(
        reference=reference,
        limits=limits,
        rules=(),
        default=RequestPlanningFresh(preparation=preparation, resources=resources),
    )


def fork_policy(
    reference: ObjectRef,
    limits: RequestPlanningLimits,
    preparation: ForkRecipientPreparation,
    *,
    resources: tuple[RequestPlanningResource, ...] = (),
) -> ConfiguredRequestPlanningPolicy:
    """Freeze app.prepare_recipient_fork data; the planner owns selection and cleanup."""
    return ConfiguredRequestPlanningPolicy(
        reference=reference,
        limits=limits,
        rules=(),
        default=RequestPlanningFork(preparation=preparation, resources=resources),
    )


async def plan_once(
    app: CayuApp, request: RequestPlanningRequest, *, context: MandateAccessContext
) -> RequestPlanningRecord:
    """One explicit planning operation, not a worker or provider retry loop.

    Preserve the complete request before calling. If observation is interrupted
    or its acknowledgement is lost, call app.reconcile_collaboration_plan with
    that same request and current context. Missing evidence is not permission to
    mint a new key. A deferred result needs a separately authorized, explicit
    successor when its prerequisite becomes eligible.
    """
    return await app.plan_collaboration_request(request, context=context)
