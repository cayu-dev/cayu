"""Provider-neutral, application-owned activation governance.

This module decides whether an exact generated knowledge candidate may become
active.  It never schedules work and never persists data; the knowledge store
atomically records validated authority beside the resulting revision.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from threading import Lock
from typing import Protocol

from cayu.storage.memory import (
    KnowledgeActivationAuthority,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationRequest,
    KnowledgeActivationSource,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    copy_knowledge_activation_decision,
    copy_knowledge_activation_request,
)

REVIEWED_ROUTING_POLICY_IDENTITY = "cayu.reviewed-routing"
REVIEWED_ROUTING_POLICY_VERSION = "1"
_MAX_RETAINED_ACTIVATION_POLICY_TASKS = 256
_RETAINED_ACTIVATION_POLICY_TASKS: set[asyncio.Task[object]] = set()
_RETAINED_ACTIVATION_POLICY_TASKS_LOCK = Lock()


class KnowledgeActivationPolicy(Protocol):
    """Application authority for one copied, bounded activation request."""

    async def decide_activation(
        self,
        request: KnowledgeActivationRequest,
    ) -> KnowledgeActivationDecision: ...


class KnowledgeActivationPolicyError(RuntimeError):
    """A configured activation policy failed closed before publication."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("Knowledge activation policy failed closed.")


def _observe_activation_policy_task(task: asyncio.Task[object]) -> None:
    """Release one settled policy task and consume detached diagnostics."""

    with _RETAINED_ACTIVATION_POLICY_TASKS_LOCK:
        _RETAINED_ACTIVATION_POLICY_TASKS.discard(task)
    if task.cancelled():
        return
    with suppress(BaseException):
        task.exception()


def _start_activation_policy_task(
    decide: Callable[[KnowledgeActivationRequest], Awaitable[object]],
    request: KnowledgeActivationRequest,
) -> asyncio.Task[object] | None:
    """Start one policy call only when bounded retained capacity is available."""

    async def invoke() -> object:
        return await decide(request)

    with _RETAINED_ACTIVATION_POLICY_TASKS_LOCK:
        if len(_RETAINED_ACTIVATION_POLICY_TASKS) >= _MAX_RETAINED_ACTIVATION_POLICY_TASKS:
            return None
        task = asyncio.create_task(invoke(), name="cayu-knowledge-activation-policy")
        _RETAINED_ACTIVATION_POLICY_TASKS.add(task)
    task.add_done_callback(_observe_activation_policy_task)
    return task


async def _invoke_activation_policy(
    decide: Callable[[KnowledgeActivationRequest], Awaitable[object]],
    request: KnowledgeActivationRequest,
    *,
    timeout_seconds: float,
) -> object:
    """Invoke an extension policy without surrendering timeout or cancellation ownership."""

    # Authenticate pending caller cancellation before the extension is dispatched.
    await asyncio.sleep(0)
    task = _start_activation_policy_task(decide, request)
    if task is None:
        raise KnowledgeActivationPolicyError("activation_policy_capacity_exhausted")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    try:
        await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        task.cancel("Knowledge activation policy was cancelled by its caller.")
        raise

    if not task.done() or loop.time() >= deadline:
        if not task.done():
            task.cancel("Knowledge activation policy exceeded its deadline.")
        raise KnowledgeActivationPolicyError("activation_policy_timed_out")

    # Completion and caller cancellation can become ready in the same event-loop
    # turn. Keep the caller signal authoritative before accepting policy output.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        task.cancel("Knowledge activation policy was cancelled by its caller.")
        raise
    if loop.time() >= deadline:
        raise KnowledgeActivationPolicyError("activation_policy_timed_out")
    if task.cancelled():
        raise KnowledgeActivationPolicyError("activation_policy_failed")
    try:
        return task.result()
    except asyncio.CancelledError:
        raise KnowledgeActivationPolicyError("activation_policy_failed") from None
    except Exception:
        raise KnowledgeActivationPolicyError("activation_policy_failed") from None


async def decide_knowledge_activation(
    request: KnowledgeActivationRequest,
    *,
    config: KnowledgeGovernanceConfig,
    policy: KnowledgeActivationPolicy | None = None,
) -> KnowledgeActivationAuthority:
    """Return validated activation authority or fail closed without persistence."""

    if type(request) is not KnowledgeActivationRequest:
        raise TypeError("request must be a KnowledgeActivationRequest.")
    if type(config) is not KnowledgeGovernanceConfig:
        raise TypeError("config must be a KnowledgeGovernanceConfig.")
    copied_request = copy_knowledge_activation_request(request)
    copied_config = KnowledgeGovernanceConfig.model_validate(config.model_dump(mode="python"))
    if copied_request.mode is not copied_config.mode:
        raise KnowledgeActivationPolicyError("governance_mode_mismatch")
    if copied_config.mode is KnowledgeGovernanceMode.REVIEWED:
        if policy is not None:
            raise KnowledgeActivationPolicyError("reviewed_mode_policy_configured")
        if copied_request.source is KnowledgeActivationSource.REVIEW_APPROVAL:
            raise KnowledgeActivationPolicyError("review_approval_requires_reviewer")
        return KnowledgeActivationAuthority(
            request=copied_request,
            decision=KnowledgeActivationDecision(
                request_sha256=copied_request.fingerprint,
                disposition=KnowledgeActivationDisposition.ROUTE_TO_REVIEW,
                policy_identity=REVIEWED_ROUTING_POLICY_IDENTITY,
                policy_version=REVIEWED_ROUTING_POLICY_VERSION,
                code="review_required",
            ),
        )

    decide = getattr(policy, "decide_activation", None)
    if not callable(decide):
        raise KnowledgeActivationPolicyError("activation_policy_missing")
    raw_decision = await _invoke_activation_policy(
        decide,
        copy_knowledge_activation_request(copied_request),
        timeout_seconds=copied_config.policy_timeout_seconds,
    )
    if type(raw_decision) is not KnowledgeActivationDecision:
        raise KnowledgeActivationPolicyError("activation_policy_output_invalid")
    try:
        decision = copy_knowledge_activation_decision(raw_decision)
        if (
            decision.policy_identity != copied_config.policy_identity
            or decision.policy_version != copied_config.policy_version
        ):
            raise ValueError("Activation policy identity does not match host configuration.")
        return KnowledgeActivationAuthority(request=copied_request, decision=decision)
    except (TypeError, ValueError):
        raise KnowledgeActivationPolicyError("activation_policy_output_invalid") from None


def reviewed_approval_authority(
    request: KnowledgeActivationRequest,
    *,
    reviewer_identity: str,
    reviewer_version: str,
    code: str = "approved",
    annotations: dict[str, object] | None = None,
) -> KnowledgeActivationAuthority:
    """Bind an explicit privileged reviewer to an exact pending revision."""

    if type(request) is not KnowledgeActivationRequest:
        raise TypeError("request must be a KnowledgeActivationRequest.")
    copied_request = copy_knowledge_activation_request(request)
    if (
        copied_request.mode is not KnowledgeGovernanceMode.REVIEWED
        or copied_request.source is not KnowledgeActivationSource.REVIEW_APPROVAL
    ):
        raise ValueError("Reviewed approval requires a review-approval request.")
    return KnowledgeActivationAuthority(
        request=copied_request,
        decision=KnowledgeActivationDecision(
            request_sha256=copied_request.fingerprint,
            disposition=KnowledgeActivationDisposition.ACTIVATE,
            policy_identity=reviewer_identity,
            policy_version=reviewer_version,
            code=code,
            annotations={} if annotations is None else annotations,
        ),
    )


__all__ = [
    "KnowledgeActivationPolicy",
    "KnowledgeActivationPolicyError",
    "decide_knowledge_activation",
    "reviewed_approval_authority",
]
