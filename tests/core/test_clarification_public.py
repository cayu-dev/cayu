"""Public question publication through the real source and request owners."""

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from uuid import uuid4

import httpx
import pytest
from tests.core.test_assistant_text_peer_export import Policy, TextProjector
from tests.core.test_budget_binding import _binding
from tests.core.test_clarification_contracts import policy as finite_policy
from tests.core.test_collaboration_request_exports import _PublicExportReader
from tests.core.test_collaboration_request_foundation import RequestResolver, setup
from tests.core.test_participant_identity import CONTEXT, app, registration
from tests.core.test_participant_lifecycle import change
from tests.core.test_peer_content import QualificationPeerExposurePolicy, _delivery_request
from tests.core.test_session_creation_fence import _collaboration_factory, _store_factory

from cayu.agents import AgentSpec
from cayu.budgets import BudgetLimit, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.collaboration._clarification_commands import (
    ClarificationCloseCommand,
    ClarificationOpenCommand,
)
from cayu.collaboration._clarification_deliveries import ClarificationDeliveryIntent
from cayu.collaboration._clarification_service_api import ClarificationServiceRequest
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration._request_store import operation_key
from cayu.collaboration._session_export_participant import SessionExportRequestReceivingOwner
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.clarifications import ClarificationQuestion
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAccessContext,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
)
from cayu.collaboration.mandates import MandateDenied, MandateResolver, ResourceSelector
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentExposureRequest, PeerContentPayload
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider
from cayu.sessions.base import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.invocation import InvocationOriginClaim
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
@pytest.mark.parametrize("terminal_wait", ["cancelled", "expired"])
async def test_terminal_original_wait_after_service(
    backend, tmp_path, request, monkeypatch, side_session, terminal_wait
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=side_session,
        terminal_wait=terminal_wait,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("worker_offset", [-3600, 3600])
async def test_public_service_clock_skew_keeps_dispatch_bounded(
    backend, tmp_path, request, monkeypatch, worker_offset
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        service_clock_offset=worker_offset,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("timing", ["before", "during_return", "foreign_settlement"])
async def test_public_side_service_final_latch_arbitration(
    backend, tmp_path, request, monkeypatch, timing
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=True,
        finish_request=True,
        final_latch_timing=timing,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("timing", ["before", "during_return", "foreign_settlement"])
async def test_public_final_latch_arbitrates_temporary_service(
    backend, tmp_path, request, monkeypatch, timing
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        finish_request=True,
        final_latch_timing=timing,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("registered_first", [False, True])
async def test_public_service_permit_orders_disablement(
    backend, tmp_path, request, monkeypatch, registered_first
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        participant_race=registered_first,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", [-1, 0, 1])
async def test_public_service_shares_prior_root_budget(
    backend, tmp_path, request, monkeypatch, boundary
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        budget_boundary=boundary,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", [-1, 0, 1])
async def test_public_service_shares_ancestor_budget(
    backend, tmp_path, request, monkeypatch, boundary
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        budget_boundary=boundary,
        ancestor_budget=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_nested_service_returns_before_parent(backend, tmp_path, request, monkeypatch):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        nested_service=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
async def test_public_second_question_preserves_original_wait(
    backend, tmp_path, request, monkeypatch, side_session
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=side_session,
        multiple_questions=True,
        finish_request=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_public_question_expiry_replay_in_fresh_process(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        question_recovery="fresh_process",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancellation", [False, True])
async def test_public_due_question_expiry_preserves_pending_delivery(
    backend, tmp_path, request, monkeypatch, cancellation
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        question_recovery="cancel" if cancellation else "lost_ack",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lost_native_ack", (False, True))
async def test_public_terminal_question_history_prunes_across_reconstruction(
    backend, tmp_path, request, monkeypatch, lost_native_ack
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=False,
        continue_questioner=True,
        finish_request=True,
        prune_history="lost_native_ack" if lost_native_ack else True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_delivery_discovery_recovers_lost_append_ack(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        delivery_recovery=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_public_delivery_discovery_in_fresh_process(backend, tmp_path, request, monkeypatch):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        delivery_recovery="fresh_process",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_delivery_recovery_requires_positive_exclusion(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        delivery_recovery="excluded",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_delivery_recovery_preserves_cancellation(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=False,
        public_reply=False,
        post_admission=False,
        side_session=False,
        delivery_recovery="cancelled",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
@pytest.mark.parametrize("phase", ["after_preparation", "before_register"])
async def test_public_prepared_service_exclusion_race(
    backend, tmp_path, request, monkeypatch, side_session, phase
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=side_session,
        prepared_recovery=phase,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_service_observation_expiry_reconciles_without_relaunch(
    backend, tmp_path, request, monkeypatch
):
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner

    observe = SessionContinuationOwner._observe
    committed_admissions = []

    async def delayed_admission_ack(owner, operation, *, key, expected, **kwargs):
        if key[-1] != "admit":
            return await observe(owner, operation, key=key, expected=expected, **kwargs)
        normal_timeout = owner.owners.observation_timeout

        async def dispatch():
            # The observer has captured its timeout. Restore the shared setting
            # before native admission so unrelated observations remain unchanged.
            owner.owners.observation_timeout = normal_timeout
            result = await operation()
            committed_admissions.append(key)
            await asyncio.sleep(0.05)
            return result

        owner.owners.observation_timeout = 0.001
        try:
            return await observe(owner, dispatch, key=key, expected=expected, **kwargs)
        finally:
            owner.owners.observation_timeout = normal_timeout

    monkeypatch.setattr(SessionContinuationOwner, "_observe", delayed_admission_ack)
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=True,
        service_observation_loss=True,
    )
    assert len(committed_admissions) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_service_cleanup_after_lost_settlement_ack(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        maintenance_recovery=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_service_cleanup_retains_repeatedly_cancelled_observation(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        maintenance_recovery="cancel_cleanup",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_public_service_cleanup_without_host_request_in_fresh_process(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        maintenance_recovery="process_cleanup",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_service_rejects_conflicting_native_wait_index(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=False,
        side_session=False,
        verify_index_integrity=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_service_requires_native_append_evidence(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        missing_append=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_reply_delivery_and_questioner_continuation(
    backend, tmp_path, request, monkeypatch
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=False,
        continue_questioner=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
async def test_public_clarification_original_final_continuation(
    backend, tmp_path, request, monkeypatch, side_session
):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=side_session,
        continue_questioner=True,
        finish_request=True,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_public_clarification_single_slot_driver(backend, tmp_path, request, monkeypatch):
    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=False,
        continue_questioner=True,
        finish_request=True,
        one_slot=True,
    )


class Mandates(MandateResolver):
    """Explicit exact actor mandates with one non-reentrant revocation guard."""

    def __init__(self, *actors):
        self.actors = actors
        self.lock = asyncio.Lock()

    @property
    def ref(self):
        return self.actors[0].ref

    @asynccontextmanager
    async def acquire(self, context):
        async with self.lock:
            actor = next((actor for actor in self.actors if actor.context == context), None)
            if actor is None:
                raise MandateDenied()
            async with actor.acquire(context) as resolution:
                yield resolution


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "temporary_service,public_reply,post_admission,side_session",
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ),
)
async def test_public_question_uses_real_assistant_export(
    backend,
    tmp_path,
    request,
    monkeypatch,
    temporary_service,
    public_reply,
    post_admission,
    side_session,
    missing_append=False,
    continue_questioner=False,
    finish_request=False,
    one_slot=False,
    maintenance_recovery=False,
    prepared_recovery=None,
    delivery_recovery=False,
    prune_history=False,
    question_recovery=False,
    multiple_questions=False,
    nested_service=False,
    budget_boundary=None,
    ancestor_budget=False,
    participant_race=None,
    final_latch_timing=None,
    human_paused=False,
    reply_cancel_before=None,
    busy_target=False,
    service_clock_offset=None,
    terminal_wait=None,
    service_observation_loss=False,
    planning_driver=None,
    planning_journey=None,
    journey_ttl_ms=300_000,
    service_driver=None,
    maintenance_driver=None,
    question_driver=None,
    delivery_prepared_driver=None,
    verify_index_integrity=False,
):
    if prune_history == "lost_native_ack":
        from cayu.runtime._session_continuation import ContinuationConflict
        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        original_ack = SessionContinuationOwner._acknowledge_temporary_settlement

        async def retain_unacknowledged(owner, retained):
            # Hold the native write at the post-foreign-commit boundary while
            # the independently owned public namespace maintenance proceeds.
            assert not retained.settlement_acknowledged
            return retained

        monkeypatch.setattr(
            SessionContinuationOwner, "_acknowledge_temporary_settlement", retain_unacknowledged
        )
    store = _store_factory(backend, tmp_path, request)()
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    collaboration = collaboration_factory()
    registered = registration(scope="clarification-" + uuid4().hex)
    values = await setup(collaboration, reg=registered, session_store=store)
    _, initialized, first, second, original, _ = values
    original = original.model_copy(update={"ttl_ms": journey_ttl_ms})
    if finish_request:
        original = original.model_copy(
            update={
                "output_contract": ObjectRef(
                    owner=initialized.owner,
                    kind="projector",
                    object_id="visible-text",
                    incarnation="v1",
                    revision=1,
                ),
                "disclosure_policy": ObjectRef(
                    owner=initialized.owner,
                    kind="export_policy",
                    object_id="policy",
                    incarnation="one",
                    revision=1,
                ),
            }
        )
    actor_a = RequestResolver(original)
    actor_b = RequestResolver(
        original.model_copy(update={"sender": second.reference, "target": first.reference})
    )
    mandates = Mandates(actor_a, actor_b)
    owner = initialized.owner

    class ExportPolicy(Policy):
        def __init__(self):
            super().__init__()
            self.pause_peer = False
            self.peer_entered = asyncio.Event()
            self.peer_release = asyncio.Event()

        @property
        def ref(self):
            return ObjectRef(
                owner=owner, kind="export_policy", object_id="policy", incarnation="one", revision=1
            )

        @asynccontextmanager
        async def acquire_peer_append(self, context, **kwargs):
            if self.pause_peer:
                self.peer_entered.set()
                await self.peer_release.wait()
            async with (
                self.disclosure_lock,
                super().acquire_peer_append(context, **kwargs) as authorization,
            ):
                yield authorization

        @asynccontextmanager
        async def acquire_peer_exposures(self, context, *, items):
            if len(items) == 1:
                async with super().acquire_peer_exposures(context, items=items) as projections:
                    yield projections
                return
            # Retained questions share one non-reentrant revocation guard.
            # Authenticate every source under it, without re-entering the
            # single-occurrence wrapper's lock or weakening the default policy.
            async with self.disclosure_lock, AsyncExitStack() as stack:
                projections = []
                for item in items:
                    await stack.enter_async_context(
                        QualificationPeerExposurePolicy.acquire_peer_append(
                            self,
                            context,
                            request=None,
                            append_key=item.origin.append_key,
                            occurrence=item.occurrence,
                        )
                    )
                    projections.append(
                        await stack.enter_async_context(
                            QualificationPeerExposurePolicy.acquire_peer_exposure(
                                self, context, **item.single_arguments()
                            )
                        )
                    )
                yield tuple(projections)

    export_policy = ExportPolicy()
    projector = TextProjector(export_policy)
    if finish_request:
        projector.identity = projector.identity.model_copy(update={"revision": 1})
    context = SessionExportAccessContext(principal="operator", mandate=actor_b.context)
    reader = _PublicExportReader(owner, context)
    receiver = SessionExportRequestReceivingOwner(audience=owner, reader=reader)
    policy = finite_policy(
        service_timeout_ms=90_000 if service_clock_offset is not None else journey_ttl_ms,
        max_questions=2 if multiple_questions or nested_service else 32,
        max_service_turns=2 if multiple_questions or nested_service else 32,
        max_depth=2 if nested_service else 4,
        reference=ObjectRef(
            owner=owner,
            kind="clarification_policy",
            object_id="finite",
            incarnation="one",
            revision=1,
        ),
    )
    # Two initial runs each settle 15 tokens at $1/million. Temporary service
    # must combine their $0.000030 with its 2048-token/$0.002048 reservation.
    # Every individual dispatch fits even the below-boundary ceiling.
    budget_key = "root-causal" if budget_boundary is None else "root-" + owner.application_scope
    binding = _binding(
        application_scope=owner.application_scope,
        binding_id="binding-1" if budget_boundary is None else "binding-" + owner.application_scope,
        root_budget_id=budget_key,
        # Compaction qualification includes a nonbillable local compactor;
        # one root binding covers it and the priced OpenAI model dispatches.
        provider_name=None if final_latch_timing is not None else "openai",
        model=None if final_latch_timing is not None else "gpt-test",
        limits=(
            BudgetLimit(
                scope="causal",
                key=budget_key,
                max_estimated_cost=Decimal("1")
                if budget_boundary is None
                else Decimal("0.002078") + Decimal(budget_boundary) / Decimal(1_000_000),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="openai",
                            model="gpt-test",
                            match="exact",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
                reservation=BudgetReservation(max_input_tokens=1024, max_output_tokens=1024),
            ),
        ),
    )
    if ancestor_budget:
        assert budget_boundary is not None
        # Keep the root comfortably below its own ceiling. Only the shared
        # ancestor can reject this otherwise affordable service dispatch.
        ancestor_key = "ancestor-" + owner.application_scope
        boundary_limit = binding.limits[0]
        binding = _binding(
            **{
                **binding.model_dump(mode="python"),
                "ancestor_budget_ids": (ancestor_key,),
                "limits": (
                    boundary_limit.model_copy(update={"max_estimated_cost": Decimal("1")}),
                    boundary_limit.model_copy(update={"key": ancestor_key}),
                ),
            }
        )

    class BudgetReceiver:
        def __init__(self):
            self.pause = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve_budget_binding(self, *, request):
            if self.pause and request.get("kind") == "clarification":
                self.entered.set()
                await self.release.wait()
            return binding

    budget_receiver = BudgetReceiver()
    budget_ledgers = []
    payloads = []
    nested_stack_probe = None
    busy_entered = asyncio.Event()
    busy_release = asyncio.Event()
    service_cancelled = asyncio.Event()
    question_call = 2 if one_slot else 1

    async def transport_response(request):
        payloads.append(request.content)
        if service_clock_offset is not None and len(payloads) == 3:
            try:
                await asyncio.Future()
            finally:
                service_cancelled.set()
        if busy_target and len(payloads) == 4:
            busy_entered.set()
            await busy_release.wait()
        if human_paused and len(payloads) == 3:
            return httpx.Response(
                200,
                json={
                    "id": "human-input-response",
                    "object": "response",
                    "model": "gpt-test",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "human-input-call",
                            "call_id": "ask-human",
                            "name": "ask_user",
                            "arguments": '{"question":"Which environment?"}',
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
                request=request,
            )
        if nested_service and len(payloads) == 4:
            assert nested_stack_probe is not None
            await nested_stack_probe()
        return httpx.Response(
            200,
            json={
                "id": "question-response",
                "object": "response",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "thinking",
                        "encrypted_content": "private-state"
                        if len(payloads) == question_call
                        else "recipient-state",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "private-thinking"
                                if len(payloads) == question_call
                                else "recipient-thinking",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "id": "question",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Which API version?"
                                if len(payloads) == question_call
                                else "Use API v2.",
                                "annotations": [],
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport_response)) as client:
        transport = HttpxOpenAITransport()
        transport._client._client = client

        def application_for(
            collaboration_store,
            sessions,
            *,
            planning_policies=(),
            prepared_admission=None,
            resource_owners=(),
        ):
            ledger = None
            if budget_boundary is not None and backend != "memory":
                if backend == "sqlite":
                    from cayu.storage.budget_ledger import SQLiteBudgetLedger

                    ledger = SQLiteBudgetLedger(tmp_path / "clarification-budget.sqlite")
                else:
                    from cayu.storage.migrations import SchemaMode
                    from cayu.storage.postgres import PostgresBudgetLedger

                    ledger = PostgresBudgetLedger(
                        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
                    )
                budget_ledgers.append(ledger)
            return app(
                collaboration_store,
                registered,
                session_store=sessions,
                collaboration_requests=RequestRegistration(
                    mandates=mandates,
                    max_ttl_ms=journey_ttl_ms,
                    receiving_owner=receiver,
                    clarification_policies=(policy,),
                    planning_policies=planning_policies,
                    prepared_admission=prepared_admission,
                    resource_owners=resource_owners,
                ),
                session_exports=SessionExportRegistration(
                    owner=owner,
                    policy=export_policy,
                    projectors=(projector,),
                    limits=ExportLimits(
                        max_exports=8,
                        max_pending=4,
                        # Multi-turn journeys retain multiple exports with
                        # distinct authority and reserved settlement capacity.
                        max_retained_bytes=262144
                        if finish_request or multiple_questions or nested_service
                        else 65536,
                    ),
                    mandates=mandates,
                ),
                budget_binding_receiver=budget_receiver,
                budget_ledger=ledger,
                enable_common_root_budget_binding=True,
            )

        current = application_for(collaboration, store)
        reader.app = current
        current.register_provider(
            OpenAIProvider(api_key="test-key", streaming=False, transport=transport), default=True
        )
        if human_paused:
            from cayu.tools.user_input import UserInputTool

            current.register_agent(
                AgentSpec(name="reviewer", model="gpt-test"), tools=[UserInputTool()]
            )
        elif final_latch_timing is not None:
            from tests.core.test_explicit_session_compaction import RecordingCompactor

            from cayu.context import CheckpointCompactionContextPolicy

            current.register_agent(
                AgentSpec(name="reviewer", model="gpt-test"),
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=RecordingCompactor(), max_user_turns=1, compact_after_messages=100
                ),
            )
        else:
            current.register_agent(AgentSpec(name="reviewer", model="gpt-test"))
        if one_slot:
            from tests.core._clarification_public_setup import SingleSlotDriver

            driver = SingleSlotDriver()
            monkeypatch.setattr(current, "_run_private", driver.wrap(current._run_private))
            monkeypatch.setattr(current, "_resume_private", driver.wrap(current._resume_private))
        try:
            await current.initialize_collaboration()
            accepted = await current.accept_collaboration_request(original, context=actor_a.context)
            from tests.core._clarification_public_setup import create_target

            if one_slot:
                target_creation, target, wait, parked = await create_target(
                    current,
                    accepted,
                    initialized,
                    first.reference,
                    actor_a.context,
                    park=True,
                )
                assert driver.active == 0 and driver.order == [target.id]
            before_admission = await current.inspect_collaboration_request(
                accepted.expected,
                context=actor_a.context,
            )
            admission = RequestAdmissionCommand(
                operation=initialized.operation("clarify"),
                expected=accepted.expected,
                expected_revision=before_admission.revision,
                expected_input_revision=0,
                expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
                generation=1,
                decision="clarify",
                evidence=(),
                initiator=_initiator(actor_b.context),
            )
            if planning_driver is None and planning_journey is None:
                await current.admit_collaboration_request(admission, context=actor_b.context)
            creation = ParticipantSessionCreationRequest(
                creation_key="question-source-" + owner.application_scope,
                request=RunRequest(
                    agent_name="reviewer",
                    messages=[Message.text("user", "Ask the API version")],
                    invocation_origin=InvocationOriginClaim(subject="operator"),
                ),
            )
            source, _ = await current.create_participant_session(
                creation, participant=second.reference, context=CONTEXT
            )
            execution = ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": source.id}),
                session_instance_id=source.instance_id,
                execution_key="question-run",
            )
            events = [
                event
                async for event in current.execute_participant_session(
                    execution, participant=second.reference, context=CONTEXT
                )
            ]
            assert any(event.type == EventType.SESSION_COMPLETED for event in events)
            rows = (await store.load_transcript_window(source.id, start_index=0, limit=16)).records
            row = next(row for row in rows if row.message.role == "assistant")
            audience = OwnerRef(
                application_scope=owner.application_scope,
                owner_id=first.reference.participant_id,
                incarnation=first.reference.incarnation,
            )
            selected_resource = ResourceSelector(
                resource=ObjectRef(
                    owner=owner,
                    kind="session_transcript_row",
                    object_id=source.id,
                    incarnation=source.instance_id,
                    revision=row.index + 1,
                )
            )
            resolution = actor_b.resolution
            actions = ("consult", "readback", "administer", "publish", "source", "expose")
            if planning_driver is not None or planning_journey is not None:
                actions = (*actions, "prepare")
            actor_b.resolution = resolution.model_copy(
                update={
                    "principal": resolution.principal.model_copy(
                        update={"actions": actions, "audiences": (owner, audience)}
                    ),
                    "chain": resolution.chain.model_copy(
                        update={
                            "entries": (
                                resolution.chain.entries[-1].model_copy(
                                    update={
                                        "actions": actions,
                                        "audiences": (owner, audience),
                                        "resources": (selected_resource,),
                                        "restrictions": resolution.chain.entries[
                                            -1
                                        ].restrictions.model_copy(
                                            update={"channels": ("prompt", "source")}
                                        ),
                                    }
                                ),
                            )
                        }
                    ),
                }
            )
            namespace = await current.initialize_session_exports(source.id, context=context)
            export = SessionExportRequest(
                ref=SessionExportRef(
                    session_id=source.id,
                    session_instance_id=source.instance_id,
                    operation=OperationRef(
                        application_scope=owner.application_scope,
                        namespace_incarnation=namespace.namespace_incarnation,
                        generation=1,
                        caller_key="question",
                    ),
                ),
                source_indices=(row.index,),
                source_selection="assistant_visible_text_v1",
                audience=audience,
                projector=projector.ref,
                policy=export_policy.ref,
            )
            exported = await current.export_session(export, context=context)
            projected = await current.inspect_clarification_source(
                export, sender=second.reference, audience=first.reference, context=context
            )
            snapshot = await current.inspect_collaboration_request(
                accepted.expected, context=actor_a.context
            )
            question = ClarificationQuestion(
                operation=initialized.operation("question"),
                request=accepted.expected.intent.selection.reference,
                request_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
                admission=admission.operation,
                initiator=_initiator(actor_b.context),
                responder=first.reference,
                receiver=receiver.ref,
                generation=1,
                input_revision=0,
                input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
                lineage=initialized.operation("lineage"),
                parent_question=None,
                depth=1,
                source=projected,
                policy=policy,
                budget_binding=ObjectRef(
                    owner=owner,
                    kind="budget_binding",
                    object_id=binding.binding_id,
                    incarnation=binding.authority_digest,
                    revision=1,
                ),
                budget_authority_sha256=binding.authority_digest,
                deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
            )
            command = ClarificationOpenCommand(
                operation=question.operation,
                expected=accepted.expected,
                expected_revision=snapshot.revision
                + (planning_driver is not None or planning_journey is not None),
                question=question,
            )
            if planning_driver is not None:
                return await planning_driver(
                    application_for=application_for,
                    collaboration=collaboration,
                    sessions=store,
                    initialized=initialized,
                    command=command,
                    source=export,
                    context=context,
                    payloads=payloads,
                    reopen_collaboration=collaboration_factory,
                    backend=backend,
                    export_policy=export_policy,
                )
            if not one_slot:
                target_creation, target, wait, parked = await create_target(
                    current,
                    accepted,
                    initialized,
                    first.reference,
                    actor_a.context,
                    park=temporary_service,
                )
            if temporary_service:
                # Wait registration retains source responsibility. Compare the
                # subsequent question rejection against that new durable state.
                snapshot = await current.inspect_collaboration_request(
                    accepted.expected, context=actor_a.context
                )
                command = command.model_copy(
                    update={"expected_revision": snapshot.revision + (planning_journey is not None)}
                )

            original_target_creation = target_creation
            if side_session:
                target_creation = ParticipantSessionCreationRequest(
                    creation_key="existing-side-target-" + owner.application_scope,
                    request=target_creation.request,
                )
                target, _ = await current.create_participant_session(
                    target_creation, participant=first.reference, context=CONTEXT
                )
                side_events = [
                    event
                    async for event in current.execute_participant_session(
                        ParticipantSessionExecutionRequest(
                            request=target_creation.request.model_copy(
                                update={"session_id": target.id}
                            ),
                            session_instance_id=target.instance_id,
                            execution_key="side-initial",
                        ),
                        participant=first.reference,
                        context=CONTEXT,
                    )
                ]
                expected_side = (
                    EventType.SESSION_INTERRUPTED if human_paused else EventType.SESSION_COMPLETED
                )
                assert any(event.type == expected_side for event in side_events)

            class FacadeOnlyReceiver:
                async def resolve_budget_binding(self, *, request):
                    pytest.fail(
                        "A changed facade field replaced configured runtime budget authority"
                    )

            current.budget_binding_receiver = FacadeOnlyReceiver()

            async def reject_ledger_registration(**kwargs):
                pytest.fail("Question publication mutated the dispatch budget ledger")

            monkeypatch.setattr(
                current.budget_ledger, "register_budget_binding", reject_ledger_registration
            )
            assert isinstance(
                await current.lookup_clarification(command, context=actor_a.context), ExactNotFound
            )
            for altered in (
                question.model_copy(update={"budget_authority_sha256": "f" * 64}),
                question.model_copy(update={"input_sha256": "f" * 64}),
                question.model_copy(
                    update={"policy": policy.model_copy(update={"max_questions": 1})}
                ),
                question.model_copy(
                    update={"source": projected.model_copy(update={"content_sha256": "f" * 64})}
                ),
            ):
                with pytest.raises(CollaborationConflict):
                    await current.open_clarification(
                        command.model_copy(update={"question": altered}),
                        source=export,
                        context=context,
                    )
                assert (
                    await current.inspect_collaboration_request(
                        accepted.expected, context=actor_a.context
                    )
                    == snapshot
                )
            if question_recovery or service_clock_offset is not None:
                async with collaboration._transaction(owner.application_scope, write=False) as tx:
                    deadline = await tx.now_ms() + (
                        90_000 if service_clock_offset is not None else 15_000
                    )
                question = question.model_copy(update={"deadline_at_ms": deadline})
                command = command.model_copy(update={"question": question})
            if planning_journey is None:
                opened = await current.open_clarification(command, source=export, context=context)
            else:
                opened, after_planned_reply = await planning_journey(
                    application_for=application_for,
                    collaboration=collaboration,
                    sessions=store,
                    initialized=initialized,
                    command=command,
                    source=export,
                    context=context,
                    payloads=payloads,
                    reopen_collaboration=collaboration_factory,
                    backend=backend,
                )
            assert opened.command == command
            payload = PeerContentPayload(
                text="Which API version?",
                content_sha256=sha256(
                    json.dumps(
                        {"text": "Which API version?", "artifact_commitments": []},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            )
            export_policy.register_export(
                exported,
                payload_sha256=payload.content_sha256,
                consumer_id=first.reference.participant_id,
            )
            peer = _delivery_request(
                suffix="question-" + owner.application_scope,
                source=source,
                target=target,
                sender=second.reference,
                consumer=first.reference,
                source_export_receipt_id=exported.event_id,
                payload=payload,
            )
            if temporary_service:
                current_target = await store.load(target.id)
                peer = peer.model_copy(
                    update={
                        "attempt_key": peer.attempt_key.model_copy(
                            update={
                                "target_run_epoch": current_target.run_epoch + post_admission,
                                "target_transcript_cursor": len(
                                    await store.load_transcript(target.id)
                                )
                                + post_admission,
                            }
                        )
                    }
                )
            key = peer.append_key.model_copy(
                update={
                    "collaboration_namespace": initialized.namespace_incarnation,
                    "collaboration_generation": question.operation.generation,
                }
            )
            peer = peer.model_copy(
                update={
                    "append_key": key,
                    "attempt_key": peer.attempt_key.model_copy(
                        update={
                            "append_key": key,
                            "deadline_at_ms": question.deadline_at_ms,
                        }
                    ),
                }
            )
            export_policy.allowed_receipts.add(peer.occurrence.producer_receipt_id)
            delivery = ClarificationDeliveryIntent(
                operation=initialized.operation("question-delivery"),
                initiator=question.initiator,
                question=question,
                sender=second.reference,
                recipient=first.reference,
                export=export,
                append=peer,
            )
            prepared_delivery = await current.prepare_clarification_delivery(
                delivery, context=context
            )
            assert prepared_delivery.status == "pending"
            if delivery_prepared_driver is not None:
                await delivery_prepared_driver(
                    current,
                    delivery=delivery,
                    context=context,
                    application_for=application_for,
                    collaboration_factory=collaboration_factory,
                )
            if maintenance_driver is not None:
                await maintenance_driver(current, context=CONTEXT)
                assert await store.read_peer_content_attempt(peer) is None
            if question_recovery:
                from tests.core._clarification_question_recovery_flow import expire_due_question
                from tests.core._clarification_recovery_flow import cleanup_in_fresh_process

                async def restart_expiry(expiry):
                    await cleanup_in_fresh_process(
                        initialized,
                        backend=backend,
                        tmp_path=tmp_path,
                        request=request,
                        kind="question",
                        expiry=expiry,
                    )

                count = len(payloads)
                await (question_driver or expire_due_question)(
                    current,
                    collaboration_factory,
                    initialized,
                    command,
                    registered,
                    monkeypatch,
                    cancellation=question_recovery == "cancel",
                    restart=restart_expiry if question_recovery == "fresh_process" else None,
                )
                assert len(payloads) == count
                assert await store.read_peer_content_attempt(peer) is None
                return
            assert (
                await current.prepare_clarification_delivery(delivery, context=context)
                == prepared_delivery
            )
            assert await store.read_peer_content_attempt(peer) is None
            assert len(payloads) == 1 + temporary_service + side_session
            export_policy.revoked = True
            with pytest.raises(CollaborationAccessDenied):
                await current.prepare_clarification_delivery(delivery, context=context)
            assert await store.read_peer_content_attempt(peer) is None
            export_policy.revoked = False
            if delivery_recovery:
                from tests.core._clarification_delivery_recovery_flow import recover_delivery

                before_recovery = len(payloads)
                assert (
                    await recover_delivery(current, expected_status="pending")
                ) == prepared_delivery
                assert len(payloads) == before_recovery
                if delivery_recovery == "excluded":
                    export_policy.revoked = True
                    other_store = (
                        store
                        if backend == "memory"
                        else _store_factory(backend, tmp_path, request)()
                    )
                    other_source = collaboration_factory()
                    other = application_for(other_source, other_store)
                    await other.initialize_collaboration()
                    try:
                        page = await other.list_pending_clarification_deliveries(context=CONTEXT)
                        selector = page.items[0].recovery
                        with pytest.raises(CollaborationAccessDenied):
                            await other.exclude_clarification_delivery(
                                selector,
                                context=CONTEXT.model_copy(update={"principal": "outsider"}),
                            )
                        export_policy.allow_cleanup = False
                        with pytest.raises(CollaborationAccessDenied):
                            await other.exclude_clarification_delivery(selector, context=CONTEXT)
                        assert await other_store.read_peer_content_attempt(peer) is None
                        export_policy.allow_cleanup = True
                        from tests.core._clarification_cleanup_race import exclude_with_lost_ack

                        from cayu.collaboration._clarification_delivery_store import (
                            DELIVERY_SETTLEMENT_BYTES,
                        )

                        async with other_source._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            before_anchor = await other_source._anchor(
                                tx, initialized, SecretRedactor()
                            )
                            before_lineage = await tx.get(
                                "clarification_lineages", operation_key(question.lineage)
                            )

                        await exclude_with_lost_ack(other, selector, peer, monkeypatch)
                        assert (
                            await other_store.read_peer_content_attempt(peer)
                        ).status == "excluded"
                        recovered = await other.reconcile_clarification_delivery(
                            selector, context=CONTEXT
                        )
                        assert recovered.status == "excluded" and recovered.reason == "withdrawn"
                        assert (
                            await other.exclude_clarification_delivery(selector, context=CONTEXT)
                            == recovered
                        )
                        assert not (
                            await other.list_pending_clarification_deliveries(context=CONTEXT)
                        ).items
                        async with other_source._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            after_anchor = await other_source._anchor(
                                tx, initialized, SecretRedactor()
                            )
                            after_lineage = await tx.get(
                                "clarification_lineages", operation_key(question.lineage)
                            )
                        assert (
                            after_anchor.reserved_bytes
                            == before_anchor.reserved_bytes - DELIVERY_SETTLEMENT_BYTES
                        )
                        assert (
                            after_lineage["usage"]["pending"]
                            == before_lineage["usage"]["pending"] - 1
                        )
                        assert export_policy.revoked and len(payloads) == before_recovery
                    finally:
                        if other_source is not collaboration:
                            await other._request_coordinator.close()
                        if other_store is not store:
                            await other_store.close()
                        if other_source is not collaboration:
                            await other_source.close()
                    return
            if not post_admission:
                # Revocation after reservation but before the peer owner's guard
                # prevents append without mistaking the pending handoff for exclusion.
                export_policy.pause_peer = True
                delivery_task = asyncio.create_task(
                    current.deliver_clarification(delivery, context=context)
                )
                try:
                    await asyncio.wait_for(export_policy.peer_entered.wait(), 10)
                    await export_policy.revoke()
                    export_policy.peer_release.set()
                    with pytest.raises(CollaborationAccessDenied):
                        await delivery_task
                finally:
                    export_policy.peer_release.set()
                    await asyncio.gather(delivery_task, return_exceptions=True)
                assert await store.read_peer_content_attempt(peer) is None
                async with collaboration._transaction(owner.application_scope, write=False) as tx:
                    pending = await tx.get(
                        "clarification_deliveries", operation_key(delivery.operation)
                    )
                    assert pending["state"] == "pending"
                export_policy.pause_peer = False
                export_policy.revoked = False  # explicit test-policy restoration
                # The receiving transaction's index authenticates its exact
                # stored record. A public delivery must refuse a conflicting
                # index before any append, even when the source is authorized.
                from cayu.storage import _peer_attempts

                permits_parked = _peer_attempts.permits_parked_delivery_append

                def conflicting_index(request, checkpoint, record, **identity):
                    from copy import deepcopy

                    from cayu.sessions._session_continuation_store import ROOT_KEY

                    changed = deepcopy(checkpoint)
                    changed[ROOT_KEY]["entries"][0]["record_sha256"] = "0" * 64
                    return permits_parked(request, changed, record, **identity)

                if verify_index_integrity:
                    with monkeypatch.context() as patch:
                        patch.setattr(
                            _peer_attempts, "permits_parked_delivery_append", conflicting_index
                        )
                        with pytest.raises(CollaborationUnavailable):
                            await current.deliver_clarification(delivery, context=context)
                assert await store.read_peer_content_attempt(peer) is None
                append = current._session_engine.append_peer_content
                append_calls = 0

                async def commit_then_lose_ack(*args, **kwargs):
                    nonlocal append_calls
                    append_calls += 1
                    await append(*args, **kwargs)
                    raise ConnectionError("peer append committed but acknowledgement was lost")

                with monkeypatch.context() as patch:
                    patch.setattr(
                        current._session_engine, "append_peer_content", commit_then_lose_ack
                    )
                    with pytest.raises(CollaborationUnavailable):
                        await current.deliver_clarification(delivery, context=context)
                    receiving_attempt = await store.read_peer_content_attempt(peer)
                    assert receiving_attempt.status == "appended", (
                        receiving_attempt.model_dump_json()
                    )
                    if delivery_recovery == "fresh_process":
                        from tests.core._clarification_recovery_flow import (
                            cleanup_in_fresh_process,
                        )

                        selected_page = await current.list_pending_clarification_deliveries(
                            context=CONTEXT
                        )
                        assert len(selected_page.items) == 1
                        export_policy.revoked = True
                        try:
                            await cleanup_in_fresh_process(
                                initialized,
                                backend=backend,
                                tmp_path=tmp_path,
                                request=request,
                                kind="delivery",
                            )
                            result = await current.reconcile_clarification_delivery(
                                selected_page.items[0].recovery, context=CONTEXT
                            )
                            assert result.status == "appended"
                            assert len(payloads) == before_recovery
                        finally:
                            export_policy.revoked = False
                    if delivery_recovery == "cancelled":
                        from tests.core._clarification_delivery_recovery_flow import (
                            cancel_delivery_recovery,
                        )

                        await cancel_delivery_recovery(
                            current, request=peer, monkeypatch=monkeypatch
                        )
                        assert len(payloads) == before_recovery
                    if delivery_recovery is True:
                        other_store = (
                            store
                            if backend == "memory"
                            else _store_factory(backend, tmp_path, request)()
                        )
                        other_source = collaboration_factory()
                        other = application_for(other_source, other_store)
                        await other.initialize_collaboration()
                        export_policy.revoked = True
                        try:
                            recovered = await recover_delivery(other, expected_status="appended")
                            assert recovered.operation == delivery.operation
                            assert len(payloads) == before_recovery
                        finally:
                            export_policy.revoked = False
                            # Memory apps share the store's mutation owner; its
                            # lifetime extends through the original app's flow.
                            if other_source is not collaboration:
                                await other._request_coordinator.close()
                            if other_store is not store:
                                await other_store.close()
                            if other_source is not collaboration:
                                await other_source.close()
                    delivered = await current.deliver_clarification(delivery, context=context)
                    assert append_calls == 1  # retry reconciles; it does not call append again
                assert delivered.status == "appended"
                assert (
                    await current.prepare_clarification_delivery(delivery, context=context)
                    == delivered
                )
                assert await current.deliver_clarification(delivery, context=context) == delivered
                altered_peer = peer.model_copy(
                    update={
                        "attempt_key": peer.attempt_key.model_copy(
                            update={"deadline_at_ms": peer.attempt_key.deadline_at_ms - 1}
                        )
                    }
                )
                with pytest.raises(CollaborationConflict):
                    await current.deliver_clarification(
                        delivery.model_copy(update={"append": altered_peer}), context=context
                    )
                assert "Which API version?" not in delivered.model_dump_json()
                assert "private-state" not in delivered.model_dump_json()
                assert (
                    await current.read_peer_content(key, context=CONTEXT, expected=peer)
                ).status == "appended"
            # Delivery itself did not invoke a provider. Explicit activation is
            # separate and exercises the real peer serializer/exposure owner.
            assert len(payloads) == 1 + temporary_service + side_session
            target_execution = ParticipantSessionExecutionRequest(
                request=target_creation.request.model_copy(update={"session_id": target.id}),
                session_instance_id=target.instance_id,
                execution_key="question-recipient-run",
            )
            with monkeypatch.context() as patch:
                # Restore class-owned admission, not an instance-bound alias:
                # producer qualification deliberately rejects overridden ledgers.
                patch.delattr(current.budget_ledger, "register_budget_binding")
                if temporary_service:
                    assert parked is not None
                    if maintenance_recovery:
                        original_settle = type(collaboration)._settle_permit

                        async def lose_service_settlement_ack(
                            self, initialized, expected, **kwargs
                        ):
                            result = await original_settle(self, initialized, expected, **kwargs)
                            if expected.intent.request.effect_scope == "clarification_service":
                                raise OSError("service permit settlement acknowledgement lost")
                            return result

                        patch.setattr(
                            type(collaboration), "_settle_permit", lose_service_settlement_ack
                        )
                    service_request = ClarificationServiceRequest(
                        operation=initialized.operation("service-question"),
                        initiator=_initiator(actor_a.context),
                        delivery=delivery,
                        ticket=parked.ticket,
                        service_generation=1,
                        parent_service=None,
                        instruction="Answer the delivered clarification.",
                    )
                    service_context = SessionExportAccessContext(
                        principal="operator", mandate=actor_a.context
                    )
                    with pytest.raises(CollaborationAccessDenied):
                        await current.service_clarification(
                            service_request,
                            context=service_context,
                            delivery_context=context if post_admission else None,
                        )
                    assert len(payloads) == 2 + side_session
                    # The recipient needs an explicit current grant for this
                    # source row. The historical delivery is not that grant.
                    read_resolution = actor_a.resolution
                    read_actions = (*read_resolution.principal.actions, "expose")
                    actor_a.resolution = read_resolution.model_copy(
                        update={
                            "principal": read_resolution.principal.model_copy(
                                update={"actions": read_actions, "audiences": (owner, audience)}
                            ),
                            "chain": read_resolution.chain.model_copy(
                                update={
                                    "entries": (
                                        read_resolution.chain.entries[-1].model_copy(
                                            update={
                                                "actions": read_actions,
                                                "audiences": (owner, audience),
                                                "resources": (selected_resource,),
                                                "restrictions": read_resolution.chain.entries[
                                                    -1
                                                ].restrictions.model_copy(
                                                    update={"channels": ("prompt", "source")}
                                                ),
                                            }
                                        ),
                                    )
                                }
                            ),
                        }
                    )
                    if post_admission:
                        with pytest.raises(CollaborationUnavailable):
                            await current.service_clarification(
                                service_request, context=service_context
                            )
                        with pytest.raises(CollaborationAccessDenied):
                            await current.service_clarification(
                                service_request,
                                context=service_context,
                                delivery_context=service_context,
                            )
                        assert len(payloads) == 2 + side_session
                    if busy_target:
                        from tests.core._clarification_busy_gate import reject_busy_target

                        await reject_busy_target(
                            current,
                            service_request,
                            context=service_context,
                            delivery_context=context,
                            payloads=payloads,
                            entered=busy_entered,
                            release=busy_release,
                        )
                        return
                    if human_paused:
                        from tests.core._clarification_human_gate import reject_human_paused

                        await reject_human_paused(
                            current,
                            service_request,
                            context=service_context,
                            delivery_context=context,
                            payloads=payloads,
                        )
                        return
                    if prepared_recovery is not None or participant_race is not None:
                        from tests.core._clarification_participant_race import disable_service
                        from tests.core._clarification_prepared_recovery import recover_prepared

                        other_store = (
                            store
                            if backend == "memory"
                            else _store_factory(backend, tmp_path, request)()
                        )
                        other_source = collaboration_factory()
                        other = application_for(other_source, other_store)
                        await other.initialize_collaboration()
                        try:
                            if participant_race is not None:
                                await disable_service(
                                    current,
                                    other,
                                    initialized,
                                    service_request,
                                    context=service_context,
                                    delivery_context=context,
                                    registered_first=participant_race,
                                    payloads=payloads,
                                    monkeypatch=monkeypatch,
                                )
                            else:
                                await recover_prepared(
                                    current,
                                    other,
                                    service_request,
                                    context=service_context,
                                    delivery_context=context,
                                    phase=prepared_recovery,
                                    payloads=payloads,
                                    monkeypatch=monkeypatch,
                                )
                        finally:
                            await other._request_coordinator.close()
                            if backend != "memory":
                                await other_store.close()
                                await other_source.close()
                        return
                    if terminal_wait is not None:
                        from tests.core._clarification_wait_cleanup import retire_terminal_wait

                        await retire_terminal_wait(
                            current,
                            service_request,
                            mode=terminal_wait,
                            service_context=service_context,
                            delivery_context=context,
                            wait=wait,
                            wait_context=actor_a.context,
                            creation=original_target_creation,
                            participant=first.reference,
                            application_for=application_for,
                            collaboration_factory=collaboration_factory,
                            session_factory=_store_factory(backend, tmp_path, request),
                            backend=backend,
                            payloads=payloads,
                            monkeypatch=monkeypatch,
                        )
                        return
                    if final_latch_timing is not None:
                        from tests.core._clarification_final_flow import (
                            consume_original_latch,
                            finish_original_wait,
                        )
                        from tests.core._clarification_latch_flow import arbitrate_latch

                        async def publish_latch():
                            return await finish_original_wait(
                                current,
                                source=source,
                                accepted=accepted,
                                initialized=initialized,
                                export_template=export,
                                source_context=service_context,
                                actor_a=actor_a,
                                actor_b=actor_b,
                                reader=reader,
                                wait=wait,
                                parked=parked,
                                payloads=payloads,
                                context=CONTEXT,
                                consume=False,
                            )

                        async def consume_latch(latched, *, blocked):
                            await consume_original_latch(
                                current,
                                initialized=initialized,
                                latched=latched,
                                payloads=payloads,
                                context=CONTEXT,
                                blocked=blocked,
                                waiting_ticket=parked.ticket if blocked else None,
                            )

                        await arbitrate_latch(
                            current,
                            service_request,
                            timing=final_latch_timing,
                            service_driver=service_driver,
                            context=service_context,
                            delivery_context=context,
                            publish=publish_latch,
                            consume=consume_latch,
                            payloads=payloads,
                            monkeypatch=monkeypatch,
                        )
                        return
                    nested_results = []
                    nested_errors = []
                    if nested_service:
                        from tests.core._clarification_nested_flow import (
                            nested_service as run_nested,
                        )

                        from cayu.runtime._session_continuation import ContinuationConflict
                        from cayu.runtime._session_continuation_owner import (
                            SessionContinuationOwner,
                        )

                        reconcile = SessionContinuationOwner.reconcile_temporary
                        parent_admissions = []
                        parent_owners = []
                        rejected_parent_returns = []
                        nested_finished = asyncio.Event()

                        async def reconcile_with_child(owner, candidate):
                            if candidate.dispatch.intent.operation != service_request.operation:
                                return await reconcile(owner, candidate)
                            retained = await owner.store._load_temporary_continuation_service(
                                candidate
                            )
                            if retained is not None and retained.state in {"returned", "excluded"}:
                                return await reconcile(owner, candidate)
                            # Hold observation, not native execution or mutation.
                            # An early exact retry cannot race through ordinary
                            # reconciliation and miss the RELEASE barrier.
                            async with asyncio.timeout(180):
                                while True:
                                    observed = (
                                        await owner.store._read_temporary_continuation_outcome(
                                            candidate
                                        )
                                    )
                                    if observed is not None and observed.state == "returned":
                                        break
                                    await asyncio.sleep(0.05)
                            if (
                                candidate.dispatch.intent.operation == service_request.operation
                                and not nested_results
                            ):
                                nested_results.append(None)
                                parent_admissions.append(candidate)
                                parent_owners.append(owner)
                                try:
                                    nested_results[0] = await run_nested(
                                        current,
                                        initialized,
                                        original,
                                        service_request,
                                        export,
                                        actor_a,
                                        actor_b,
                                        source,
                                        target,
                                        export_policy,
                                        payloads,
                                    )
                                except BaseException as error:
                                    nested_errors.append(error)
                                    raise
                                finally:
                                    nested_finished.set()
                            elif candidate.dispatch.intent.operation == service_request.operation:
                                # Public observation may time out while the
                                # owned parent callback is still working. Its
                                # exact retry must honor this test barrier too.
                                await nested_finished.wait()
                                if nested_errors:
                                    raise nested_errors[0]
                            return await reconcile(owner, candidate)

                        async def nested_stack_probe():
                            # The real child transport is in flight. The parent
                            # has released, but the child still owns the stack.
                            before_child_return = await store.load_continuation_ticket(
                                parked.ticket.session_id,
                                session_instance_id=parked.ticket.session_instance_id,
                                registration_key=parked.ticket.registration_key,
                            )
                            assert [item.state for item in before_child_return.services] == [
                                "admitted",
                                "admitted",
                            ]
                            with pytest.raises(ContinuationConflict):
                                await reconcile(parent_owners[0], parent_admissions[0])
                            assert (
                                await store.load_continuation_ticket(
                                    parked.ticket.session_id,
                                    session_instance_id=parked.ticket.session_instance_id,
                                    registration_key=parked.ticket.registration_key,
                                )
                                == before_child_return
                            )
                            rejected_parent_returns.append(True)

                        patch.setattr(
                            SessionContinuationOwner, "reconcile_temporary", reconcile_with_child
                        )
                    if service_clock_offset is not None:
                        import cayu.deadlines as deadlines
                        import cayu.runtime._temporary_service_execution as service_execution

                        class WorkerClock(datetime):
                            @classmethod
                            def now(cls, tz=None):
                                return datetime.now(tz) + timedelta(seconds=service_clock_offset)

                        patch.setattr(deadlines, "datetime", WorkerClock)
                        patch.setattr(service_execution, "datetime", WorkerClock, raising=False)
                        with pytest.raises(CollaborationUnavailable) as interrupted:
                            await current.service_clarification(
                                service_request, context=service_context, delivery_context=context
                            )
                        await asyncio.wait_for(service_cancelled.wait(), 100)
                        assert len(payloads) == 3, repr(interrupted.value.__cause__)
                        records = await current.inspect_clarification_services(
                            parked.ticket, context=CONTEXT
                        )
                        assert records.items
                        # A cancelled active provider is not proof of exclusion.
                        assert all(item.state != "excluded" for item in records.items)
                        return
                    # Nested qualification includes two bounded service turns
                    # plus export/admission and return settlement for both.
                    if missing_append:
                        from cayu.collaboration._clarification_deliveries import (
                            ClarificationDeliveryReceipt,
                        )

                        async def false_delivery_ack(*args, **kwargs):
                            return ClarificationDeliveryReceipt(
                                operation=delivery.operation,
                                question=question.operation,
                                kind="question",
                                status="appended",
                                queue_id="untrusted-success",
                            )

                        # A faulty callback's projection is not receiving
                        # authority. The real native dispatch gate must reject.
                        patch.setattr(
                            current._clarification_coordinator, "deliver", false_delivery_ack
                        )
                    from tests.core._clarification_service_observation import (
                        await_service_return,
                    )

                    service_launches = []
                    lost_service_observations = []
                    if service_observation_loss:
                        original_service = current.service_clarification
                        coordinator = current._clarification_coordinator
                        original_delivery = coordinator.deliver
                        original_held_delivery = coordinator._deliver_held
                        delivery_waits = []
                        observation_owners = current._request_coordinator._owners
                        normal_observation_timeout = observation_owners.observation_timeout

                        async def delayed_held_delivery(value):
                            # run() has captured this observer's short timeout
                            # before scheduling us. Do not shorten concurrent
                            # maintenance observations throughout delivery.
                            observation_owners.observation_timeout = normal_observation_timeout
                            # Outlive the observer, not the admitted service
                            # deadline. Native append/authentication still run.
                            await asyncio.sleep(0.05)
                            return await original_held_delivery(value)

                        async def short_delivery_observation(*args, **kwargs):
                            owners = current._request_coordinator._owners
                            owners.observation_timeout = 0.001
                            try:
                                receipt = await original_delivery(*args, **kwargs)
                                delivery_waits.append(receipt.status)
                                return receipt
                            finally:
                                owners.observation_timeout = normal_observation_timeout

                        patch.setattr(coordinator, "_deliver_held", delayed_held_delivery)
                        patch.setattr(coordinator, "deliver", short_delivery_observation)

                        async def expire_launch_observation(*args, **kwargs):
                            service_launches.append(True)
                            owners = current._request_coordinator._owners
                            owners.observation_timeout = 0.001
                            try:
                                return await original_service(*args, **kwargs)
                            except CollaborationUnavailable:
                                lost_service_observations.append(True)
                                raise
                            finally:
                                owners.observation_timeout = normal_observation_timeout

                        patch.setattr(current, "service_clarification", expire_launch_observation)
                    serviced = await (service_driver or await_service_return)(
                        current,
                        service_request,
                        context=service_context,
                        delivery_context=context if post_admission else None,
                        recovery_context=CONTEXT,
                        timeout=660 if nested_service else 360,
                        nested_errors=nested_errors,
                        retain_settlement_debt=bool(maintenance_recovery),
                    )
                    if service_observation_loss:
                        assert service_launches == [True]
                        assert lost_service_observations == [True]
                        assert delivery_waits == ["appended"]
                        patch.setattr(current, "service_clarification", original_service)
                        patch.setattr(coordinator, "deliver", original_delivery)
                        patch.setattr(coordinator, "_deliver_held", original_held_delivery)
                    assert serviced.state == "returned"
                    if budget_boundary is not None:
                        from tests.core._clarification_budget_flow import verify_shared_budget

                        await verify_shared_budget(
                            current, source, target, binding, payloads, allowed=budget_boundary >= 0
                        )
                        assert (
                            await current.service_clarification(
                                service_request,
                                context=service_context,
                                delivery_context=context,
                            )
                            == serviced
                        )
                        assert len(payloads) == 2 + (budget_boundary >= 0)
                        return
                    if nested_service:
                        if nested_errors:
                            raise nested_errors[0]
                        assert nested_results and nested_results[0].state == "returned"
                        assert rejected_parent_returns
                        assert serviced.released_session_status == "completed"
                        retained = await store.load_continuation_ticket(
                            parked.ticket.session_id,
                            session_instance_id=parked.ticket.session_instance_id,
                            registration_key=parked.ticket.registration_key,
                        )
                        assert retained.ticket.state == "WAITING" and retained.latch is None
                        assert [item.state for item in retained.services] == [
                            "returned",
                            "returned",
                        ]
                        assert len(payloads) == 4
                        return
                    assert serviced.released_session_status == (
                        "failed" if missing_append else "completed"
                    )
                    assert (
                        await current.service_clarification(
                            service_request,
                            context=service_context,
                            delivery_context=context if post_admission else None,
                        )
                        == serviced
                    )
                    with pytest.raises(CollaborationConflict):
                        await current.service_clarification(
                            service_request.model_copy(update={"instruction": "Different work."}),
                            context=service_context,
                            delivery_context=context if post_admission else None,
                        )
                    retained_wait = await store.load_continuation_ticket(
                        parked.ticket.session_id,
                        session_instance_id=parked.ticket.session_instance_id,
                        registration_key=parked.ticket.registration_key,
                    )
                    assert retained_wait.ticket.state == "WAITING"
                    assert retained_wait.ticket.registration_key == parked.ticket.registration_key
                    if missing_append:
                        assert len(payloads) == 2
                        assert await store.read_peer_content_attempt(peer) is None
                        failed = [
                            event
                            for event in await store.load_events(target.id)
                            if event.type == EventType.SESSION_FAILED
                        ]
                        assert len(failed) == 1
                        assert failed[0].payload["error_type"] == "ContinuationConflict"
                        assert "exact durable peer append" in failed[0].payload["error"]
                        async with collaboration._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            responsibility = await tx.get(
                                "clarification_deliveries", operation_key(delivery.operation)
                            )
                        assert (
                            responsibility["state"] == "pending"
                            and responsibility["receipt"] is None
                        )
                        return
                else:
                    recipient_events = [
                        event
                        async for event in current.execute_participant_session(
                            target_execution, participant=first.reference, context=CONTEXT
                        )
                    ]
                    assert any(
                        event.type == EventType.SESSION_COMPLETED for event in recipient_events
                    )
            assert len(payloads) == 2 + temporary_service + side_session
            assert b"Which API version?" in payloads[-1]
            assert b"private-state" not in payloads[-1] and b"private-thinking" not in payloads[-1]
            assert len(export_policy.calls) == 1
            exposure_call = export_policy.calls[0]
            exposure_request = PeerContentExposureRequest.for_model_attempt(
                append_key=key,
                append_operation_key=peer.operation_key,
                model_attempt_id=exposure_call["model_attempt_id"],
                provider_name=exposure_call["provider_name"],
                capability_version=exposure_call["capability_version"],
            )
            exposure = await store.read_peer_content_exposure(key, exposure_request.exposure_id)
            assert exposure is not None and exposure.outcome == "exposed"
            if post_admission:
                delivered = await current.prepare_clarification_delivery(delivery, context=context)
                assert delivered.status == "appended"
            assert await current.deliver_clarification(delivery, context=context) == delivered
            if service_observation_loss:
                # This regression qualifies observation loss through native
                # return/replay and actual peer exposure. Reply acceptance and
                # the rest of the public journey have their own matrix below;
                # do not repeat them for a delivery-observation timing test.
                assert len(payloads) == 2 + temporary_service + side_session
                return
            exact = await current.lookup_clarification(command, context=actor_a.context)
            assert isinstance(exact, ExactMatch) and exact.receipt == opened
            inspection = await current.inspect_clarification(command, context=actor_a.context)
            assert isinstance(inspection, ExactMatch) and inspection.receipt == opened.decision
            assert isinstance(
                await current.lookup_clarification(
                    command.model_copy(update={"expected_revision": command.expected_revision + 1}),
                    context=actor_a.context,
                ),
                ExactConflict,
            )
            if planning_journey is None:
                assert (
                    await current.open_clarification(command, source=export, context=context)
                    == opened
                )
            assert len(payloads) == 2 + temporary_service + side_session
            assert "private-state" not in opened.model_dump_json()
            assert "private-thinking" not in opened.model_dump_json()
            retained = await current.inspect_collaboration_request(
                accepted.expected, context=actor_a.context
            )
            if public_reply:
                from cayu import ClarificationReplyRequest
                from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                from cayu.sessions._model_completion_publication import (
                    model_step_publication_from_checkpoint,
                )

                checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
                    target.id
                )
                pointer = model_step_publication_from_checkpoint(checkpoint)
                assert pointer is not None
                reply_audience = OwnerRef(
                    application_scope=owner.application_scope,
                    owner_id=second.reference.participant_id,
                    incarnation=second.reference.incarnation,
                )
                reply_resource = ResourceSelector(
                    resource=ObjectRef(
                        owner=owner,
                        kind="session_transcript_row",
                        object_id=target.id,
                        incarnation=target.instance_id,
                        revision=pointer.source_transcript_cursor + 1,
                    )
                )
                resolution = actor_a.resolution
                reply_audiences = (owner, audience, reply_audience)
                actor_a.resolution = resolution.model_copy(
                    update={
                        "principal": resolution.principal.model_copy(
                            update={
                                "actions": actions,
                                "audiences": reply_audiences,
                            }
                        ),
                        "chain": resolution.chain.model_copy(
                            update={
                                "entries": (
                                    resolution.chain.entries[-1].model_copy(
                                        update={
                                            "actions": actions,
                                            "audiences": reply_audiences,
                                            "resources": (selected_resource, reply_resource),
                                        }
                                    ),
                                )
                            }
                        ),
                    }
                )
                reply_namespace = await current.initialize_session_exports(
                    target.id, context=service_context
                )
                reply_export = export.model_copy(
                    update={
                        "ref": SessionExportRef(
                            session_id=target.id,
                            session_instance_id=target.instance_id,
                            operation=OperationRef(
                                application_scope=owner.application_scope,
                                namespace_incarnation=reply_namespace.namespace_incarnation,
                                generation=1,
                                caller_key="reply-export",
                            ),
                        ),
                        "source_indices": (pointer.source_transcript_cursor,),
                        "audience": reply_audience,
                    }
                )
                reply_exported = await current.export_session(reply_export, context=service_context)
                reply_request = ClarificationReplyRequest(
                    operation=initialized.operation("accept-reply"),
                    initiator=_initiator(actor_a.context),
                    expected=accepted.expected,
                    service=service_request,
                    source=reply_export,
                    production_stage_id=pointer.stage_id,
                    expected_input_revision=retained.clarification.input_revision,
                    expected_input_sha256=retained.clarification.input_sha256
                    or question.input_sha256,
                )
                before_reply = await current.inspect_collaboration_request(
                    accepted.expected, context=actor_a.context
                )
                with pytest.raises(CollaborationConflict):
                    await current.reply_to_clarification(
                        reply_request.model_copy(update={"production_stage_id": "wrong-stage"}),
                        context=service_context,
                    )
                assert (
                    await current.inspect_collaboration_request(
                        accepted.expected, context=actor_a.context
                    )
                    == before_reply
                )
                if reply_cancel_before is not None:
                    from tests.core._clarification_reply_cancel import arbitrate_reply_cancel

                    closure = ClarificationCloseCommand(
                        operation=initialized.operation("cancel-produced-question"),
                        expected=accepted.expected,
                        question=question.operation,
                        question_sha256=clarification_commitment(question, SecretRedactor()),
                        kind="cancelled",
                        initiator=_initiator(actor_a.context),
                    )
                    await arbitrate_reply_cancel(
                        current,
                        command,
                        reply_request,
                        closure,
                        cancel_first=reply_cancel_before,
                        context=actor_a.context,
                        export_context=service_context,
                        ticket=parked.ticket,
                        payloads=payloads,
                    )
                    return
                from cayu.collaboration import _clarification_reply_api as reply_api

                accept_reply = reply_api.accept_reply
                acknowledgements_lost = []

                async def commit_then_lose_reply_ack(*args, **kwargs):
                    result = await accept_reply(*args, **kwargs)
                    acknowledgements_lost.append(result)
                    raise OSError("reply acknowledgement lost after durable election")

                with monkeypatch.context() as patch:
                    patch.setattr(reply_api, "accept_reply", commit_then_lose_reply_ack)
                    with pytest.raises(CollaborationUnavailable):
                        await current.reply_to_clarification(reply_request, context=service_context)
                assert len(acknowledgements_lost) == 1
                accepted_reply = await current.reply_to_clarification(
                    reply_request, context=service_context
                )
                assert accepted_reply == acknowledgements_lost[0]
                assert (
                    accepted_reply.input_revision == before_reply.clarification.input_revision + 1
                )
                assert (
                    await current.reply_to_clarification(reply_request, context=service_context)
                    == accepted_reply
                )
                if planning_journey is not None:
                    return await after_planned_reply(
                        current=current,
                        reply=accepted_reply,
                        wait=wait,
                        parked=parked,
                        wait_context=actor_a.context,
                    )
                if multiple_questions:
                    from tests.core._clarification_final_flow import finish_original_wait
                    from tests.core._clarification_multiple_flow import second_question
                    from tests.core._clarification_public_flow import continue_with_reply

                    monkeypatch.delattr(current.budget_ledger, "register_budget_binding")
                    (
                        second_opening,
                        second_export,
                        second_exported,
                        second_resource,
                    ) = await second_question(
                        current,
                        initialized,
                        command,
                        service_request,
                        export,
                        reply_export,
                        actor_a,
                        actor_b,
                        context,
                        service_context,
                        export_policy,
                        payloads,
                        service_driver=service_driver,
                    )
                    await continue_with_reply(
                        current,
                        opening=second_opening,
                        question=second_opening.question,
                        reply_export=second_export,
                        reply_exported=second_exported,
                        service_context=service_context,
                        actor_a=actor_a,
                        actor_b=actor_b,
                        reply_resource=second_resource,
                        audiences=reply_audiences,
                        initialized=initialized,
                        responder=first.reference,
                        questioner=second.reference,
                        source=target,
                        destination=source,
                        export_policy=export_policy,
                        reply_text="Use API v2.",
                        payloads=payloads,
                        context=CONTEXT,
                    )
                    await finish_original_wait(
                        current,
                        source=source,
                        accepted=accepted,
                        initialized=initialized,
                        export_template=export,
                        source_context=service_context,
                        actor_a=actor_a,
                        actor_b=actor_b,
                        reader=reader,
                        wait=wait,
                        parked=parked,
                        payloads=payloads,
                        context=CONTEXT,
                    )
                    return
                if continue_questioner:
                    from tests.core._clarification_public_flow import continue_with_reply

                    # Publication is inert; this next phase is real execution
                    # and must use the normal causal budget ledger admission.
                    monkeypatch.delattr(current.budget_ledger, "register_budget_binding")
                    reply_delivery, delivered_reply = await continue_with_reply(
                        current,
                        opening=command,
                        question=question,
                        reply_export=reply_export,
                        reply_exported=reply_exported,
                        service_context=service_context,
                        actor_a=actor_a,
                        actor_b=actor_b,
                        reply_resource=reply_resource,
                        audiences=reply_audiences,
                        initialized=initialized,
                        responder=first.reference,
                        questioner=second.reference,
                        source=target,
                        destination=source,
                        export_policy=export_policy,
                        reply_text="Use API v2.",
                        payloads=payloads,
                        context=CONTEXT,
                    )
                    source_replay = await current.lookup_session_export(export, context=context)
                    assert isinstance(source_replay, ExactMatch)
                    assert source_replay.receipt == exported
                    if finish_request:
                        from tests.core._clarification_final_flow import finish_original_wait

                        await finish_original_wait(
                            current,
                            source=source,
                            accepted=accepted,
                            initialized=initialized,
                            export_template=export,
                            source_context=service_context,
                            actor_a=actor_a,
                            actor_b=actor_b,
                            reader=reader,
                            wait=wait,
                            parked=parked,
                            payloads=payloads,
                            context=CONTEXT,
                        )
                        if one_slot:
                            assert driver.active == 0
                            assert driver.order == [
                                parked.ticket.session_id,
                                source.id,
                                *([target.id] if side_session else []),
                                target.id,
                                source.id,
                                parked.ticket.session_id,
                            ]
                reopened_store = (
                    store if backend == "memory" else _store_factory(backend, tmp_path, request)()
                )
                reopened_collaboration = collaboration_factory()
                reopened_app = application_for(reopened_collaboration, reopened_store)
                # Reconstruct public values too: no private in-process wrapper
                # or cached admission can serve as receiving authority.
                reconstructed_reply = ClarificationReplyRequest.model_validate_json(
                    reply_request.model_dump_json()
                )
                reconstructed_context = SessionExportAccessContext.model_validate_json(
                    service_context.model_dump_json()
                )
                try:
                    await reopened_app.initialize_collaboration()
                    if continue_questioner:
                        reconstructed_delivery = ClarificationDeliveryIntent.model_validate_json(
                            reply_delivery.model_dump_json()
                        )
                        assert (
                            await reopened_app.deliver_clarification(
                                reconstructed_delivery, context=reconstructed_context
                            )
                            == delivered_reply
                        )
                    assert (
                        await reopened_app.reply_to_clarification(
                            reconstructed_reply, context=reconstructed_context
                        )
                        == accepted_reply
                    )
                    reopened_snapshot = await reopened_app.inspect_collaboration_request(
                        accepted.expected, context=actor_a.context
                    )
                    assert (
                        reopened_snapshot.clarification.input_revision
                        == accepted_reply.input_revision
                    )
                    export_policy.revoked = True
                    await reopened_app.change_participant_lifecycle(
                        change(
                            initialized,
                            first.reference,
                            key="disable-after-service-return",
                            revision=1,
                            state="disabled",
                        ),
                        context=CONTEXT,
                    )
                    # Maintenance discharges retained receiving responsibility;
                    # it never renews this revoked source disclosure grant.
                    if maintenance_recovery:
                        async with reopened_collaboration._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            debt = await tx.get(
                                "clarification_services", operation_key(service_request.operation)
                            )
                            assert debt["state"] == "pending"
                        from cayu import ClarificationServiceRecovery
                        from cayu.collaboration._contracts import CollaborationContractError

                        with pytest.raises(CollaborationAccessDenied):
                            await reopened_app.list_pending_clarification_services(
                                context=CONTEXT.model_copy(update={"principal": "outsider"})
                            )
                        for invalid_limit in (False, 0, 65):
                            with pytest.raises(CollaborationContractError):
                                await reopened_app.list_pending_clarification_services(
                                    context=CONTEXT, limit=invalid_limit
                                )
                        page = await reopened_app.list_pending_clarification_services(
                            context=CONTEXT, limit=1
                        )
                        assert len(page.items) == 1 and page.next_cursor is not None
                        summary = page.items[0]
                        assert summary.question == service_request.delivery.question.operation
                        assert summary.recovery.operation == service_request.operation
                        assert not (
                            await reopened_app.list_pending_clarification_services(
                                context=CONTEXT, cursor=page.next_cursor, limit=1
                            )
                        ).items
                        public_summary = page.model_dump_json()
                        for private in (
                            "Answer the delivered clarification.",
                            "Which API version?",
                            "Use API v2.",
                            "participant_permit",
                            "admission_commitment",
                        ):
                            assert private not in public_summary
                        recovery = ClarificationServiceRecovery.model_validate_json(
                            summary.recovery.model_dump_json()
                        )
                        for field, changed in (
                            ("selection_sha256", "e" * 64),
                            ("dispatch_sha256", "f" * 64),
                            ("session_instance_id", "different-incarnation"),
                        ):
                            with pytest.raises(CollaborationConflict):
                                await reopened_app.reconcile_clarification_service(
                                    recovery.model_copy(update={field: changed}), context=CONTEXT
                                )
                    with pytest.raises(CollaborationAccessDenied):
                        await reopened_app.reconcile_clarification_service(
                            service_request,
                            context=CONTEXT.model_copy(update={"principal": "outsider"}),
                        )
                    with pytest.raises(CollaborationConflict):
                        await reopened_app.reconcile_clarification_service(
                            service_request.model_copy(update={"instruction": "Different work."}),
                            context=CONTEXT,
                        )
                    if maintenance_recovery:
                        async with reopened_collaboration._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            assert (
                                await tx.get(
                                    "clarification_services",
                                    operation_key(service_request.operation),
                                )
                                == debt
                            )
                    if maintenance_recovery == "cancel_cleanup":
                        from tests.core._clarification_recovery_flow import cancel_cleanup_observer

                        await cancel_cleanup_observer(
                            reopened_app,
                            reopened_collaboration,
                            recovery,
                            context=CONTEXT,
                            monkeypatch=monkeypatch,
                        )
                    elif maintenance_recovery == "process_cleanup":
                        from tests.core._clarification_recovery_flow import cleanup_in_fresh_process

                        await cleanup_in_fresh_process(
                            initialized, backend=backend, tmp_path=tmp_path, request=request
                        )
                    for _ in range(2):
                        assert (
                            await reopened_app.reconcile_clarification_service(
                                recovery if maintenance_recovery else service_request,
                                context=CONTEXT,
                            )
                            == serviced
                        )
                    if maintenance_recovery:
                        async with reopened_collaboration._transaction(
                            owner.application_scope, write=False
                        ) as tx:
                            debt = await tx.get(
                                "clarification_services", operation_key(service_request.operation)
                            )
                            assert debt["state"] == "settled"
                        assert not (
                            await reopened_app.list_pending_clarification_services(context=CONTEXT)
                        ).items
                    with pytest.raises(CollaborationAccessDenied):
                        await reopened_app.reply_to_clarification(
                            reconstructed_reply, context=reconstructed_context
                        )
                    with pytest.raises(CollaborationAccessDenied):
                        await current.reply_to_clarification(reply_request, context=service_context)
                    if prune_history:
                        from tests.core._clarification_retention_flow import prune_closed_question

                        if prune_history == "lost_native_ack":
                            retained_services = await reopened_app.inspect_clarification_services(
                                service_request.ticket, context=CONTEXT
                            )
                            recovery = retained_services.items[0].recovery
                            with pytest.raises(
                                ContinuationConflict, match="settlement acknowledgement"
                            ):
                                await reopened_app.validate_session_closure(
                                    service_request.ticket.session_id
                                )
                        await prune_closed_question(
                            current,
                            collaboration_factory,
                            initialized,
                            command,
                            actor_a,
                            mandates,
                            registered,
                        )
                        if prune_history == "lost_native_ack":
                            monkeypatch.setattr(
                                SessionContinuationOwner,
                                "_acknowledge_temporary_settlement",
                                original_ack,
                            )
                            before = len(payloads)
                            for _ in range(2):
                                assert (
                                    await reopened_app.reconcile_clarification_service(
                                        recovery, context=CONTEXT
                                    )
                                    == serviced
                                )
                            assert len(payloads) == before
                            for field, value in (
                                ("selection_sha256", "e" * 64),
                                ("dispatch_sha256", "f" * 64),
                                ("session_instance_id", "other-incarnation"),
                            ):
                                with pytest.raises(CollaborationConflict):
                                    await reopened_app.reconcile_clarification_service(
                                        recovery.model_copy(update={field: value}), context=CONTEXT
                                    )
                            from cayu.collaboration.exports import SessionExportConflict
                            from cayu.runtime._temporary_continuation import temporary_service_key

                            terminal = await reopened_store.load_session_operation(
                                service_request.ticket.session_id,
                                temporary_service_key(service_request.operation),
                            )
                            assert terminal is not None
                            assert terminal["settlement_acknowledged"] is True
                            # The continuation fence has cleared. This public
                            # export journey still owns separate export pins;
                            # service settlement must not erase those obligations.
                            with pytest.raises(SessionExportConflict):
                                await reopened_app.validate_session_closure(
                                    service_request.ticket.session_id
                                )
                finally:
                    await reopened_app.drain_session_exports()
                    await reopened_app._request_coordinator.close()
                    if backend != "memory":
                        await reopened_store.close()
                        await reopened_collaboration.close()
                assert len(payloads) == 3 + side_session + continue_questioner + finish_request
                assert "recipient-state" not in accepted_reply.model_dump_json()
                return
            competing_store = collaboration_factory()
            other = app(competing_store, registered)
            await other.initialize_collaboration()
            next_question = question.model_copy(
                update={"operation": initialized.operation("second-question"), "generation": 2}
            )
            next_command = command.model_copy(
                update={"operation": next_question.operation, "question": next_question}
            )
            budget_receiver.pause = True
            publication = asyncio.create_task(
                current.open_clarification(next_command, source=export, context=context)
            )
            try:
                await asyncio.wait_for(budget_receiver.entered.wait(), 5)
                await other.change_participant_lifecycle(
                    change(
                        initialized,
                        second.reference,
                        key="disable-before-publication",
                        revision=1,
                        state="disabled",
                    ),
                    context=CONTEXT,
                )
                budget_receiver.release.set()
                with pytest.raises(CollaborationAccessDenied):
                    await publication
                assert (
                    await current.inspect_collaboration_request(
                        accepted.expected, context=actor_a.context
                    )
                    == retained
                )
            finally:
                budget_receiver.release.set()
                await asyncio.gather(publication, return_exceptions=True)
                if backend != "memory":
                    await competing_store.close()
            # Publication already committed before disablement remains the same
            # historical fact; replay neither admits service nor reopens a question.
            assert (
                await current.open_clarification(command, source=export, context=context) == opened
            )
            export_policy.revoked = True
            with pytest.raises(CollaborationAccessDenied):
                await current.open_clarification(command, source=export, context=context)
            with pytest.raises(CollaborationAccessDenied):
                await current.deliver_clarification(delivery, context=context)
            assert len(payloads) == 2 + temporary_service + side_session
            # Metadata readback uses current request read authority, not a new
            # disclosure/execution grant or the retired publication registration.
            read_store = collaboration_factory()
            reader_app = app(
                read_store,
                registered,
                collaboration_requests=RequestRegistration(mandates=mandates, max_ttl_ms=300000),
            )
            await reader_app.initialize_collaboration()
            try:
                reconstructed = ClarificationOpenCommand.model_validate_json(
                    command.model_dump_json()
                )
                exact = await reader_app.lookup_clarification(
                    reconstructed, context=actor_a.context
                )
                assert isinstance(exact, ExactMatch) and exact.receipt == opened
                actor_a.denied = True
                with pytest.raises(CollaborationAccessDenied):
                    await reader_app.lookup_clarification(reconstructed, context=actor_a.context)
                actor_a.denied = False
                closure = ClarificationCloseCommand(
                    operation=initialized.operation("cancel-question"),
                    expected=accepted.expected,
                    question=question.operation,
                    question_sha256=clarification_commitment(question, SecretRedactor()),
                    kind="cancelled",
                    initiator=_initiator(actor_a.context),
                )
                assert isinstance(
                    await reader_app.lookup_clarification(closure, context=actor_a.context),
                    ExactNotFound,
                )
                with pytest.raises(CollaborationConflict):
                    await reader_app.close_clarification(
                        closure.model_copy(update={"kind": "expired"}), context=actor_a.context
                    )
                closed = await reader_app.close_clarification(closure, context=actor_a.context)
                assert closed.decision.state == "cancelled"
                after_closure = await reader_app.inspect_collaboration_request(
                    accepted.expected, context=actor_a.context
                )
                assert after_closure.state == "open" and after_closure.admission == "clarifying"
                assert after_closure.clarification == retained.clarification
                assert (
                    await reader_app.close_clarification(closure, context=actor_a.context) == closed
                )
                assert (
                    await reader_app.inspect_collaboration_request(
                        accepted.expected, context=actor_a.context
                    )
                    == after_closure
                )
                exact_close = await reader_app.lookup_clarification(
                    closure, context=actor_a.context
                )
                assert isinstance(exact_close, ExactMatch) and exact_close.receipt == closed
                historical_open = await reader_app.lookup_clarification(
                    command, context=actor_a.context
                )
                assert isinstance(historical_open, ExactMatch) and historical_open.receipt == opened
                inspection = await reader_app.inspect_clarification(
                    command, context=actor_a.context
                )
                assert isinstance(inspection, ExactMatch) and inspection.receipt == closed.decision
                async with read_store._transaction(owner.application_scope, write=True) as tx:
                    await tx.delete("clarification_questions", operation_key(question.operation))
                assert isinstance(
                    await reader_app.lookup_clarification(reconstructed, context=actor_a.context),
                    ExactUnavailable,
                )
                assert len(payloads) == 2 + temporary_service + side_session
            finally:
                actor_a.denied = False
                if backend != "memory":
                    await read_store.close()
        finally:
            await current.drain_session_exports()
            await current._request_coordinator.close()
            for ledger in budget_ledgers:
                await ledger.close()
            if backend != "memory":
                await store.close()
                await collaboration.close()


@pytest.mark.anyio
async def test_clarification_service_permits_belong_to_the_servicing_app(
    tmp_path, request, monkeypatch
):
    from cayu.collaboration._ownership import _MutationScope

    scoped_keys: list[object] = []
    run = _MutationScope.run

    async def recording_run(self, operation, **kwargs):
        scoped_keys.append(kwargs["key"])
        return await run(self, operation, **kwargs)

    monkeypatch.setattr(_MutationScope, "run", recording_run)
    from cayu.runtime import _session_continuation_owner

    trackers: list[object] = []
    owner_init = _session_continuation_owner.SessionContinuationOwner.__init__

    def recording_init(self, *args, **kwargs):
        trackers.append(kwargs.get("track"))
        owner_init(self, *args, **kwargs)

    monkeypatch.setattr(
        _session_continuation_owner.SessionContinuationOwner, "__init__", recording_init
    )
    await test_public_question_uses_real_assistant_export(
        "memory",
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
    )

    # The app's shutdown waits for its service permit mutations, not just requests.
    # Permit keys carry the application scope; the coordinator's own key does not.
    assert any(key[0] == "clarification-service" and isinstance(key[1], str) for key in scoped_keys)
    # Continuation owners retain work past their observers; the app waits for it.
    assert trackers
    assert all(isinstance(getattr(track, "__self__", None), _MutationScope) for track in trackers)
