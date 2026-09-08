"""Permission-to-store control coordination; native acquisition is a separate phase."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_browser_control import operator_purpose, request
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPolicyResult,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
    BrowserSensitiveEntryIntent,
    BrowserTakeoverIntent,
    BrowserTextInputIntent,
    BrowserViewIntent,
)
from cayu.vaults.redaction import SecretRedactor


def intent_for(command):
    return BrowserTakeoverIntent.model_validate(
        {
            **request().model_dump(mode="python", exclude={"operator"}),
            "identity": command.changed_record.identity,
        }
    )


def coordinator(store, policy):
    return BrowserControlCoordinator(
        purpose=operator_purpose(),
        store=store,
        policy=policy,
        redactor=SecretRedactor(),
        clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_sensitive_entry_admission_is_pending_and_fences_viewing(backend, tmp_path):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            policy = Policy(True)
            owner = coordinator(store, policy)
            principal = BrowserControlPrincipal(subject="operator")
            pending = await owner.request_takeover(
                principal=principal,
                operator_session_id="server-session",
                intent=intent_for(bootstrap),
            )
            # Acquisition is established separately by the real channel tests.
            acquired = await owner._publish_guest_acquisition(expected=pending, lease_until_ms=2000)
            intent = BrowserSensitiveEntryIntent(
                identity=acquired.identity,
                expected_record_revision=acquired.revision,
                expected_control_epoch=acquired.control_epoch,
                request_id=acquired.request.request_id,
            )
            before = await store.load_checkpoint("session")
            with pytest.raises(BrowserControlConflict):
                await owner.request_sensitive_entry(
                    principal=principal, operator_session_id="other-session", intent=intent
                )
            policy.allowed = False
            with pytest.raises(BrowserControlPermissionDenied):
                await owner.request_sensitive_entry(
                    principal=principal, operator_session_id="server-session", intent=intent
                )
            assert await store.load_checkpoint("session") == before
            policy.allowed = True
            result = await owner.request_sensitive_entry(
                principal=principal, operator_session_id="server-session", intent=intent
            )
            assert result.sensitive_entry_pending and result.capture_restricted
            assert not result.sensitive_entry
            assert result.control_epoch == acquired.control_epoch
            restarted = coordinator(store, policy)
            assert (
                await restarted.request_sensitive_entry(
                    principal=principal, operator_session_id="server-session", intent=intent
                )
                == result
            )
            with pytest.raises(BrowserControlPermissionDenied):
                await BrowserViewTickets(restarted).issue(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=BrowserViewIntent(
                        identity=result.identity,
                        expected_record_revision=result.revision,
                        page=result.request.pages[0],
                    ),
                )
            with pytest.raises(BrowserControlConflict):
                await restarted.request_handback(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=BrowserHandbackIntent(
                        **intent.model_dump(exclude={"expected_record_revision"}),
                        expected_record_revision=result.revision,
                    ),
                )
            # Positive guest/capture settlement is covered by the ASGI corpus.
            settled = await restarted._publish_guest_sensitive_entry(expected=result)
            input_intent = BrowserTextInputIntent(
                identity=settled.identity,
                request_id=settled.request.request_id,
                expected_record_revision=settled.revision,
                expected_control_epoch=settled.control_epoch,
                input_sequence=1,
                page=settled.request.pages[0],
            )
            before = await store.load_checkpoint("session")
            clock = [1.0]
            tickets = BrowserInputTickets(restarted, clock=lambda: clock[0])
            token = await tickets.issue(
                principal=principal, operator_session_id="server-session", intent=input_intent
            )
            assert await store.load_checkpoint("session") == before
            bound_input = tickets.consume(token)
            assert bound_input.intent == input_intent
            assert bound_input.principal == principal
            with pytest.raises(BrowserControlPermissionDenied):
                tickets.consume(token)
            expired = await tickets.issue(
                principal=principal, operator_session_id="server-session", intent=input_intent
            )
            clock[0] += 10
            with pytest.raises(BrowserControlPermissionDenied):
                tickets.consume(expired)
            policy.allowed = False
            with pytest.raises(BrowserControlPermissionDenied):
                await tickets.issue(
                    principal=principal, operator_session_id="server-session", intent=input_intent
                )
            policy.allowed = True
            with pytest.raises(BrowserControlConflict):
                await restarted._admit_text_input(
                    principal=principal,
                    operator_session_id="other-session",
                    intent=input_intent,
                )
            assert await store.load_checkpoint("session") == before
            reserved = await restarted._admit_text_input(
                principal=principal,
                operator_session_id="server-session",
                intent=input_intent,
            )
            assert reserved.pending_input_sequence == 1
            assert reserved.pending_input_page == input_intent.page
            assert reserved.manual_mutation_uncertain
            assert reserved.settled_input_sequence == 0
            assert [
                (item.page_id, item.operations) for item in reserved.operator_page_operations
            ] == [(input_intent.page.page_id, 1)]
            with pytest.raises(ValueError):
                reserved.model_copy(update={"pending_input_page": None})
            recovery = coordinator(store, policy)
            # Identical acknowledgement-loss retry cannot authorize retyping.
            with pytest.raises(BrowserControlConflict):
                await recovery._admit_text_input(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=input_intent,
                )
            with pytest.raises(BrowserControlConflict):
                await recovery._admit_text_input(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=input_intent.model_copy(
                        update={"expected_record_revision": reserved.revision}
                    ),
                )
            assert (await recovery._load(reserved.identity))[1] == reserved
            acknowledged = await recovery._publish_guest_input(expected=reserved)
            assert (
                await coordinator(store, policy)._publish_guest_input(expected=reserved)
                == acknowledged
            )
            second = await recovery._admit_text_input(
                principal=principal,
                operator_session_id="server-session",
                intent=input_intent.model_copy(
                    update={"expected_record_revision": acknowledged.revision, "input_sequence": 2}
                ),
            )
            acknowledged = await recovery._publish_guest_input(expected=second)
            assert [
                (item.page_id, item.operations) for item in acknowledged.operator_page_operations
            ] == [(input_intent.page.page_id, 2)]
            returning = await recovery.request_handback(
                principal=principal,
                operator_session_id="server-session",
                intent=BrowserHandbackIntent(
                    identity=acknowledged.identity,
                    expected_record_revision=acknowledged.revision,
                    expected_control_epoch=acknowledged.control_epoch,
                    request_id=input_intent.request_id,
                ),
            )
            returned = await recovery._publish_guest_handback(expected=returning)
            assert returned.operator_page_operations == acknowledged.operator_page_operations
            assert (await coordinator(store, policy)._load(returned.identity))[1] == returned
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_disconnect_cannot_overwrite_a_newer_takeover_revision(backend, tmp_path):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            initial = await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            pending = await owner.request_takeover(
                principal=BrowserControlPrincipal(subject="operator"),
                operator_session_id="server-session",
                intent=intent_for(bootstrap),
            )
            with pytest.raises(BrowserControlConflict):
                await owner.mark_channel_uncertain(expected=initial)
            fenced = await owner.mark_channel_uncertain(expected=pending)
            assert fenced.state == "control_uncertain"
            assert fenced.request == pending.request
            assert fenced.control_epoch == pending.control_epoch
            assert fenced.pending_input_sequence == pending.pending_input_sequence
            assert fenced.lease_until_ms == pending.lease_until_ms
            restarted = coordinator(store, Policy(True))
            assert await restarted.mark_channel_uncertain(expected=pending) == fenced
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "changed",
    [
        None,
        "capture_restricted",
        "fresh_observation_required",
        "manual_mutation_uncertain",
        "revision",
    ],
)
def test_idle_channel_fences_only_exact_queued_request(backend, tmp_path, changed, monkeypatch):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            initial = await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            pending = await owner.request_takeover(
                principal=BrowserControlPrincipal(subject="operator"),
                operator_session_id="server-session",
                intent=intent_for(bootstrap),
            )
            if changed is not None:
                # A malformed reconstructed record must not gain cleanup
                # authority solely from its matching request/allocation ID.
                altered = pending.model_copy(
                    update={changed: pending.revision + 1 if changed == "revision" else True}
                )
                read = owner._read_record

                async def reconstructed(identity):
                    session, raw, controls, _ = await read(identity)
                    return session, raw, controls, altered

                monkeypatch.setattr(owner, "_read_record", reconstructed)
                with pytest.raises(BrowserControlConflict):
                    await owner._fence_idle_channel(expected=initial)
                assert (await read(initial.identity))[3] == pending
            else:
                fenced = await owner._fence_idle_channel(expected=initial)
                assert fenced == pending.model_copy(
                    update={"revision": pending.revision + 1, "state": "control_uncertain"}
                )
                restarted = coordinator(store, Policy(True))
                assert await restarted._fence_idle_channel(expected=initial) == fenced
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_authorized_request_persists_derived_actor_but_does_not_grant_input(backend, tmp_path):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, Policy(True))
            principal = BrowserControlPrincipal(subject="authenticated", tenant="tenant")
            intent = intent_for(bootstrap)
            result = await owner.request_takeover(
                principal=principal, operator_session_id="server-session", intent=intent
            )
            assert result.state == "takeover_requested"
            assert result.lease_until_ms is None
            assert result.control_epoch == 1
            assert result.request.operator.subject == "authenticated"
            assert result.request.operator.operator_session_id == "server-session"
            restarted = coordinator(store, Policy(True))
            assert (
                await restarted.request_takeover(
                    principal=principal, operator_session_id="server-session", intent=intent
                )
                == result
            )
            with pytest.raises(BrowserControlConflict):
                await restarted.request_takeover(
                    principal=BrowserControlPrincipal(subject="different", tenant="tenant"),
                    operator_session_id="server-session",
                    intent=intent,
                )
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_view_permission_does_not_authorize_takeover(backend, tmp_path):
    class ViewOnly(Policy):
        async def decide(self, request):
            return BrowserControlPolicyResult(allowed=request.action == "view")

    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owner = coordinator(store, ViewOnly(True))
            principal = BrowserControlPrincipal(subject="viewer")
            permission = await owner.authorize_view(
                principal=principal,
                operator_session_id="server-session",
                identity=bootstrap.changed_record.identity,
                expected_record_revision=1,
            )
            assert permission.action == "view"
            before = await store.load_checkpoint("session")
            with pytest.raises(BrowserControlPermissionDenied):
                await owner.request_takeover(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=intent_for(bootstrap),
                )
            assert await store.load_checkpoint("session") == before

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_competing_authorized_operators_have_only_one_durable_winner(backend, tmp_path):
    async def scenario():
        barrier = asyncio.Barrier(2)

        class ConcurrentPolicy(Policy):
            async def decide(self, request):
                await barrier.wait()
                return BrowserControlPolicyResult(allowed=True)

        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            await BrowserControlPublisher(store).publish(bootstrap)
            owners = [coordinator(store, ConcurrentPolicy(True)) for _ in range(2)]
            intents = [
                intent_for(bootstrap).model_copy(update={"request_id": "bt_" + str(n) * 32})
                for n in (1, 2)
            ]
            results = await asyncio.gather(
                *(
                    owner.request_takeover(
                        principal=BrowserControlPrincipal(subject=f"operator-{n}"),
                        operator_session_id=f"server-{n}",
                        intent=intent,
                    )
                    for n, (owner, intent) in enumerate(zip(owners, intents, strict=True))
                ),
                return_exceptions=True,
            )
            assert sum(not isinstance(result, BaseException) for result in results) == 1
            assert (
                sum(
                    isinstance(result, (BrowserControlConflict, ExceptionGroup))
                    for result in results
                )
                == 1
            )

    asyncio.run(scenario())
