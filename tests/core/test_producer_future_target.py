"""Producer delivery exclusions survive real future-target creation and ID reuse."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from tests.core import producer_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_peer_content import QualifiedPeerProvider
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu import AgentSpec, ProducerDeliveryRecovery
from cayu.collaboration._producer_contracts import ProducerOutputRegistration
from cayu.collaboration._session_export_store import digest
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.sessions import RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("ordering", "revoked"),
    [("exclusion-first", True), ("exclusion-first", False), ("creation-first", True)],
)
async def test_public_producer_future_target_orders_delivery_exclusion(
    native_stores, monkeypatch, ordering, revoked
):
    original_scenario = producer_export_scenario.output_scenario
    entered, release = asyncio.Event(), asyncio.Event()
    tasks, creations, targets, target_apps = [], [], [], []
    reopened_sessions = []
    peer_entered, peer_release = asyncio.Event(), asyncio.Event()
    late_observer, late_owners = None, ()

    async def with_future_target(*args, **kwargs):
        values = list(await original_scenario(*args, **kwargs))
        app, _, _, _, _, initialized, command, _ = values
        target_app = make_app(
            native_stores[0],
            app._participant_coordinator._registration,
            session_store=app.session_store,
        )
        target_app.register_provider(QualifiedPeerProvider([], name="provider"), default=True)
        target_app.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
        assert await target_app.initialize_collaboration() == initialized
        target_apps.append(target_app)
        destination = command.destinations[0]
        creation = RecipientSessionCreationRequest(
            request=RunRequest(
                agent_name="reviewer",
                session_id="future-" + initialized.owner.application_scope,
                messages=[],
            ),
            creation_key="future-output-consumer:" + initialized.owner.application_scope,
            recipient=destination.recipient,
        )
        creations.append(creation)
        create = app.session_store.create_participant_owned_session

        async def paused(*args, **kwargs):
            targets.append(kwargs["creation_target"])
            entered.set()
            await release.wait()
            return await create(*args, **kwargs)

        monkeypatch.setattr(app.session_store, "create_participant_owned_session", paused)
        tasks.append(
            asyncio.create_task(target_app.create_recipient_session(creation, context=CONTEXT))
        )
        await asyncio.wait_for(entered.wait(), 60)
        key = destination.attempt.append_key.model_copy(
            update={
                "target_session_id": None,
                "target_session_instance_id": None,
                "creation_target": targets[0],
            }
        )
        destination = destination.model_copy(
            update={"attempt": destination.attempt.model_copy(update={"append_key": key})}
        )
        values[6] = ProducerOutputRegistration.model_validate(
            command.model_copy(update={"destinations": (destination,)})
        )
        return tuple(values)

    monkeypatch.setattr(producer_export_scenario, "output_scenario", with_future_target)
    app = None
    try:
        values, context = await asyncio.wait_for(
            producer_export_scenario.completed_export_scenario(native_stores, monkeypatch),
            180,
        )
        app, resolver, _, provider, _, _, command, _ = values
        destination = command.destinations[0]
        exported = await app.export_producer_output(command, destination.operation, context=context)
        await app.publish_producer_outcome(
            command, destination=destination.operation, context=context
        )
        source = await app.lookup_session_export(exported.request, context=context)
        policy = app._session_export_coordinator.registration.policy
        policy.register_export(
            source.receipt,
            payload_sha256=digest({"text": "retained answer", "artifact_commitments": []}),
            consumer_id=destination.recipient.participant_id,
        )
        policy.allowed_receipts.add(command.operation.caller_key)
        prepared = await app.deliver_producer_output(
            command, destination.operation, context=context, prepare_only=True
        )
        assert prepared.receipt is None and not tasks[0].done()
        page = await app.pending_producer_outputs(
            command.admission.prepared.recipient, context=CONTEXT
        )
        token = next(
            item.recovery for item in page.items if item.recovery.registration == command.operation
        )
        recovery = ProducerDeliveryRecovery(
            registration=token.registration,
            registration_commitment=token.registration_commitment,
            destination=destination.operation,
        )
        if ordering == "exclusion-first":
            append = app.append_peer_content

            async def delayed_append(*args, **kwargs):
                peer_entered.set()
                await peer_release.wait()
                return await append(*args, **kwargs)

            monkeypatch.setattr(app, "append_peer_content", delayed_append)
            late_observer = asyncio.create_task(
                app.deliver_producer_output(command, destination.operation, context=context)
            )
            await asyncio.wait_for(peer_entered.wait(), 60)
            late_observer.cancel()
            late_observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await late_observer
            assert late_observer.cancelled() and late_observer.cancelling() == 2
            late_owners = tuple(app._request_coordinator._owners.pending)
            assert late_owners and not any(owner.done() for owner in late_owners)
        if ordering == "creation-first":
            release.set()
            target, _ = await asyncio.wait_for(tasks[0], 60)
            appended = await app.deliver_producer_output(
                command, destination.operation, context=context
            )
            assert appended.receipt.status == "appended"
        if revoked:
            resolver.recipient.denied = True
            policy.denied.add("append")
        if ordering == "exclusion-first":
            transaction = native_stores[0]._transaction
            lost = []

            @asynccontextmanager
            async def commit_then_lose_ack(scope, *, write):
                accepted = False
                async with transaction(scope, write=write) as tx:
                    put = tx.put

                    async def track(family, key, value, *, insert):
                        nonlocal accepted
                        await put(family, key, value, insert=insert)
                        accepted |= getattr(value, "mode", None) == "producer_delivery_accepted"

                    tx.put = track
                    yield tx
                if accepted and not lost:
                    lost.append(True)
                    raise ConnectionError("Delivery exclusion acknowledgement lost after commit")

            with monkeypatch.context() as patch:
                patch.setattr(native_stores[0], "_transaction", commit_then_lose_ack)
                with pytest.raises(CollaborationUnavailable):
                    await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
            assert lost == [True]
        backend, address = native_stores[3]
        sessions = app.session_store
        if backend == "sqlite":
            from cayu.storage.sqlite import SQLiteSessionStore

            sessions = SQLiteSessionStore(Path(address).with_name("sessions.sqlite"))
            reopened_sessions.append(sessions)
        elif backend == "postgres":
            from cayu.storage.postgres import PostgresSessionStore

            sessions = PostgresSessionStore(address)
            reopened_sessions.append(sessions)
        other = make_app(
            native_stores[2](),
            app._participant_coordinator._registration,
            session_store=sessions,
            collaboration_requests=app._request_coordinator._registration,
            session_exports=app._session_export_coordinator.registration,
        )
        await other.initialize_collaboration()
        assert not other._providers
        app = other
        settled = await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
        assert settled.state == ("excluded" if ordering == "exclusion-first" else "appended")
        assert (
            await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
            == settled
        )
        if ordering == "exclusion-first":
            release.set()
            target, receipt = await asyncio.wait_for(tasks[0], 60)
            assert await target_apps[0].create_recipient_session(creations[0], context=CONTEXT) == (
                target,
                receipt,
            )
            assert await app.session_store.load_transcript(target.id) == []
            await app.session_store.delete_session(target.id)
            replacement, _ = await target_apps[0].create_recipient_session(
                RecipientSessionCreationRequest(
                    request=creations[0].request,
                    creation_key="unrelated-public-id-reuse:" + command.operation.application_scope,
                    recipient=destination.recipient,
                ),
                context=CONTEXT,
            )
            assert replacement.id == target.id and replacement.instance_id != target.instance_id
            assert (
                await app.reconcile_producer_delivery(recovery, context=CONTEXT, exclude=True)
                == settled
            )
            assert await app.session_store.load_transcript(replacement.id) == []
            peer_release.set()
            outcomes = await asyncio.wait_for(
                asyncio.gather(*late_owners, return_exceptions=True), 60
            )
            if revoked:
                assert all(isinstance(outcome, Exception) for outcome in outcomes)
            else:
                assert all(outcome.receipt.status == "excluded" for outcome in outcomes)
            assert await app.session_store.load_transcript(replacement.id) == []
        assert len(provider.requests) == 1
    finally:
        release.set()
        peer_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *late_owners,
            *((late_observer,) if late_observer is not None else ()),
            return_exceptions=True,
        )
        if app is not None:
            await app.drain_collaboration_requests()
        for sessions in reopened_sessions:
            await sessions.close()
