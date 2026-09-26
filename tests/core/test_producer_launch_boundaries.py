"""Live producer admission, exact attachment replay and successor isolation."""

from datetime import UTC, datetime

import pytest
from tests.core.producer_export_scenario import completed_export_scenario
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._request_store import retained_request
from cayu.collaboration.requests import RequestControl
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _producer_output_store as native
from cayu.sessions.base import (
    EnqueueSessionMessageRequest,
    ResumeRequest,
    SessionMessageDeliveryMode,
    SessionMessageQuery,
    _invocation_lifecycle_authority_read_scope,
)


@pytest.mark.anyio
@pytest.mark.parametrize("queued", [False, True])
async def test_live_producer_preserves_original_invocation_and_registration_replay(
    native_stores, monkeypatch, queued
):
    original = native.retain_native_output
    applications = []
    from cayu.applications import CayuApp

    register = CayuApp.register_producer_output

    async def remember(app, command, execution, *, context):
        result = await register(app, command, execution, context=context)
        applications.append((app, command, execution, context))
        return result

    monkeypatch.setattr(CayuApp, "register_producer_output", remember)
    replayed = []

    async def before_completion(store, session_id, **kwargs):
        app, command, execution, context = applications[0]
        with _invocation_lifecycle_authority_read_scope():
            before = await store.load_checkpoint(session_id)
        replayed.append(await register(app, command, execution, context=context))
        with _invocation_lifecycle_authority_read_scope():
            assert await store.load_checkpoint(session_id) == before
        if queued:
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="separate-successor",
                    content="This input is not the registered producer invocation.",
                    delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
                )
            )
        return await original(store, session_id, **kwargs)

    monkeypatch.setattr(native, "retain_native_output", before_completion)
    values, _ = await completed_export_scenario(native_stores, monkeypatch)
    app, resolver, admission, provider, session, initialized, command, execution = values
    assert len(replayed) == 1 and len(provider.requests) == 1
    with _invocation_lifecycle_authority_read_scope():
        before = await native_stores[1].load_checkpoint(session.id)
    await register(app, command, execution, context=resolver.recipient.context)
    with _invocation_lifecycle_authority_read_scope():
        assert await native_stores[1].load_checkpoint(session.id) == before
    if queued:
        records = await native_stores[1].inspect_session_messages(
            SessionMessageQuery(session_id=session.id)
        )
        assert len(records.records) == 1 and records.records[0].status.value == "queued"
        assert (await native_stores[1].load(session.id)).status.value == "interrupted"

    monkeypatch.setattr(native, "retain_native_output", original)
    completion = await app.retain_producer_completion(command, context=CONTEXT)
    release = await native_stores[1]._read_native_producer_release(command)
    followup = ResumeRequest(
        session_id=session.id, messages=[Message.text("user", "authorized follow-up")]
    )
    with pytest.raises(PermissionError, match="registered native handoff"):
        async for _ in app.resume(followup, context=CONTEXT):
            pass
    assert len(provider.requests) == 1

    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        snapshot = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            app._secret_redactor,
        )
    assert snapshot is not None
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-boundary-producer"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    final = await app.settle_producer_output(command, context=CONTEXT)
    assert await app.settle_producer_output(command, context=CONTEXT) == final
    assert len(provider.requests) == 1

    with pytest.raises(PermissionError):
        async for _ in app.resume(followup):
            pass
    assert len(provider.requests) == 1
    provider._batches = (
        *provider._batches,
        (ModelStreamEvent.text_delta("successor answer"), ModelStreamEvent.completed()),
        *((ModelStreamEvent.text_delta("queued answer"), ModelStreamEvent.completed()),) * queued,
    )
    events = [event async for event in app.resume(followup, context=CONTEXT)]
    from cayu.events import EventType

    assert any(event.type is EventType.SESSION_COMPLETED for event in events), [
        event.model_dump()
        for event in events
        if event.type in (EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED)
    ]
    # The explicit follow-up and (when present) the queued input each execute
    # once; neither is another invocation of the registered producer.
    assert len(provider.requests) == 2 + int(queued)
    assert await app.retain_producer_completion(command, context=CONTEXT) == completion
    assert await native_stores[1]._read_native_producer_release(command) == release
    assert await app.settle_producer_output(command, context=CONTEXT) == final
    if queued:
        records = await native_stores[1].inspect_session_messages(
            SessionMessageQuery(session_id=session.id)
        )
        assert len(records.records) == 1 and records.records[0].status.value == "delivered"
    if native_stores[3][0] != "memory":
        import asyncio
        import json
        import sys

        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.recovery.producer_completion_reader_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(
                child.communicate(
                    json.dumps(
                        {
                            "backend": native_stores[3][0],
                            "address": native_stores[3][1],
                            "expected": command.model_dump(mode="json"),
                            "successor": True,
                        }
                    ).encode()
                ),
                90,
            )
            assert child.returncode == 0, error.decode()
            assert json.loads(output) == {
                "completion": completion.model_dump(mode="json"),
                "release": release.model_dump(mode="json"),
                "cleanup": final.model_dump(mode="json"),
            }
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
    await native_stores[1].delete_session(session.id)


@pytest.mark.anyio
async def test_producer_expiry_is_checked_inside_native_admission(native_stores, monkeypatch):
    values = await output_scenario(native_stores, with_exports=True)
    app, resolver, admission, provider, session, initialized, command, execution = values
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    resolution = resolver.recipient.resolution
    actions = (*resolution.principal.actions, "execute")
    resolver.recipient.resolution = resolution.model_copy(
        update={
            "principal": resolution.principal.model_copy(update={"actions": actions}),
            "chain": resolution.chain.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"actions": actions})
                        for entry in resolution.chain.entries
                    ),
                }
            ),
        }
    )
    original = native.require_native_admission
    admit = native.admit_native_producer
    observed = []
    expired_at = datetime.fromtimestamp(command.limits.deadline_at_ms / 1000, UTC)

    async def expired_owner_clock(store, registration, lifecycle):
        # Advance only the receiving owner's transaction clock, not the worker
        # or the source election clock. The expiry must cross the real store API.
        with monkeypatch.context() as clock:
            if native_stores[3][0] == "postgres":

                async def now(cur):
                    return expired_at

                clock.setattr(native_stores[1], "_session_store_now", now)
            else:
                clock.setattr(native_stores[1], "_ownership_clock", lambda: expired_at)
            return await admit(store, registration, lifecycle)

    def after_election(checkpoint, lifecycle, *, now):
        # The source election has committed. Advance the receiving clock only
        # inside the native transaction, after its reads and lock acquisition.
        observed.append(True)
        assert now == expired_at
        assert datetime.now(UTC) < now
        return original(checkpoint, lifecycle, now=now)

    monkeypatch.setattr(native, "require_native_admission", after_election)
    monkeypatch.setattr(native, "admit_native_producer", expired_owner_clock)
    with pytest.raises(Exception):
        async for _ in app.execute_producer_output(
            command, execution, context=CONTEXT, producer_context=resolver.recipient.context
        ):
            pass
    assert observed and not provider.requests
    assert (await native_stores[1].load(session.id)).run_epoch == 0
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await native_stores[1].load_checkpoint(session.id)
    assert checkpoint[native.ROOT_KEY]["state"] == "prepared"
    pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    assert any(item.recovery.registration == command.operation for item in pending.items)
    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        snapshot = await retained_request(
            native_stores[0],
            tx,
            initialized,
            admission.expected.intent.request,
            admission.expected.initiator,
            app._secret_redactor,
        )
    assert snapshot is not None
    await app.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("close-expired-launch"),
            expected=admission.expected,
            expected_revision=snapshot.revision,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    pending = await app.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
    assert not any(item.recovery.registration == command.operation for item in pending.items)
    assert not provider.requests
    await native_stores[1].delete_session(session.id)
