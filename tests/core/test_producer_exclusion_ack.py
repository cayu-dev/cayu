"""Pre-launch exclusion stays discoverable until native retention is released."""

import asyncio
import json
import sys

import pytest
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import app as make_app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration._producer_cleanup_finalization import ProducerCleanupFinalized
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.sessions import RunRequest, SessionIdentity


@pytest.mark.anyio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_excluded_producer_cleanup_reopens_after_native_ack_loss(
    native_stores, monkeypatch, after_commit
):
    (
        app,
        resolver,
        admission,
        provider,
        session,
        initialized,
        command,
        execution,
    ) = await output_scenario(native_stores)
    monkeypatch.setattr(app._request_coordinator._owners, "observation_timeout", 60)
    await app.register_producer_output(command, execution, context=resolver.recipient.context)
    prior = await app.inspect_collaboration_request(
        admission.expected, context=resolver.sender.context
    )
    control = RequestControl(
        operation=initialized.operation("exclude-before-native-ack"),
        expected=admission.expected,
        expected_revision=prior.revision,
        kind="cancel",
    )
    receiver = app._request_coordinator._registration.receiving_owner
    acknowledge = receiver._acknowledge_producer_cleanup

    async def lose_ack(*args):
        if after_commit:
            await acknowledge(*args)
        raise ConnectionError("Native cleanup acknowledgement unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(receiver, "_acknowledge_producer_cleanup", lose_ack)
        with pytest.raises(CollaborationUnavailable):
            await app.control_collaboration_request(control, context=resolver.sender.context)
    other = make_app(
        native_stores[2](),
        app._participant_coordinator._registration,
        session_store=native_stores[1],
        collaboration_requests=app._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    monkeypatch.setattr(other._request_coordinator._owners, "observation_timeout", 60)
    try:
        pending = await other.pending_producer_outputs(
            admission.prepared.recipient, context=CONTEXT
        )
        assert any(item.recovery.registration == command.operation for item in pending.items)
        participant = await other.inspect_participant(admission.prepared.recipient, context=CONTEXT)
        assert participant.outstanding_obligations >= 2
        if after_commit:
            await native_stores[1].delete_session(session.id)
            replacement = await native_stores[1].create(
                RunRequest(session_id=session.id, agent_name="unrelated", messages=[]),
                identity=SessionIdentity(provider_name="unused", model="unused"),
            )
            assert replacement.instance_id != session.instance_id
        else:
            with pytest.raises(ValueError, match="producer output responsibility"):
                await native_stores[1].delete_session(session.id)
        if native_stores[3][0] == "memory":
            finalized = await other.settle_producer_output(command, context=CONTEXT)
        else:
            finalized = await _recover_in_fresh_process(native_stores[3], command)
        assert finalized.delivery == "excluded"
        assert await other.settle_producer_output(command, context=CONTEXT) == finalized
        assert not (
            await other.pending_producer_outputs(admission.prepared.recipient, context=CONTEXT)
        ).items
        assert not provider.requests
        if after_commit:
            assert await native_stores[1].load(session.id) == replacement
        if not after_commit:
            await native_stores[1].delete_session(session.id)
        assert await other.control_collaboration_request(control, context=resolver.sender.context)
    finally:
        await other.drain_collaboration_requests()


async def _recover_in_fresh_process(backend, command):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.producer_completion_reader_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, error = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    {
                        "backend": backend[0],
                        "address": backend[1],
                        "expected": command.model_dump(mode="json"),
                        "cleanup": True,
                        "excluded": True,
                    }
                ).encode()
            ),
            90,
        )
        assert process.returncode == 0, error.decode()
        return ProducerCleanupFinalized.model_validate_json(output)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
