"""Fresh-process native readback and continuation rejection after human closure."""

import asyncio
import json
import sys
from pathlib import Path

from cayu import ProducerOutputRegistration
from cayu.runtime._invocation_lifecycle import prepare_rebind_invocation_command
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions.base import (
    SessionRunFenced,
    SessionStatus,
    _invocation_lifecycle_authority_read_scope,
)
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    material = json.loads(sys.stdin.read())
    command = ProducerOutputRegistration.model_validate(material["command"])
    store = (
        SQLiteSessionStore(Path(material["address"]).with_name("sessions.sqlite"))
        if material["backend"] == "sqlite"
        else PostgresSessionStore(material["address"])
    )
    try:
        output = await store._read_retained_native_producer_output(command)
        assert output.disposition == "stopped"
        sid = command.admission.prepared.target.session_id
        session = await store.load(sid)
        with _invocation_lifecycle_authority_read_scope():
            checkpoint = await store.load_checkpoint(sid)
        active = active_invocation_execution_profile_from_checkpoint(checkpoint)

        def next_epoch(session, checkpoint):
            return checkpoint_with_active_invocation_execution_profile(
                checkpoint,
                session_id=sid,
                interaction_id=active.interaction_id,
                run_epoch=session.run_epoch + 1,
                profile=active.profile,
                expected=active,
            )

        rebind = prepare_rebind_invocation_command(
            session,
            checkpoint,
            expected_statuses={SessionStatus.INTERRUPTED},
            checkpoint_transform=next_epoch,
            target_status=SessionStatus.RUNNING,
        )
        try:
            await store.apply_invocation_lifecycle_command(rebind)
        except SessionRunFenced as error:
            assert "durably stopped" in str(error)
        else:
            raise AssertionError("Stopped producer regained execution after restart")
        assert await store.load(sid) == session
        with _invocation_lifecycle_authority_read_scope():
            assert await store.load_checkpoint(sid) == checkpoint
        assert await store._read_retained_native_producer_output(command) == output
        print("paused-stop-reconstructed", flush=True)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
