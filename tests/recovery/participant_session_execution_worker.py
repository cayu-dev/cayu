"""Fresh-process participant-root execution fixture for SIGKILL recovery."""

from __future__ import annotations

import asyncio
import json
import sys

from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.agents import AgentSpec
from cayu.collaboration.participants import ParticipantRef
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import Message, RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


def _provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("hello"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )


def _application(session_store, collaboration_store, scope: str, provider) -> object:
    application = app(
        collaboration_store,
        registration(scope=scope),
        session_store=session_store,
    )
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model"))
    return application


async def _execute(application, execution, participant) -> None:
    async for _event in application.execute_participant_session(
        execution, participant=participant, context=CONTEXT
    ):
        pass


async def main() -> None:
    backend, session_address, collaboration_address, scope, phase = sys.argv[1:]
    if backend == "sqlite":
        session_store = SQLiteSessionStore(session_address)
        collaboration_store = SQLiteCollaborationStore(collaboration_address)
    elif backend == "postgres":
        session_store = PostgresSessionStore(session_address, schema_mode=SchemaMode.CREATE)
        collaboration_store = PostgresCollaborationStore(
            collaboration_address, schema_mode=SchemaMode.CREATE
        )
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")
    application = _application(session_store, collaboration_store, scope, _provider())
    initialized = await application.initialize_collaboration()
    if phase == "replay":
        payload = json.loads(sys.stdin.read())
        participant = ParticipantRef.model_validate(payload["participant"])
        raw_execution = payload["execution"]
        execution = ParticipantSessionExecutionRequest(
            request=RunRequest.model_validate(raw_execution["request"]),
            session_instance_id=raw_execution["session_instance_id"],
            execution_key=raw_execution["execution_key"],
        )
        await _execute(application, execution, participant)
        restored = await session_store.load(execution.request.session_id)
        assert restored is not None and restored.status.value == "completed"
        print(json.dumps({"status": restored.status.value}), flush=True)
        await session_store.close()
        await collaboration_store.close()
        return

    _, participant_receipt = await create(application, initialized, key="sigkill-execution")
    participant = participant_receipt.participants[0].reference
    creation = ParticipantSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
        creation_key="sigkill-execution-create",
    )
    session, _ = await application.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    execution = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="sigkill-execution-run",
    )
    if phase != "admission":
        raise RuntimeError(f"Unknown phase: {phase}")
    original_apply = application._runtime_session_store.apply_invocation_lifecycle_command
    payload = {
        "participant": participant.model_dump(mode="json"),
        "execution": {
            "request": execution.request.model_dump(mode="json"),
            "session_instance_id": execution.session_instance_id,
            "execution_key": execution.execution_key,
        },
    }
    held = True

    async def commit_then_hold(command):
        nonlocal held
        result = await original_apply(command)
        if held:
            held = False
            print(json.dumps(payload), flush=True)
            await asyncio.Event().wait()
        return result

    application._runtime_session_store.apply_invocation_lifecycle_command = commit_then_hold
    await _execute(application, execution, participant)


if __name__ == "__main__":
    asyncio.run(main())
