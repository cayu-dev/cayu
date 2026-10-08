"""Secret-safe invocation publication and runner evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from cayu._exception_groups import (
    iter_exception_tree,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.execution_profiles import (
    event_with_execution_profile_authority,
)
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_writer import RuntimeEventWriter, prepare_runtime_event
from cayu.runtime._tool_invocation.resources import ToolInvocationCall
from cayu.runtime._tool_round_staging import (
    _event_with_tool_round_authority,
)
from cayu.sessions.base import (
    SessionStore,
)
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.vaults.redaction import SecretRedactor


class InvocationPublication:
    """Record one final publication scope before exposing invocation evidence."""

    def __init__(
        self,
        *,
        tool_call: runtime_records.ToolCallRequest,
        redactor: SecretRedactor,
        observer: Callable[[str, invocation_secrets.InvocationPublicationSnapshot], Awaitable[None]]
        | None,
    ) -> None:
        self._tool_call = tool_call
        self._redactor = redactor
        self._observer = observer
        self._recorded = False

    async def record(
        self,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        if self._recorded:
            return
        if self._observer is not None:
            await self._observer(self._tool_call.id, snapshot)
        self._recorded = True

    async def static_scope(
        self,
    ) -> invocation_secrets.InvocationPublicationSnapshot:
        snapshot = invocation_secrets.InvocationPublicationSnapshot(
            redactor=self._redactor,
            unsafe_output=False,
        )
        await self.record(snapshot)
        return snapshot


class InvocationEvidence:
    """Retain runner evidence until the invocation secret scope is sealed."""

    def __init__(
        self,
        *,
        call: ToolInvocationCall,
        secret_scope: invocation_secrets.InvocationSecretTracker,
        publication: InvocationPublication,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        resolved_redactor_observer: Callable[[str, InvocationRedactorSnapshot], Awaitable[None]]
        | None,
    ) -> None:
        self._call = call
        self._secret_scope = secret_scope
        self._publication = publication
        self._session_store = session_store
        self._event_writer = event_writer
        self._resolved_redactor_observer = resolved_redactor_observer
        self._staged_runner_completion_events: list[tuple[Event, int]] = []
        self.runner_events: list[Event] = []

    def redactor(self) -> SecretRedactor:
        return self._secret_scope.redactor

    async def observe_runner(
        self,
        phase: Literal["started", "completed"],
        payload: dict[str, Any],
        command_evidence_revision: int,
    ) -> None:
        if type(command_evidence_revision) is not int or command_evidence_revision < 0:
            raise TypeError("Runner command evidence revision must be a non-negative integer.")
        event = Event(
            type=(
                EventType.RUNNER_EXEC_STARTED
                if phase == "started"
                else EventType.RUNNER_EXEC_COMPLETED
            ),
            session_id=self._call.session.id,
            agent_name=self._call.registered_agent.spec.name,
            environment_name=self._call.environment_name,
            tool_name=self._call.tool_call.name,
            payload={
                **payload,
                "tool_call_id": self._call.tool_call.id,
                "idempotency_key": self._call.idempotency_key,
                **self._call.tool_round_identity.payload(),
                **(
                    {"approval_id": self._call.approval_id}
                    if self._call.approval_id is not None
                    else {}
                ),
                **({"input_id": self._call.input_id} if self._call.input_id is not None else {}),
            },
        )
        event = event_with_execution_profile_authority(event, self._call.execution_profile)
        event = _event_with_tool_round_authority(
            event,
            self._call.tool_round_identity,
            *(field for field in ("approval_id", "input_id") if field in event.payload),
        )

        command = event.payload.get("command")
        if not isinstance(command, dict) or command.get("kind") not in {
            "process",
            "shell",
        }:
            raise RuntimeError("Runner command evidence is malformed.")
        if phase == "started":
            # A command must not cross the runner boundary until its exact
            # invocation/profile linkage is durable. Command arguments can
            # become secret only after dispatch, so the pre-dispatch record
            # is deliberately content-free instead of being held in memory.
            event = event.model_copy(
                update={
                    "payload": {
                        **event.payload,
                        "command": {
                            "kind": command["kind"],
                            "arguments_state": "unavailable",
                        },
                    }
                },
                deep=True,
            )
            self.runner_events.append(
                await self._event_writer.emit(
                    prepare_runtime_event(
                        event,
                        redactor=self._secret_scope.redactor,
                    )
                )
            )
            return
        self._staged_runner_completion_events.append((event, command_evidence_revision))

    async def publish_runner(
        self,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        """Publish runner completion detail after late secret registration closes."""

        if not self._staged_runner_completion_events:
            return
        final_revision = self._secret_scope.snapshot().revision
        captured_events = self._staged_runner_completion_events
        self._staged_runner_completion_events = []
        prepared_events: list[Event] = []
        for event, command_evidence_revision in captured_events:
            if snapshot.secret_scope_incomplete or command_evidence_revision != final_revision:
                command = event.payload.get("command")
                if not isinstance(command, dict) or command.get("kind") not in {
                    "process",
                    "shell",
                }:
                    raise RuntimeError("Runner command evidence is malformed.")
                event = event.model_copy(
                    update={
                        "payload": {
                            **event.payload,
                            "command": {
                                "kind": command["kind"],
                                "arguments_state": "unavailable",
                            },
                        }
                    },
                    deep=True,
                )
            prepared_events.append(prepare_runtime_event(event, redactor=snapshot.redactor))
        captured_events.clear()
        for event in prepared_events:
            self.runner_events.append(await self._event_writer.emit(event))

    async def persist_sealed(
        self,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        await self._publication.record(snapshot)
        await self.publish_runner(snapshot)

    async def persist_resolved(
        self,
        snapshot: InvocationRedactorSnapshot,
    ) -> None:
        if self._resolved_redactor_observer is not None:
            await self._resolved_redactor_observer(self._call.tool_call.id, snapshot)
            return
        await self._session_store.transform_checkpoint(
            self._call.session.id,
            tool_round_recovery.assistant_publication_redactor_transform(
                tool_round_identity=self._call.tool_round_identity,
                tool_call_id=self._call.tool_call.id,
                redactor=snapshot.redactor,
            ),
        )

    async def persist_interrupted(
        self,
        interrupt: BaseException,
    ) -> bool:
        """Persist the final scope without replacing an owned interrupt."""

        def record_abandoned_publication_diagnostic(error: BaseException) -> None:
            if type(error) is GeneratorExit:
                error_type = "GeneratorExit"
            elif isinstance(error, BaseExceptionGroup):
                error_type = "BaseExceptionGroup"
            elif isinstance(error, asyncio.CancelledError):
                error_type = "CancelledError"
            else:
                # Exception subclass names and messages are extension-owned
                # and can contain workload-derived values. Keep only a fixed
                # classification at this public cancellation boundary.
                error_type = "BaseException"
            note = (
                "Assistant publication projection terminated with "
                f"{error_type} while preserving cancellation."
            )
            interrupt.add_note(note)
            if isinstance(interrupt, BaseExceptionGroup):
                for candidate in iter_exception_tree(interrupt):
                    if isinstance(candidate, asyncio.CancelledError):
                        candidate.add_note(note)

        publication_task = asyncio.create_task(
            self.persist_sealed(self._secret_scope.seal_for_publication())
        )
        publication_outcome = await await_shielded_task_outcome(publication_task)
        publication_error = publication_outcome.error
        if publication_error is not None and not isinstance(publication_error, Exception):
            if type(publication_error) in (KeyboardInterrupt, SystemExit):
                # Process-level interpreter signals retain their ordinary
                # semantics; this issue does not redefine process shutdown.
                raise publication_error
            # Caller cancellation remains authoritative over task-contained
            # snapshot abandonment. Retain only a fixed diagnostic because
            # raw extension-owned exception state is unsafe to publish.
            record_abandoned_publication_diagnostic(publication_error)
        if publication_error is not None and isinstance(publication_error, Exception):
            interrupt.add_note(
                "Failed to persist the assistant publication projection while "
                "preserving cancellation."
            )
        later_cancellation = publication_outcome.cancellation
        if later_cancellation is None:
            return False
        current_task = asyncio.current_task()
        if current_task is None:  # pragma: no cover - coroutine execution invariant
            interrupt.add_note("A later cancellation could not be redelivered after publication.")
            return False
        cancellation_args = later_cancellation.args
        if not cancellation_args:
            current_task.cancel()
        else:
            current_task.cancel(cancellation_args[0])
        return True
