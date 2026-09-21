from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    interrupt_and_release_test_invocation,
    runtime_interaction_started_event,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._contracts import (
    CollaborationContractError,
    ExpectedOperation,
    HandoffIntent,
    HandoffSlot,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._invocation_lifecycle import (
    AdmitInvocationCommand,
    AdmittedInvocationBinding,
    InvocationCheckpointPatch,
    InvocationMutationResult,
    PreparedInvocationBinding,
    _authenticated_invocation_context,
    invocation_checkpoint_state_sha256,
    runtime_publication_checkpoint_mutation,
)
from cayu.runtime._session_continuation import (
    ContinuationConflict,
    ContinuationConsumption,
    ContinuationLatch,
    ContinuationNamespace,
    ContinuationPreparation,
    ContinuationRecord,
    ContinuationRetirement,
    ContinuationService,
    ContinuationTicket,
    ContinuationUnavailable,
    ContinuationWait,
    admit_continuation,
    continuation_admission_digest,
    continuation_admission_inputs,
    continuation_namespace_id,
    continuation_operation_key,
    continuation_registration_operation,
)
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.runtime._session_continuation_scope import consumption_scope, preparation_scope
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
)
from cayu.sessions.base import (
    ForkSessionRequest,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    Session,
    SessionOperationPublication,
    SessionStatus,
    SessionStore,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


class _QualifiedReceiver:
    async def authenticate_continuation_latch(self, latch):
        return latch


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "terminal",
    ["consumed", "retired", "superseded", "prepared", "latched-inline", "latched-queued"],
)
def test_continuation_history_forks_and_overtaken_wait_cleanup(
    backend, terminal, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("continuation_postgres_dsn") if backend == "postgres" else None

    async def run():
        def open_store():
            if backend == "memory":
                return InMemorySessionStore()
            if backend == "sqlite":
                return SQLiteSessionStore(tmp_path / "fork-continuations.sqlite")
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            return PostgresSessionStore(dsn, schema_mode=SchemaMode.VALIDATE)

        store = open_store()

        def runtime():
            provider = ScriptedModelProvider(
                (ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})),
                name="continuation-provider",
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="continuation-model"))
            return app, provider

        def receiving_owner(ticket):
            return SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )

        try:
            app, provider = runtime()
            latched_takeover = terminal.startswith("latched-")
            session_id = f"history-{backend}-{terminal}"
            admitted = await create_admitted_session(
                store,
                app=app,
                request=RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "wait")],
                ),
                provider_name=provider.name,
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            invocation = await _context(store, session_id)
            owner = receiving_owner(ticket)
            await owner.prepare(
                ContinuationWait.model_validate(
                    ticket.model_dump(include=set(ContinuationWait.model_fields))
                ),
                invocation=invocation,
            )
            waiting = await owner.park(ticket, invocation=invocation)
            retirement = ContinuationRetirement(
                ticket=waiting.ticket,
                control_id="history-retirement",
                reason=(
                    "superseded"
                    if terminal in {"superseded", "prepared"} or latched_takeover
                    else "cancelled"
                ),
                retired_at=datetime.now(UTC).isoformat(),
            )
            latch = ContinuationLatch(
                ticket=ticket,
                wait_receipt_digest="wait-receipt",
                outcome_kind="success",
                selected_manifest=(),
                disclosure_digest="disclosure",
                latch_key="history-latch",
                outcome_digest="outcome",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            if terminal == "retired":
                await owner.retire(retirement, invocation=invocation)
            if terminal == "superseded":
                with pytest.raises(ContinuationConflict):
                    await owner.retire(retirement)
            if terminal == "prepared":
                latched = await owner.latch(latch)
                assert latched.latch is not None
                await _consume(store, _consumption(waiting.ticket, latched.latch, "pending"))
            if latched_takeover:
                await owner.latch(latch)
            await interrupt_and_release_test_invocation(store, session_id)
            resume = ResumeRequest(session_id=session_id, messages=[Message.text("user", "go")])
            if terminal == "consumed":
                latched = await owner.latch(latch)
                assert latched.latch is not None
                settled = await owner.service(
                    app,
                    resume,
                    ContinuationService(
                        ticket=latched.ticket,
                        latch=latched.latch,
                        continuation_id="history-resume",
                        mode="inline",
                        accepted_at=datetime.now(UTC).isoformat(),
                    ),
                )
                assert settled.ticket.state == "CONSUMED"
            else:
                if terminal == "superseded":
                    with pytest.raises(ContinuationConflict):
                        await owner.retire(retirement)
                events = [event async for event in app.resume(resume)]
                assert events[-1].type is EventType.SESSION_COMPLETED
            await owner.drain()
            if backend != "memory":
                await store.close()
                store = open_store()
                app, provider = runtime()
            owner = receiving_owner(ticket)
            if latched_takeover:
                before = await store.load_continuation_ticket(
                    session_id,
                    registration_key=ticket.registration_key,
                    session_instance_id=ticket.session_instance_id,
                )
                assert before is not None and before.latch is not None
                assert before.consumption is None
                session_before = await store.load(session_id)
                requests_before = len(provider.requests)
                stale_service = ContinuationService(
                    ticket=before.ticket,
                    latch=before.latch,
                    continuation_id="stale-resume",
                    mode="inline" if terminal == "latched-inline" else "queued",
                    accepted_at=datetime.now(UTC).isoformat(),
                )
                for _ in range(2):
                    with pytest.raises(ContinuationConflict) as error:
                        await owner.service(app, resume, stale_service)
                    failure = error.value
                    diagnostics = []
                    while failure is not None:
                        diagnostics.append(str(failure))
                        failure = failure.__cause__
                    assert any("stale writer generation" in value for value in diagnostics)
                    assert (
                        await store.load_continuation_ticket(
                            session_id,
                            registration_key=ticket.registration_key,
                            session_instance_id=ticket.session_instance_id,
                        )
                        == before
                    )
                    assert (await store.load(session_id)).run_epoch == session_before.run_epoch
                    assert len(provider.requests) == requests_before
                retired = await owner.retire(retirement)
                assert retired.ticket.state == "RETIRED" and retired.consumption is None
                await store.validate_session_closure_admission(session_id)
                await store.delete_session(session_id)
                await owner.drain()
                return
            if terminal == "prepared":
                before = await store.load_continuation_ticket(
                    session_id,
                    registration_key=ticket.registration_key,
                    session_instance_id=ticket.session_instance_id,
                )
                with pytest.raises(ContinuationConflict):
                    await owner.retire(retirement)
                assert (
                    await store.load_continuation_ticket(
                        session_id,
                        registration_key=ticket.registration_key,
                        session_instance_id=ticket.session_instance_id,
                    )
                    == before
                )
                await owner.drain()
                return
            if terminal == "superseded":
                with pytest.raises(ContinuationConflict):
                    await owner.latch(latch)
                with pytest.raises(ContinuationConflict):
                    await store.validate_session_closure_admission(session_id)
                with pytest.raises(PermissionError):
                    await store.retire_continuation(retirement)
                with pytest.raises(PermissionError):
                    await owner.retire(retirement.model_copy(update={"reason": "cancelled"}))
                publish = store.publish_session_operation

                async def lose_ack(session_id, **kwargs):
                    await publish(session_id, **kwargs)
                    raise RuntimeError("supersession committed before acknowledgement loss")

                with monkeypatch.context() as patch:
                    patch.setattr(store, "publish_session_operation", lose_ack)
                    with pytest.raises(ContinuationUnavailable):
                        await owner.retire(retirement)
                retired = await owner.retire(retirement)
                assert retired.ticket.state == "RETIRED"
                assert retired.consumption is None
                assert retired.latch is None
                assert await owner.retire(retirement) == retired
                with pytest.raises(ContinuationConflict):
                    await owner.retire(retirement.model_copy(update={"control_id": "changed"}))
                with pytest.raises(ContinuationConflict):
                    await owner.latch(latch)
                await store.validate_session_closure_admission(session_id)
                await store.delete_session(session_id)
                assert await store.load(session_id) is None
            else:
                child_id = session_id + "-child"
                await store.validate_session_closure_admission(session_id)
                original_root = (await store.load_checkpoint(session_id))["session_continuations"]
                with pytest.raises(ContinuationConflict):
                    await store.transform_checkpoint(
                        session_id, lambda _session, _checkpoint: {"session_continuations": {}}
                    )
                assert (await store.load_checkpoint(session_id))[
                    "session_continuations"
                ] == original_root
                events = [
                    event
                    async for event in app.fork_session(
                        ForkSessionRequest(
                            source_session_id=session_id, session_id=child_id, copy_checkpoint=True
                        )
                    )
                ]
                assert events
                child = await store.load(child_id)
                assert child is not None
                assert "session_continuations" not in (await store.load_checkpoint(child_id) or {})
                assert (
                    await store.load_continuation_ticket(
                        child_id,
                        registration_key=ticket.registration_key,
                        session_instance_id=child.instance_id,
                    )
                    is None
                )
                assert "session_continuations" in (await store.load_checkpoint(session_id) or {})
            await owner.drain()
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


async def _consume(store, consumption):
    # White-box store transition characterization. Owner/gate acceptance tests
    # below use SessionContinuationOwner and the actual runtime admission path.
    with consumption_scope(consumption):
        return await store.consume_continuation(consumption)


async def _admit_fixture_continuation(store, consumption, command):
    with consumption_scope(consumption):
        return await admit_continuation(store, consumption, command)


async def _latch(store, latch, *, receiver):
    owner = SessionContinuationOwner(
        store=store,
        owner=latch.ticket.owner,
        receiver=receiver,
        receiver_capability=CapabilityDescriptor(
            owner=latch.ticket.owner,
            mutations=(),
            readbacks=(LATCH_FAMILY,),
        ),
        redactor=SecretRedactor(),
    )
    return await owner.latch(latch)


async def _park(store, ticket):
    owner = SessionContinuationOwner(
        store=store,
        owner=ticket.owner,
        receiver=_QualifiedReceiver(),
        receiver_capability=CapabilityDescriptor(
            owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=SecretRedactor(),
    )
    return await owner.park(ticket, invocation=await _context(store, ticket.session_id))


async def _retire(store, retirement, *, invocation=None):
    owner = SessionContinuationOwner(
        store=store,
        owner=retirement.ticket.owner,
        receiver=_QualifiedReceiver(),
        receiver_capability=CapabilityDescriptor(
            owner=retirement.ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
        ),
        redactor=SecretRedactor(),
    )
    return await owner.retire(
        retirement,
        invocation=(
            await _context(store, retirement.ticket.session_id)
            if invocation is None
            else invocation
        ),
    )


class _AcknowledgementLostStore(InMemorySessionStore):
    session_continuation_version = 1
    invocation_lifecycle_command_version = 1
    lose_next_publication = False

    async def publish_session_operation(self, session_id: str, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        if self.lose_next_publication:
            self.lose_next_publication = False
            raise RuntimeError("simulated continuation acknowledgement loss")
        return result


@pytest.fixture(scope="module")
def continuation_postgres_dsn(postgres_dsn) -> Iterator[str]:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def reset_schema() -> None:
        import psycopg
        from psycopg import sql

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT tablename FROM pg_catalog.pg_tables "
                    "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
                )
                for (table_name,) in await cursor.fetchall():
                    await cursor.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table_name)
                        )
                    )
            await connection.commit()
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.ensure_schema()
        finally:
            await store.close()

    asyncio.run(reset_schema())
    yield postgres_dsn


def _ticket(
    session,
    interaction_id: str,
    *,
    registration_key: str = "wait-1",
) -> ContinuationTicket:
    owner = OwnerRef(
        application_scope="tests",
        owner_id=session.agent_name,
        incarnation=session.instance_id,
    )
    namespace = ContinuationNamespace(
        session_id=session.id,
        session_instance_id=session.instance_id,
        owner=owner,
        namespace_id=continuation_namespace_id(session.id, session.instance_id, owner),
        generation=1,
    )
    target = ObjectRef(
        owner=owner,
        kind="fixture",
        object_id="target-1",
        incarnation="target-1",
        revision=1,
    )
    return ContinuationTicket(
        namespace=namespace,
        session_id=session.id,
        session_instance_id=session.instance_id,
        owner=owner,
        registration_key=registration_key,
        targets=(target,),
        predicate_kind="ANY_SUCCESS",
        predicate_version=1,
        deadline="2030-01-01T00:00:00+00:00",
        failure_policy="unavailable",
        service_policy="none",
        wait_edge_revision=1,
        interaction_id=interaction_id,
        writer_generation=session.run_epoch,
        purpose="fixture continuation",
        state="ARMING",
        revision=1,
    )


def test_wait_intent_rejects_keys_that_cannot_fit_a_durable_ticket() -> None:
    owner = OwnerRef(application_scope="tests", owner_id="owner", incarnation="one")
    target = ObjectRef(
        owner=owner,
        kind="fixture",
        object_id="target-1",
        incarnation="target-1",
        revision=1,
    )

    with pytest.raises(ValueError):
        ContinuationWait(
            registration_key="x" * 257,
            targets=(target,),
            predicate_kind="ANY_SUCCESS",
            predicate_version=1,
            deadline="2030-01-01T00:00:00+00:00",
            failure_policy="unavailable",
            service_policy="none",
            wait_edge_revision=1,
            purpose="fixture continuation",
        )


async def _preparation(store, ticket: ContinuationTicket) -> ContinuationPreparation:
    session = await store.load(ticket.session_id)
    assert session is not None
    operation = OperationRef(
        application_scope=ticket.owner.application_scope,
        namespace_incarnation=ticket.namespace.namespace_id,
        generation=ticket.namespace.generation,
        caller_key=ticket.registration_key,
    )
    initiator = InitiatorBinding(
        issuer=ticket.owner,
        principal=ticket.owner.owner_id,
        participant=None,
        mandate=None,
        invocation_id=session.invocation.root_invocation_id,
        interaction_id=ticket.interaction_id,
    )
    return ContinuationPreparation(
        operation=operation,
        source=ticket.owner,
        destination=ticket.owner,
        initiator=initiator,
        registration=HandoffIntent[ContinuationTicket](
            slot=HandoffSlot(source=ticket.owner, parent=operation, slot="wait-registration"),
            child=ExpectedOperation[ContinuationTicket](
                operation=continuation_registration_operation(operation),
                kind="wait.register",
                schema_version=1,
                mode="wait",
                source=ticket.owner,
                destination=ticket.owner,
                initiator=initiator,
                receipt_stage="registered",
                intent=ticket,
            ),
        ),
        intent=ticket,
    )


async def _context(store, session_id, *, admission=None, app=None):
    session = await store.load(session_id)
    assert session is not None
    active = active_invocation_execution_profile_from_checkpoint(
        await runtime_checkpoint_session_store(store).load_checkpoint(session_id)
    )
    assert active is not None
    if admission is not None:
        active = admission.target_active_profile
    binding_type = AdmittedInvocationBinding if admission is None else PreparedInvocationBinding
    if app is None:
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider((), name=session.provider_name), default=True)
        app.register_agent(AgentSpec(name=session.agent_name, model=session.model))
    return _authenticated_invocation_context(
        active_profile=active,
        binding=binding_type(
            session_id=session.id,
            session_instance_id=session.instance_id,
            interaction_id=active.interaction_id,
            run_epoch=active.run_epoch,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            runtime_build_provenance=session.runtime_build_provenance,
            environment_name=session.environment_name,
        ),
        validated_profile=active.profile,
        registered_agent=app._agents[session.agent_name],
        registered_provider=app._providers[session.provider_name],
        registered_environment=None,
        runtime_hooks=(),
        loop_policies=(),
        request_loop_policies=(),
        budget_policy=None,
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
    )


async def _prepare_command(store, command):
    invocation = await _context(store, command.intent.session_id)
    with preparation_scope(command, invocation):
        return await store.prepare_continuation_ticket(command)


async def _bootstrap(store, ticket):
    command = await _preparation(store, ticket)
    invocation = await _context(store, ticket.session_id)
    with preparation_scope(command, invocation):
        return await store._initialize_continuation_namespace(ticket.namespace)


async def _admitted_store() -> tuple[InMemorySessionStore, Session, str]:
    store = InMemorySessionStore()
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id="continuation-session",
            messages=[Message.text("user", "wait")],
        ),
        provider_name="continuation-provider",
        model="continuation-model",
    )
    return store, admitted.session, admitted.active_invocation_profile.interaction_id


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registered_owner_prepares_from_runtime_context(backend, tmp_path) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "owner-preparation.sqlite")
        )
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="owner-preparation",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            invocation = await _context(store, admitted.session.id)
            owner = SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            intent = ContinuationWait.model_validate(
                ticket.model_dump(include=set(ContinuationWait.model_fields))
            )
            record = await owner.prepare(intent, invocation=invocation)
            assert record.ticket == ticket
            assert record.preparation.initiator.invocation_id == (
                admitted.session.invocation.root_invocation_id
            )
            # Identical public data does not carry the owner's private provenance.
            with pytest.raises(PermissionError):
                await store.prepare_continuation_ticket(record.preparation)
            await interrupt_and_release_test_invocation(store, admitted.session.id)
            assert await owner.prepare(intent, invocation=invocation) == record
            with pytest.raises(PermissionError):
                await owner.prepare(intent, invocation=object())  # ty: ignore[invalid-argument-type]
            await owner.drain()
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


async def _test_early_latch_survives_arm_wait_and_exact_consumption() -> None:
    store, session, interaction_id = await _admitted_store()
    ticket = _ticket(session, interaction_id)
    armed = await _prepare_command(store, await _preparation(store, ticket))
    assert armed.ticket.state == "ARMING"

    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    await _latch(store, latch, receiver=_QualifiedReceiver())
    with pytest.raises(PermissionError):
        await store.mark_continuation_waiting(ticket)
    waiting = await _park(store, ticket)
    assert waiting.ticket.state == "WAITING"
    assert waiting.latch is not None

    prepared = ContinuationConsumption(
        ticket=waiting.ticket,
        latch=waiting.latch,
        continuation_id="continuation-1",
        mode="inline",
        input_digest="input",
        profile_digest="profile",
        budget_digest="budget",
        admission_command_digest="command",
        admission_expected_run_epoch=ticket.writer_generation,
        receipt_stage="prepared",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    retained = await _consume(store, prepared)
    assert retained.ticket.state == "WAITING"
    assert retained.consumption == prepared

    with pytest.raises(ContinuationConflict):
        await _consume(store, prepared.model_copy(update={"receipt_stage": "admitted"}))


def test_early_latch_survives_arm_wait_and_exact_consumption() -> None:
    asyncio.run(_test_early_latch_survives_arm_wait_and_exact_consumption())


async def _test_fixed_key_changes_conflict_and_retirement_is_exact() -> None:
    store, session, interaction_id = await _admitted_store()
    ticket = _ticket(session, interaction_id)
    await _prepare_command(store, await _preparation(store, ticket))
    changed = ticket.model_copy(update={"purpose": "different"})
    with pytest.raises(ContinuationConflict):
        await _prepare_command(store, await _preparation(store, changed))

    current = await store.load_continuation_ticket(
        session.id,
        registration_key=ticket.registration_key,
        session_instance_id=session.instance_id,
    )
    assert current is not None
    retirement = ContinuationRetirement(
        ticket=current.ticket,
        control_id="retire-1",
        reason="cancelled",
        retired_at=datetime.now(UTC).isoformat(),
    )
    with pytest.raises(PermissionError):
        await store.retire_continuation(retirement)
    retired = await _retire(store, retirement)
    assert retired.ticket.state == "RETIRED"
    assert await _retire(store, retirement) == retired

    with pytest.raises(ContinuationConflict):
        await _park(store, ticket)


def test_fixed_key_changes_conflict_and_retirement_is_exact() -> None:
    asyncio.run(_test_fixed_key_changes_conflict_and_retirement_is_exact())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_retirement_owner_retains_publication_through_cancellation(backend, tmp_path) -> None:
    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        class DelayedPublication(SessionStore):
            delay = False
            dispatches = 0

            async def publish_session_operation(self, session_id, **kwargs):
                if self.delay:
                    self.dispatches += 1
                    entered.set()
                    await release.wait()
                return await super().publish_session_operation(session_id, **kwargs)

        class Memory(DelayedPublication, InMemorySessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        class SQLite(DelayedPublication, SQLiteSessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        store = Memory() if backend == "memory" else SQLite(tmp_path / "retirement.sqlite")
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="retirement-cancellation",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            await _prepare_command(store, await _preparation(store, ticket))
            invocation = await _context(store, ticket.session_id)
            owner = SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            retirement = ContinuationRetirement(
                ticket=ticket,
                control_id="explicit-retirement",
                reason="cancelled",
                retired_at=datetime.now(UTC).isoformat(),
            )
            store.delay = True
            task = asyncio.create_task(owner.retire(retirement, invocation=invocation))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel("private cancellation message")
            task.cancel("second cancellation")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled() and task.cancelling() == 2
            assert caught.value.__context__ is None
            assert "private cancellation message" not in str(caught.value)
            assert len(owner.owners.pending) == 1
            current = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert current is not None and current.retirement is None
            owner.owners.observation_timeout = 0.01
            from cayu.runtime._session_continuation import ContinuationUnavailable

            with pytest.raises(ContinuationUnavailable):
                await owner.retire(retirement, invocation=invocation)
            assert store.dispatches == 1
            release.set()
            owner.owners.observation_timeout = 5
            await owner.drain()
            settled = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert settled is not None and settled.ticket.state == "RETIRED"
            assert settled.retirement is not None
            assert settled.retirement.control_id == retirement.control_id
            assert store.dispatches == 1
        finally:
            release.set()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_preparation_exact_lookup_and_initiator_replay(backend, tmp_path) -> None:
    async def run() -> None:
        class ReadFailure(SessionStore):
            read_failure = None

            async def load_session_operation(
                self, session_id, idempotency_key, *, checkpoint_root_guard=None
            ):
                if self.read_failure == "malformed":
                    return {"schema_version": False}
                if self.read_failure == "unavailable":
                    raise OSError("dependency unavailable")
                if self.read_failure == "denied":
                    raise PermissionError("denied")
                return await super().load_session_operation(
                    session_id, idempotency_key, checkpoint_root_guard=checkpoint_root_guard
                )

        class Memory(ReadFailure, InMemorySessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        class SQLite(ReadFailure, SQLiteSessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        store = Memory() if backend == "memory" else SQLite(tmp_path / "lookup.sqlite")
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="lookup-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            command = await _preparation(store, ticket)
            assert (await store.lookup_continuation_ticket(command)).status == "not_found"
            record = await _prepare_command(store, command)
            matched = await store.lookup_continuation_ticket(command)
            assert matched.status == "match" and matched.receipt == record
            changed_initiator = command.initiator.model_copy(update={"principal": "other-owner"})
            changed = command.model_copy(
                update={
                    "initiator": changed_initiator,
                    "registration": command.registration.model_copy(
                        update={
                            "child": command.registration.child.model_copy(
                                update={
                                    "initiator": changed_initiator,
                                }
                            ),
                        }
                    ),
                }
            )
            assert (await store.lookup_continuation_ticket(changed)).status == "conflict"
            with pytest.raises(ContinuationConflict):
                await _prepare_command(store, changed)
            for failure in ("malformed", "unavailable"):
                store.read_failure = failure
                assert (await store.lookup_continuation_ticket(command)).status == "unavailable"
            store.read_failure = "denied"
            with pytest.raises(PermissionError):
                await store.lookup_continuation_ticket(command)
            store.read_failure = None
            # Acknowledged preparation replay is independent of subsequent
            # writer release; it neither creates a new wait nor reclaims a writer.
            await interrupt_and_release_test_invocation(store, admitted.session.id)
            assert await _prepare_command(store, command) == record
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predicate_kind", "ALL_SUCCESS"),
        ("predicate_version", 2),
        ("threshold", 1),
        (
            "targets",
            (
                ObjectRef(
                    owner=OwnerRef(
                        application_scope="tests",
                        owner_id="assistant",
                        incarnation="target-2",
                    ),
                    kind="fixture",
                    object_id="target-2",
                    incarnation="target-2",
                    revision=1,
                ),
            ),
        ),
        ("deadline", "2031-01-01T00:00:00+00:00"),
        ("failure_policy", "fail"),
        ("service_policy", "clarify"),
        ("wait_edge_revision", 2),
        ("interaction_id", "different-interaction"),
        ("writer_generation", 2),
        ("purpose", "different-purpose"),
    ],
)
def test_fixed_key_mutation_fields_conflict(field: str, value: object) -> None:
    async def run() -> None:
        store, session, interaction_id = await _admitted_store()
        ticket = _ticket(session, interaction_id)
        if field == "threshold":
            ticket = ticket.model_copy(
                update={
                    "predicate_kind": "QUORUM_SUCCESS",
                    "threshold": 2,
                    "targets": (
                        *ticket.targets,
                        ticket.targets[0].model_copy(update={"object_id": "other-target"}),
                    ),
                }
            )
        await _prepare_command(store, await _preparation(store, ticket))
        changed = ticket.model_copy(update={field: value})
        error_type = (
            PermissionError
            if field in {"writer_generation", "interaction_id"}
            else ContinuationConflict
        )
        with pytest.raises(error_type):
            await _prepare_command(store, await _preparation(store, changed))
        assert (
            await store.lookup_continuation_ticket(await _preparation(store, changed))
        ).status == "conflict"

    asyncio.run(run())


async def _ready_continuation() -> tuple[
    InMemorySessionStore,
    ContinuationTicket,
    ContinuationLatch,
]:
    store, session, interaction_id = await _admitted_store()
    ticket = _ticket(session, interaction_id)
    await _prepare_command(store, await _preparation(store, ticket))
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    await _latch(store, latch, receiver=_QualifiedReceiver())
    waiting = await _park(store, ticket)
    assert waiting.latch is not None
    return store, waiting.ticket, waiting.latch


def _consumption(
    ticket: ContinuationTicket,
    latch: ContinuationLatch,
    continuation_id: str,
) -> ContinuationConsumption:
    return ContinuationConsumption(
        ticket=ticket,
        latch=latch,
        continuation_id=continuation_id,
        mode="inline",
        input_digest="input",
        profile_digest="profile",
        budget_digest="budget",
        admission_command_digest="command",
        admission_expected_run_epoch=ticket.writer_generation,
        receipt_stage="prepared",
        accepted_at=datetime.now(UTC).isoformat(),
    )


async def _test_concurrent_consumption_and_retirement_elect_once() -> None:
    store, ticket, latch = await _ready_continuation()
    first = _consumption(ticket, latch, "continuation-1")
    second = _consumption(ticket, latch, "continuation-2")
    results = await asyncio.gather(
        _consume(store, first),
        _consume(store, second),
        return_exceptions=True,
    )
    assert sum(type(result) is ContinuationRecord for result in results) == 1
    assert sum(isinstance(result, ContinuationConflict) for result in results) == 1

    store, ticket, latch = await _ready_continuation()
    consumption = _consumption(ticket, latch, "continuation-race")
    retirement = ContinuationRetirement(
        ticket=ticket,
        control_id="retire-race",
        reason="cancelled",
        retired_at=datetime.now(UTC).isoformat(),
    )
    results = await asyncio.gather(
        _consume(store, consumption),
        _retire(store, retirement),
        return_exceptions=True,
    )
    successes = [result for result in results if type(result) is ContinuationRecord]
    assert 1 <= len(successes) <= 2
    assert sum(isinstance(result, ContinuationConflict) for result in results) == 2 - len(successes)
    final = await store.load_continuation_ticket(
        ticket.session_id,
        registration_key=ticket.registration_key,
        session_instance_id=ticket.session_instance_id,
    )
    assert final is not None
    if len(successes) == 2:
        assert final.ticket.state == "RETIRED"
        assert final.consumption is not None
        assert final.consumption.receipt_stage == "excluded"


def test_concurrent_consumption_and_retirement_elect_once() -> None:
    asyncio.run(_test_concurrent_consumption_and_retirement_elect_once())


def test_consumption_replay_rejects_changed_latch() -> None:
    async def run() -> None:
        store, ticket, latch = await _ready_continuation()
        original = _consumption(ticket, latch, "exact-continuation")
        await _consume(store, original)
        expected = original
        before = await store.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        forged = expected.model_copy(
            update={"latch": latch.model_copy(update={"outcome_digest": "different"})}
        )
        with pytest.raises(ContinuationConflict):
            await _consume(store, forged)
        assert (
            await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            == before
        )

    asyncio.run(run())


@pytest.mark.parametrize("signal", ["cancel", "timeout"])
def test_latch_observer_interruption_keeps_ticket_pending(signal: str) -> None:
    async def run() -> None:
        store, session, interaction_id = await _admitted_store()
        ticket = _ticket(session, interaction_id)
        await _prepare_command(store, await _preparation(store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        deadline = asyncio.get_running_loop().create_future()

        class WaitingReceiver:
            async def authenticate_continuation_latch(self, latch):
                entered.set()
                await release.wait()
                return latch

        async def observe():
            if signal == "timeout":
                async with asyncio.timeout(None) as scope:
                    deadline.set_result(scope)
                    return await _latch(store, latch, receiver=WaitingReceiver())
            return await _latch(store, latch, receiver=WaitingReceiver())

        observer = asyncio.create_task(observe())
        await entered.wait()
        if signal == "cancel":
            observer.cancel()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled()
            assert observer.cancelling() == 2
        else:
            (await deadline).reschedule(asyncio.get_running_loop().time())
            with pytest.raises(TimeoutError):
                await observer
            assert not observer.cancelled()
            assert observer.cancelling() == 0
        record = await store.load_continuation_ticket(
            session.id,
            registration_key=ticket.registration_key,
            session_instance_id=session.instance_id,
        )
        assert record is not None
        assert record.ticket.state == "ARMING"
        assert record.latch is record.consumption is record.retirement is None
        release.set()
        recovered = await _latch(store, latch, receiver=WaitingReceiver())
        assert recovered.latch == latch

    asyncio.run(run())


async def _test_namespace_is_bound_to_session_incarnation() -> None:
    store, session, interaction_id = await _admitted_store()
    ticket = _ticket(session, interaction_id)
    bad = ticket.model_copy(
        update={
            "namespace": ticket.namespace.model_copy(update={"namespace_id": "forged"}),
        }
    )
    with pytest.raises(CollaborationContractError):
        await _prepare_command(
            store, (await _preparation(store, ticket)).model_copy(update={"intent": bad})
        )


def test_namespace_is_bound_to_session_incarnation() -> None:
    asyncio.run(_test_namespace_is_bound_to_session_incarnation())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_namespace_bootstrap_converges_and_rejects_changed_owner(backend, tmp_path) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "namespace.sqlite")
        )
        peer = store if backend == "memory" else SQLiteSessionStore(tmp_path / "namespace.sqlite")
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="namespace-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            namespaces = await asyncio.gather(
                _bootstrap(store, ticket),
                _bootstrap(peer, ticket),
            )
            assert namespaces == [ticket.namespace, ticket.namespace]
            other_owner = ticket.owner.model_copy(update={"application_scope": "other-application"})
            other_namespace = ticket.namespace.model_copy(
                update={
                    "owner": other_owner,
                    "namespace_id": continuation_namespace_id(
                        ticket.session_id, ticket.session_instance_id, other_owner
                    ),
                }
            )
            foreign_ticket = ticket.model_copy(
                update={"owner": other_owner, "namespace": other_namespace}
            )
            with pytest.raises(ContinuationConflict):
                await _bootstrap(peer, foreign_ticket)
            with pytest.raises(ContinuationConflict):
                await _prepare_command(peer, await _preparation(store, foreign_ticket))
            assert (
                await _prepare_command(store, await _preparation(store, ticket))
            ).namespace == namespaces[0]
        finally:
            if isinstance(store, SQLiteSessionStore) and isinstance(peer, SQLiteSessionStore):
                await store.close()
                await peer.close()

    asyncio.run(run())


async def _test_acknowledgement_loss_replays_the_same_ticket() -> None:
    store = _AcknowledgementLostStore()
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id="ack-session",
            messages=[Message.text("user", "wait")],
        ),
        provider_name="continuation-provider",
        model="continuation-model",
    )
    ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
    store.lose_next_publication = True
    with pytest.raises(RuntimeError, match="acknowledgement loss"):
        await _prepare_command(store, await _preparation(store, ticket))
    replay = await _prepare_command(store, await _preparation(store, ticket))
    assert replay.ticket == ticket


def test_acknowledgement_loss_replays_the_same_ticket() -> None:
    asyncio.run(_test_acknowledgement_loss_replays_the_same_ticket())


def test_latched_replay_does_not_reacquire_the_receiving_owner() -> None:
    async def run() -> None:
        store, session, interaction_id = await _admitted_store()
        ticket = _ticket(session, interaction_id)
        await _prepare_command(store, await _preparation(store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        calls = 0

        class Receiver:
            async def authenticate_continuation_latch(self, latch):
                nonlocal calls
                calls += 1
                return latch

        receiver = Receiver()

        class SubstitutingReceiver:
            async def authenticate_continuation_latch(self, latch):
                return latch.model_copy(update={"outcome_digest": "different-outcome"})

        with pytest.raises(ContinuationConflict):
            await _latch(store, latch, receiver=SubstitutingReceiver())
        unchanged = await store.load_continuation_ticket(
            session.id,
            registration_key=ticket.registration_key,
            session_instance_id=session.instance_id,
        )
        assert unchanged is not None and unchanged.latch is None
        await _latch(store, latch, receiver=receiver)
        assert calls == 1

        class UnavailableReceiver:
            async def authenticate_continuation_latch(self, candidate):
                raise RuntimeError("receiving owner unavailable")

        replay = await _latch(store, latch, receiver=UnavailableReceiver())
        assert replay.latch == latch

        waiting = await _park(store, ticket)
        assert waiting.latch is not None
        revised_latch = latch.model_copy(update={"ticket": waiting.ticket})
        replay_after_parking = await _latch(
            store,
            revised_latch,
            receiver=UnavailableReceiver(),
        )
        assert replay_after_parking.latch == latch

    asyncio.run(run())


def test_registered_receiver_owns_opaque_work_and_rejects_raw_lookalikes() -> None:
    async def run() -> None:
        store, session, interaction_id = await _admitted_store()
        ticket = _ticket(session, interaction_id)
        armed = await _prepare_command(store, await _preparation(store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        dispatched = threading.Event()
        release = threading.Event()
        calls = 0
        source_latch = latch

        class SourceOwner:
            async def authenticate_continuation_latch(self, latch):
                nonlocal calls
                calls += 1
                # This callback has no receiving or publication authority.
                with pytest.raises(PermissionError):
                    await store.latch_continuation(latch)
                if latch != source_latch:
                    raise PermissionError("Source receipt does not match.")
                dispatched.set()
                assert await asyncio.to_thread(release.wait, 5)
                return source_latch

        owner = SessionContinuationOwner(
            store=store,
            owner=ticket.owner,
            receiver=SourceOwner(),
            receiver_capability=CapabilityDescriptor(
                owner=ticket.owner,
                mutations=(),
                readbacks=(LATCH_FAMILY,),
            ),
            redactor=SecretRedactor(),
        )
        try:
            with pytest.raises(PermissionError):
                await store.latch_continuation(latch)
            with pytest.raises(PermissionError):
                await store.publish_session_operation(
                    session.id,
                    idempotency_key=continuation_operation_key(ticket),
                    operation_transform=lambda _session, checkpoint, _current: (
                        SessionOperationPublication(
                            checkpoint=checkpoint or {},
                            operation_records={
                                continuation_operation_key(ticket): armed.model_copy(
                                    update={"latch": latch}
                                ).model_dump(mode="json")
                            },
                        )
                    ),
                    events=[],
                )
            caller = asyncio.create_task(owner.latch(latch))
            async with asyncio.timeout(2):
                while not dispatched.is_set():
                    await asyncio.sleep(0.001)
            caller.cancel("private cancellation canary")
            caller.cancel("private cancellation canary")
            with pytest.raises(asyncio.CancelledError) as caught:
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
            assert "private cancellation canary" not in str(caught.value)
            assert len(owner.owners.pending) == 1
            retry = asyncio.create_task(owner.latch(latch))
            await asyncio.sleep(0)
            assert calls == 1 and not retry.done()
            release.set()
            completed = await retry
            assert completed.latch == latch and calls == 1
            with pytest.raises(PermissionError):
                await store.latch_continuation(latch)
            assert await owner.latch(latch) == completed
            assert calls == 1
        finally:
            release.set()
            await owner.drain()

    asyncio.run(run())


@pytest.mark.parametrize(
    "method",
    [
        "publish_session_operation",
        "latch_continuation",
        "_initialize_continuation_namespace",
        "apply_invocation_lifecycle_command",
    ],
)
def test_continuation_capability_requires_override_qualification(method) -> None:
    async def unqualified(*args, **kwargs):
        raise AssertionError("Capability checks must not dispatch methods.")

    store = type("UnqualifiedStore", (InMemorySessionStore,), {method: unqualified})()
    assert not store._supports_session_continuation_protocol()
    assert not runtime_checkpoint_session_store(store)._supports_session_continuation_protocol()
    qualified = InMemorySessionStore()
    assert qualified._supports_session_continuation_protocol()
    assert runtime_checkpoint_session_store(qualified)._supports_session_continuation_protocol()


def test_sqlite_fresh_process_acknowledgement_loss_preserves_latch(tmp_path) -> None:
    async def prepare_session(path):
        store = SQLiteSessionStore(path)
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id="sqlite-process-continuation-session",
                messages=[Message.text("user", "wait")],
            ),
            provider_name="continuation-provider",
            model="continuation-model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        await store.close()
        return ticket

    path = tmp_path / "continuation-process.sqlite"
    ticket = asyncio.run(prepare_session(path))
    ticket_json = json.dumps(ticket.model_dump(mode="json"), separators=(",", ":"))
    child = r"""
import asyncio, json, sys
from datetime import UTC, datetime
from cayu.runtime._session_continuation import ContinuationLatch, ContinuationUnavailable
from cayu.storage.sqlite import SQLiteSessionStore
from tests.core.test_session_continuation import _latch, _preparation, _prepare_command

class Receiver:
    async def authenticate_continuation_latch(self, latch):
        return latch

class LostAckSQLiteStore(SQLiteSessionStore):
    session_continuation_version = 1
    lose_next = False

    async def publish_session_operation(self, session_id, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        if self.lose_next:
            self.lose_next = False
            raise RuntimeError("simulated child acknowledgement loss")
        return result

async def main():
    path, ticket_json = sys.argv[1:]
    ticket_type = __import__("cayu.runtime._session_continuation", fromlist=["ContinuationTicket"]).ContinuationTicket
    ticket = ticket_type.model_validate(json.loads(ticket_json))
    store = LostAckSQLiteStore(path)
    await _prepare_command(store, await _preparation(store, ticket))
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    store.lose_next = True
    try:
        await _latch(store, latch, receiver=Receiver())
    except ContinuationUnavailable as exc:
        if "simulated child acknowledgement loss" not in str(exc.__cause__):
            raise
        # Simulate abrupt owner loss after the transaction committed and
        # before the caller observed its acknowledgement.
        import os
        os._exit(17)
    raise AssertionError("latch acknowledgement should have been lost")

asyncio.run(main())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(("src", ".", env.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [sys.executable, "-c", child, str(path), ticket_json],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    # The child intentionally exits non-zero after the durable latch commit,
    # modeling a lost acknowledgement/process loss.  The parent must still
    # recover the committed latch from a fresh store instance.
    assert completed.returncode == 17, completed.stderr

    async def reopen_and_read() -> None:
        store = SQLiteSessionStore(path)
        try:
            record = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert record is not None
            assert record.latch is not None
            assert record.latch.latch_key == "latch-1"
        finally:
            await store.close()

    asyncio.run(reopen_and_read())


async def _test_continuation_admission_uses_typed_lifecycle_boundary(
    *,
    lose_admission_acknowledgement: bool = False,
    sqlite_path=None,
    postgres_dsn=None,
    reconcile_from_receipt: bool = False,
    use_owner: bool = False,
    session_id: str = "admission-session",
    race_retirement: bool = False,
    race_released_claim: bool = False,
    peer_commits_first: bool = False,
    race_supersession: bool = False,
    recover_supersession: bool = False,
    release_before_supersession: bool = False,
    fail_admission_before_commit: bool = False,
    fail_admission_attempts: int = 0,
    lose_claim_readback: bool = False,
    retire_after_failed_admission: bool = False,
    compact_before_recovery: str | None = None,
) -> None:
    admission_enabled = False
    admission_entered = asyncio.Event()
    admission_release = asyncio.Event()
    claim_readback_loss = lose_claim_readback

    class AcknowledgementLoss(SessionStore):
        lose_acknowledgement = lose_admission_acknowledgement
        lose_claim_readback = claim_readback_loss
        lose_exclusion = False

        async def publish_session_operation(self, session_id, **kwargs):
            transform = kwargs.get("operation_transform")
            if transform is not None:

                def fail_finalization(session, checkpoint, current):
                    publication = transform(session, checkpoint, current)
                    if self.lose_exclusion and any(
                        (value.get("retirement") or {}).get("reason") == "superseded"
                        for value in publication.operation_records.values()
                    ):
                        self.lose_exclusion = False
                        raise RuntimeError("supersession publication unavailable")
                    if self.lose_acknowledgement and any(
                        value.get("ticket", {}).get("state") == "CONSUMED"
                        for value in publication.operation_records.values()
                    ):
                        self.lose_acknowledgement = False
                        raise RuntimeError("admission committed without acknowledgement")
                    return publication

                kwargs["operation_transform"] = fail_finalization
            return await super().publish_session_operation(session_id, **kwargs)

        async def load_continuation_ticket(self, session_id, **kwargs):
            record = await super().load_continuation_ticket(session_id, **kwargs)
            if (
                self.lose_claim_readback
                and record is not None
                and record.consumption is not None
                and record.consumption.admission_claimed
            ):
                self.lose_claim_readback = False
                return None
            return record

    class Memory(AcknowledgementLoss, InMemorySessionStore):
        session_continuation_version = 1
        invocation_lifecycle_command_version = 1

    class SQLite(AcknowledgementLoss, SQLiteSessionStore):
        session_continuation_version = 1
        invocation_lifecycle_command_version = 1

    if postgres_dsn is not None:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        class Postgres(AcknowledgementLoss, PostgresSessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        store = Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
    else:
        store = Memory() if sqlite_path is None else SQLite(sqlite_path)
    persistent_store = sqlite_path is not None or postgres_dsn is not None

    async def close_store() -> None:
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    app = CayuApp(session_store=store, enable_logging=False)
    admitted = await create_admitted_session(
        store,
        app=app,
        request=RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "wait")],
        ),
        provider_name="continuation-provider",
        model="continuation-model",
    )
    ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
    await _prepare_command(store, await _preparation(store, ticket))
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    await _latch(store, latch, receiver=_QualifiedReceiver())
    waiting = await _park(store, ticket)
    assert waiting.latch is not None
    retirement_invocation = await _context(store, admitted.session.id)
    await interrupt_and_release_test_invocation(store, admitted.session.id)
    session = await store.load(admitted.session.id)
    checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(admitted.session.id)
    assert session is not None
    profile = execution_profile_from_session_metadata(session.metadata)
    event = runtime_interaction_started_event(
        app,
        session_id=session.id,
        interaction_id="continuation-next-interaction",
        agent_name=session.agent_name,
    )
    command = AdmitInvocationCommand(
        session_id=session.id,
        expected_session_instance_id=session.instance_id,
        expected_statuses=(SessionStatus.INTERRUPTED,),
        expected_run_epoch=session.run_epoch,
        expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
        target_active_profile=admitted.active_invocation_profile.model_copy(
            update={
                "interaction_id": event.interaction_id,
                "run_epoch": session.run_epoch + 1,
                "profile": profile,
            }
        ),
        checkpoint_patch=InvocationCheckpointPatch(
            mutation=runtime_publication_checkpoint_mutation(checkpoint, checkpoint)
        ),
        tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
        interaction_started_event=event,
        expected_active_profile=admitted.active_invocation_profile,
    )
    input_digest, profile_digest, budget_digest = continuation_admission_inputs(command)
    prepared = ContinuationConsumption(
        ticket=waiting.ticket,
        latch=waiting.latch,
        continuation_id="typed-admission-1",
        mode="inline",
        input_digest=input_digest,
        profile_digest=profile_digest,
        budget_digest=budget_digest,
        admission_command_digest=continuation_admission_digest(command),
        admission_expected_run_epoch=command.expected_run_epoch,
        receipt_stage="prepared",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    if use_owner:
        owner = SessionContinuationOwner(
            store=store,
            owner=ticket.owner,
            receiver=_QualifiedReceiver(),
            receiver_capability=CapabilityDescriptor(
                owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
        )
        invocation = await _context(store, session.id, admission=command)
        for field in ("input_digest", "profile_digest", "budget_digest"):
            with pytest.raises(ContinuationConflict):
                await owner.admit(
                    prepared.model_copy(update={field: "foreign"}), command, invocation=invocation
                )
        with pytest.raises(PermissionError):
            await owner.admit(prepared, command, invocation=object())  # ty: ignore[invalid-argument-type]
        unchanged = await store.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        assert unchanged is not None and unchanged.consumption is None
        try:
            if compact_before_recovery is not None:
                import contextvars

                from tests.core._execution_profile_fixtures import admit_test_invocation

                from cayu.events import Event, event_with_runtime_envelope_authority
                from cayu.runtime import _invocation_lifecycle as lifecycle_module

                original_apply = lifecycle_module.apply_invocation_lifecycle_command
                if compact_before_recovery == "superseded":

                    async def competing_admission(store_arg, command_arg):
                        winner = command_arg.model_copy(
                            update={
                                "interaction_source_messages": (Message.text("user", "winner"),)
                            }
                        )
                        await asyncio.create_task(
                            original_apply(store_arg, winner), context=contextvars.Context()
                        )
                        raise RuntimeError("another admission won")

                    store.lose_exclusion = True
                    lifecycle_module.apply_invocation_lifecycle_command = competing_admission  # ty: ignore[invalid-assignment]
                else:
                    store.lose_acknowledgement = True
                try:
                    with pytest.raises(ContinuationUnavailable):
                        await owner.admit(prepared, command, invocation=invocation)
                finally:
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
                pending = await store.load_continuation_ticket(
                    session.id,
                    registration_key=ticket.registration_key,
                    session_instance_id=session.instance_id,
                )
                assert pending is not None and pending.consumption is not None
                assert pending.consumption.receipt_stage == "prepared"
                from cayu.runtime._session_continuation_store import ROOT_KEY
                from cayu.sessions.base import _invocation_lifecycle_authority_read_scope

                def generic_read(_session, current):
                    assert current is not None and ROOT_KEY not in current
                    return current

                await store.transform_checkpoint(session.id, generic_read)

                def forbidden_index_write(_session, current):
                    assert current is not None and ROOT_KEY in current
                    return current | {ROOT_KEY: current[ROOT_KEY] | {"entries": []}}

                with (
                    _invocation_lifecycle_authority_read_scope(),
                    pytest.raises(ContinuationConflict),
                ):
                    await store.transform_checkpoint(session.id, forbidden_index_write)
                identity = (
                    f"admit:{session.id}:{session.instance_id}:"
                    f"{prepared.admission_expected_run_epoch + 1}"
                )

                async def advance(prefix):
                    for index in range(4):
                        await store.append_event(
                            session.id,
                            event_with_runtime_envelope_authority(
                                Event(type=EventType.SESSION_RESUMED, session_id=session.id),
                                "session_id",
                            ),
                        )
                        await interrupt_and_release_test_invocation(store, session.id)
                        await admit_test_invocation(
                            runtime_checkpoint_session_store(store),
                            session.id,
                            interaction_started_event=runtime_interaction_started_event(
                                app,
                                session_id=session.id,
                                interaction_id=f"{prefix}-{index}",
                                agent_name=session.agent_name,
                            ),
                        )
                    current_checkpoint = await runtime_checkpoint_session_store(
                        store
                    ).load_checkpoint(session.id)
                    ledger = lifecycle_module._invocation_lifecycle_receipt_ledger_from_checkpoint(
                        current_checkpoint
                    )
                    assert len(ledger.receipts) <= 6
                    return {item.command_identity for item in ledger.receipts}

                assert identity in await advance("pending")
                await owner.drain()
                if persistent_store:
                    await close_store()
                    store = (
                        SQLite(sqlite_path)
                        if sqlite_path is not None
                        else Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
                    )
                owner = SessionContinuationOwner(
                    store=store,
                    owner=ticket.owner,
                    receiver=_QualifiedReceiver(),
                    receiver_capability=CapabilityDescriptor(
                        owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                    ),
                    redactor=SecretRedactor(),
                )
                settled = await owner.reconcile_admission(prepared)
                assert settled.consumption is not None
                assert settled.consumption.receipt_stage == (
                    "excluded" if compact_before_recovery == "superseded" else "admitted"
                )
                assert identity not in await advance("settled")
                assert await owner.reconcile_admission(prepared) == settled
                await store.append_event(
                    session.id,
                    event_with_runtime_envelope_authority(
                        Event(type=EventType.SESSION_RESUMED, session_id=session.id), "session_id"
                    ),
                )
                await interrupt_and_release_test_invocation(store, session.id)
                await store.validate_session_closure_admission(session.id)
                await store.delete_session(session.id)
                await owner.drain()
                return
            if race_supersession:
                from cayu.runtime import _invocation_lifecycle as lifecycle_module

                original_apply = lifecycle_module.apply_invocation_lifecycle_command
                entered, release = asyncio.Event(), asyncio.Event()

                async def blocked_admission(store_arg, command_arg):
                    entered.set()
                    await release.wait()
                    if release_before_supersession:
                        raise RuntimeError("known pre-commit admission rejection")
                    return await original_apply(store_arg, command_arg)

                lifecycle_module.apply_invocation_lifecycle_command = blocked_admission  # ty: ignore[invalid-assignment]
                peer_store = store
                if sqlite_path is not None:
                    peer_store = SQLiteSessionStore(sqlite_path)
                elif postgres_dsn is not None:
                    peer_store = Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
                try:
                    losing = asyncio.create_task(
                        owner.admit(prepared, command, invocation=invocation)
                    )
                    await asyncio.wait_for(entered.wait(), 5)
                    if release_before_supersession:
                        release.set()
                        with pytest.raises(ContinuationUnavailable):
                            await losing
                        unclaimed = await store.load_continuation_ticket(
                            ticket.session_id,
                            registration_key=ticket.registration_key,
                            session_instance_id=ticket.session_instance_id,
                        )
                        assert unclaimed is not None and unclaimed.consumption is not None
                        assert not unclaimed.consumption.admission_claimed
                        # Ordinary absence is not proof of supersession.
                        with pytest.raises(ContinuationUnavailable):
                            await owner.reconcile_admission(prepared)
                    winner_event = runtime_interaction_started_event(
                        app,
                        session_id=session.id,
                        interaction_id="competing-resume",
                        agent_name=session.agent_name,
                    )
                    winner_command = command.model_copy(
                        update={
                            "interaction_started_event": winner_event,
                            "target_active_profile": command.target_active_profile.model_copy(
                                update={
                                    "interaction_id": winner_event.interaction_id,
                                }
                            ),
                        }
                    )
                    winner = await original_apply(
                        runtime_checkpoint_session_store(peer_store), winner_command
                    )
                    assert type(winner) is InvocationMutationResult and not winner.replayed
                    from cayu.runtime._invocation_lifecycle import (
                        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                        superseding_invocation_admission_digest_from_state,
                    )

                    winner_checkpoint = await runtime_checkpoint_session_store(
                        peer_store
                    ).load_checkpoint(session.id)
                    assert winner_checkpoint is not None

                    def classify(observed_session, observed_checkpoint):
                        return superseding_invocation_admission_digest_from_state(
                            observed_session,
                            observed_checkpoint,
                            session_id=session.id,
                            session_instance_id=session.instance_id,
                            expected_run_epoch=command.expected_run_epoch,
                            command_sha256=prepared.admission_command_digest,
                        )

                    # Absence is not supersession; malformed and wrong-epoch
                    # evidence cannot authorize the new exclusion transition.
                    missing = dict(winner_checkpoint)
                    missing.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
                    assert classify(winner.session, missing) is None
                    malformed = dict(winner_checkpoint)
                    malformed[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = {
                        "record_sha256": "wrong"
                    }
                    with pytest.raises((RuntimeError, ValueError)):
                        classify(winner.session, malformed)
                    with pytest.raises((RuntimeError, ValueError)):
                        classify(session, winner_checkpoint)
                    store.lose_exclusion = recover_supersession
                    release.set()
                    if not release_before_supersession:
                        with pytest.raises(ContinuationUnavailable):
                            await losing
                    elif recover_supersession:
                        with pytest.raises(ContinuationUnavailable):
                            await owner.reconcile_admission(prepared)
                    await owner.drain()
                    if recover_supersession and persistent_store:
                        await close_store()
                        store = (
                            SQLite(sqlite_path)
                            if sqlite_path is not None
                            else Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
                        )
                    owner = SessionContinuationOwner(
                        store=store,
                        owner=ticket.owner,
                        receiver=_QualifiedReceiver(),
                        receiver_capability=CapabilityDescriptor(
                            owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                        ),
                        redactor=SecretRedactor(),
                    )
                    excluded = await owner.reconcile_admission(prepared)
                    assert excluded.ticket.state == "RETIRED"
                    assert excluded.consumption is not None
                    assert excluded.consumption.receipt_stage == "excluded"
                    assert not excluded.consumption.admission_claimed
                    assert excluded.retirement is not None
                    assert (
                        excluded.retirement.control_id
                        == "admission-superseded:" + continuation_admission_digest(winner_command)
                    )
                    assert await owner.reconcile_admission(prepared) == excluded
                    current = await store.load(session.id)
                    assert current is not None and current.run_epoch == winner.session.run_epoch
                    replay = await original_apply(
                        runtime_checkpoint_session_store(store), winner_command
                    )
                    assert type(replay) is InvocationMutationResult and replay.replayed
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
                    from cayu.events import Event, event_with_runtime_envelope_authority

                    # The winner bypasses the session loop in this admission
                    # race fixture; publish its normal lifecycle-start boundary
                    # before terminalizing it for the deletion check.
                    await store.append_event(
                        session.id,
                        event_with_runtime_envelope_authority(
                            Event(type=EventType.SESSION_RESUMED, session_id=session.id),
                            "session_id",
                        ),
                    )
                    await interrupt_and_release_test_invocation(store, session.id)
                    await store.validate_session_closure_admission(session.id)
                    await store.delete_session(session.id)
                    assert await store.load(session.id) is None
                    await owner.drain()
                    return
                finally:
                    release.set()
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
                    if sqlite_path is not None:
                        assert isinstance(peer_store, SQLiteSessionStore)
                        await peer_store.close()
                    elif postgres_dsn is not None:
                        assert isinstance(peer_store, Postgres)
                        await peer_store.close()
            if lose_claim_readback:
                with pytest.raises(ContinuationUnavailable):
                    await owner.admit(prepared, command, invocation=invocation)
                claimed = await store.load_continuation_ticket(
                    prepared.ticket.session_id,
                    registration_key=prepared.ticket.registration_key,
                    session_instance_id=prepared.ticket.session_instance_id,
                )
                assert claimed is not None and claimed.consumption is not None
                assert claimed.consumption.admission_claimed
                result, settled = await owner.admit(prepared, command, invocation=invocation)
                assert not result.replayed
                assert settled.ticket.state == "CONSUMED"
                await owner.drain()
                return
            if fail_admission_before_commit or fail_admission_attempts:
                from cayu.runtime import _invocation_lifecycle as lifecycle_module

                original_apply = lifecycle_module.apply_invocation_lifecycle_command
                remaining_failures = max(1, fail_admission_attempts)

                async def fail_apply(
                    store: SessionStore, lifecycle_command: AdmitInvocationCommand
                ):
                    nonlocal remaining_failures
                    if remaining_failures > 0:
                        remaining_failures -= 1
                        raise RuntimeError("admission rejected before commit")
                    return await original_apply(store, lifecycle_command)

                lifecycle_module.apply_invocation_lifecycle_command = fail_apply  # ty: ignore[invalid-assignment]
                try:
                    attempts = max(1, fail_admission_attempts)
                    for _ in range(attempts):
                        with pytest.raises(ContinuationUnavailable):
                            await owner.admit(prepared, command, invocation=invocation)
                        retained = await store.load_continuation_ticket(
                            prepared.ticket.session_id,
                            registration_key=prepared.ticket.registration_key,
                            session_instance_id=prepared.ticket.session_instance_id,
                        )
                        assert retained is not None
                        assert retained.consumption is not None
                        assert retained.consumption.receipt_stage == "prepared"
                        assert not retained.consumption.admission_claimed
                        if fail_admission_attempts:
                            assert len(retained.events) == 4
                finally:
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
                retained = await store.load_continuation_ticket(
                    prepared.ticket.session_id,
                    registration_key=prepared.ticket.registration_key,
                    session_instance_id=prepared.ticket.session_instance_id,
                )
                assert retained is not None and retained.consumption is not None
                assert retained.consumption.receipt_stage == "prepared"
                # The lifecycle boundary was read back successfully and no
                # admission receipt exists, so this is a known pre-commit
                # rejection rather than an acknowledgement-loss case.  The
                # claim must be released, making the continuation retryable
                # (and allowing exclusion to race only after that release).
                assert not retained.consumption.admission_claimed
                changed_command = command.model_copy(
                    update={
                        "expected_statuses": (*command.expected_statuses, SessionStatus.COMPLETED)
                    }
                )
                changed_consumption = prepared.model_copy(
                    update={
                        "admission_command_digest": continuation_admission_digest(changed_command)
                    }
                )
                with pytest.raises(ContinuationConflict):
                    await owner.admit(changed_consumption, changed_command, invocation=invocation)
                from cayu.runtime._session_continuation_store import require_history

                try:
                    require_history(retained)
                except ContinuationConflict as error:
                    pytest.fail(
                        f"released continuation history invalid: {error}; "
                        f"events={retained.events!r}; "
                        f"record={retained.model_dump(mode='json', exclude={'events'})!r}"
                    )
                if retire_after_failed_admission:
                    retirement = ContinuationRetirement(
                        ticket=prepared.ticket,
                        control_id="retire-after-admission-retries",
                        reason="failed",
                        retired_at=datetime.now(UTC).isoformat(),
                    )
                    retired = await owner.exclude(retirement, invocation=retirement_invocation)
                    assert retired.ticket.state == "RETIRED"
                    assert retired.consumption is not None
                    assert retired.consumption.receipt_stage == "excluded"
                    await owner.drain()
                    return
                result, settled = await owner.admit(prepared, command, invocation=invocation)
                assert result.session.run_epoch == prepared.admission_expected_run_epoch + 1
                assert settled.ticket.state == "CONSUMED"
                await owner.drain()
                return
            if race_released_claim:
                from cayu.runtime import _invocation_lifecycle as lifecycle_module

                original_apply = lifecycle_module.apply_invocation_lifecycle_command
                first_entered, peer_entered = asyncio.Event(), asyncio.Event()
                fail_first, release_peer = asyncio.Event(), asyncio.Event()
                calls = 0
                peer_store = SQLiteSessionStore(sqlite_path) if sqlite_path is not None else store
                if postgres_dsn is not None:
                    peer_store = Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
                peer = SessionContinuationOwner(
                    store=peer_store,
                    owner=ticket.owner,
                    receiver=_QualifiedReceiver(),
                    receiver_capability=CapabilityDescriptor(
                        owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                    ),
                    redactor=SecretRedactor(),
                )

                async def racing_apply(store_arg, command_arg):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        first_entered.set()
                        await fail_first.wait()
                        raise RuntimeError("first worker rejected before commit")
                    peer_entered.set()
                    await release_peer.wait()
                    return await original_apply(store_arg, command_arg)

                lifecycle_module.apply_invocation_lifecycle_command = racing_apply  # ty: ignore[invalid-assignment]
                try:
                    first = asyncio.create_task(
                        owner.admit(prepared, command, invocation=invocation)
                    )
                    await asyncio.wait_for(first_entered.wait(), 5)
                    second = asyncio.create_task(
                        peer.admit(prepared, command, invocation=invocation)
                    )
                    await asyncio.wait_for(peer_entered.wait(), 5)
                    if peer_commits_first:
                        release_peer.set()
                        _, settled = await second
                        assert settled.ticket.state == "CONSUMED"
                    fail_first.set()
                    with pytest.raises(ContinuationUnavailable):
                        await first
                    retirement = ContinuationRetirement(
                        ticket=prepared.ticket,
                        control_id="retire-after-claim-release",
                        reason="failed",
                        retired_at=datetime.now(UTC).isoformat(),
                    )
                    if peer_commits_first:
                        with pytest.raises(ContinuationConflict):
                            await owner.exclude(retirement, invocation=retirement_invocation)
                    else:
                        retired = await owner.exclude(retirement, invocation=retirement_invocation)
                        assert retired.ticket.state == "RETIRED"
                        release_peer.set()
                        with pytest.raises((ContinuationConflict, ContinuationUnavailable)):
                            await second
                    current = await store.load(session.id)
                    assert current is not None
                    assert current.run_epoch == command.expected_run_epoch + int(peer_commits_first)
                    await owner.drain()
                    await peer.drain()
                    return
                finally:
                    fail_first.set()
                    release_peer.set()
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
                    if peer_store is not store and isinstance(peer_store, SQLiteSessionStore):
                        await peer_store.close()
                    elif postgres_dsn is not None:
                        assert isinstance(peer_store, Postgres)
                        await peer_store.close()
            if race_retirement:
                from cayu.runtime import _invocation_lifecycle as lifecycle_module

                original_apply = lifecycle_module.apply_invocation_lifecycle_command

                async def blocked_apply(
                    store: SessionStore, lifecycle_command: AdmitInvocationCommand
                ):
                    if admission_enabled:
                        admission_entered.set()
                        await admission_release.wait()
                    return await original_apply(store, lifecycle_command)

                try:
                    lifecycle_module.apply_invocation_lifecycle_command = blocked_apply  # ty: ignore[invalid-assignment]
                    admission_enabled = True
                    admission_task = asyncio.create_task(
                        owner.admit(prepared, command, invocation=invocation)
                    )
                    try:
                        await asyncio.wait_for(admission_entered.wait(), timeout=5)
                    except TimeoutError:
                        if admission_task.done():
                            admission_task.result()
                        raise
                    claimed_record = await store.load_continuation_ticket(
                        prepared.ticket.session_id,
                        registration_key=prepared.ticket.registration_key,
                        session_instance_id=prepared.ticket.session_instance_id,
                    )
                    assert claimed_record is not None
                    assert claimed_record.consumption is not None
                    assert claimed_record.consumption.admission_claimed
                    assert claimed_record.consumption.receipt_stage == "prepared"
                    retirement = ContinuationRetirement(
                        ticket=prepared.ticket,
                        control_id="retire-during-admission",
                        reason="failed",
                        retired_at=datetime.now(UTC).isoformat(),
                    )
                    with pytest.raises(ContinuationConflict):
                        await asyncio.wait_for(
                            owner.exclude(retirement, invocation=retirement_invocation), timeout=5
                        )
                    admission_release.set()
                    result, settled = await admission_task
                    assert settled.ticket.state == "CONSUMED"
                    assert settled.consumption is not None
                    assert settled.consumption.receipt_stage == "admitted"
                    await owner.drain()
                    return
                finally:
                    admission_release.set()
                    lifecycle_module.apply_invocation_lifecycle_command = original_apply
            result, settled = await owner.admit(prepared, command, invocation=invocation)
            assert settled.ticket.state == "CONSUMED"
            assert result.session.run_epoch == prepared.admission_expected_run_epoch + 1
            replay_result, replay = await owner.admit(prepared, command, invocation=invocation)
            assert replay == settled
            assert replay_result.replayed
            await owner.drain()
        finally:
            if persistent_store:
                await close_store()
        return
    with pytest.raises(ContinuationConflict):
        await _admit_fixture_continuation(
            runtime_checkpoint_session_store(store),
            prepared.model_copy(update={"admission_command_digest": "wrong-command"}),
            command,
        )
    await _consume(store, prepared)
    with (
        pytest.raises(ContinuationConflict, match="durable admission evidence"),
        consumption_scope(prepared),
    ):
        await store._finalize_continuation_admission(
            prepared.model_copy(update={"receipt_stage": "admitted"}), command
        )
    if lose_admission_acknowledgement:
        with pytest.raises(RuntimeError, match="admission committed without acknowledgement"):
            await _admit_fixture_continuation(
                runtime_checkpoint_session_store(store), prepared, command
            )
        retained = await store.load_continuation_ticket(
            session.id,
            registration_key=ticket.registration_key,
            session_instance_id=session.instance_id,
        )
        assert retained is not None and retained.consumption is not None
        assert retained.consumption.receipt_stage == "prepared"
        committed_session = await store.load(session.id)
        assert committed_session is not None
        assert committed_session.run_epoch == command.expected_run_epoch + 1
    if reconcile_from_receipt:
        # Recovery retains only the prepared source receipt, not the original
        # invocation command or any in-process event provenance.
        del command
        if sqlite_path is not None:
            await close_store()
            store = SQLiteSessionStore(sqlite_path)
        elif postgres_dsn is not None:
            await close_store()
            store = Postgres(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
            store.lose_acknowledgement = False
        owner = SessionContinuationOwner(
            store=store,
            owner=ticket.owner,
            receiver=_QualifiedReceiver(),
            receiver_capability=CapabilityDescriptor(
                owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
        )
        try:
            settled = await owner.reconcile_admission(prepared)
            assert settled.ticket.state == "CONSUMED"
            assert settled.consumption is not None
            assert settled.consumption.receipt_stage == "admitted"
            assert await owner.reconcile_admission(prepared) == settled
            replayed_session = await store.load(session.id)
            assert replayed_session is not None
            assert replayed_session.run_epoch == prepared.admission_expected_run_epoch + 1
            with pytest.raises(ContinuationConflict):
                await owner.reconcile_admission(prepared.model_copy(update={"mode": "queued"}))
            for field, value in (
                ("outcome_digest", "foreign"),
                ("disclosure_digest", "foreign"),
                ("latch_key", "foreign"),
            ):
                with pytest.raises(ContinuationConflict):
                    await owner.reconcile_admission(
                        prepared.model_copy(
                            update={"latch": prepared.latch.model_copy(update={field: value})}
                        )
                    )
            await owner.drain()
        finally:
            if persistent_store:
                await close_store()
        return
    result, settled = await _admit_fixture_continuation(
        runtime_checkpoint_session_store(store),
        prepared,
        command,
    )
    assert type(result) is InvocationMutationResult
    assert settled.ticket.state == "CONSUMED"
    assert settled.consumption is not None
    assert settled.consumption.receipt_stage == "admitted"
    forged = prepared.model_copy(
        update={"latch": waiting.latch.model_copy(update={"outcome_digest": "foreign"})}
    )
    with pytest.raises(ContinuationConflict):
        await _admit_fixture_continuation(runtime_checkpoint_session_store(store), forged, command)
    replay_result, replay_record = await _admit_fixture_continuation(
        runtime_checkpoint_session_store(store), prepared, command
    )
    assert replay_result.session.run_epoch == result.session.run_epoch
    assert replay_record == settled


def test_continuation_admission_uses_typed_lifecycle_boundary() -> None:
    asyncio.run(_test_continuation_admission_uses_typed_lifecycle_boundary())


def test_continuation_reconciles_committed_admission_after_acknowledgement_loss() -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            lose_admission_acknowledgement=True,
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_continuation_replays_admission_after_claim_readback_loss(backend, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            lose_claim_readback=True,
            sqlite_path=None if backend == "memory" else tmp_path / "claim-loss.sqlite",
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("peer_commits_first", [False, True])
def test_released_claim_fences_concurrent_admission(backend, peer_commits_first, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            race_released_claim=True,
            peer_commits_first=peer_commits_first,
            sqlite_path=None if backend == "memory" else tmp_path / "claim-race.sqlite",
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("unclaimed", [False, True])
def test_competing_admission_excludes_losing_continuation(
    backend, recover, unclaimed, tmp_path
) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            race_supersession=True,
            recover_supersession=recover,
            release_before_supersession=unclaimed,
            sqlite_path=None if backend == "memory" else tmp_path / "supersession.sqlite",
        )
    )


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("unclaimed", [False, True])
def test_postgres_competing_admission_excludes_losing_continuation(
    continuation_postgres_dsn, recover, unclaimed
) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            race_supersession=True,
            recover_supersession=recover,
            release_before_supersession=unclaimed,
            postgres_dsn=continuation_postgres_dsn,
            session_id=f"postgres-supersession-{recover}-{unclaimed}",
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["admitted", "superseded"])
def test_pending_continuation_pins_receipt_until_settlement(
    backend, outcome, tmp_path, monkeypatch
):
    from cayu.runtime import _invocation_lifecycle as lifecycle_module

    monkeypatch.setattr(lifecycle_module, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS", 6)
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            compact_before_recovery=outcome,
            sqlite_path=None if backend == "memory" else tmp_path / "receipt-retention.sqlite",
        )
    )


@pytest.mark.parametrize("outcome", ["admitted", "superseded"])
def test_postgres_pending_continuation_pins_receipt_until_settlement(
    outcome, continuation_postgres_dsn, monkeypatch
):
    from cayu.runtime import _invocation_lifecycle as lifecycle_module

    monkeypatch.setattr(lifecycle_module, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS", 6)
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            compact_before_recovery=outcome,
            postgres_dsn=continuation_postgres_dsn,
            session_id=f"postgres-retention-{outcome}",
        )
    )


def test_continuation_compacts_repeated_failed_admission_attempts() -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            fail_admission_attempts=2,
        )
    )


def test_continuation_can_be_retired_after_repeated_failed_admission_attempts() -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            use_owner=True,
            fail_admission_attempts=2,
            retire_after_failed_admission=True,
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_owner_reconciles_without_command_after_reopen(backend, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            lose_admission_acknowledgement=True,
            sqlite_path=tmp_path / "admission-recovery.sqlite" if backend == "sqlite" else None,
            reconcile_from_receipt=True,
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_owner_requires_runtime_context_and_exact_attribution(backend, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            sqlite_path=tmp_path / "owner-admission.sqlite" if backend == "sqlite" else None,
            use_owner=True,
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_claim_fences_retirement(backend, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            sqlite_path=tmp_path / "admission-retirement-race.sqlite"
            if backend == "sqlite"
            else None,
            use_owner=True,
            race_retirement=True,
            session_id=f"{backend}-admission-retirement-race-session",
        )
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_admission_releases_claim_for_retry(backend, tmp_path) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            sqlite_path=tmp_path / "admission-failure.sqlite" if backend == "sqlite" else None,
            use_owner=True,
            fail_admission_before_commit=True,
            session_id=f"{backend}-admission-failure-session",
        )
    )


def test_postgres_failed_admission_releases_claim_for_retry(continuation_postgres_dsn) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            postgres_dsn=continuation_postgres_dsn,
            use_owner=True,
            fail_admission_before_commit=True,
            session_id="postgres-admission-failure-session",
        )
    )


def test_postgres_admission_claim_fences_retirement(continuation_postgres_dsn) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            postgres_dsn=continuation_postgres_dsn,
            use_owner=True,
            race_retirement=True,
            session_id="postgres-admission-retirement-race-session",
        )
    )


@pytest.mark.parametrize("peer_commits_first", [False, True])
def test_postgres_released_claim_fences_concurrent_admission(
    continuation_postgres_dsn, peer_commits_first
) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            postgres_dsn=continuation_postgres_dsn,
            use_owner=True,
            race_released_claim=True,
            peer_commits_first=peer_commits_first,
            session_id=f"postgres-released-claim-{peer_commits_first}",
        )
    )


def test_postgres_owner_admission_uses_typed_lifecycle_boundary(continuation_postgres_dsn) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            postgres_dsn=continuation_postgres_dsn,
            use_owner=True,
            session_id="postgres-owner-admission-session",
        )
    )


def test_postgres_admission_reconciles_acknowledgement_loss(continuation_postgres_dsn) -> None:
    asyncio.run(
        _test_continuation_admission_uses_typed_lifecycle_boundary(
            postgres_dsn=continuation_postgres_dsn,
            lose_admission_acknowledgement=True,
            reconcile_from_receipt=True,
            session_id="postgres-recovery-admission-session",
        )
    )


def test_postgres_prepared_continuation_exclusion_settles_responsibility(
    continuation_postgres_dsn,
) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run() -> None:
        store = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            await _test_prepared_continuation_can_be_excluded(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_latch_cancellation_retains_pending_ownership(continuation_postgres_dsn) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run() -> None:
        store = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="postgres-cancellation-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            await _prepare_command(store, await _preparation(store, ticket))
            latch = ContinuationLatch(
                ticket=ticket,
                wait_receipt_digest="wait-receipt",
                outcome_kind="success",
                selected_manifest=(),
                disclosure_digest="disclosure",
                latch_key="latch-1",
                outcome_digest="outcome",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            entered = asyncio.Event()
            release = asyncio.Event()

            class WaitingReceiver:
                async def authenticate_continuation_latch(self, latch):
                    entered.set()
                    await release.wait()
                    return latch

            owner = SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=WaitingReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            task = asyncio.create_task(owner.latch(latch))
            await entered.wait()
            task.cancel("postgres cancellation")
            with pytest.raises(asyncio.CancelledError):
                await task
            pending = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert pending is not None and pending.latch is None
            release.set()
            recovered = await _latch(store, latch, receiver=_QualifiedReceiver())
            assert recovered.latch == latch
            await owner.drain()
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_fresh_process_reconciles_prepared_admission(tmp_path) -> None:
    async def prepare(path):
        store = SQLiteSessionStore(path)
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id="sqlite-admission-process-session",
                messages=[Message.text("user", "wait")],
            ),
            provider_name="continuation-provider",
            model="continuation-model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        await _prepare_command(store, await _preparation(store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        await _latch(store, latch, receiver=_QualifiedReceiver())
        waiting = await _park(store, ticket)
        await interrupt_and_release_test_invocation(store, admitted.session.id)
        session = await store.load(admitted.session.id)
        checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint(
            admitted.session.id
        )
        assert session is not None
        profile = execution_profile_from_session_metadata(session.metadata)
        event = runtime_interaction_started_event(
            CayuApp(session_store=store, enable_logging=False),
            session_id=session.id,
            interaction_id="continuation-process-interaction",
            agent_name=session.agent_name,
        )
        command = AdmitInvocationCommand(
            session_id=session.id,
            expected_session_instance_id=session.instance_id,
            expected_statuses=(SessionStatus.INTERRUPTED,),
            expected_run_epoch=session.run_epoch,
            expected_checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
            target_active_profile=admitted.active_invocation_profile.model_copy(
                update={
                    "interaction_id": event.interaction_id,
                    "run_epoch": session.run_epoch + 1,
                    "profile": profile,
                }
            ),
            checkpoint_patch=InvocationCheckpointPatch(
                mutation=runtime_publication_checkpoint_mutation(checkpoint, checkpoint)
            ),
            tool_capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
            interaction_started_event=event,
            expected_active_profile=admitted.active_invocation_profile,
        )
        input_digest, profile_digest, budget_digest = continuation_admission_inputs(command)
        prepared = ContinuationConsumption(
            ticket=waiting.ticket,
            latch=waiting.latch,
            continuation_id="process-admission-1",
            mode="inline",
            input_digest=input_digest,
            profile_digest=profile_digest,
            budget_digest=budget_digest,
            admission_command_digest=continuation_admission_digest(command),
            admission_expected_run_epoch=command.expected_run_epoch,
            receipt_stage="prepared",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        await _consume(store, prepared)
        await store.close()
        return prepared, command

    path = tmp_path / "continuation-admission-process.sqlite"
    prepared, command = asyncio.run(prepare(path))
    child = r"""
import asyncio, json, sys
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.runtime._session_continuation import ContinuationConsumption
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.runtime._invocation_lifecycle import AdmitInvocationCommand
from cayu.storage.sqlite import SQLiteSessionStore
from tests.core.test_session_continuation import _context
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata
from cayu.events import Event
from cayu.vaults.redaction import SecretRedactor

async def main():
    path, encoded, command_json = sys.argv[1:]
    store = SQLiteSessionStore(path)
    try:
        candidate = ContinuationConsumption.model_validate(json.loads(encoded))
        raw_command = json.loads(command_json)
        session = await store.load(candidate.ticket.session_id)
        assert session is not None
        raw_command["tool_capability_ceiling"] = tool_capability_ceiling_from_session_metadata(
            session.metadata
        )
        raw_command["interaction_started_event"] = Event.model_validate(
            raw_command["interaction_started_event"]
        )
        command = AdmitInvocationCommand.model_validate(raw_command)
        owner = SessionContinuationOwner(
            store=store,
            owner=candidate.ticket.owner,
            receiver=object(),
            receiver_capability=CapabilityDescriptor(
                owner=candidate.ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
            ),
            redactor=SecretRedactor(),
        )
        invocation = await _context(store, candidate.ticket.session_id, admission=command)
        _, record = await owner.admit(candidate, command, invocation=invocation)
        assert record.ticket.state == "CONSUMED"
        assert record.consumption is not None
        assert record.consumption.receipt_stage == "admitted"
        await owner.drain()
    finally:
        await store.close()

asyncio.run(main())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(("src", ".", env.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(path),
            json.dumps(prepared.model_dump(mode="json"), separators=(",", ":")),
            json.dumps(command.model_dump(mode="json"), separators=(",", ":")),
        ],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_postgres_fresh_process_reconciles_lost_latch_acknowledgement(
    continuation_postgres_dsn,
) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def prepare() -> ContinuationTicket:
        store = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id="postgres-process-continuation-session",
                messages=[Message.text("user", "wait")],
            ),
            provider_name="continuation-provider",
            model="continuation-model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        await _prepare_command(store, await _preparation(store, ticket))
        await store.close()
        return ticket

    ticket = asyncio.run(prepare())
    child = r"""
import asyncio, json, os, sys
from datetime import UTC, datetime
from cayu.runtime._session_continuation import ContinuationLatch, ContinuationUnavailable
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from tests.core.test_session_continuation import _latch

class Receiver:
    async def authenticate_continuation_latch(self, value):
        return value

class LostAckPostgres(PostgresSessionStore):
    session_continuation_version = 1
    invocation_lifecycle_command_version = 1

    async def publish_session_operation(self, session_id, **kwargs):
        await super().publish_session_operation(session_id, **kwargs)
        raise RuntimeError("simulated child acknowledgement loss")

async def main():
    dsn, encoded = sys.argv[1:]
    ticket_type = __import__(
        "cayu.runtime._session_continuation", fromlist=["ContinuationTicket"]
    ).ContinuationTicket
    ticket = ticket_type.model_validate(json.loads(encoded))
    store = LostAckPostgres(dsn, schema_mode=SchemaMode.VALIDATE)
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    try:
        await _latch(store, latch, receiver=Receiver())
    except ContinuationUnavailable as exc:
        if "simulated child acknowledgement loss" not in str(exc.__cause__):
            raise
        os._exit(17)
    raise AssertionError("latch acknowledgement should have been lost")

asyncio.run(main())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(("src", ".", env.get("PYTHONPATH", "")))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            continuation_postgres_dsn,
            json.dumps(ticket.model_dump(mode="json"), separators=(",", ":")),
        ],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 17, completed.stderr

    async def reopen_and_read() -> None:
        store = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            record = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert record is not None
            assert record.latch is not None
            assert record.latch.latch_key == "latch-1"
        finally:
            await store.close()

    asyncio.run(reopen_and_read())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["inline", "queued"])
@pytest.mark.parametrize(
    "fault",
    ["precommit", "claim_readback", "finalization", "supersession", "released_supersession"],
)
def test_owner_service_uses_real_resume_and_never_redispatches(
    backend, mode, fault, tmp_path
) -> None:
    async def run() -> None:
        class Loss(SessionStore):
            lose_claim = False
            lose_finalization = False
            lose_exclusion = False

            async def load_continuation_ticket(self, session_id, **kwargs):
                record = await super().load_continuation_ticket(session_id, **kwargs)
                if (
                    self.lose_claim
                    and record is not None
                    and record.consumption is not None
                    and record.consumption.admission_claimed
                ):
                    self.lose_claim = False
                    return None
                return record

            async def publish_session_operation(self, session_id, **kwargs):
                transform = kwargs.get("operation_transform")
                if transform is not None:

                    def fail_finalization(session, checkpoint, current):
                        publication = transform(session, checkpoint, current)
                        if self.lose_exclusion and any(
                            (value.get("retirement") or {}).get("reason") == "superseded"
                            for value in publication.operation_records.values()
                        ):
                            self.lose_exclusion = False
                            raise RuntimeError("supersession settlement unavailable")
                        if self.lose_finalization and any(
                            value.get("ticket", {}).get("state") == "CONSUMED"
                            for value in publication.operation_records.values()
                        ):
                            self.lose_finalization = False
                            raise RuntimeError("lost continuation finalization")
                        return publication

                    kwargs["operation_transform"] = fail_finalization
                return await super().publish_session_operation(session_id, **kwargs)

        class Memory(Loss, InMemorySessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        class SQLite(Loss, SQLiteSessionStore):
            session_continuation_version = 1
            invocation_lifecycle_command_version = 1

        store = Memory() if backend == "memory" else SQLite(tmp_path / "service.sqlite")
        try:
            provider = ScriptedModelProvider(
                (
                    ModelStreamEvent.text_delta("resumed"),
                    ModelStreamEvent.completed(
                        {"finish_reason": "stop", "model": "continuation-model"}
                    ),
                ),
                name="continuation-provider",
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="continuation-model"))
            admitted = await create_admitted_session(
                store,
                app=app,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="service-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name=provider.name,
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            invocation = await _context(store, ticket.session_id)
            owner = SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            intent = ContinuationWait.model_validate(
                ticket.model_dump(include=set(ContinuationWait.model_fields))
            )
            await owner.prepare(intent, invocation=invocation)
            latch = ContinuationLatch(
                ticket=ticket,
                wait_receipt_digest="wait-receipt",
                outcome_kind="success",
                selected_manifest=(),
                disclosure_digest="disclosure",
                latch_key="latch-1",
                outcome_digest="outcome",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            await owner.latch(latch)
            waiting = await owner.park(ticket, invocation=invocation)
            await interrupt_and_release_test_invocation(store, ticket.session_id)
            assert waiting.latch is not None
            service = ContinuationService(
                ticket=waiting.ticket,
                latch=waiting.latch,
                continuation_id="service-1",
                mode=mode,
                accepted_at=datetime.now(UTC).isoformat(),
            )
            request = ResumeRequest(
                session_id=ticket.session_id, messages=[Message.text("user", "continue")]
            )
            from cayu.runtime._session_continuation import ContinuationUnavailable

            incompatible_app = CayuApp(session_store=store, enable_logging=False)
            incompatible_app.register_provider(provider, default=True)
            incompatible_app.register_agent(
                AgentSpec(name="assistant", model="continuation-model"),
                tools=[UserInputTool()],
            )
            with pytest.raises(ContinuationUnavailable):
                await owner.service(incompatible_app, request, service)
            pending = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert pending is not None and pending.consumption is None
            assert len(provider.requests) == 0
            from cayu.runtime import _invocation_lifecycle as lifecycle_module

            original_apply = lifecycle_module.apply_invocation_lifecycle_command
            failed_once = False
            original_admit = owner.admit
            admission_commands = []
            rejected_commands = []

            async def exact_admit(consumption, command, *, invocation):
                admission_commands.append(command.model_dump(mode="json"))
                if len(admission_commands) > 1:
                    assert admission_commands[-1] == admission_commands[0]
                return await original_admit(consumption, command, invocation=invocation)

            owner.admit = exact_admit  # ty: ignore[invalid-assignment]

            async def fail_first_admission(store_arg, command_arg):
                nonlocal failed_once
                if (
                    fault == "supersession"
                    and type(command_arg) is AdmitInvocationCommand
                    and not failed_once
                ):
                    import contextvars

                    failed_once = True
                    winner = command_arg.model_copy(
                        update={
                            "interaction_source_messages": (
                                *command_arg.interaction_source_messages,
                                Message.text("user", "competing admission"),
                            ),
                        }
                    )
                    # Independent dispatch must not inherit the loser's private
                    # continuation claim. The typed command still owns provenance.
                    await asyncio.create_task(
                        original_apply(store_arg, winner), context=contextvars.Context()
                    )
                    raise RuntimeError("another invocation won admission")
                if (
                    fault in {"precommit", "released_supersession"}
                    and type(command_arg).__name__ == "AdmitInvocationCommand"
                    and not failed_once
                ):
                    failed_once = True
                    rejected_commands.append(command_arg)
                    raise RuntimeError("known pre-commit admission rejection")
                return await original_apply(store_arg, command_arg)

            lifecycle_module.apply_invocation_lifecycle_command = fail_first_admission  # ty: ignore[invalid-assignment]
            store.lose_claim = fault == "claim_readback"
            store.lose_finalization = fault == "finalization"
            store.lose_exclusion = fault == "supersession"
            try:
                with pytest.raises(ContinuationUnavailable):
                    await owner.service(app, request, service)
            finally:
                lifecycle_module.apply_invocation_lifecycle_command = original_apply
            retryable = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert retryable is not None
            assert retryable.consumption is not None
            assert retryable.consumption.receipt_stage == "prepared"
            assert retryable.consumption.admission_claimed == (
                fault not in {"precommit", "released_supersession"}
            )
            assert len(provider.requests) == 0
            if fault == "released_supersession":
                winner = rejected_commands[0].model_copy(
                    update={
                        "interaction_source_messages": (
                            *rejected_commands[0].interaction_source_messages,
                            Message.text("user", "competing admission after claim release"),
                        ),
                    }
                )
                await original_apply(runtime_checkpoint_session_store(store), winner)
            if fault == "claim_readback":
                # Changed application gates may refuse reconstruction. The old
                # generation must first be fenced so exclusion remains possible.
                with pytest.raises(ContinuationUnavailable):
                    await owner.service(incompatible_app, request, service)
                released = await store.load_continuation_ticket(
                    ticket.session_id,
                    registration_key=ticket.registration_key,
                    session_instance_id=ticket.session_instance_id,
                )
                assert released is not None and released.consumption is not None
                assert not released.consumption.admission_claimed
                assert (
                    released.consumption.admission_command_digest
                    == retryable.consumption.admission_command_digest
                )
            if backend == "sqlite":
                await owner.drain()
                assert isinstance(store, SQLiteSessionStore)
                await store.close()
                store = SQLite(tmp_path / "service.sqlite")
                app = CayuApp(session_store=store, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="assistant", model="continuation-model"))
                owner = SessionContinuationOwner(
                    store=store,
                    owner=ticket.owner,
                    receiver=_QualifiedReceiver(),
                    receiver_capability=CapabilityDescriptor(
                        owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                    ),
                    redactor=SecretRedactor(),
                )
            result = await owner.service(app, request, service)
            assert result.ticket.state == (
                "RETIRED" if fault in {"supersession", "released_supersession"} else "CONSUMED"
            )
            assert result.consumption is not None and result.consumption.mode == mode
            assert (
                result.consumption.admission_command_digest
                == retryable.consumption.admission_command_digest
            )
            expected_dispatches = (
                0 if fault in {"finalization", "supersession", "released_supersession"} else 1
            )
            assert len(provider.requests) == expected_dispatches
            assert await owner.service(app, request, service) == result
            assert len(provider.requests) == expected_dispatches
            with pytest.raises(ContinuationConflict):
                await owner.service(
                    app,
                    request.model_copy(update={"messages": [Message.text("user", "different")]}),
                    service,
                )
            with pytest.raises(ContinuationConflict):
                await owner.service(
                    app,
                    request,
                    service.model_copy(update={"mode": "queued" if mode == "inline" else "inline"}),
                )
            assert len(provider.requests) == expected_dispatches
            await owner.drain()
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_real_user_input_pause_blocks_continuation_admission(backend, tmp_path) -> None:
    async def run() -> None:
        from cayu.runtime._session_continuation import ContinuationUnavailable

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "human-pause.sqlite")
        )
        try:
            provider = ScriptedModelProvider(
                (
                    ModelStreamEvent.tool_call(
                        id="ask-1", name="ask_user", arguments={"question": "Approve?"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ),
                name="continuation-provider",
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="continuation-model"), tools=[UserInputTool()]
            )
            owner = None
            ticket = None
            invocation = None
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="human-wait",
                    messages=[Message.text("user", "ask")],
                )
            ):
                if event.type is EventType.SESSION_STARTED:
                    invocation = await _context(store, "human-wait", app=app)
                    session = await store.load("human-wait")
                    assert session is not None
                    ticket = _ticket(session, invocation.binding.interaction_id)
                    owner = SessionContinuationOwner(
                        store=store,
                        owner=ticket.owner,
                        receiver=_QualifiedReceiver(),
                        receiver_capability=CapabilityDescriptor(
                            owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                        ),
                        redactor=SecretRedactor(),
                    )
                    intent = ContinuationWait.model_validate(
                        ticket.model_dump(include=set(ContinuationWait.model_fields))
                    )
                    await owner.prepare(intent, invocation=invocation)
            assert owner is not None and ticket is not None and invocation is not None
            latch = ContinuationLatch(
                ticket=ticket,
                wait_receipt_digest="wait-receipt",
                outcome_kind="success",
                selected_manifest=(),
                disclosure_digest="disclosure",
                latch_key="latch-1",
                outcome_digest="outcome",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            await owner.latch(latch)
            waiting = await owner.park(ticket, invocation=invocation)
            assert waiting.latch is not None
            service = ContinuationService(
                ticket=waiting.ticket,
                latch=waiting.latch,
                continuation_id="human-blocked",
                mode="inline",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            checkpoint_before = await runtime_checkpoint_session_store(store).load_checkpoint(
                ticket.session_id
            )
            with pytest.raises(ContinuationUnavailable):
                await owner.service(
                    app,
                    ResumeRequest(
                        session_id=ticket.session_id, messages=[Message.text("user", "continue")]
                    ),
                    service,
                )
            retained = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert retained is not None and retained.consumption is None
            assert retained.ticket.state == "WAITING"
            checkpoint_after = await runtime_checkpoint_session_store(store).load_checkpoint(
                ticket.session_id
            )
            assert checkpoint_after == checkpoint_before
            assert len(provider.requests) == 1
            await owner.drain()
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_pending_continuation_fences_session_erasure_until_retired(backend, tmp_path) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "deletion.sqlite")
        )
        try:
            admitted = await create_admitted_session(
                store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="deletion-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            command = await _preparation(store, ticket)
            await _prepare_command(store, command)
            invocation = await _context(store, ticket.session_id)
            await interrupt_and_release_test_invocation(store, ticket.session_id)
            with pytest.raises(ValueError, match="continuation|responsibility"):
                await store.validate_session_closure_admission(ticket.session_id)
            current = await store.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert current is not None
            retirement = ContinuationRetirement(
                ticket=current.ticket,
                control_id="delete-retirement",
                reason="unavailable",
                retired_at=datetime.now(UTC).isoformat(),
            )
            owner = SessionContinuationOwner(
                store=store,
                owner=ticket.owner,
                receiver=_QualifiedReceiver(),
                receiver_capability=CapabilityDescriptor(
                    owner=ticket.owner, mutations=(), readbacks=(LATCH_FAMILY,)
                ),
                redactor=SecretRedactor(),
            )
            excluded = await owner.exclude(retirement, invocation=invocation)
            assert excluded.ticket.state == "RETIRED"
            await owner.drain()
            await store.validate_session_closure_admission(ticket.session_id)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


async def _test_prepared_continuation_can_be_excluded(store: SessionStore) -> None:
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id="prepared-exclusion-session",
            messages=[Message.text("user", "wait")],
        ),
        provider_name="continuation-provider",
        model="continuation-model",
    )
    ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
    await _prepare_command(store, await _preparation(store, ticket))
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    await _latch(store, latch, receiver=_QualifiedReceiver())
    waiting = await _park(store, ticket)
    assert waiting.latch is not None
    prepared = await _consume(store, _consumption(waiting.ticket, waiting.latch, "refused"))
    assert prepared.consumption is not None
    assert prepared.consumption.receipt_stage == "prepared"

    retirement = ContinuationRetirement(
        ticket=prepared.ticket,
        control_id="destination-refused",
        reason="failed",
        retired_at=datetime.now(UTC).isoformat(),
    )
    excluded = await _retire(store, retirement)
    assert excluded.ticket.state == "RETIRED"
    assert excluded.consumption is not None
    assert excluded.consumption.receipt_stage == "excluded"
    assert excluded.retirement is not None
    assert excluded.retirement.control_id == "destination-refused"
    assert await _retire(store, retirement) == excluded
    await interrupt_and_release_test_invocation(store, admitted.session.id)
    await store.validate_session_closure_admission(admitted.session.id)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_prepared_continuation_exclusion_settles_responsibility(backend, tmp_path) -> None:
    async def run() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "prepared-exclusion.sqlite")
        )
        try:
            await _test_prepared_continuation_can_be_excluded(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


async def _test_sqlite_reopen_preserves_latched_ticket(path) -> None:
    store = SQLiteSessionStore(path)
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id="sqlite-continuation-session",
            messages=[Message.text("user", "wait")],
        ),
        provider_name="continuation-provider",
        model="continuation-model",
    )
    ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
    await _prepare_command(store, await _preparation(store, ticket))
    latch = ContinuationLatch(
        ticket=ticket,
        wait_receipt_digest="wait-receipt",
        outcome_kind="success",
        selected_manifest=(),
        disclosure_digest="disclosure",
        latch_key="latch-1",
        outcome_digest="outcome",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    await _latch(store, latch, receiver=_QualifiedReceiver())
    await store.close()
    reopened = SQLiteSessionStore(path)
    record = await reopened.load_continuation_ticket(
        ticket.session_id,
        registration_key=ticket.registration_key,
        session_instance_id=ticket.session_instance_id,
    )
    assert record is not None
    assert record.latch == latch
    await reopened.close()


def test_sqlite_reopen_preserves_latched_ticket(tmp_path) -> None:
    asyncio.run(_test_sqlite_reopen_preserves_latched_ticket(tmp_path / "continuation.sqlite"))


def test_sqlite_independent_workers_elect_one_continuation(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "continuation-workers.sqlite"
        first_store = SQLiteSessionStore(path)
        admitted = await create_admitted_session(
            first_store,
            request=RunRequest(
                agent_name="assistant",
                session_id="sqlite-worker-continuation-session",
                messages=[Message.text("user", "wait")],
            ),
            provider_name="continuation-provider",
            model="continuation-model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        await _prepare_command(first_store, await _preparation(first_store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        await _latch(first_store, latch, receiver=_QualifiedReceiver())
        waiting = await _park(first_store, ticket)
        assert waiting.latch is not None
        second_store = SQLiteSessionStore(path)
        try:
            first = _consumption(waiting.ticket, waiting.latch, "worker-1")
            second = _consumption(waiting.ticket, waiting.latch, "worker-2")
            results = await asyncio.gather(
                _consume(first_store, first),
                _consume(second_store, second),
                return_exceptions=True,
            )
            assert sum(type(result) is ContinuationRecord for result in results) == 1
            assert sum(isinstance(result, ContinuationConflict) for result in results) == 1
        finally:
            await second_store.close()
            await first_store.close()

    asyncio.run(run())


def test_postgres_independent_workers_elect_one_continuation(continuation_postgres_dsn) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run() -> None:
        first_store = PostgresSessionStore(
            continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE
        )
        second_store = PostgresSessionStore(
            continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE
        )
        try:
            admitted = await create_admitted_session(
                first_store,
                request=RunRequest(
                    agent_name="assistant",
                    session_id="postgres-worker-continuation-session",
                    messages=[Message.text("user", "wait")],
                ),
                provider_name="continuation-provider",
                model="continuation-model",
            )
            ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
            await _prepare_command(first_store, await _preparation(first_store, ticket))
            latch = ContinuationLatch(
                ticket=ticket,
                wait_receipt_digest="wait-receipt",
                outcome_kind="success",
                selected_manifest=(),
                disclosure_digest="disclosure",
                latch_key="latch-1",
                outcome_digest="outcome",
                accepted_at=datetime.now(UTC).isoformat(),
            )
            await _latch(first_store, latch, receiver=_QualifiedReceiver())
            waiting = await _park(first_store, ticket)
            assert waiting.latch is not None
            first = _consumption(waiting.ticket, waiting.latch, "postgres-worker-1")
            second = _consumption(waiting.ticket, waiting.latch, "postgres-worker-2")
            results = await asyncio.gather(
                _consume(first_store, first),
                _consume(second_store, second),
                return_exceptions=True,
            )
            assert sum(type(result) is ContinuationRecord for result in results) == 1
            assert sum(isinstance(result, ContinuationConflict) for result in results) == 1
        finally:
            await second_store.close()
            await first_store.close()

    asyncio.run(run())


def test_postgres_reopen_preserves_latched_ticket(continuation_postgres_dsn) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run() -> None:
        store = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        admitted = await create_admitted_session(
            store,
            request=RunRequest(
                agent_name="assistant",
                session_id="postgres-continuation-session",
                messages=[Message.text("user", "wait")],
            ),
            provider_name="continuation-provider",
            model="continuation-model",
        )
        ticket = _ticket(admitted.session, admitted.active_invocation_profile.interaction_id)
        await _prepare_command(store, await _preparation(store, ticket))
        latch = ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest="wait-receipt",
            outcome_kind="success",
            selected_manifest=(),
            disclosure_digest="disclosure",
            latch_key="latch-1",
            outcome_digest="outcome",
            accepted_at=datetime.now(UTC).isoformat(),
        )
        await _latch(store, latch, receiver=_QualifiedReceiver())
        await store.close()

        reopened = PostgresSessionStore(continuation_postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            record = await reopened.load_continuation_ticket(
                ticket.session_id,
                registration_key=ticket.registration_key,
                session_instance_id=ticket.session_instance_id,
            )
            assert record is not None
            assert record.latch == latch
        finally:
            await reopened.close()

    asyncio.run(run())
