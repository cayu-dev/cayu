"""Session state contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
    run_request_with_registered_tool_ceiling,
)
from tests.core._run_lineage_support import assert_root_run_lineage
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events, collect_fork_events

import cayu.runtime._session_engine as session_engine_module
import cayu.sessions._model_completion_publication as model_completion_publication_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.base import project_runtime_managed_context_checkpoint
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions._checkpoint_preservation import _reserve_completion_result_event_publication
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    AUTOMATIC_RECALL_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.forks import fork_session_invocation
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.vaults.redaction import SecretRedactor


def _test_session_identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def test_checkpoint_replacement_preserves_current_runtime_state_without_resurrection():
    owner_expiry = datetime.now(UTC) + timedelta(hours=1)

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        session_id = "sess_preserve_current_cascade_marker"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        current_marker = {
            "attempt_id": "current-attempt",
            "interrupt_payload": {"interruption_type": "operator_requested"},
        }
        stale_marker = {
            "attempt_id": "stale-attempt",
            "interrupt_payload": {"interruption_type": "operator_requested"},
        }
        current_operations = {
            "version": 1,
            "active_operation_id": None,
            "records": {"current-operation": {"status": "completed"}},
        }
        stale_operations = {
            "version": 1,
            "active_operation_id": None,
            "records": {"stale-operation": {"status": "completed"}},
        }
        current_prompt_intents = {
            "version": 1,
            "records": {"current-transition": {"status": "prepared"}},
        }
        stale_prompt_intents = {
            "version": 1,
            "records": {"stale-transition": {"status": "prepared"}},
        }
        current_publication_digest = "a" * 64
        stale_publication_digest = "b" * 64
        current_owner_id = f"completion-result-owner:v1:{'c' * 64}"
        stale_owner_id = f"completion-result-owner:v1:{'d' * 64}"
        stale_publications = {
            "schema_version": 2,
            "reservations": {
                f"completion-result-publication:v1:{stale_publication_digest}": {
                    "schema_version": 2,
                    "publication_id": (
                        f"completion-result-publication:v1:{stale_publication_digest}"
                    ),
                    "authority_sha256": stale_publication_digest,
                    "owners": {
                        stale_owner_id: {
                            "schema_version": 2,
                            "owner_id": stale_owner_id,
                            "expires_at": owner_expiry.isoformat(),
                        }
                    },
                }
            },
        }
        await store.checkpoint(
            session_id,
            {
                "pending_session_interrupt": {
                    "reason": "current reason",
                    "interruption_type": "operator_requested",
                },
                "pending_interruption_cascade": current_marker,
                "session_operations": current_operations,
                "prompt_anatomy_transition_intents": current_prompt_intents,
                "current": True,
            },
        )
        await store._publish_completion_result_event_publication(
            session_id,
            checkpoint_transform=lambda _session, checkpoint, store_now: (
                _reserve_completion_result_event_publication(
                    checkpoint,
                    publication_id=(
                        f"completion-result-publication:v1:{current_publication_digest}"
                    ),
                    authority_sha256=current_publication_digest,
                    owner_id=current_owner_id,
                    owner_expires_at=owner_expiry,
                    now=store_now,
                )
            ),
            events=[],
        )
        stale_replacement = {
            "pending_session_interrupt": {
                "reason": "stale reason",
                "interruption_type": "operator_requested",
            },
            "pending_interruption_cascade": stale_marker,
            "session_operations": stale_operations,
            "prompt_anatomy_transition_intents": stale_prompt_intents,
            COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY: stale_publications,
            "replacement": True,
        }
        await store.transform_checkpoint(
            session_id,
            session_engine_module._replace_checkpoint_preserving_runtime_state(stale_replacement),
        )
        preserved = await store.load_checkpoint(session_id)
        await app._session_finalization.clear_pending_session_interrupt(session_id)
        await app._background_interruption_coordinator.clear_pending_interruption_cascade(
            session_id
        )
        await store._publish_completion_result_event_publication(
            session_id,
            checkpoint_transform=lambda _session, checkpoint, _store_now: {
                key: value
                for key, value in (checkpoint or {}).items()
                if key
                not in {
                    "session_operations",
                    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
                }
            },
            events=[],
        )
        await store.transform_checkpoint(
            session_id,
            session_engine_module._replace_checkpoint_preserving_runtime_state(stale_replacement),
        )
        return preserved, await store.load_checkpoint(session_id)

    preserved, after_clear = asyncio.run(run())

    assert preserved == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "pending_session_interrupt": {
            "reason": "current reason",
            "interruption_type": "operator_requested",
        },
        "pending_interruption_cascade": {
            "attempt_id": "current-attempt",
            "interrupt_payload": {"interruption_type": "operator_requested"},
        },
        "session_operations": {
            "version": 1,
            "active_operation_id": None,
            "records": {"current-operation": {"status": "completed"}},
        },
        "prompt_anatomy_transition_intents": {
            "version": 1,
            "records": {"current-transition": {"status": "prepared"}},
        },
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY: {
            "schema_version": 2,
            "reservations": {
                f"completion-result-publication:v1:{'a' * 64}": {
                    "schema_version": 2,
                    "publication_id": f"completion-result-publication:v1:{'a' * 64}",
                    "authority_sha256": "a" * 64,
                    "owners": {
                        f"completion-result-owner:v1:{'c' * 64}": {
                            "schema_version": 2,
                            "owner_id": f"completion-result-owner:v1:{'c' * 64}",
                            "expires_at": owner_expiry.isoformat(),
                        }
                    },
                }
            },
        },
        "replacement": True,
    }
    assert after_clear == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "prompt_anatomy_transition_intents": {
            "version": 1,
            "records": {"current-transition": {"status": "prepared"}},
        },
        "replacement": True,
    }


def test_fork_does_not_inherit_source_result_event_publication_reservation() -> None:
    from cayu.runtime._fork_source_snapshot import fork_source_checkpoint_sha256

    authority_sha256 = "c" * 64
    publication_id = f"completion-result-publication:v1:{authority_sha256}"
    owner_id = f"completion-result-owner:v1:{'d' * 64}"
    owner_expiry = datetime.now(UTC) + timedelta(hours=1)
    checkpoint_without_reservation = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "resumable": {"value": True},
    }
    checkpoint_with_reservation = {
        **checkpoint_without_reservation,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY: {
            "schema_version": 2,
            "reservations": {
                publication_id: {
                    "schema_version": 2,
                    "publication_id": publication_id,
                    "authority_sha256": authority_sha256,
                    "owners": {
                        owner_id: {
                            "schema_version": 2,
                            "owner_id": owner_id,
                            "expires_at": owner_expiry.isoformat(),
                        }
                    },
                }
            },
        },
    }
    assert fork_source_checkpoint_sha256(
        checkpoint_with_reservation
    ) == fork_source_checkpoint_sha256(checkpoint_without_reservation)

    async def run():
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor("completion"),
        )
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.text_delta("source complete"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source_session_id = "sess_result_publication_fork_source"
        child_session_id = "sess_result_publication_fork_child"
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=source_session_id,
                messages=[Message.text("user", "finish the source")],
            ),
        )
        await store._publish_completion_result_event_publication(
            source_session_id,
            checkpoint_transform=lambda _session, checkpoint, store_now: (
                _reserve_completion_result_event_publication(
                    checkpoint,
                    publication_id=publication_id,
                    authority_sha256=authority_sha256,
                    owner_id=owner_id,
                    owner_expires_at=owner_expiry,
                    now=store_now,
                )
            ),
            events=[],
        )

        events = await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id=source_session_id,
                session_id=child_session_id,
            ),
        )
        return (
            events,
            await store.load_checkpoint(source_session_id),
            await store.load_checkpoint(child_session_id),
        )

    events, source_checkpoint, child_checkpoint = asyncio.run(run())

    assert any(event.type is EventType.SESSION_FORKED for event in events)
    assert source_checkpoint is not None
    assert COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY in source_checkpoint
    assert child_checkpoint is not None
    assert "invocation_configuration" in source_checkpoint
    assert "invocation_configuration" not in child_checkpoint
    assert COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY not in child_checkpoint


@pytest.mark.parametrize(
    "source_owned_key",
    (
        "session_operations",
        "session_run_operation",
        "queued_dispatch_terminal_receipts",
        "prompt_anatomy_transition_intents",
        "environment_factory_pending_disposals",
        "environment_factory_retired_disposals",
        "environment_factory_allocation_generations",
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        "invocation_configuration",
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        model_completion_publication_module.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
        AUTOMATIC_RECALL_CHECKPOINT_KEY,
    ),
)
def test_fork_source_checkpoint_digest_ignores_source_owned_runtime_state(
    source_owned_key: str,
) -> None:
    from cayu.runtime._fork_source_snapshot import (
        fork_source_checkpoint_projection,
        fork_source_checkpoint_sha256,
    )

    child_state = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "application_checkpoint": 14,
    }

    source_state = {**child_state, source_owned_key: {"source_owned": True}}

    assert fork_source_checkpoint_projection(source_state) == child_state
    assert fork_source_checkpoint_sha256(source_state) == fork_source_checkpoint_sha256(child_state)


def test_context_projection_hides_terminal_decision_authority() -> None:
    ordinary_checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "application_checkpoint": 14,
    }
    source = {
        **ordinary_checkpoint,
        "invocation_configuration": {"private": "admitted"},
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: {"private": "active"},
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: {"private": "settled"},
    }

    assert project_runtime_managed_context_checkpoint(source) == ordinary_checkpoint


def test_pending_terminal_interruption_blocks_fork_snapshot_and_publication() -> None:
    async def run() -> tuple[Session | None, Session | None, dict[str, Any] | None]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        source_session_id = "sess_pending_terminal_interruption_fork_guard"
        child_session_id = "sess_pending_terminal_interruption_fork"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=source_session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(source_session_id, SessionStatus.INTERRUPTED)
        interrupt_payload = {
            "reason": "operator interruption lost its terminal acknowledgement",
            "metadata": {"owner": "source"},
            "interruption_type": "operator_requested",
            "interruption_request_id": "source-interruption-request",
        }
        await store.checkpoint(
            source_session_id,
            {"pending_session_interrupt": interrupt_payload},
        )

        rejection = "incomplete terminal interruption evidence"
        with pytest.raises(RuntimeError, match=rejection):
            await app.snapshot_fork_source(source_session_id)
        with pytest.raises(RuntimeError, match=rejection):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=source_session_id,
                        session_id=child_session_id,
                    )
                )
            ]

        return (
            await store.load(source_session_id),
            await store.load(child_session_id),
            await store.load_checkpoint(source_session_id),
        )

    source, child, checkpoint = asyncio.run(run())

    assert source is not None
    assert source.status is SessionStatus.INTERRUPTED
    assert child is None
    assert checkpoint == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "pending_session_interrupt": {
            "reason": "operator interruption lost its terminal acknowledgement",
            "metadata": {"owner": "source"},
            "interruption_type": "operator_requested",
            "interruption_request_id": "source-interruption-request",
        },
    }


def test_pending_interruption_cascade_blocks_resume_and_fork():
    async def run():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(FakeProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "sess_pending_cascade_continuation_guard"
        await store.create(
            run_request_with_registered_tool_ceiling(
                app,
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)
        await store.checkpoint(
            session_id,
            {
                "pending_interruption_cascade": {
                    "attempt_id": "pending-cascade-attempt",
                    "interrupt_payload": {"interruption_type": "operator_requested"},
                }
            },
        )

        with pytest.raises(RuntimeError, match="incomplete background interruption cascade"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
        with pytest.raises(RuntimeError, match="incomplete background interruption cascade"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=session_id,
                        session_id="sess_pending_cascade_fork",
                    )
                )
            ]
        await store.update_status(session_id, SessionStatus.FAILED)
        with pytest.raises(RuntimeError, match="incomplete background interruption cascade"):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=session_id,
                        session_id="sess_pending_cascade_fork_without_checkpoint",
                        copy_checkpoint=False,
                    )
                )
            ]
        return (
            await store.load(session_id),
            await store.load("sess_pending_cascade_fork"),
            await store.load("sess_pending_cascade_fork_without_checkpoint"),
            await store.load_checkpoint(session_id),
        )

    source, copied_fork, checkpointless_fork, checkpoint = asyncio.run(run())

    assert source.status == SessionStatus.FAILED
    assert copied_fork is None
    assert checkpointless_fork is None
    assert checkpoint == {
        "pending_interruption_cascade": {
            "attempt_id": "pending-cascade-attempt",
            "interrupt_payload": {"interruption_type": "operator_requested"},
        }
    }


def test_in_memory_session_store_rejects_fork_status_mismatch():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_status_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="Fork status must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_store_fork_status_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.RUNNING,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    asyncio.run(run_operations())


def test_in_memory_session_store_rejects_fork_provider_mismatch():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_provider_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        with pytest.raises(ValueError, match="Fork provider_name must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_store_fork_provider_child",
                    agent_name="assistant",
                    provider_name="other",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )

    asyncio.run(run_operations())


def test_in_memory_session_store_transforms_current_checkpoint_during_fork():
    store = InMemorySessionStore()

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_store_fork_checkpoint_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})

        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_store_fork_checkpoint_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=None,
            checkpoint_transform=lambda _session, checkpoint: {
                "copied_version": checkpoint["version"] if checkpoint else None
            },
        )

        assert await store.load_checkpoint("sess_store_fork_checkpoint_child") == {
            "copied_version": 2
        }

    asyncio.run(run_operations())


def test_in_memory_session_store_rejects_duplicate_session_ids():
    store = InMemorySessionStore()
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_duplicate",
        messages=[Message.text("user", "hi")],
    )

    asyncio.run(store.create(request, identity=_test_session_identity()))

    with pytest.raises(ValueError, match="Session already exists"):
        asyncio.run(store.create(request, identity=_test_session_identity()))


def test_in_memory_session_store_create_rejects_invalid_request_type():
    store = InMemorySessionStore()

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create({"agent_name": "assistant"}, identity=_test_session_identity()))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="RunRequest"):
        asyncio.run(store.create(None, identity=_test_session_identity()))  # type: ignore[arg-type]


def test_in_memory_session_store_revalidates_constructed_run_requests():
    class BadString(str):
        def __bool__(self):
            raise RuntimeError("session id bool should not run")

        def strip(self):
            raise RuntimeError("session id strip should not run")

    class BadMessages(list):
        def __iter__(self):
            raise RuntimeError("messages iteration should not run")

    store = InMemorySessionStore()

    session = asyncio.run(
        store.create(
            RunRequest.model_construct(
                agent_name="assistant",
                session_id=BadString("sess_bad"),
                messages=[Message.text("user", "hi")],
                metadata={},
                max_steps=1,
            ),
            identity=_test_session_identity(),
        )
    )
    assert session.id == "sess_bad"
    assert type(session.id) is str

    with pytest.raises(ValueError, match="messages must be a list"):
        asyncio.run(
            store.create(
                RunRequest.model_construct(
                    agent_name="assistant",
                    session_id="sess_bad_messages",
                    messages=BadMessages([]),
                    metadata={},
                    max_steps=1,
                ),
                identity=_test_session_identity(),
            )
        )

    assert asyncio.run(store.load("sess_bad")) is not None
    assert asyncio.run(store.load("sess_bad_messages")) is None


def test_in_memory_session_store_rejects_loading_events_for_missing_session():
    store = InMemorySessionStore()

    with pytest.raises(KeyError, match="Session not found"):
        asyncio.run(store.load_events("missing"))


@pytest.mark.parametrize("session_id", ["", " ", 123])
def test_in_memory_session_store_rejects_invalid_session_ids(session_id):
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_id_target",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
    )
    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_invalid_id_target",
    )

    for method in [
        store.load,
        store.load_events,
        lambda value: store.update_status(value, SessionStatus.COMPLETED),
        lambda value: store.append_event(value, event),
        lambda value: store.checkpoint(value, {"ok": True}),
        store.load_checkpoint,
    ]:
        with pytest.raises(ValueError, match="session_id"):
            asyncio.run(method(session_id))  # type: ignore[arg-type]


def test_in_memory_session_store_rejects_invalid_event_type_on_append():
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_event_type",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
    )

    with pytest.raises(TypeError, match="Event"):
        asyncio.run(store.append_event("sess_event_type", {"session_id": "sess_event_type"}))  # type: ignore[arg-type]


def test_in_memory_session_store_revalidates_constructed_events_on_append():
    class BadPayload(dict):
        def items(self):
            raise RuntimeError("store event payload traversal should not run")

    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_constructed_event",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
    )

    with pytest.raises(ValueError, match="`id` cannot be blank"):
        asyncio.run(
            store.append_event(
                "sess_constructed_event",
                Event.model_construct(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="sess_constructed_event",
                    id=" ",
                    payload={},
                ),
            )
        )

    invalid_event = Event(type=EventType.MODEL_TEXT_DELTA, session_id="sess_constructed_event")
    object.__setattr__(invalid_event, "payload", BadPayload({"delta": "hello"}))
    with pytest.raises(ValueError, match="JSON-compatible"):
        asyncio.run(store.append_event("sess_constructed_event", invalid_event))


def test_in_memory_session_store_isolates_request_metadata():
    store = InMemorySessionStore()
    metadata = {"nested": {"value": "original"}}
    identity = _test_session_identity()
    request = RunRequest(
        agent_name="assistant",
        session_id="sess_metadata_isolation",
        messages=[Message.text("user", "hi")],
        metadata=metadata,
    )

    asyncio.run(store.create(request, identity=identity))
    metadata["nested"]["value"] = "mutated"
    session = asyncio.run(store.load("sess_metadata_isolation"))

    assert session is not None
    assert_root_run_lineage(session)
    assert {
        key: value
        for key, value in session.metadata.items()
        if key not in {"cayu:runtime_build_provenance", "cayu:run_lineage"}
    } == {"nested": {"value": "original"}}
    assert session.runtime_build_provenance == identity.runtime_build_provenance


@pytest.mark.parametrize("status", ["bad", "completed", 123])
def test_in_memory_session_store_rejects_invalid_status_values(status):
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_status_validation",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
    )

    with pytest.raises(ValueError, match="SessionStatus"):
        asyncio.run(store.update_status("sess_status_validation", status))  # type: ignore[arg-type]


def test_in_memory_session_store_checkpoints_json_state_and_isolates_it():
    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_checkpoint",
                messages=[Message.text("user", "hi")],
            ),
            identity=_test_session_identity(),
        )
    )
    state = {"nested": {"value": "original"}}

    asyncio.run(store.checkpoint("sess_checkpoint", state))
    state["nested"]["value"] = "mutated"
    loaded = asyncio.run(store.load_checkpoint("sess_checkpoint"))

    assert loaded == {"nested": {"value": "original"}}

    with pytest.raises(ValueError, match="JSON-compatible"):
        asyncio.run(store.checkpoint("sess_checkpoint", {"bad": object()}))

    with pytest.raises(ValueError, match="dictionary"):
        asyncio.run(store.checkpoint("sess_checkpoint", []))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dictionary"):
        asyncio.run(store.checkpoint("sess_checkpoint", "bad"))  # type: ignore[arg-type]


def test_resume_request_rejects_blank_model_target():
    with pytest.raises(ValueError, match="model"):
        ResumeRequest(
            session_id="sess_blank_model",
            messages=[Message.text("user", "hi")],
            target=ModelTarget(provider_name="fake", model=" "),
        )
