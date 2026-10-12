"""Context-view persistence composes with explicit state and one lock."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.test_context_views import _manifest as _example_manifest
from tests.core.test_context_views import _replace_manifest
from tests.core.test_memory_queued_message_ownership import _assert_import_without_stores, _session

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import _memory_context_views as views
from cayu.sessions._execution_profile_checkpoint import (
    EXECUTION_PROFILE_METADATA_KEY,
    execution_profile_session_metadata,
)
from cayu.sessions._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    ModelStepPublicationCheckpoint,
)
from cayu.sessions._model_stage import MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
from cayu.sessions.context_views import ParticipantSessionBinding
from cayu.sessions.messaging import SessionMessageDeliveryMode
from cayu.sessions.records import RUNTIME_BUILD_PROVENANCE_METADATA_KEY, EventRecord


def _manifest():
    return _replace_manifest(
        _example_manifest(), source_session_instance_id="11111111-1111-4111-8111-111111111111"
    )


class _Composition:
    """Independent mutable state, without constructing or retaining a session store."""

    def __init__(self):
        self.lock = asyncio.Lock()
        session = _session("session").model_copy(
            update={"instance_id": _manifest().source_session_instance_id}
        )
        self.state = views.MemoryContextViewState(
            context_publication_keys={}, context_views={}, sessions={session.id: session}
        )

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state


class _SourceComposition:
    def __init__(self):
        self.lock = asyncio.Lock()
        session = _session("session").model_copy(
            update={"instance_id": _manifest().source_session_instance_id}
        )
        identity = profiled_session_identity(provider_name="fake", model="fake")
        profile = identity.execution_profile
        session.metadata[RUNTIME_BUILD_PROVENANCE_METADATA_KEY] = (
            identity.runtime_build_provenance.model_dump(mode="json")
        )
        assert profile is not None
        session.metadata[EXECUTION_PROFILE_METADATA_KEY] = execution_profile_session_metadata(
            profile
        )
        binding = ParticipantSessionBinding(
            application_scope="app",
            participant=_manifest().participant,
            session_id=session.id,
            session_instance_id=session.instance_id,
            lifecycle_revision=1,
            configuration_revision=1,
            admission_generation=1,
            creator_commitment="creator",
            authorization_commitment="authorization",
            initial_input_commitment="initial",
            request_commitment="request",
            execution_profile_commitment=profile.fingerprint,
            historical_definition_json="{}",
            creation_key="creation",
        )
        event = Event(
            id="completion",
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            interaction_id="interaction",
            payload={"execution_profile_fingerprint": profile.fingerprint},
        )
        pointer = ModelStepPublicationCheckpoint(
            logical_step_id="step",
            stage_id="stage",
            source_transcript_cursor=0,
            transcript_end_cursor=1,
            completion_event_id=event.id,
            classification={"type": "final"},
            assistant_message_published=True,
        )
        record = EventRecord(sequence=1, event=event)
        self.state = views.MemoryContextViewSourceState(
            checkpoints={
                session.id: {
                    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY: pointer.model_dump(mode="json")
                }
            },
            event_records_by_id={(session.id, event.id): record},
            participant_session_bindings={session.id: binding},
            pending_session_messages={},
            session_closure_progress={},
            session_event_records={session.id: [record]},
            session_operation_records={},
            sessions={session.id: session},
            transcript_interaction_ids={session.id: ["interaction"]},
            transcripts={session.id: [Message.text("assistant", "done")]},
            validate_publication_material=self.validate_material,
        )

    def validate_material(self, receipt):
        assert self.lock.locked()
        raise AssertionError("A final model turn has no tool publication to validate.")

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state


def test_memory_context_views_import_without_stores():
    _assert_import_without_stores("cayu.sessions._memory_context_views")


def test_memory_context_view_publication_replay_and_detached_readback():
    async def run():
        h = _Composition()
        manifest = _manifest()
        published = await views.publish_context_view(
            h.locked, manifest, publication_key="publication"
        )
        object.__setattr__(published, "view_id", "changed")
        assert await views.lookup_context_view_publication(h.locked, "publication") == manifest
        assert (
            await views.publish_context_view(h.locked, manifest, publication_key="publication")
            == manifest
        )
        read = await views.read_context_view(
            h.locked, manifest.view_id, source_session_id="session"
        )
        object.__setattr__(read.view, "view_id", "changed")
        assert h.state.context_views[manifest.view_id] == manifest
        before = deepcopy(h.state)
        with pytest.raises(ValueError, match="conflicts with its manifest"):
            await views.publish_context_view(
                h.locked,
                _replace_manifest(manifest, view_id="different"),
                publication_key="publication",
            )
        with pytest.raises(ValueError, match="already bound"):
            await views.publish_context_view(h.locked, manifest, publication_key="different")
        with pytest.raises(LookupError, match="unavailable"):
            await views.read_context_view(h.locked, manifest.view_id, source_session_id="wrong")
        assert h.state == before
        assert not h.lock.locked()

    asyncio.run(run())


def test_memory_context_view_resolves_replaced_indexes_after_lock():
    async def run():
        h = _Composition()
        async with h.lock:
            pending = asyncio.create_task(
                views.publish_context_view(h.locked, _manifest(), publication_key="publication")
            )
            await asyncio.sleep(0)
            assert not pending.done()
            old = h.state
            h.state = h.state._replace(context_publication_keys={}, context_views={})
        assert await pending == _manifest()
        assert not old.context_views and not old.context_publication_keys
        assert h.state.context_publication_keys == {"publication": "view-1"}

    asyncio.run(run())


def test_memory_context_view_source_snapshot_detaches_and_keeps_current_flags():
    async def run():
        h = _SourceComposition()
        mode = next(iter(SessionMessageDeliveryMode))
        h.state.pending_session_messages[("session", mode)] = deque()
        h.state.session_operation_records["session"] = {
            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY: {}
        }
        h.state.transcripts["session"].append(Message.text("user", "later input"))
        h.state.transcript_interaction_ids["session"].append("later-interaction")
        before = deepcopy(h.state.checkpoints)
        snapshot = await views.capture_completed_turn_snapshot(h.locked, "session")
        assert snapshot.has_queued_input and snapshot.has_active_model_stage
        assert not snapshot.has_closure_owner
        assert snapshot.current_transcript_cursor == 2
        assert snapshot.publication.transcript_end_cursor == 1
        assert len(snapshot.publication.records) == 1
        assert snapshot.publication.records[0].message == Message.text("assistant", "done")
        snapshot.current_session.labels["changed"] = "current"
        snapshot.publication.session.labels["changed"] = "historical"
        snapshot.publication.completion_event.payload["changed"] = True
        snapshot.checkpoint.clear()
        assert h.state.checkpoints == before
        assert not h.state.sessions["session"].labels
        assert "changed" not in h.state.event_records_by_id[("session", "completion")].event.payload
        assert not h.lock.locked()

    asyncio.run(run())


@pytest.mark.parametrize("broken", ["binding", "event", "transcript"])
def test_memory_context_view_source_rejects_inexact_evidence(broken):
    async def run():
        h = _SourceComposition()
        if broken == "binding":
            h.state.participant_session_bindings.clear()
        elif broken == "event":
            h.state.event_records_by_id.clear()
        else:
            h.state.transcripts["session"].clear()
        before = deepcopy(tuple(value for value in h.state if not callable(value)))
        with pytest.raises((PermissionError, ValueError)):
            await views.capture_completed_turn_snapshot(h.locked, "session")
        assert tuple(value for value in h.state if not callable(value)) == before
        assert not h.lock.locked()

    asyncio.run(run())


def test_memory_context_view_publication_cancellation_does_not_publish():
    async def run():
        h = _Composition()
        async with h.lock:
            pending = asyncio.create_task(
                views.publish_context_view(h.locked, _manifest(), publication_key="publication")
            )
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert not h.state.context_views and not h.state.context_publication_keys
        assert (
            await views.publish_context_view(h.locked, _manifest(), publication_key="publication")
            == _manifest()
        )

    asyncio.run(run())
