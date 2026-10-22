"""Context-view persistence composes with explicit state and one lock."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core.test_context_view_admission import _ownership, _selection
from tests.core.test_context_views import _manifest as _example_manifest
from tests.core.test_context_views import _replace_manifest
from tests.core.test_memory_queued_message_ownership import _assert_import_without_stores, _session

from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import _memory_context_views as views
from cayu.sessions._closure_records import ClosureProgressIndex
from cayu.sessions._context_selection_fence import (
    _CONTEXT_SELECTION_AUTHORITY,
    ContextViewSelectionExcluded,
)
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
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.restore_count = 0
        self.state = views.MemoryContextViewState(
            context_lifecycle_events={},
            context_ownership_operation_commitments={},
            context_ownership_operation_receipts={},
            context_selection_exclusions={},
            context_selection_receipts={},
            context_selection_request_commitments={},
            ownership_clock=self.clock,
            participant_session_bindings={},
            restore_selection=self.restore,
            context_publication_keys={},
            context_views={},
            sessions={session.id: session},
        )

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state

    def clock(self):
        assert self.lock.locked()
        return self.now

    def restore(self, snapshot):
        assert self.lock.locked()
        self.restore_count += 1
        self.state = self.state._replace(
            context_selection_receipts=snapshot.receipts,
            context_selection_request_commitments=snapshot.commitments,
            context_lifecycle_events=snapshot.events,
        )

    def snapshot(self):
        return deepcopy(tuple(dict(value) for value in self.state if not callable(value)))


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
            session_closure_progress=ClosureProgressIndex(),
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
        before = h.snapshot()
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
        assert h.snapshot() == before
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


def test_memory_context_view_selection_ownership_and_retention_compose():
    async def run():
        h = _Composition()
        manifest = _replace_manifest(
            _manifest(),
            compaction_json='{"input_frontier":0,"retained_output_frontier":1,"retained_suffix_frontier":1,"state":"uncompacted"}',
        )
        await views.publish_context_view(h.locked, manifest, publication_key="publication")
        request = _selection(manifest, "select")
        selected = await views.select_context_view(h.locked, request)
        assert await views.select_context_view(h.locked, request) == selected
        assert (
            await views.lookup_context_view_selection(h.locked, request.selection_key) == selected
        )
        decision = await views.read_context_view_selection_decision(h.locked, request)
        assert isinstance(decision, ExactMatch) and decision.receipt.state == "selected"
        with pytest.raises(ValueError, match="retention pin"):
            await views.validate_context_view_source_closure(h.locked, "session")
        await views.validate_context_view_compaction(
            h.locked, "session", manifest.transcript_cursor
        )
        # The publication owner calls this under a lock it already holds.
        async with h.lock:
            views.validate_compaction(
                "session",
                manifest.transcript_cursor,
                ownership_clock=h.clock,
                receipts=h.state.context_selection_receipts,
                views=h.state.context_views,
            )
        del h.state.context_views[manifest.view_id]
        with pytest.raises(ValueError, match="material is unavailable"):
            await views.validate_context_view_compaction(
                h.locked, "session", manifest.transcript_cursor
            )
        h.state.context_views[manifest.view_id] = manifest
        current = selected
        for operation in ("adopt", "transfer", "release"):
            intent = _ownership(current, operation, operation)
            current = await views.transition_ownership(h.locked, intent)
            assert await views.transition_ownership(h.locked, intent) == current
            assert (
                current.ownership_revision == {"adopt": 2, "transfer": 3, "release": 4}[operation]
            )
        await views.validate_context_view_source_closure(h.locked, "session")
        events = await views.read_context_view_lifecycle_events(
            h.locked, manifest.view_id, owner_participant=manifest.participant
        )
        assert [event.state for event in events] == ["adopted", "transferred", "released"]
        assert (
            len(await views.read_context_view_lifecycle_events(h.locked, manifest.view_id, limit=2))
            == 2
        )
        assert await views.select_context_view(h.locked, request) == current
        assert not h.lock.locked() and h.restore_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_memory_context_view_selection_restores_all_evidence_after_write_failure(failure):
    class RejectWrite(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise failure("selection write rejected")

    async def run():
        h = _Composition()
        manifest = _manifest()
        await views.publish_context_view(h.locked, manifest, publication_key="publication")
        old_request = _selection(manifest, "expiring", max_lifetime_seconds=1)
        old = await views.select_context_view(h.locked, old_request)
        h.now += timedelta(seconds=2)
        h.state = h.state._replace(
            context_selection_request_commitments=RejectWrite(
                h.state.context_selection_request_commitments
            )
        )
        before = h.snapshot()
        old_maps = h.state
        request = _selection(manifest, "new")
        with pytest.raises(failure, match="selection write rejected"):
            await views.select_context_view(h.locked, request)
        assert h.restore_count == 1 and not h.lock.locked()
        assert h.snapshot() == before
        assert h.state.context_selection_receipts is not old_maps.context_selection_receipts
        assert (
            h.state.context_selection_request_commitments
            is not old_maps.context_selection_request_commitments
        )
        assert h.state.context_lifecycle_events is not old_maps.context_lifecycle_events
        assert await views.lookup_context_view_selection(h.locked, old.selection_key) == old
        assert await views.lookup_context_view_selection(h.locked, request.selection_key) is None
        # A fresh operation resolves the restored dictionaries and can commit.
        selected = await views.select_context_view(h.locked, request)
        assert selected.state == "selected"
        assert (
            await views.lookup_context_view_selection(h.locked, old.selection_key)
        ).state == "expired"
        assert len(h.state.context_lifecycle_events) == 1

    asyncio.run(run())


def test_memory_context_view_ownership_failure_retains_expiry_evidence():
    async def run():
        h = _Composition()
        manifest = _manifest()
        await views.publish_context_view(h.locked, manifest, publication_key="publication")
        request = _selection(manifest, "expiring", max_lifetime_seconds=1)
        selected = await views.select_context_view(h.locked, request)
        h.now += timedelta(seconds=2)
        with pytest.raises(LookupError, match="has expired"):
            await views.transition_ownership(h.locked, _ownership(selected, "adopt", "adopt"))
        assert h.restore_count == 0
        assert (
            await views.lookup_context_view_selection(h.locked, request.selection_key)
        ).state == "expired"
        assert [
            event.state
            for event in await views.read_context_view_lifecycle_events(h.locked, manifest.view_id)
        ] == ["expired"]
        assert not h.state.context_ownership_operation_commitments
        assert not h.state.context_ownership_operation_receipts
        await views.validate_context_view_source_closure(h.locked, "session")
        assert not h.lock.locked()

    asyncio.run(run())


def test_memory_context_view_control_requires_authority_and_reports_exact_evidence():
    async def run():
        h = _Composition()
        manifest = _manifest()
        request = _selection(manifest, "exclude")
        assert isinstance(
            await views.read_context_view_selection_decision(h.locked, request), ExactNotFound
        )
        with pytest.raises(PermissionError):
            await views.control_selection(h.locked, request, authority=object(), exclude=True)
        assert not h.state.context_selection_exclusions
        reserved = await views.control_selection(
            h.locked, request, authority=_CONTEXT_SELECTION_AUTHORITY, exclude=False
        )
        assert reserved.state == "reserved"
        excluded = await views.control_selection(
            h.locked, request, authority=_CONTEXT_SELECTION_AUTHORITY, exclude=True
        )
        assert excluded.state == "excluded"
        assert (
            await views.control_selection(
                h.locked, request, authority=_CONTEXT_SELECTION_AUTHORITY, exclude=True
            )
            == excluded
        )
        with pytest.raises(ContextViewSelectionExcluded):
            await views.select_context_view(h.locked, request)
        changed = request.model_copy(update={"minimum_transcript_cursor": 99})
        assert isinstance(
            await views.read_context_view_selection_decision(h.locked, changed), ExactConflict
        )
        h.state.context_selection_exclusions[request.selection_key] = {}
        assert isinstance(
            await views.read_context_view_selection_decision(h.locked, request), ExactUnavailable
        )
        assert not h.lock.locked()

    asyncio.run(run())


def test_memory_context_view_selection_snapshot_failure_releases_lock():
    class RejectSnapshot(dict):
        def copy(self):
            raise RuntimeError("snapshot rejected")

    async def run():
        h = _Composition()
        h.state = h.state._replace(context_selection_receipts=RejectSnapshot())
        with pytest.raises(RuntimeError, match="snapshot rejected"):
            await views.select_context_view(h.locked, _selection(_manifest(), "select"))
        assert h.restore_count == 0 and not h.lock.locked()
        h.state = h.state._replace(context_selection_receipts={})
        await views.publish_context_view(h.locked, _manifest(), publication_key="publication")
        assert (
            await views.select_context_view(h.locked, _selection(_manifest(), "select"))
        ).state == "selected"

    asyncio.run(run())
