"""Shared session-store mcp contract cases."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _identity,
    _mcp_test_manifest_hash,
    _open_store,
)

from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    _mcp_manifest_session_ref,
)
from cayu.sessions.requests import (
    RunRequest,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_atomically_fences_mcp_manifest_baselines(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = []
            for index in range(2):
                sessions.append(
                    await store.create(
                        RunRequest(
                            agent_name="assistant",
                            session_id=f"mcp_manifest_cas_{index}",
                            messages=[Message.text("user", "check manifest")],
                        ),
                        identity=_identity(),
                    )
                )

            history_key = "sha256:" + "1" * 64
            publications = []
            for index, session in enumerate(sessions):
                event = Event(
                    id=f"evt_mcp_manifest_cas_{index}",
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "2" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "6" * 64,
                            server_hash=f"sha256:{index + 5}" + "0" * 63,
                        ),
                        "source_manifest_hash": "sha256:" + "6" * 64,
                        "server_hash": f"sha256:{index + 5}" + "0" * 63,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                baseline = McpManifestBaseline(
                    history_key=history_key,
                    generation=1,
                    manifest_identity=event.payload["manifest_identity"],
                    manifest_hash=event.payload["manifest_hash"],
                    source_manifest_hash=event.payload["source_manifest_hash"],
                    server_hash=event.payload["server_hash"],
                    tools=(),
                    exposed_tools=(),
                    accepted_session_ref=_mcp_manifest_session_ref(session.id),
                    accepted_event_id=event.id,
                    accepted_at=event.timestamp,
                )
                publications.append(
                    store.compare_and_publish_mcp_manifest_checks(
                        session.id,
                        expected_generations={history_key: None},
                        baseline_updates={history_key: baseline},
                        events=[event],
                    )
                )

            results = await asyncio.gather(*publications)
            assert [result.published for result in results].count(True) == 1
            assert [result.published for result in results].count(False) == 1

            loaded = await store.load_mcp_manifest_baselines((history_key,))
            baselines = loaded.baselines
            assert baselines[history_key].generation == 1
            accepted_events = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert len(accepted_events) == 1
            assert baselines[history_key].accepted_event_id == accepted_events[0].event.id
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rolls_back_mcp_baseline_when_event_insert_fails(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_atomic_failure",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            duplicate_event_id = "evt_mcp_manifest_duplicate"
            await store.append_event(
                session.id,
                Event(
                    id=duplicate_event_id,
                    type=EventType.SESSION_STARTED,
                    session_id=session.id,
                ),
            )
            history_key = "sha256:" + "6" * 64
            candidate = Event(
                id=duplicate_event_id,
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "7" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash="sha256:" + "a" * 64,
                        server_hash="sha256:" + "9" * 64,
                    ),
                    "source_manifest_hash": "sha256:" + "a" * 64,
                    "server_hash": "sha256:" + "9" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=candidate.payload["manifest_identity"],
                manifest_hash=candidate.payload["manifest_hash"],
                source_manifest_hash=candidate.payload["source_manifest_hash"],
                server_hash=candidate.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=candidate.id,
                accepted_at=candidate.timestamp,
            )

            with pytest.raises(ValueError, match="Event already exists"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=[candidate],
                )

            loaded = await store.load_mcp_manifest_baselines((history_key,))
            assert loaded.baselines == {}
            manifest_events = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert manifest_events == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_revalidates_constructed_mcp_baselines(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_constructed_baseline",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "a" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "b" * 64,
                    "manifest_hash": "sha256:" + "c" * 64,
                    "source_manifest_hash": "sha256:" + "e" * 64,
                    "server_hash": "sha256:" + "d" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            invalid = McpManifestBaseline.model_construct(
                history_key=history_key,
                generation=1,
                manifest_identity="not-a-hash",
                manifest_hash=event.payload["manifest_hash"],
                source_manifest_hash=event.payload["source_manifest_hash"],
                server_hash=event.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=event.id,
                accepted_at=event.timestamp,
            )

            with pytest.raises(ValueError, match="SHA-256"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: invalid},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("status", ["first_seen", "changed"])
def test_session_store_conformance_rejects_accepted_mcp_transition_without_baseline(
    session_store_case,
    status: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_manifest_missing_baseline_{status}",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "1" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "2" * 64,
                    "manifest_hash": "sha256:" + "3" * 64,
                    "source_manifest_hash": "sha256:" + "5" * 64,
                    "server_hash": "sha256:" + "4" * 64,
                    "status": status,
                    "outcome": "accepted",
                },
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_partially_updated_mcp_transition_batch(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_partial_baseline_batch",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_keys = ("sha256:" + "5" * 64, "sha256:" + "6" * 64)
            events = [
                Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + ("7" if index == 0 else "8") * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "c" * 64,
                            server_hash="sha256:" + "b" * 64,
                        ),
                        "source_manifest_hash": "sha256:" + "c" * 64,
                        "server_hash": "sha256:" + "b" * 64,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                for index, history_key in enumerate(history_keys)
            ]
            first = events[0]
            baseline = McpManifestBaseline(
                history_key=history_keys[0],
                generation=1,
                manifest_identity=first.payload["manifest_identity"],
                manifest_hash=first.payload["manifest_hash"],
                source_manifest_hash=first.payload["source_manifest_hash"],
                server_hash=first.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=first.id,
                accepted_at=first.timestamp,
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={key: None for key in history_keys},
                    baseline_updates={history_keys[0]: baseline},
                    events=events,
                )

            assert (await store.load_mcp_manifest_baselines(history_keys)).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_duplicate_accepted_mcp_transitions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_manifest_duplicate_accepted_transitions",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "c" * 64
            events = [
                Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "d" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash="sha256:" + "2" * 64,
                            server_hash="sha256:" + "1" * 64,
                        ),
                        "source_manifest_hash": "sha256:" + "2" * 64,
                        "server_hash": "sha256:" + "1" * 64,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                for index in range(2)
            ]
            first = events[0]
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=first.payload["manifest_identity"],
                manifest_hash=first.payload["manifest_hash"],
                source_manifest_hash=first.payload["source_manifest_hash"],
                server_hash=first.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=first.id,
                accepted_at=first.timestamp,
            )

            with pytest.raises(
                ValueError,
                match="Every accepted MCP manifest transition",
            ):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=events,
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_accepts_unchanged_mcp_event_without_update(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = [
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"mcp_manifest_unchanged_publication_{index}",
                        messages=[Message.text("user", "check manifest")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            history_key = "sha256:" + "2" * 64
            accepted = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[0].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "3" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash="sha256:" + "6" * 64,
                        server_hash="sha256:" + "5" * 64,
                    ),
                    "source_manifest_hash": "sha256:" + "6" * 64,
                    "server_hash": "sha256:" + "5" * 64,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=accepted.payload["manifest_identity"],
                manifest_hash=accepted.payload["manifest_hash"],
                source_manifest_hash=accepted.payload["source_manifest_hash"],
                server_hash=accepted.payload["server_hash"],
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(sessions[0].id),
                accepted_event_id=accepted.id,
                accepted_at=accepted.timestamp,
            )
            first_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[0].id,
                expected_generations={history_key: None},
                baseline_updates={history_key: baseline},
                events=[accepted],
            )
            assert first_result.published is True

            unchanged = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[1].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": baseline.manifest_identity,
                    "manifest_hash": baseline.manifest_hash,
                    "source_manifest_hash": baseline.source_manifest_hash,
                    "server_hash": baseline.server_hash,
                    "status": "unchanged",
                    "outcome": "accepted",
                },
            )
            second_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[1].id,
                expected_generations={history_key: 1},
                baseline_updates={},
                events=[unchanged],
            )

            assert second_result.published is True
            assert second_result.baselines == {history_key: baseline}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert [record.event.id for record in records] == [accepted.id, unchanged.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    [
        "manifest_identity",
        "manifest_hash",
        "source_manifest_hash",
        "server_hash",
    ],
)
def test_session_store_conformance_rejects_false_unchanged_mcp_evidence(
    session_store_case,
    mismatch: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            sessions = [
                await store.create(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"mcp_false_unchanged_{mismatch}_{index}",
                        messages=[Message.text("user", "check manifest")],
                    ),
                    identity=_identity(),
                )
                for index in range(2)
            ]
            history_key = "sha256:" + "2" * 64
            source_hash = "sha256:" + "6" * 64
            server_hash = "sha256:" + "5" * 64
            accepted = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[0].id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "3" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=accepted.payload["manifest_identity"],
                manifest_hash=accepted.payload["manifest_hash"],
                source_manifest_hash=source_hash,
                server_hash=server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(sessions[0].id),
                accepted_event_id=accepted.id,
                accepted_at=accepted.timestamp,
            )
            first_result = await store.compare_and_publish_mcp_manifest_checks(
                sessions[0].id,
                expected_generations={history_key: None},
                baseline_updates={history_key: baseline},
                events=[accepted],
            )
            assert first_result.published is True

            false_payload = {
                "history_key": history_key,
                "manifest_identity": baseline.manifest_identity,
                "manifest_hash": baseline.manifest_hash,
                "source_manifest_hash": baseline.source_manifest_hash,
                "server_hash": baseline.server_hash,
                "status": "unchanged",
                "outcome": "accepted",
            }
            false_payload[mismatch] = "sha256:" + "f" * 64
            false_unchanged = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=sessions[1].id,
                payload=false_payload,
            )
            with pytest.raises(ValueError, match="unchanged|identity"):
                await store.compare_and_publish_mcp_manifest_checks(
                    sessions[1].id,
                    expected_generations={history_key: 1},
                    baseline_updates={},
                    events=[false_unchanged],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {
                history_key: baseline
            }
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert [record.event.id for record in records] == [accepted.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("status", ["first_seen", "changed"])
def test_session_store_conformance_rejects_mcp_status_incompatible_with_current_state(
    session_store_case,
    status: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_invalid_{status}_state",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "7" * 64
            source_hash = "sha256:" + "8" * 64
            server_hash = "sha256:" + "9" * 64
            expected_generation: int | None = None
            generation = 1
            if status == "first_seen":
                seed_event = Event(
                    type=EventType.MCP_MANIFEST_CHECKED,
                    session_id=session.id,
                    payload={
                        "history_key": history_key,
                        "manifest_identity": "sha256:" + "a" * 64,
                        "manifest_hash": _mcp_test_manifest_hash(
                            source_manifest_hash=source_hash,
                            server_hash=server_hash,
                        ),
                        "source_manifest_hash": source_hash,
                        "server_hash": server_hash,
                        "status": "first_seen",
                        "outcome": "accepted",
                    },
                )
                seed_baseline = McpManifestBaseline(
                    history_key=history_key,
                    generation=1,
                    manifest_identity=seed_event.payload["manifest_identity"],
                    manifest_hash=seed_event.payload["manifest_hash"],
                    source_manifest_hash=source_hash,
                    server_hash=server_hash,
                    tools=(),
                    exposed_tools=(),
                    accepted_session_ref=_mcp_manifest_session_ref(session.id),
                    accepted_event_id=seed_event.id,
                    accepted_at=seed_event.timestamp,
                )
                seeded = await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: seed_baseline},
                    events=[seed_event],
                )
                assert seeded.published is True
                expected_generation = 1
                generation = 2

            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "a" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": status,
                    "outcome": "accepted",
                },
            )
            update = McpManifestBaseline(
                history_key=history_key,
                generation=generation,
                manifest_identity=event.payload["manifest_identity"],
                manifest_hash=event.payload["manifest_hash"],
                source_manifest_hash=source_hash,
                server_hash=server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=event.id,
                accepted_at=event.timestamp,
            )
            with pytest.raises(ValueError, match="first_seen|changed"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: expected_generation},
                    baseline_updates={history_key: update},
                    events=[event],
                )

            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
            assert len(records) == (1 if status == "first_seen" else 0)
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("event_type", "outcome"),
    [
        (EventType.MCP_MANIFEST_BLOCKED, "blocked"),
        (EventType.MCP_MANIFEST_CHECKED, "batch_blocked"),
    ],
)
def test_session_store_conformance_accepts_mcp_block_without_baseline_update(
    session_store_case,
    event_type: EventType,
    outcome: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_{outcome}_publication",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "b" * 64
            source_hash = "sha256:" + "c" * 64
            server_hash = "sha256:" + "d" * 64
            event = Event(
                type=event_type,
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "e" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": outcome,
                },
            )
            result = await store.compare_and_publish_mcp_manifest_checks(
                session.id,
                expected_generations={history_key: None},
                baseline_updates={},
                events=[event],
            )

            assert result.published is True
            assert result.baselines == {}
            records = await store.query_events(EventQuery(event_type=event_type, limit=10))
            assert [record.event.id for record in records] == [event.id]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_rejects_missing_mcp_history_outcome(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="mcp_missing_history_outcome",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_keys = ("sha256:" + "1" * 64, "sha256:" + "2" * 64)
            source_hash = "sha256:" + "3" * 64
            server_hash = "sha256:" + "4" * 64
            event = Event(
                type=EventType.MCP_MANIFEST_BLOCKED,
                session_id=session.id,
                payload={
                    "history_key": history_keys[0],
                    "manifest_identity": "sha256:" + "5" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=source_hash,
                        server_hash=server_hash,
                    ),
                    "source_manifest_hash": source_hash,
                    "server_hash": server_hash,
                    "status": "first_seen",
                    "outcome": "blocked",
                },
            )
            with pytest.raises(ValueError, match="exactly one outcome"):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={key: None for key in history_keys},
                    baseline_updates={},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines(history_keys)).baselines == {}
            records = await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "mismatch",
    ["server_hash", "source_manifest_hash", "accepted_event_id", "blocked_event"],
)
def test_session_store_conformance_rejects_mcp_baselines_without_matching_accepted_events(
    session_store_case,
    mismatch: str,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"mcp_manifest_event_mismatch_{mismatch}",
                    messages=[Message.text("user", "check manifest")],
                ),
                identity=_identity(),
            )
            history_key = "sha256:" + "e" * 64
            event_source_hash = "sha256:" + "4" * 64
            event_server_hash = "sha256:" + "3" * 64
            event = Event(
                type=(
                    EventType.MCP_MANIFEST_BLOCKED
                    if mismatch == "blocked_event"
                    else EventType.MCP_MANIFEST_CHECKED
                ),
                session_id=session.id,
                payload={
                    "history_key": history_key,
                    "manifest_identity": "sha256:" + "1" * 64,
                    "manifest_hash": _mcp_test_manifest_hash(
                        source_manifest_hash=event_source_hash,
                        server_hash=event_server_hash,
                    ),
                    "source_manifest_hash": event_source_hash,
                    "server_hash": event_server_hash,
                    "status": "first_seen",
                    "outcome": "accepted",
                },
            )
            baseline_source_hash = (
                "sha256:" + "5" * 64 if mismatch == "source_manifest_hash" else event_source_hash
            )
            baseline_server_hash = (
                "sha256:" + "6" * 64 if mismatch == "server_hash" else event_server_hash
            )
            baseline = McpManifestBaseline(
                history_key=history_key,
                generation=1,
                manifest_identity=event.payload["manifest_identity"],
                manifest_hash=_mcp_test_manifest_hash(
                    source_manifest_hash=baseline_source_hash,
                    server_hash=baseline_server_hash,
                ),
                source_manifest_hash=baseline_source_hash,
                server_hash=baseline_server_hash,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(session.id),
                accepted_event_id=(
                    "evt_missing_manifest_acceptance"
                    if mismatch == "accepted_event_id"
                    else event.id
                ),
                accepted_at=event.timestamp,
            )

            expected_error = (
                "matching baseline update"
                if mismatch == "accepted_event_id"
                else "must match exactly one accepted"
            )
            with pytest.raises(ValueError, match=expected_error):
                await store.compare_and_publish_mcp_manifest_checks(
                    session.id,
                    expected_generations={history_key: None},
                    baseline_updates={history_key: baseline},
                    events=[event],
                )

            assert (await store.load_mcp_manifest_baselines((history_key,))).baselines == {}
            records = await store.query_events(
                EventQuery(
                    event_types=(
                        EventType.MCP_MANIFEST_CHECKED,
                        EventType.MCP_MANIFEST_BLOCKED,
                    ),
                    limit=10,
                )
            )
            assert records == []
        finally:
            await _close_store(store)

    asyncio.run(run())
