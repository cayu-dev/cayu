from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu.core import Event, EventType, Message
from cayu.runtime import RunRequest, SessionIdentity, SessionStore
from cayu.runtime.sessions import (
    SESSION_LINEAGE_MAX_CHILD_LIMIT,
    SessionLineageQuery,
    SessionTopologyQuery,
    decode_session_lineage_cursor,
    decode_session_topology_cursor,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


async def _create_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
) -> None:
    await store.create(
        RunRequest(
            agent_name=f"agent-{session_id}",
            session_id=session_id,
            parent_session_id=parent_session_id,
            causal_budget_id="budget-topology",
            environment_name="topology-test",
            labels={"private-label": session_id},
            metadata={"private_metadata": session_id},
            messages=[Message.text("user", session_id)],
        ),
        identity=_identity(),
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-started",
            type=EventType.SESSION_STARTED,
            session_id=session_id,
            payload={"excluded_from_topology": "x" * 4096},
        ),
    )


async def assert_session_topology_store_conformance(store: SessionStore) -> None:
    """Exercise the backend-neutral bounded topology contract."""

    await _create_session(store, "topology-root")
    await _create_session(
        store,
        "topology-parent",
        parent_session_id="topology-root",
    )
    await _create_session(
        store,
        "topology-focus",
        parent_session_id="topology-parent",
    )
    await _create_session(
        store,
        "topology-root-sibling",
        parent_session_id="topology-root",
    )
    for suffix in ("a", "b", "c", "d"):
        await _create_session(
            store,
            f"topology-child-{suffix}",
            parent_session_id="topology-focus",
        )

    query = SessionTopologyQuery(
        focus_session_id="topology-focus",
        expanded_parent_ids=("topology-root", "topology-focus"),
        child_limit=2,
    )
    first = await store.query_session_topology(query)
    repeated = await store.query_session_topology(query)

    assert first == repeated
    assert first.focus.id == "topology-focus"
    assert [node.id for node in first.ancestors] == [
        "topology-root",
        "topology-parent",
    ]
    assert [node.id for node in first.expanded_parents] == [
        "topology-root",
        "topology-focus",
    ]
    assert [branch.parent_session_id for branch in first.branches] == [
        "topology-root",
        "topology-focus",
    ]
    assert [node.id for node in first.branches[0].children] == [
        "topology-parent",
        "topology-root-sibling",
    ]
    assert first.branches[0].has_more is False
    assert first.branches[0].next_cursor is None

    focus_branch = first.branches[1]
    assert [node.id for node in focus_branch.children] == [
        "topology-child-a",
        "topology-child-b",
    ]
    assert focus_branch.has_more is True
    assert focus_branch.next_cursor is not None
    cursor_created_at, cursor_id = decode_session_topology_cursor(
        focus_branch.next_cursor,
        parent_session_id="topology-focus",
    )
    assert cursor_created_at == focus_branch.children[-1].created_at
    assert cursor_id == "topology-child-b"

    lineage = await store.query_session_lineage(
        SessionLineageQuery(parent_session_id="topology-focus", limit=2)
    )
    assert lineage == await store.query_session_lineage(
        SessionLineageQuery(parent_session_id="topology-focus", limit=2)
    )
    assert [node.id for node in lineage.children] == [
        "topology-child-a",
        "topology-child-b",
    ]
    assert lineage.has_more is True
    assert lineage.next_cursor is not None
    lineage_cursor_created_at, lineage_cursor_id = decode_session_lineage_cursor(
        lineage.next_cursor,
        parent_session_id="topology-focus",
    )
    assert lineage_cursor_created_at == lineage.children[-1].created_at
    assert lineage_cursor_id == "topology-child-b"
    for node in lineage.children:
        assert set(node.model_dump()) == {
            "id",
            "parent_session_id",
            "created_at",
            "origin_events",
        }
        assert len(node.origin_events) == 1
        assert node.origin_events[0].event_id == f"{node.id}-started"
        assert node.origin_events[0].event_type is EventType.SESSION_STARTED

    await store.append_event(
        "topology-child-a",
        Event(
            id="topology-child-a-duplicate-origin",
            type=EventType.SESSION_STARTED,
            session_id="topology-child-a",
        ),
    )
    duplicate_origins = await store.query_session_lineage(
        SessionLineageQuery(parent_session_id="topology-focus", limit=2)
    )
    duplicate_child = duplicate_origins.children[0]
    assert duplicate_child.id == "topology-child-a"
    assert len(duplicate_child.origin_events) == 2

    lineage_continuation = await store.query_session_lineage(
        SessionLineageQuery(
            parent_session_id="topology-focus",
            cursor=lineage.next_cursor,
            limit=2,
        )
    )
    assert [node.id for node in lineage_continuation.children] == [
        "topology-child-c",
        "topology-child-d",
    ]
    assert lineage_continuation.has_more is False
    assert lineage_continuation.next_cursor is None

    try:
        await store.query_session_lineage(
            SessionLineageQuery(
                parent_session_id="topology-root",
                cursor=lineage.next_cursor,
            )
        )
    except ValueError as exc:
        assert "cursor" in str(exc).lower()
    else:  # pragma: no cover - a backend contract violation
        raise AssertionError("A lineage cursor was accepted for the wrong parent branch.")

    try:
        await store.query_session_lineage(SessionLineageQuery(parent_session_id="topology-missing"))
    except KeyError:
        pass
    else:  # pragma: no cover - a backend contract violation
        raise AssertionError("A missing lineage parent was returned as an empty branch.")

    # Store boundaries must revalidate exact-type Pydantic instances before
    # touching storage. model_copy() intentionally skips field validation and
    # is therefore a useful regression probe for forged public inputs.
    valid_missing_query = SessionLineageQuery(parent_session_id="topology-missing")
    for invalid_limit in (-2, SESSION_LINEAGE_MAX_CHILD_LIMIT + 1):
        forged_query = valid_missing_query.model_copy(update={"limit": invalid_limit})
        with pytest.raises(ValidationError, match="limit"):
            await store.query_session_lineage(forged_query)

    continuation = await store.query_session_topology(
        SessionTopologyQuery(
            focus_session_id="topology-focus",
            expanded_parent_ids=("topology-focus",),
            child_cursors={"topology-focus": focus_branch.next_cursor},
            child_limit=2,
        )
    )
    assert [node.id for node in continuation.branches[0].children] == [
        "topology-child-c",
        "topology-child-d",
    ]
    assert continuation.branches[0].has_more is False
    assert continuation.branches[0].next_cursor is None

    # A later insert remains reachable through the existing keyset cursor; no
    # already-returned child is duplicated.
    await _create_session(
        store,
        "topology-child-e",
        parent_session_id="topology-focus",
    )
    after_insert = await store.query_session_topology(
        SessionTopologyQuery(
            focus_session_id="topology-focus",
            expanded_parent_ids=("topology-focus",),
            child_cursors={"topology-focus": focus_branch.next_cursor},
            child_limit=10,
        )
    )
    assert [node.id for node in after_insert.branches[0].children] == [
        "topology-child-c",
        "topology-child-d",
        "topology-child-e",
    ]

    # Topology nodes deliberately exclude labels, metadata, prompts, histories,
    # and output payloads.
    assert set(first.focus.model_dump()) == {
        "id",
        "agent_name",
        "provider_name",
        "model",
        "parent_session_id",
        "causal_budget_id",
        "runtime_name",
        "runtime_version",
        "runtime_build_provenance",
        "environment_name",
        "status",
        "created_at",
        "updated_at",
        "last_activity_at",
    }

    try:
        await store.query_session_topology(
            SessionTopologyQuery(
                focus_session_id="topology-focus",
                expanded_parent_ids=("topology-root",),
                child_cursors={"topology-root": focus_branch.next_cursor},
            )
        )
    except ValueError as exc:
        assert "cursor" in str(exc).lower()
    else:  # pragma: no cover - a backend contract violation
        raise AssertionError("A topology cursor was accepted for the wrong parent branch.")

    try:
        await store.query_session_topology(
            SessionTopologyQuery(
                focus_session_id="topology-focus",
                ancestor_depth_limit=1,
            )
        )
    except ValueError as exc:
        assert "ancestor" in str(exc).lower()
    else:  # pragma: no cover - a backend contract violation
        raise AssertionError("An over-depth ancestor chain was returned as complete.")

    try:
        await store.query_session_topology(
            SessionTopologyQuery(
                focus_session_id="topology-focus",
                expanded_parent_ids=("topology-missing",),
            )
        )
    except KeyError:
        pass
    else:  # pragma: no cover - a backend contract violation
        raise AssertionError("A missing expanded parent was returned as an empty branch.")
