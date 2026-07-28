from __future__ import annotations

from cayu.core import Message
from cayu.runtime import RunRequest, SessionIdentity, SessionStore
from cayu.runtime.sessions import (
    SessionTopologyQuery,
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
