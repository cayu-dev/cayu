"""Supervision events behave the same on every built-in session store."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.runtime.runtime_session_support import (
    _reviewer_actor,
    collect_interrupt_events,
    collect_tool_approval_events,
)
from tests.runtime.runtime_supervision_support import (
    approval_app,
    pause_for_approval,
    resolve_approval,
    run_across_releases,
    supervision_page,
    supervision_window,
    two_call_provider,
)
from tests.runtime.runtime_tool_support import RequireApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import approval_revision_metadata
from cayu.approvals.tools import ToolApprovalDecision
from cayu.releases import ReleaseIdentity
from cayu.resource_access import ResourceAccessPolicy
from cayu.sessions import SessionStore, SupervisionEventCursorInvalid, SupervisionEventKind
from cayu.sessions._access_rules import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.requests import InterruptSessionRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]


def test_store_declares_supervision_event_support(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            return store.supports_supervision_events
        finally:
            await _close_store(store)

    assert asyncio.run(run()) is True


def test_every_store_classifies_and_pages_the_same_touches(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            app = approval_app(store, sessions=3)
            approved = await pause_for_approval(app, store, "sess_parity_approve")
            await collect_tool_approval_events(
                app,
                resolve_approval(
                    "sess_parity_approve",
                    approved,
                    ToolApprovalDecision.APPROVE,
                    reason="checked",
                    resolved_by=_reviewer_actor(),
                ),
            )
            denied = await pause_for_approval(app, store, "sess_parity_deny")
            await collect_tool_approval_events(
                app,
                resolve_approval("sess_parity_deny", denied, ToolApprovalDecision.DENY),
            )
            await pause_for_approval(app, store, "sess_parity_interrupt")
            await collect_interrupt_events(
                app,
                InterruptSessionRequest(
                    session_id="sess_parity_interrupt",
                    reason="stop",
                    requested_by=_reviewer_actor(),
                ),
            )
            first = await supervision_page(store, limit=2)
            second = await supervision_page(store, limit=2, cursor=first.next_cursor)
            return approved, first, second
        finally:
            await _close_store(store)

    approved, first, second = asyncio.run(run())

    records = [*first.events, *second.events]
    assert [(event.session_id, event.kind) for event in records] == [
        ("sess_parity_approve", SupervisionEventKind.APPROVE),
        ("sess_parity_deny", SupervisionEventKind.DENY),
        ("sess_parity_interrupt", SupervisionEventKind.INTERRUPT),
    ]
    assert first.has_more is True
    assert second.has_more is False
    assert records[0].approval_id == approved.approval_id
    assert records[0].reason == "checked"
    assert records[0].actor is not None
    assert records[0].actor.subject == "reviewer@example.com"
    assert all(event.execution_profile_fingerprint for event in records)
    sequences = [event.sequence for event in records]
    assert sequences == sorted(sequences)


def test_every_store_lists_both_calls_of_one_denial(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            app = CayuApp(session_store=store)
            app.register_provider(two_call_provider(), default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[SideEffectTool()],
                tool_policy=RequireApprovalPolicy(),
            )
            pending = await pause_for_approval(app, store, "sess_parity_two_calls")
            await collect_tool_approval_events(
                app,
                resolve_approval("sess_parity_two_calls", pending, ToolApprovalDecision.DENY),
            )
            return await supervision_page(store)
        finally:
            await _close_store(store)

    page = asyncio.run(run())

    assert [event.kind for event in page.events] == [SupervisionEventKind.DENY]
    assert set(page.events[0].tool_call_ids) == {"call_1", "call_2"}


def test_every_store_reports_declared_edits_and_attention_identity(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            app = approval_app(store, sessions=3)
            attention = {}
            revision = approval_revision_metadata(
                proposed_digest="a" * 64, approved_digest="b" * 64
            )
            # Ordinary metadata ahead of the declaration overflows the bounded audit
            # copy; the declared revision must still decide the classification.
            crowded = {**{f"field_{index:03d}": "x" * 64 for index in range(200)}, **revision}
            for session_id, metadata in (
                ("sess_parity_edit", revision),
                ("sess_parity_crowded_edit", crowded),
                ("sess_parity_plain", {}),
            ):
                pending = await pause_for_approval(app, store, session_id)
                actions = await store.query_pending_actions(
                    PendingActionQuery(session_id=session_id)
                )
                attention[session_id] = actions.actions[0].attention_id
                await collect_tool_approval_events(
                    app,
                    resolve_approval(
                        session_id, pending, ToolApprovalDecision.APPROVE, metadata=metadata
                    ),
                )
            return attention, await supervision_page(store)
        finally:
            await _close_store(store)

    attention, page = asyncio.run(run())

    assert [(event.session_id, event.kind) for event in page.events] == [
        ("sess_parity_edit", SupervisionEventKind.EDIT),
        ("sess_parity_crowded_edit", SupervisionEventKind.EDIT),
        ("sess_parity_plain", SupervisionEventKind.APPROVE),
    ]
    for event in page.events:
        assert event.attention_id == attention[event.session_id]
        assert event.attention_state == "resolved"
    for event in page.events[:2]:
        assert event.approval_revision is not None
        assert event.approval_revision.approved_digest == "b" * 64
    assert page.events[2].approval_revision is None


def test_every_store_filters_touches_by_the_release_their_invocation_recorded(
    session_store_case,
):
    async def run():
        store = await _open_store(session_store_case)
        try:
            await run_across_releases(
                store,
                "sess_parity_release",
                ReleaseIdentity(id="release-1"),
                ReleaseIdentity(id="release-2"),
            )
            operator = CayuApp(session_store=store, release_identity=ReleaseIdentity(id="reader"))
            first = await supervision_page(store, release_id="release-1")
            with pytest.raises(SupervisionEventCursorInvalid):
                await supervision_page(store, release_id="release-2", cursor=first.next_cursor)
            return (
                first,
                await operator.query_supervision_events(supervision_window(release_id="release-2")),
                await supervision_page(store, release_id="reader"),
            )
        finally:
            await _close_store(store)

    first, second, unrecorded = asyncio.run(run())

    assert [(event.tool_call_ids, event.release_id) for event in first.events] == [
        (("call_1",), "release-1"),
        (("call_2",), "release-1"),
    ]
    assert [(event.tool_call_ids, event.release_id) for event in second.events] == [
        (("call_3",), "release-2")
    ]
    assert unrecorded.events == ()


def test_store_without_the_capability_fails_closed_before_reading():
    store = SimpleNamespace(supports_supervision_events=False)

    assert SessionStore.supports_supervision_events is False
    with pytest.raises(NotImplementedError, match="supervision events"):
        asyncio.run(
            SessionStore.query_supervision_events(
                store,  # ty: ignore[invalid-argument-type]
                supervision_window(),
            )
        )


def _organization_scope(*organizations: str) -> SessionAccessScope:
    rules = tuple(
        SessionAccessRule(
            selectors=(SessionAccessSelector(key="organization", values=(organization,)),)
        )
        for organization in organizations
    )
    return SessionAccessScope(read=rules, execute=rules, inspect_state=rules)


class _OrganizationPolicy(ResourceAccessPolicy):
    authority = "tests:supervision-organization:v1"

    def __init__(self) -> None:
        self.current = _organization_scope("acme")

    async def resolve(self, subject):
        assert subject == "auditor"
        return self.current


def test_scoped_access_returns_only_admitted_sessions(session_store_case):
    async def run():
        store = await _open_store(session_store_case)
        try:
            policy = _OrganizationPolicy()
            app = approval_app(store, sessions=2, resource_access_policy=policy)
            for session_id, organization in (
                ("sess_scope_acme", "acme"),
                ("sess_scope_globex", "globex"),
            ):
                pending = await pause_for_approval(
                    app, store, session_id, labels={"organization": organization}
                )
                await collect_tool_approval_events(
                    app, resolve_approval(session_id, pending, ToolApprovalDecision.APPROVE)
                )
            access = await app.access("auditor")
            admitted = await access.sessions.supervision_events(supervision_window())
            policy.current = SessionAccessScope()
            revoked = await access.sessions.supervision_events(supervision_window())
            operator = await app.query_supervision_events(supervision_window())
            return admitted, revoked, operator
        finally:
            await _close_store(store)

    admitted, revoked, operator = asyncio.run(run())

    assert [event.session_id for event in admitted.events] == ["sess_scope_acme"]
    assert revoked.events == ()
    assert {event.session_id for event in operator.events} == {
        "sess_scope_acme",
        "sess_scope_globex",
    }


def test_scoped_access_has_no_content_option():
    import inspect

    from cayu.sessions.access import ScopedSessionAccess

    parameters = inspect.signature(ScopedSessionAccess.supervision_events).parameters
    assert "include_content" not in parameters
