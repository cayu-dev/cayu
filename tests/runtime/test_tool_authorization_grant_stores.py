"""One contract suite for in-memory, SQLite, and PostgreSQL grant stores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu.approvals import (
    InMemoryToolAuthorizationGrantStore,
    ProcessCommandGrantMatcher,
    ToolAuthorizationGrant,
    ToolAuthorizationGrantActor,
    ToolAuthorizationGrantBoundary,
    ToolAuthorizationGrantConflict,
    ToolAuthorizationGrantNotFound,
    ToolAuthorizationGrantOrigin,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantScope,
    ToolAuthorizationGrantScopeKind,
    ToolAuthorizationGrantStatus,
    ToolAuthorizationGrantUse,
    ToolAuthorizationGrantUseCursor,
    ToolAuthorizationGrantUseDecision,
    ToolCallGrantMatcher,
    tool_authorization_grant_id,
)
from cayu.approvals.authorization_grants import (
    TOOL_AUTHORIZATION_GRANT_MAX_LIST,
    use_order_key,
)
from cayu.storage import SQLiteToolAuthorizationGrantStore
from cayu.storage import migrations as schema

pytestmark = pytest.mark.anyio

T0 = datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=UTC)
OWNER = ToolAuthorizationGrantActor(subject="owner", tenant="acme")


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def open_store(request, tmp_path):
    """Return a factory; each call opens a new store instance on one database."""

    backend = request.param
    address = (
        request.getfixturevalue("postgres_dsn")
        if backend == "postgres"
        else str(tmp_path / "grants.sqlite")
    )
    memory = InMemoryToolAuthorizationGrantStore()
    opened = []

    def factory():
        if backend == "memory":
            return memory
        if backend == "sqlite":
            store = SQLiteToolAuthorizationGrantStore(address)
        else:
            from cayu.storage import PostgresToolAuthorizationGrantStore

            store = PostgresToolAuthorizationGrantStore(
                address, schema_mode=schema.SchemaMode.CREATE
            )
        opened.append(store)
        return store

    yield factory
    for store in opened:
        await store.close()


def grant(
    *,
    session_id: str | None = None,
    scope: ToolAuthorizationGrantScopeKind = ToolAuthorizationGrantScopeKind.SESSION,
    created_at: datetime = T0,
    lifetime_seconds: int | None = None,
    argv: tuple[str, ...] = ("git", "status"),
    digest: str = "sha256:first",
    rate: int = 0,
) -> ToolAuthorizationGrant:
    # Postgres databases are shared per module, so every grant binds fresh ids.
    session_id = session_id or f"session-{uuid4().hex}"
    approval_id = f"approval-{uuid4().hex}"
    workspace_id = f"ws-{session_id}"
    return ToolAuthorizationGrant(
        grant_id=tool_authorization_grant_id(session_id=session_id, approval_id=approval_id),
        policy_id="commands",
        tool_name="exec_command",
        approval_gate="commands",
        matcher=ProcessCommandGrantMatcher.from_arguments({"argv": list(argv)}),
        scope=ToolAuthorizationGrantScope(
            kind=scope,
            id=session_id if scope is ToolAuthorizationGrantScopeKind.SESSION else workspace_id,
        ),
        boundary=ToolAuthorizationGrantBoundary(
            agent_name="assistant",
            workspace_id=workspace_id,
            workspace_explicit=True,
            labels={"tenant": "acme"},
        ),
        origin=ToolAuthorizationGrantOrigin(
            session_id=session_id,
            session_instance_id=f"instance-{session_id}",
            approval_id=approval_id,
            tool_round_id="round",
            tool_call_id="call",
        ),
        resolution_request_digests=(digest,),
        created_by=OWNER,
        created_at=created_at,
        lifetime_seconds=lifetime_seconds,
        expires_at=(
            None
            if lifetime_seconds is None
            else created_at.replace(microsecond=123000) + timedelta(seconds=lifetime_seconds)
        ),
        sampled_review_rate_bps=rate,
        justification="approved unchanged 20 times",
        evidence={"approvals": 20, "changed": 0},
    )


async def test_stage_is_idempotent_by_approval_and_rejects_conflicts(open_store) -> None:
    store = open_store()
    requested = grant()
    staged = await store.stage(requested)
    assert staged.status is ToolAuthorizationGrantStatus.STAGED
    assert staged.created_at == T0.replace(microsecond=123000)
    assert await store.stage(requested) == staged

    retried = await store.stage(
        requested.model_copy(update={"resolution_request_digests": ("sha256:second",)})
    )
    assert retried.resolution_request_digests == ("sha256:first", "sha256:second")

    conflicting = requested.model_copy(update={"sampled_review_rate_bps": 100})
    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.stage(conflicting)
    assert await open_store().get(requested.grant_id) == retried


async def test_lifecycle_transitions_are_shared_and_reconstructable(open_store) -> None:
    store = open_store()
    requested = grant(lifetime_seconds=60)
    await store.stage(requested)
    active = await store.activate(requested.grant_id, resolved_by=OWNER, activated_at=T0)
    assert active.status is ToolAuthorizationGrantStatus.ACTIVE
    assert await store.activate(requested.grant_id, resolved_by=None, activated_at=T0) == active
    assert (
        active.effective_status(T0 + timedelta(seconds=59)) is ToolAuthorizationGrantStatus.ACTIVE
    )
    assert (
        active.effective_status(T0 + timedelta(seconds=60)) is ToolAuthorizationGrantStatus.EXPIRED
    )

    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.abandon(requested.grant_id, reason="late", abandoned_at=T0)
    revoked = await store.revoke(
        requested.grant_id, revoked_by=OWNER, reason="errors found", revoked_at=T0
    )
    assert revoked.status is ToolAuthorizationGrantStatus.REVOKED
    assert (
        await store.revoke(requested.grant_id, revoked_by=None, reason=None, revoked_at=T0)
        == revoked
    )
    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.activate(requested.grant_id, resolved_by=OWNER, activated_at=T0)
    # A new store instance observes the same durable record.
    assert await open_store().get(requested.grant_id) == revoked

    abandoned = grant()
    await store.stage(abandoned)
    settled = await store.abandon(abandoned.grant_id, reason="approval_denied", abandoned_at=T0)
    assert settled.status is ToolAuthorizationGrantStatus.ABANDONED
    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.revoke(abandoned.grant_id, revoked_by=OWNER, reason=None, revoked_at=T0)
    with pytest.raises(ToolAuthorizationGrantNotFound):
        await store.activate("tag_missing", resolved_by=None, activated_at=T0)


async def test_listing_filters_and_orders_identically(open_store) -> None:
    store = open_store()
    session_id = f"session-{uuid4().hex}"
    later = grant(session_id=session_id, created_at=T0 + timedelta(seconds=5))
    earlier = grant(session_id=session_id, created_at=T0, argv=("ls",))
    workspace = grant(session_id=session_id, scope=ToolAuthorizationGrantScopeKind.WORKSPACE)
    for item in (later, earlier, workspace):
        await store.stage(item)
    await store.activate(earlier.grant_id, resolved_by=OWNER, activated_at=T0)

    session_grants = await store.list(
        ToolAuthorizationGrantQuery(
            scope_kind=ToolAuthorizationGrantScopeKind.SESSION, scope_id=session_id
        )
    )
    assert [item.grant_id for item in session_grants] == [earlier.grant_id, later.grant_id]
    active = await store.list(
        ToolAuthorizationGrantQuery(
            origin_session_id=session_id, statuses=(ToolAuthorizationGrantStatus.ACTIVE,)
        )
    )
    assert [item.grant_id for item in active] == [earlier.grant_id]
    workspace_grants = await store.list(
        ToolAuthorizationGrantQuery(
            policy_id="commands",
            tool_name="exec_command",
            scope_kind=ToolAuthorizationGrantScopeKind.WORKSPACE,
            scope_id=f"ws-{session_id}",
        )
    )
    assert [item.grant_id for item in workspace_grants] == [workspace.grant_id]
    limited = await store.list(ToolAuthorizationGrantQuery(origin_session_id=session_id, limit=1))
    assert len(limited) == 1


async def test_listing_pages_by_cursor_through_same_millisecond_ties(open_store) -> None:
    store = open_store()
    session_id = f"session-{uuid4().hex}"
    grants = [
        grant(session_id=session_id, created_at=T0 + timedelta(milliseconds=index // 3))
        for index in range(7)
    ]
    for item in grants:
        await store.stage(item)
    expected = sorted(grants, key=lambda item: (item.created_at, item.grant_id))

    query = ToolAuthorizationGrantQuery(origin_session_id=session_id, limit=2)
    paged: list = []
    while query is not None:
        page = await store.list(query)
        paged.extend(page)
        query = query.next_page(page)

    assert [item.grant_id for item in paged] == [item.grant_id for item in expected]


async def test_use_evidence_is_first_write_wins_and_ordered(open_store) -> None:
    store = open_store()
    requested = grant(rate=10_000)
    await store.stage(requested)

    def use(call_id: str, recorded_at: datetime) -> ToolAuthorizationGrantUse:
        return ToolAuthorizationGrantUse(
            grant_id=requested.grant_id,
            policy_id="commands",
            scope=requested.scope,
            session_id=requested.origin.session_id,
            session_instance_id="instance-1",
            tool_round_id="round-1",
            tool_call_id=call_id,
            tool_name="exec_command",
            arguments_digest="sha256:args",
            decision=ToolAuthorizationGrantUseDecision.SAMPLED_REVIEW,
            sample_bucket=42,
            sampled_review_rate_bps=10_000,
            recorded_at=recorded_at,
        )

    first = await store.record_use(use("b", T0 + timedelta(seconds=1)))
    assert await store.record_use(use("b", T0 + timedelta(seconds=9))) == first
    await store.record_use(use("a", T0))
    uses = await open_store().list_uses(requested.grant_id)
    assert [item.tool_call_id for item in uses] == ["a", "b"]
    assert uses[1] == first


async def test_use_listing_pages_by_cursor_past_the_maximum_page(open_store) -> None:
    store = open_store()
    requested = grant(rate=0)
    await store.stage(requested)
    recorded = []
    for index in range(TOOL_AUTHORIZATION_GRANT_MAX_LIST + 3):
        recorded.append(
            await store.record_use(
                ToolAuthorizationGrantUse(
                    grant_id=requested.grant_id,
                    policy_id="commands",
                    scope=requested.scope,
                    session_id=requested.origin.session_id,
                    session_instance_id="instance-1",
                    tool_round_id=f"round-{index:04d}",
                    tool_call_id="call",
                    tool_name="exec_command",
                    arguments_digest="sha256:args",
                    decision=ToolAuthorizationGrantUseDecision.ALLOWED,
                    sample_bucket=9_999,
                    sampled_review_rate_bps=0,
                    # Equal timestamps: the use identity breaks the ties.
                    recorded_at=T0 + timedelta(milliseconds=index // 100),
                )
            )
        )

    seen = []
    after = None
    while True:
        page = await store.list_uses(
            requested.grant_id, limit=TOOL_AUTHORIZATION_GRANT_MAX_LIST, after=after
        )
        seen.extend(page)
        if len(page) < TOOL_AUTHORIZATION_GRANT_MAX_LIST:
            break
        after = ToolAuthorizationGrantUseCursor.after(page[-1])

    assert [use.use_key for use in seen] == [
        use.use_key for use in sorted(recorded, key=use_order_key)
    ]


async def test_abandon_refuses_a_changed_request_digest_set(open_store) -> None:
    store = open_store()
    first = grant(digest="sha256:first")
    await store.stage(first)
    retried = first.model_copy(update={"resolution_request_digests": ("sha256:second",)})
    await store.stage(retried)

    with pytest.raises(ToolAuthorizationGrantConflict):
        await store.abandon(
            first.grant_id,
            reason="approved_without_grant",
            abandoned_at=T0,
            expected_request_digests=("sha256:first",),
        )
    current = await store.get(first.grant_id)
    assert current.status is ToolAuthorizationGrantStatus.STAGED
    assert current.resolution_request_digests == ("sha256:first", "sha256:second")


def test_matchers_are_exact_and_reject_shell_text() -> None:
    matcher = ProcessCommandGrantMatcher.from_arguments({"argv": ["git", "status"]})
    assert matcher.matches({"kind": "process", "argv": ["git", "status"], "timeout_s": 30})
    assert not matcher.matches({"argv": ["git", "status", "--short"]})
    assert not matcher.matches({"argv": ["git"]})
    assert not matcher.matches({"shell": "git status"})
    assert not matcher.matches({"argv": ["git", "status"], "env": {"GIT_DIR": "/tmp"}})
    assert not matcher.matches({"argv": ["git", "status"], "cwd": "/other"})
    assert not matcher.matches({"argv": ["git", "status"], "unexpected": True})

    pinned = ToolCallGrantMatcher.from_arguments(
        {"vendor": {"id": "v-1"}, "amount": 10, "flag": True}, ["vendor.id", "flag"]
    )
    assert pinned.matches({"vendor": {"id": "v-1"}, "amount": 99, "flag": True})
    assert not pinned.matches({"vendor": {"id": "v-2"}, "amount": 10, "flag": True})
    assert not pinned.matches({"vendor": {"id": "v-1"}, "flag": 1})
    assert not pinned.matches({"amount": 10, "flag": True})
