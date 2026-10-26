"""Focused real-server Knowledge page acceptance; Chromium explicitly enabled."""

import asyncio
import os

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_server_inbox import _app

from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_KNOWLEDGE_LIVE") != "1",
    reason="Opt-in Knowledge browser acceptance.",
)


def test_kept_knowledge_real_read_and_revision_fenced_remove():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="kept-entry", text="Verified knowledge", status=KnowledgeStatus.ACTIVE
                )
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope), nested=True
        ) as (page, base, _requests, _responses):
            mutations = []
            page.on(
                "request",
                lambda request: (
                    mutations.append(request)
                    if request.url.endswith("/knowledge/kept-entry/retire")
                    else None
                ),
            )
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="Kept", exact=True)).to_be_visible()
            await expect(page.get_by_text("Verified knowledge", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Remove", exact=True).click()
            await page.get_by_role("button", name="Confirm remove", exact=True).click()
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_be_visible()
            assert (await store.get_entry("kept-entry")).status is KnowledgeStatus.ARCHIVED
            assert len(mutations) == 1
            assert mutations[0].post_data_json == {"expected_revision": 1}

    asyncio.run(scenario())


def test_member_reviews_preserve_evidence_and_have_no_remove_controls():
    from tests.server.test_server_inbox import _knowledge_app

    async def scenario():
        runtime, store = _knowledge_app()
        await store.create_entry(
            KnowledgeEntry(
                id="long-public-entry",
                text="Unbroken" * 1000,
                status=KnowledgeStatus.ACTIVE,
                source_uri="https://example.test/" + "source" * 200,
            ),
            access_scope=KnowledgeAccessScope.privileged(),
        )
        async with _home_browser(role="member", mobile=True, seeded_runtime=runtime) as (
            page,
            base,
            requests,
            _responses,
        ):
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(
                page.get_by_role("heading", name="Waiting for review", exact=True)
            ).to_be_visible()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(2)
            await page.get_by_text("Proposal evidence", exact=True).first.click()
            await expect(page.get_by_role("heading", name="Metadata", exact=True)).to_be_visible()
            await page.get_by_text("Decision options", exact=True).first.click()
            await page.get_by_label("Reason (optional)", exact=True).first.fill(
                "Reviewed on Knowledge page"
            )
            await page.get_by_role("button", name="Keep", exact=True).first.click()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(1)
            await page.get_by_role("button", name="Drop", exact=True).first.click()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_have_count(0)
            await expect(page.get_by_role("heading", name="Kept", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Remove", exact=True)).to_have_count(0)
            assert not any(path.endswith("/retire") for path in requests)
            statuses = [
                (await store.get_entry(key)).status for key in ("vendor-codes", "freight-accruals")
            ]
            assert set(statuses) == {KnowledgeStatus.ACTIVE, KnowledgeStatus.ARCHIVED}
            await (
                page.get_by_role("article", name="Knowledge long-public-entry", exact=True)
                .get_by_text("Entry details", exact=True)
                .click()
            )
            for width in (390, 768, 1100, 1440):
                await page.set_viewport_size({"width": width, "height": 900})
                search = page.get_by_role("textbox", name="Search loaded knowledge", exact=True)
                await search.focus()
                await expect(search).to_be_focused()
                bounds = await search.bounding_box()
                assert bounds and bounds["width"] >= 140
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                if width in (390, 1440):
                    await page.screenshot(path=f".agents/knowledge-{width}.png", full_page=True)

    asyncio.run(scenario())


def test_empty_scoped_pages_continue_without_false_empty_or_capped_total():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="later", text="Visible after scoped pages", status=KnowledgeStatus.ACTIVE
                )
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            calls = []

            async def scoped(route):
                from urllib.parse import parse_qs, urlsplit

                query = parse_qs(urlsplit(route.request.url).query)
                cursor = query.get("cursor", [None])[0]
                calls.append(cursor)
                response = await route.fetch(url=base + "/api/knowledge")
                body = await response.json()
                if cursor is None:
                    body["entries"] = []
                    body["next_cursor"] = "scoped-next"
                await route.fulfill(response=response, json=body)

            await page.route("**/api/knowledge?*", scoped)
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(page.get_by_text("Visible after scoped pages", exact=True)).to_be_visible()
            assert calls[:2] == [None, "scoped-next"]
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_have_count(0)
            await expect(
                page.get_by_text("1 loaded. Text is a public preview.", exact=True)
            ).to_be_visible()

    asyncio.run(scenario())


def test_kept_loading_failure_and_stale_remove_are_honest():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="stale", text="Revision guarded entry", status=KnowledgeStatus.ACTIVE
                )
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            release = asyncio.Event()

            async def failed_read(route):
                await release.wait()
                await route.fulfill(status=503, json={"detail": "Knowledge unavailable"})

            await page.route("**/api/knowledge?*", failed_read)
            await page.goto(base + "/cayu/knowledge", wait_until="domcontentloaded")
            await expect(page.get_by_text("Loading kept knowledge…", exact=True)).to_be_visible()
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_have_count(0)
            release.set()
            await expect(
                page.get_by_text("Knowledge unavailable", exact=False).first
            ).to_be_visible()
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_have_count(0)
            await page.unroute("**/api/knowledge?*", failed_read)
            await page.get_by_role("button", name="Refresh knowledge", exact=True).click()
            await expect(page.get_by_text("Revision guarded entry", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Remove", exact=True).click()
            requests = []

            async def conflict(route):
                requests.append(route.request.post_data_json)
                await route.fulfill(status=409, json={"detail": "Revision changed"})

            await page.route("**/api/knowledge/stale/retire", conflict)
            await page.get_by_role("button", name="Confirm remove", exact=True).click()
            await expect(
                page.get_by_role("button", name="Confirm remove", exact=True)
            ).to_be_disabled()
            assert requests == [{"expected_revision": 1}]
            assert (await store.get_entry("stale")).status is KnowledgeStatus.ACTIVE
            await page.get_by_role("button", name="Cancel", exact=True).click()
            await expect(page.get_by_role("button", name="Remove", exact=True)).to_be_enabled()

    asyncio.run(scenario())


def test_focused_reference_knowledge_journey(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    from examples.dashboard_behavior_live import _exercise_dashboard_knowledge

    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="dashboard-kept",
                    text="Dashboard retained knowledge",
                    status=KnowledgeStatus.ACTIVE,
                ),
                KnowledgeEntry(
                    id="dashboard-proposal",
                    text="Dashboard proposed knowledge",
                    status=KnowledgeStatus.PENDING,
                ),
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            await _exercise_dashboard_knowledge(page, base)

    asyncio.run(scenario())


def test_empty_scan_pauses_and_navigation_cancels_owned_read():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(access_scope=scope)
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            calls = []

            async def empty(route):
                from urllib.parse import parse_qs, urlsplit

                cursor = parse_qs(urlsplit(route.request.url).query).get("cursor", ["0"])[0]
                calls.append(cursor)
                await route.fulfill(
                    json={
                        "entries": [],
                        "next_cursor": str(int(cursor) + 1),
                        "total_count": None,
                        "usage_through_sequence": None,
                        "usage_unavailable_reason": "access_limited",
                    }
                )

            await page.route("**/api/knowledge?*", empty)
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(
                page.get_by_role("button", name="Load more knowledge", exact=True)
            ).to_be_enabled()
            assert len(calls) == 25
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Load more knowledge", exact=True).click()
            await expect(
                page.get_by_role("button", name="Load more knowledge", exact=True)
            ).to_be_enabled()
            # Explicit continuation begins another bounded window.
            assert len(calls) > 25
            await page.unroute("**/api/knowledge?*", empty)
            requested = asyncio.Event()
            release = asyncio.Event()
            aborted = asyncio.Event()

            async def delayed(route):
                requested.set()
                await release.wait()
                await route.fulfill(
                    json={
                        "entries": [],
                        "next_cursor": None,
                        "usage_through_sequence": None,
                        "usage_unavailable_reason": None,
                    }
                )

            page.on(
                "requestfailed",
                lambda request: aborted.set() if "/api/knowledge?" in request.url else None,
            )
            await page.route("**/api/knowledge?*", delayed)
            try:
                await page.get_by_role("button", name="Refresh knowledge", exact=True).click()
                await asyncio.wait_for(requested.wait(), 5)
                await (
                    page.get_by_role("navigation", name="Main")
                    .get_by_role("link", name="Home", exact=True)
                    .click()
                )
                await asyncio.wait_for(aborted.wait(), 5)
            finally:
                release.set()

    asyncio.run(scenario())


def test_kept_capability_blocks_reads_and_lost_ack_never_replays_retirement():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="ack", text="Acknowledgement boundary", status=KnowledgeStatus.ACTIVE
                )
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, requests, _responses):

            async def capability(route):
                response = await route.fetch()
                body = await response.json()
                body["capabilities"]["surfaces"]["kept_knowledge"]["read"] = {
                    "enabled": False,
                    "unavailable_reason": "unsupported",
                }
                await route.fulfill(response=response, json=body)

            await page.route("**/api/contract", capability)
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(
                page.get_by_text("Kept knowledge is unavailable.", exact=False)
            ).to_be_visible()
            assert "/api/knowledge" not in requests
            await page.unroute("**/api/contract", capability)
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_text("Acknowledgement boundary", exact=True)).to_be_visible()
            dispatched = []

            async def lose_ack(route):
                dispatched.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.status == 200
                await route.abort("failed")

            await page.route("**/api/knowledge/ack/retire", lose_ack)
            await page.get_by_role("button", name="Remove", exact=True).click()
            await page.get_by_role("button", name="Confirm remove", exact=True).click()
            await expect(
                page.get_by_text("Removal was not confirmed.", exact=False)
            ).to_be_visible()
            assert (await store.get_entry("ack")).status is KnowledgeStatus.ARCHIVED
            await page.get_by_role("button", name="Refresh knowledge", exact=True).click()
            await expect(page.get_by_text("No knowledge kept yet", exact=True)).to_be_visible()
            assert dispatched == [{"expected_revision": 1}]

    asyncio.run(scenario())


def test_loaded_history_does_not_amplify_automatic_refresh():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id=f"history-{i}", text=f"History entry {i}", status=KnowledgeStatus.ACTIVE
                )
                for i in range(120)
            ],
            access_scope=scope,
        )
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, requests, _responses):
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await page.get_by_role("button", name="Load more knowledge", exact=True).click()
            await expect(
                page.get_by_role("button", name="Load more knowledge", exact=True)
            ).to_be_enabled()
            await page.get_by_role("button", name="Load more knowledge", exact=True).click()
            await expect(
                page.get_by_role("button", name="Load more knowledge", exact=True)
            ).to_have_count(0)
            before = requests.count("/api/knowledge")
            await asyncio.sleep(5.5)
            assert requests.count("/api/knowledge") == before, (
                "Historical pages must not be automatically replayed"
            )
            await expect(page.get_by_text("Viewing an older page.", exact=False)).to_be_visible()
            await store.create_entry(
                KnowledgeEntry(id="newest", text="Newly kept entry", status=KnowledgeStatus.ACTIVE),
                access_scope=scope,
            )
            await page.get_by_role("button", name="Refresh knowledge", exact=True).click()
            await expect(page.get_by_text("Newly kept entry", exact=True)).to_be_visible()
            assert await page.get_by_role("article").count() <= 50
            newest_reads = requests.count("/api/knowledge")
            await asyncio.sleep(5.5)
            assert requests.count("/api/knowledge") - newest_reads <= 2

    asyncio.run(scenario())


def test_member_proposal_preserves_source_identity_only_at_matching_revision():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id="document-proposal",
                    title="Use account A",
                    text="Review document knowledge",
                    status=KnowledgeStatus.PENDING,
                    source_type="document",
                    source_id="invoice-2026-10",
                    source_uri=None,
                )
            ],
            access_scope=scope,
        )
        async with _home_browser(
            role="member", seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(page.get_by_text("Use account A", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("Review document knowledge", exact=True).first
            ).to_be_visible()
            await expect(page.get_by_text("document", exact=True)).to_be_visible()
            await expect(page.get_by_text("invoice-2026-10", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_be_enabled()

            async def mismatch(route):
                response = await route.fetch()
                body = await response.json()
                body["revision"] += 1
                await route.fulfill(response=response, json=body)

            await page.route("**/api/knowledge/pending/document-proposal", mismatch)
            await page.reload(wait_until="networkidle")
            await expect(page.get_by_text("This proposal changed.", exact=False)).to_be_visible()
            await expect(page.get_by_text("invoice-2026-10", exact=True)).to_have_count(0)
            await expect(page.get_by_text("Use account A", exact=True)).to_have_count(0)
            await expect(page.get_by_role("button", name="Keep", exact=True)).to_be_disabled()

    asyncio.run(scenario())


def test_review_scoped_history_has_bounded_automatic_refresh():
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(access_scope=scope)
        async with _home_browser(
            seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            calls = []

            async def empty(route):
                from urllib.parse import parse_qs, urlsplit

                cursor = parse_qs(urlsplit(route.request.url).query).get("cursor", ["0"])[0]
                calls.append(cursor)
                await route.fulfill(
                    json={
                        "items": [],
                        "issues": [],
                        "next_cursor": str(int(cursor) + 1),
                        "has_more": True,
                    }
                )

            await page.route("**/api/inbox?*", empty)
            await page.goto(base + "/cayu/knowledge", wait_until="networkidle")
            await expect(
                page.get_by_role("button", name="Load more proposals", exact=True)
            ).to_be_enabled()
            assert len(calls) == 25
            await asyncio.sleep(5.5)
            assert len(calls) == 25, "Historical proposal windows must not replay on the timer"
            await page.get_by_role("button", name="Load more proposals", exact=True).click()
            await expect(
                page.get_by_role("button", name="Load more proposals", exact=True)
            ).to_be_enabled()
            assert len(calls) == 50

    asyncio.run(scenario())


@pytest.mark.parametrize("route_path", ["/inbox?kind=approval", "/sessions/approval"])
def test_shared_review_window_keeps_decisions_and_scoped_refresh(route_path):
    async def scenario():
        from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

        async with _home_browser(role="member", decisions=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            calls = []

            async def window(route):
                url = urlsplit(route.request.url)
                query = parse_qs(url.query)
                cursor = query.pop("cursor", [None])[0]
                calls.append((cursor, query))
                if cursor is None:
                    await route.fulfill(
                        json={"items": [], "issues": [], "next_cursor": "older", "has_more": True}
                    )
                else:
                    response = await route.fetch(
                        url=urlunsplit(url._replace(query=urlencode(query, doseq=True)))
                    )
                    await route.fulfill(response=response)

            await page.route("**/api/inbox?*", window)
            await page.goto(base + "/cayu" + route_path, wait_until="networkidle")
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_be_enabled()
            before = len(calls)
            await asyncio.sleep(5.5)
            assert len(calls) == before
            assert (
                all(query.get("session_id") == ["approval"] for _, query in calls)
                if route_path.startswith("/sessions")
                else all(query.get("kind") == ["approval"] for _, query in calls)
            )
            await page.get_by_role("button", name="Approve", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            await expect(
                page.get_by_text("No actions on this older page.", exact=False)
            ).to_be_visible()
            assert all(cursor == "older" for cursor, _ in calls[before:]), (
                "Decision invalidation must refresh only the active window"
            )
            refresh = page.get_by_role("button", name="Refresh actions", exact=True)
            await refresh.focus()
            await expect(refresh).to_be_focused()
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await refresh.click()
            await expect(
                page.get_by_role("button", name="Refresh actions", exact=True)
            ).to_be_visible()
            assert any(cursor is None for cursor, _ in calls[before:])

    asyncio.run(scenario())


@pytest.mark.parametrize("route_path", ["/knowledge", "/inbox", "/sessions/approval"])
def test_review_source_issue_page_is_not_discarded(route_path):
    async def scenario():
        async with _home_browser(decisions=True, knowledge=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            calls = []

            async def issue_page(route):
                calls.append(route.request.url)
                await route.fulfill(
                    json={
                        "items": [],
                        "issues": [
                            {
                                "code": "source_incomplete",
                                "detail": "Checkpoint evidence could not be read",
                                "session_id": "approval",
                            }
                        ],
                        "next_cursor": "continue-after-issue",
                        "has_more": True,
                    }
                )

            await page.route("**/api/inbox?*", issue_page)
            await page.goto(base + "/cayu" + route_path, wait_until="networkidle")
            await expect(
                page.get_by_text("Checkpoint evidence could not be read", exact=False)
            ).to_be_visible()
            await expect(
                page.get_by_role("link", name="Inspect session approval", exact=True)
            ).to_be_visible()
            assert len(calls) == 1, "Issue-bearing pages require explicit continuation"
            await expect(page.get_by_text("Nothing is waiting for you", exact=True)).to_have_count(
                0
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("route_path", ["/knowledge", "/inbox?kind=knowledge"])
def test_dropping_last_historical_proposal_does_not_claim_queue_empty(route_path):
    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = InMemoryKnowledgeStore(
            [
                KnowledgeEntry(
                    id=f"pending-{i:02}", text=f"Proposal {i}", status=KnowledgeStatus.PENDING
                )
                for i in range(51)
            ],
            access_scope=scope,
        )
        async with _home_browser(
            role="member", seeded_runtime=_app(knowledge_store=store, knowledge_access_scope=scope)
        ) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu" + route_path, wait_until="networkidle")
            next_label = "Load more proposals" if route_path == "/knowledge" else "Load more"
            await page.get_by_role("button", name=next_label, exact=True).click()
            await expect(page.get_by_role("button", name="Drop", exact=True)).to_have_count(1)
            await page.get_by_role("button", name="Drop", exact=True).click()
            await expect(page.get_by_role("button", name="Drop", exact=True)).to_have_count(0)
            await expect(
                page.get_by_text("No proposals waiting for review.", exact=True)
            ).to_have_count(0)
            await expect(page.get_by_text("Nothing is waiting for you", exact=True)).to_have_count(
                0
            )
            await expect(
                page.get_by_text("No actions on this older page.", exact=False)
            ).to_be_visible()
            await page.get_by_role(
                "button",
                name="Refresh proposals" if route_path == "/knowledge" else "Refresh actions",
                exact=True,
            ).click()
            await expect(page.get_by_role("button", name="Drop", exact=True)).to_have_count(50)

    asyncio.run(scenario())


@pytest.mark.parametrize("route_path", ["/knowledge", "/inbox", "/sessions/approval"])
def test_terminal_source_failure_never_becomes_empty_queue(route_path):
    async def scenario():
        async with _home_browser(decisions=True, knowledge=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            failed = True

            async def terminal(route):
                await route.fulfill(
                    json={
                        "items": [],
                        "issues": [
                            {
                                "code": "source_incomplete",
                                "detail": "Public source is unavailable",
                                "session_id": "approval",
                            }
                        ]
                        if failed
                        else [],
                        "next_cursor": None,
                        "has_more": False,
                    }
                )

            await page.route("**/api/inbox?*", terminal)
            await page.goto(base + "/cayu" + route_path, wait_until="networkidle")
            await expect(
                page.get_by_text("Public source is unavailable", exact=False)
            ).to_be_visible()
            await expect(page.get_by_text("Nothing is waiting for you", exact=True)).to_have_count(
                0
            )
            await expect(
                page.get_by_text("No proposals waiting for review.", exact=True)
            ).to_have_count(0)
            failed = False
            await page.get_by_role(
                "button",
                name="Refresh proposals" if route_path == "/knowledge" else "Refresh actions",
                exact=True,
            ).click()
            await expect(
                page.get_by_text("Public source is unavailable", exact=False)
            ).to_have_count(0)
            if route_path != "/sessions/approval":
                await expect(
                    page.get_by_text(
                        "No proposals waiting for review."
                        if route_path == "/knowledge"
                        else "Nothing is waiting for you",
                        exact=True,
                    )
                ).to_be_visible()

    asyncio.run(scenario())


def test_shipped_knowledge_capability_matrix(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    from examples.dashboard_behavior_live import _exercise_knowledge_capability_contract

    async def scenario():
        async with _home_browser(knowledge=True, tasks=True) as (page, base, _requests, _responses):
            await _exercise_knowledge_capability_contract(page, base)

    asyncio.run(scenario())


def test_shipped_optional_surfaces_keep_configured_knowledge_visible(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    from examples.dashboard_behavior_live import _exercise_optional_surface_capabilities, _seed_app

    async def scenario():
        runtime, _provider, _store, _tasks, directory = await _seed_app()
        try:
            async with _home_browser(seeded_runtime=runtime) as (page, base, _requests, _responses):
                await _exercise_optional_surface_capabilities(page, base)
        finally:
            directory.cleanup()

    asyncio.run(scenario())


def test_workflow_request_boundary_tears_down_previous_document(monkeypatch):
    from pathlib import Path
    from urllib.parse import urlsplit

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    from examples import dashboard_behavior_live as example

    class NavigationObserved(Exception):
        pass

    async def scenario():
        async with _home_browser() as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/sessions", wait_until="networkidle")
            previous_path = "/api/sessions/previous-document/transcript"
            current_path = "/api/sessions/unrelated-current-document/transcript"
            seen = []
            observing = False
            original_on = page.on
            original_goto = page.goto

            def on(event, callback):
                nonlocal observing
                if event == "request" and callback.__name__ == "record_workflow_request":
                    observing = True
                return original_on(event, callback)

            original_on(
                "request",
                lambda request: seen.append(urlsplit(request.url).path) if observing else None,
            )
            monkeypatch.setattr(page, "on", on)

            async def transcript(route):
                await route.fulfill(json={"messages": [], "offset": 0, "total": 0})

            await page.route("**/api/sessions/*-document/transcript", transcript)
            workflow_url = base + f"/cayu/sessions/{example.WORKFLOW_FOCUS_SESSION_ID}/workflow"

            async def goto(url, **kwargs):
                # Deliver the previous document's delayed read at the navigation
                # boundary, after the production observer has been installed.
                if url == workflow_url and page.url.endswith("/cayu/sessions"):
                    await page.evaluate("path => fetch(path)", previous_path)
                response = await original_goto(url, **kwargs)
                if url == workflow_url:
                    await page.evaluate("path => fetch(path)", current_path)
                    raise NavigationObserved
                return response

            monkeypatch.setattr(page, "goto", goto)
            with pytest.raises(NavigationObserved):
                await example._exercise_workflow(page, base, None, None, [])
            assert previous_path not in seen, (
                "Previous-document reads contaminated Workflow ownership"
            )
            assert current_path in seen, (
                "The guard must still observe unrelated current-document reads"
            )

    asyncio.run(scenario())
