"""Public HTTP and browser acceptance for the conversation-first session detail."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress

import pytest
from tests.core.test_cloud_reporting import _app
from tests.server.test_dashboard_session_search import search_browser

from cayu import Message
from cayu.sessions import RunRequest, SessionIdentity

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_SESSION_DETAIL_LIVE") != "1",
    reason="Opt-in packaged Chromium session-detail acceptance.",
)


def test_detail_bounded_latest_history_and_role_filters():
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(
                session_id="long-conversation",
                agent_name="writer",
                title="Long conversation",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await runtime.session_store.append_transcript_messages(
            "long-conversation",
            [
                Message.text("user" if index % 2 == 0 else "assistant", f"message-{index}")
                for index in range(10002)
            ],
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            response = await page.request.get(
                f"{base}/api/sessions/long-conversation/transcript?limit=1"
            )
            assert response.ok
            assert (await response.json())["total_messages"] == 10002
            reads = []
            page.on(
                "request",
                lambda request: (
                    reads.append(request.url) if "/transcript?" in request.url else None
                ),
            )
            await page.goto(f"{base}/cayu/sessions/long-conversation")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await expect(page.get_by_text("message-10001", exact=True)).to_be_visible()
            await expect(page.get_by_text("message-0", exact=True)).to_have_count(0)
            assert len(reads) <= 3, reads
            await page.get_by_role("button", name="Older messages", exact=True).click()
            await expect(page.get_by_text("message-9901", exact=True)).to_be_visible()
            await expect(page.get_by_text("message-10001", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Latest messages", exact=True).click()
            await expect(page.get_by_text("message-10001", exact=True)).to_be_visible()
            await page.get_by_label("Message role", exact=True).select_option("user")
            await expect(page.get_by_text("message-10000", exact=True)).to_be_visible()
            await expect(page.get_by_text("message-10001", exact=True)).to_have_count(0)
            await expect(page.get_by_role("tab", name="Overview", exact=True)).to_have_count(0)

    asyncio.run(run())


def test_detail_rejected_followup_retains_draft_then_submits_once():
    from playwright.async_api import expect
    from starlette.responses import JSONResponse
    from tests.core.test_cloud_reporting import _drain

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="followup",
                    agent_name="writer",
                    messages=[Message.text("user", "original")],
                    title="Follow-up",
                )
            )
        )
        submissions = []

        def reject_first(app):
            @app.middleware("http")
            async def reject(request, call_next):
                if request.url.path == "/api/resume" and request.method == "POST":
                    submissions.append(await request.json())
                    if len(submissions) == 1:
                        return JSONResponse({"detail": "Prompt rejected for test"}, status_code=422)
                return await call_next(request)

            return app

        async with search_browser(runtime=runtime, wrapper=reject_first) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/followup")
            composer = page.locator("#session-composer")
            await composer.fill("keep this draft")
            await page.get_by_role("button", name="Send", exact=True).click()
            await expect(page.get_by_text("Prompt rejected for test", exact=False)).to_be_visible()
            await expect(composer).to_have_value("keep this draft")
            await composer.fill("corrected follow-up")
            await page.get_by_role("button", name="Send", exact=True).click()
            await expect(composer).to_have_value("")
            await expect(page.get_by_text("corrected follow-up", exact=True)).to_be_visible()
            assert len(submissions) == 2
            assert [body["prompt"] for body in submissions] == [
                "keep this draft",
                "corrected follow-up",
            ]
            assert {body["session_id"] for body in submissions} == {"followup"}

    asyncio.run(run())


def test_detail_unknown_followup_reobserves_without_resubmission():
    import json

    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import _drain

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="uncertain",
                    agent_name="writer",
                    messages=[Message.text("user", "original")],
                    title="Uncertain",
                )
            )
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            submissions = []
            resume_requests = []
            page.on(
                "request",
                lambda request: (
                    resume_requests.append(request) if request.url.endswith("/api/resume") else None
                ),
            )

            async def lose_observation(route):
                submissions.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.ok
                await page.route(
                    "**/api/sessions/*/events?**",
                    lambda route: route.fulfill(
                        status=422, json={"detail": "Injected event observation unavailable"}
                    ),
                )
                error = {
                    "type": "stream.error",
                    "kind": "observer",
                    "code": "event_frame_too_large",
                    "error": "Injected observation loss",
                    "error_type": "ObserverFailure",
                    "retryable": False,
                    "session_id": "uncertain",
                }
                await route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body=f"event: error\ndata: {json.dumps(error)}\n\n",
                )

            await page.route("**/api/resume", lose_observation)
            await page.goto(f"{base}/cayu/sessions/uncertain")
            composer = page.locator("#session-composer")
            await composer.fill("accepted despite lost response")
            await page.get_by_role("button", name="Send", exact=True).click()
            await expect(
                page.locator('[data-mutation-transport-phase="transport_failed"]')
            ).to_be_visible(timeout=25_000)
            await expect(composer).to_have_value("accepted despite lost response")
            await expect(composer).to_be_disabled()
            await expect(page.get_by_role("button", name="Send", exact=True)).to_be_disabled()
            assert len(submissions) == 1
            await page.unroute("**/api/sessions/*/events?**")
            await page.unroute("**/api/resume", lose_observation)
            await page.get_by_role("button", name="Retry observation", exact=True).click()
            await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
                timeout=15_000
            )
            await expect(composer).to_have_value("")
            await expect(composer).to_be_enabled()
            assert len(submissions) == 1
            assert sum("last-event-id" not in request.headers for request in resume_requests) == 1
            transcript = await page.request.get(
                f"{base}/api/sessions/uncertain/transcript?role=user"
            )
            assert transcript.ok
            body = await transcript.json()
            assert (
                sum(
                    part.get("text") == "accepted despite lost response"
                    for message in body["messages"]
                    for part in message["content"]
                )
                == 1
            )

    asyncio.run(run())


def test_detail_member_mobile_reads_without_execution_controls():
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import _drain

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="member-read",
                    agent_name="writer",
                    messages=[Message.text("user", "Member conversation")],
                    title="Member access",
                )
            )
        )
        async with search_browser(runtime=runtime, member=True) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/member-read")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await expect(page.get_by_text("Member conversation", exact=True)).to_be_visible()
            await expect(page.locator("#session-composer")).to_have_count(0)
            await page.get_by_role("button", name="More actions", exact=True).click()
            await expect(page.get_by_text("Run again", exact=True)).to_have_count(0)
            await expect(page.get_by_text("Add to evals", exact=True)).to_have_count(0)
            await page.keyboard.press("Escape")
            await page.get_by_role("button", name="Session details", exact=True).click()
            await expect(page.get_by_role("tab", name="Overview", exact=True)).to_be_visible()
            await expect(page.get_by_role("tab", name="Browser", exact=True)).to_have_count(0)
            await expect(
                page.get_by_role("button", name="Edit labels", exact=True)
            ).to_be_disabled()
            await expect(
                page.get_by_role("button", name="Edit metadata", exact=True)
            ).to_be_disabled()
            await expect(page.get_by_test_id("annotations-unavailable")).to_be_visible()
            assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    asyncio.run(run())


def test_detail_member_settles_inline_approval_with_reason():
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import _drain

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="member-decide",
                    agent_name="approver",
                    messages=[Message.text("user", "go")],
                    title="Member decision",
                )
            )
        )
        async with search_browser(runtime=runtime, member=True) as (page, base, _, _, _):
            settlements = []
            page.on(
                "request",
                lambda request: (
                    settlements.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/settle")
                    else None
                ),
            )
            await page.goto(f"{base}/cayu/sessions/member-decide")
            await expect(page.get_by_role("button", name="Deny", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("Approve with changed arguments", exact=True)
            ).to_have_count(0)
            await page.get_by_label("Reason (optional)", exact=True).fill("Reviewed and declined")
            await page.get_by_role("button", name="Deny", exact=True).click()
            await expect(page.get_by_role("button", name="Deny", exact=True)).to_have_count(0)
            assert len(settlements) == 1
            assert settlements[0]["reason"] == "Reviewed and declined"
            assert settlements[0]["decision"] == "deny"
            pending = await page.request.get(f"{base}/api/inbox?session_id=member-decide")
            assert pending.ok
            assert (await pending.json())["items"] == []

    asyncio.run(run())


@pytest.mark.parametrize("execution_enabled", [True, False])
def test_detail_evaluation_uses_existing_captured_evidence_flow(execution_enabled):
    from pathlib import Path

    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import _drain

    from cayu.project_control_plane import (
        ProjectControlPlaneAccess,
        _create_project_control_plane_context,
    )
    from cayu.server import DashboardConfig, ServerConfig, create_server

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="evaluate",
                    agent_name="writer",
                    messages=[Message.text("user", "Retain this evidence")],
                    title="Evaluate this run",
                )
            )
        )
        context = _create_project_control_plane_context(
            project_root=Path(__file__).resolve().parents[2],
            project_id="detail-evaluation",
            eval_store=None,
            store_backend=None,
            store_source=None,
            access=ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT,
        )

        def configured(_app):
            return create_server(
                runtime,
                config=ServerConfig.local_development(
                    dashboard=DashboardConfig(
                        directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY")
                    )
                ),
                project_context=context,
            )

        async with search_browser(runtime=runtime, wrapper=configured) as (page, base, _, _, _):
            if not execution_enabled:

                async def disable_execution(route):
                    response = await route.fetch()
                    body = await response.json()
                    body["capabilities"]["mutations"]["session_execution"] = {
                        "enabled": False,
                        "unavailable_reason": "unsupported",
                    }
                    await route.fulfill(response=response, json=body)

                await page.route("**/api/contract", disable_execution)
            await page.goto(f"{base}/cayu/sessions/evaluate")
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Add to evals", exact=True).click()
            await expect(page.get_by_text("Evaluate captured session", exact=True)).to_be_visible()
            await expect(page.get_by_test_id("promotion-sheet")).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("allowed", "lost_ack"),
    [
        (False, None),
        (True, None),
        (True, "enqueue"),
        (True, "withdraw"),
        (True, "navigate"),
        (True, "deliver"),
    ],
)
def test_detail_running_queue_respects_policy_and_withdraws_exact_record(allowed, lost_ack):
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import PRICE, _drain, _Provider, _respond
    from tests.server.test_session_message_lifecycle_api import OwnershipPolicy

    from cayu import AgentSpec, CayuApp
    from cayu.server import AuthContext, DashboardConfig, ServerConfig, create_server

    async def run():
        reached, release = asyncio.Event(), asyncio.Event()

        class BlockingProvider(_Provider):
            async def stream(self, request):
                reached.set()
                await release.wait()
                async for event in super().stream(request):
                    yield event

        policy = OwnershipPolicy()
        runtime = CayuApp(session_message_access_policy=policy, enable_logging=False)
        runtime.register_provider(
            BlockingProvider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        runtime.register_agent(AgentSpec(name="writer", model=PRICE.model))
        execution = asyncio.create_task(
            _drain(
                runtime.run(
                    RunRequest(
                        session_id="queue",
                        agent_name="writer",
                        messages=[Message.text("user", "original")],
                        title="Running queue",
                    )
                )
            )
        )
        try:
            await asyncio.wait_for(reached.wait(), 10)
            session = await runtime.session_store.load("queue")
            assert session is not None
            if allowed:
                for action in ("inspect", "enqueue", "withdraw"):
                    policy.grants.add(("operator", None, "queue", session.instance_id, action))

            def configured(_app):
                return create_server(
                    runtime,
                    config=ServerConfig.protected(
                        lambda _request: AuthContext(subject="operator"),
                        dashboard=DashboardConfig(
                            directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY")
                        ),
                    ),
                )

            async with search_browser(runtime=runtime, wrapper=configured) as (page, base, _, _, _):
                try:
                    mutations = []
                    page.on(
                        "request",
                        lambda request: (
                            mutations.append(request)
                            if request.method == "POST" and "/messages" in request.url
                            else None
                        ),
                    )
                    await page.goto(f"{base}/cayu/sessions/queue")
                    composer = page.locator("#session-composer")
                    if not allowed:
                        await expect(
                            page.get_by_text(
                                "Queued messages require authenticated access", exact=False
                            )
                        ).to_be_visible()
                        await expect(composer).to_be_disabled()
                        assert mutations == []
                        return
                    await expect(composer).to_be_enabled()

                    accepted, acknowledgement, acknowledged = (
                        asyncio.Event(),
                        asyncio.Event(),
                        asyncio.Event(),
                    )

                    async def hold_acceptance(route):
                        if route.request.method != "POST":
                            await route.continue_()
                            return
                        from playwright.async_api import Error as PlaywrightError

                        response = await route.fetch()
                        assert response.ok
                        accepted.set()
                        await acknowledgement.wait()
                        # The real browser aborted observation on navigation.
                        try:
                            with suppress(PlaywrightError):
                                await route.fulfill(response=response)
                        finally:
                            acknowledged.set()

                    async def lose_enqueue(route):
                        if route.request.method != "POST":
                            await route.continue_()
                            return
                        response = await route.fetch()
                        assert response.ok
                        await route.fulfill(status=200, content_type="text/event-stream", body="")

                    async def lose_withdraw(route):
                        response = await route.fetch()
                        assert response.ok
                        await route.fulfill(
                            status=503, json={"detail": "Lost withdrawal acknowledgement"}
                        )

                    if lost_ack == "navigate":
                        await page.route("**/api/sessions/queue/messages", hold_acceptance)
                    if lost_ack == "enqueue":
                        await page.route("**/api/sessions/queue/messages", lose_enqueue)
                    if lost_ack == "withdraw":
                        await page.route("**/api/sessions/queue/messages/*/withdraw", lose_withdraw)
                    await composer.fill("Please verify the new entry")
                    await page.get_by_role("button", name="Send", exact=True).click()
                    if lost_ack == "navigate":
                        try:
                            await asyncio.wait_for(accepted.wait(), 10)
                            await (
                                page.locator("main")
                                .get_by_role("link", name="Sessions", exact=True)
                                .click()
                            )
                            response = await page.request.get(f"{base}/api/sessions/queue/messages")
                            assert response.ok
                            assert [
                                record["status"] for record in (await response.json())["records"]
                            ] == ["queued"]
                            assert len(mutations) == 1
                            assert not execution.done()
                        finally:
                            acknowledgement.set()
                            await asyncio.wait_for(acknowledged.wait(), 10)
                            await page.unroute("**/api/sessions/queue/messages", hold_acceptance)
                        return
                    if lost_ack == "enqueue":
                        await expect(
                            page.get_by_role("button", name="Retry message acceptance", exact=True)
                        ).to_be_visible()
                        await expect(composer).to_have_value("Please verify the new entry")
                        await expect(composer).to_be_disabled()
                        await page.unroute("**/api/sessions/queue/messages", lose_enqueue)
                        await page.get_by_role(
                            "button", name="Retry message acceptance", exact=True
                        ).click()
                    await expect(page.get_by_text("1 message queued", exact=True)).to_be_visible()
                    await expect(composer).to_have_value("")
                    if lost_ack == "deliver":
                        release.set()
                        await asyncio.wait_for(execution, 10)
                        await expect(
                            page.get_by_text("1 message queued", exact=True)
                        ).to_have_count(0)
                        await expect(
                            page.get_by_text("Please verify the new entry", exact=True)
                        ).to_be_visible()
                        response = await page.request.get(f"{base}/api/sessions/queue/messages")
                        assert [
                            record["status"] for record in (await response.json())["records"]
                        ] == ["delivered"]
                        assert len(mutations) == 1
                        return
                    await page.get_by_role(
                        "button", name="Withdraw queued message", exact=True
                    ).click()
                    if lost_ack == "withdraw":
                        await expect(
                            page.get_by_text("Withdrawal was not confirmed.", exact=False)
                        ).to_be_visible()
                        await page.unroute(
                            "**/api/sessions/queue/messages/*/withdraw", lose_withdraw
                        )
                        await page.get_by_role(
                            "button", name="Withdraw queued message", exact=True
                        ).click()
                    await expect(page.get_by_text("1 message queued", exact=True)).to_have_count(0)
                    assert len(mutations) == (3 if lost_ack else 2)
                    enqueue_requests = [
                        request for request in mutations if request.url.endswith("/messages")
                    ]
                    withdraw_requests = [
                        request for request in mutations if request.url.endswith("/withdraw")
                    ]
                    assert all(
                        request.post_data_json == enqueue_requests[0].post_data_json
                        for request in enqueue_requests
                    )
                    assert all(
                        request.post_data_json == withdraw_requests[0].post_data_json
                        for request in withdraw_requests
                    )
                    enqueue, withdraw = (
                        enqueue_requests[0].post_data_json,
                        withdraw_requests[0].post_data_json,
                    )
                    assert enqueue["delivery_mode"] == "next_turn"
                    assert enqueue["content"] == "Please verify the new entry"
                    assert "last-event-id" not in mutations[0].headers
                    assert withdraw["session_instance_id"]
                    assert len(withdraw["expected_revision"]) == 64
                    response = await page.request.get(f"{base}/api/sessions/queue/messages")
                    assert response.ok
                    assert [record["status"] for record in (await response.json())["records"]] == [
                        "withdrawn"
                    ]
                finally:
                    release.set()
                    await asyncio.wait_for(asyncio.shield(execution), 10)
        finally:
            release.set()
            await asyncio.wait_for(execution, 10)

    asyncio.run(run())


@pytest.mark.parametrize("missing_authority", [False, True])
def test_detail_recovery_uses_exact_public_authority(missing_authority):
    from playwright.async_api import expect
    from tests.server.test_pending_recovery_linkage import _app as recovery_app
    from tests.server.test_pending_recovery_linkage import _crash

    from cayu.sessions import InMemorySessionStore

    async def run():
        runtime, tool = recovery_app(InMemorySessionStore())
        await _crash(runtime)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            responses = []
            page.on("response", lambda response: responses.append(response))
            if missing_authority:

                async def omit_authority(route):
                    response = await route.fetch()
                    body = await response.json()
                    for action in body["actions"]:
                        action["tool_call_id"] = None
                    await route.fulfill(response=response, json=body)

                await page.route("**/api/pending-actions?**", omit_authority)
            await page.goto(f"{base}/cayu/sessions/crashed")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            if missing_authority:
                await expect(
                    page.get_by_text(
                        "Recovery is unavailable because its exact tool-call authority is not exposed.",
                        exact=False,
                    )
                ).to_be_visible()
                await expect(page.get_by_role("button", name="Recover", exact=True)).to_have_count(
                    0
                )
                assert not any(
                    response.request.method == "POST" and "/recover" in response.url
                    for response in responses
                )
                return
            await page.get_by_label("The person's answer", exact=True).fill("staging")
            await page.get_by_label("Verified result or failure reason", exact=True).fill(
                "Verified sibling completed externally."
            )
            async with page.expect_response(
                lambda response: (
                    response.request.method == "POST" and "/user-input/recover" in response.url
                )
            ) as response_info:
                await page.get_by_role("button", name="Recover", exact=True).click()
            response = await response_info.value
            assert response.status == 200, await response.text()
            await expect(page.get_by_role("button", name="Recover", exact=True)).to_have_count(0)
            assert (await runtime.session_store.load("crashed")).status == "completed"
            recovered = [
                response
                for response in responses
                if response.request.method == "POST" and "/user-input/recover" in response.url
            ]
            assert len(recovered) == 1
            assert recovered[0].status == 200
            assert recovered[0].request.post_data_json["tool_call_id"].startswith("cayu_event_")
            assert not any(
                response.request.method == "POST" and response.url.endswith("/settle")
                for response in responses
            )
            assert tool.calls == 0

    asyncio.run(run())


def test_detail_interrupt_confirms_reason_and_stops_live_runtime():
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import PRICE, _drain, _Provider, _respond

    from cayu import AgentSpec, CayuApp

    async def run():
        reached, release = asyncio.Event(), asyncio.Event()

        class BlockingProvider(_Provider):
            async def stream(self, request):
                reached.set()
                await release.wait()
                async for event in super().stream(request):
                    yield event

        runtime = CayuApp(enable_logging=False)
        runtime.register_provider(
            BlockingProvider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        runtime.register_agent(AgentSpec(name="writer", model=PRICE.model))
        execution = asyncio.create_task(
            _drain(
                runtime.run(
                    RunRequest(
                        session_id="interrupt",
                        agent_name="writer",
                        messages=[Message.text("user", "original")],
                        title="Stop this run",
                    )
                )
            )
        )
        try:
            await asyncio.wait_for(reached.wait(), 10)
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                try:
                    requests = []
                    page.on(
                        "request",
                        lambda request: (
                            requests.append(request.post_data_json)
                            if request.method == "POST" and request.url.endswith("/interrupt")
                            else None
                        ),
                    )
                    await page.goto(f"{base}/cayu/sessions/interrupt")
                    await page.get_by_role("button", name="Stop", exact=True).click()
                    await expect(
                        page.get_by_role("button", name="Keep running", exact=True)
                    ).to_be_visible()
                    assert not requests
                    await page.locator("#stop-reason").fill("Operator verified this should stop")
                    await page.get_by_role("button", name="Stop session", exact=True).click()
                    await expect(
                        page.get_by_role("button", name="Stop session", exact=True)
                    ).to_have_count(0)
                    await expect(
                        page.get_by_text(
                            "This session was stopped. Send a message to continue it.", exact=True
                        )
                    ).to_be_visible()
                    assert requests == [{"reason": "Operator verified this should stop"}]
                    response = await page.request.get(f"{base}/api/sessions/interrupt")
                    assert response.ok
                    assert (await response.json())["status"] == "interrupted"
                finally:
                    release.set()
                    await asyncio.wait_for(asyncio.gather(execution, return_exceptions=True), 10)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(execution, return_exceptions=True), 10)

    asyncio.run(run())


def test_detail_tool_results_keep_exact_round_evidence():
    from playwright.async_api import expect

    from cayu.messages import ToolCallPart, ToolResultPart

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="tool-evidence", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        messages = []
        for label, output in [
            ("round-one", "first exact result"),
            ("round-two", "second exact result"),
        ]:
            round_id = f"tround_{'1' * 32}" if label == "round-one" else f"tround_{'2' * 32}"
            messages.extend(
                [
                    Message(
                        role="assistant",
                        content=[
                            ToolCallPart(
                                tool_call_id="same",
                                tool_name="read",
                                arguments={"path": label},
                                tool_round_id=round_id,
                                model_step_id=f"mstep_{'3' * 32}",
                                model_attempt_id=f"matt_{'4' * 32}",
                            )
                        ],
                    ),
                    Message(
                        role="tool",
                        content=[
                            ToolResultPart(
                                tool_call_id="same",
                                tool_name="read",
                                content=output,
                                tool_round_id=round_id,
                                model_step_id=f"mstep_{'3' * 32}",
                                model_attempt_id=f"matt_{'4' * 32}",
                            )
                        ],
                    ),
                ]
            )
        await runtime.session_store.append_transcript_messages("tool-evidence", messages)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/tool-evidence")
            await page.get_by_role("button", name="Used read 2 times", exact=False).click()
            await page.get_by_role("button", name="read round-one", exact=False).click()
            await expect(page.get_by_text("first exact result", exact=True)).to_be_visible()
            await expect(page.get_by_text("second exact result", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="read round-two", exact=False).click()
            await expect(page.get_by_text("second exact result", exact=True)).to_be_visible()
            await expect(
                page.get_by_role("button", name="View in trace", exact=True)
            ).to_have_count(2)

    asyncio.run(run())


def test_detail_redacted_tool_identity_keeps_result_as_independent_evidence():
    from playwright.async_api import expect

    from cayu.messages import ToolCallPart, ToolResultPart
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        runtime = _app(secret_redactor=SecretRedactor("hidden-call-secret"))
        await runtime.session_store.create(
            RunRequest(session_id="hidden-tool", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await runtime.session_store.append_transcript_messages(
            "hidden-tool",
            [
                Message(
                    role="assistant",
                    content=[
                        ToolCallPart(
                            tool_call_id="hidden-call-secret",
                            tool_name="read",
                            arguments={"path": "report"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content=[
                        ToolResultPart(
                            tool_call_id="hidden-call-secret",
                            tool_name="read",
                            content="Independent redacted outcome",
                        )
                    ],
                ),
            ],
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            response = await page.request.get(f"{base}/api/sessions/hidden-tool/transcript")
            assert response.ok
            payload = await response.json()
            assert payload["messages"][0]["content"][0]["tool_call_id"] == "[REDACTED_SECRET]"
            await page.goto(f"{base}/cayu/sessions/hidden-tool")
            await expect(
                page.get_by_text("result Independent redacted outcome", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("hidden-call-secret", exact=False)).to_have_count(0)

    asyncio.run(run())


def test_detail_structured_output_and_long_user_message_preserve_exact_content():
    from playwright.async_api import expect

    from cayu.messages import StructuredOutputPart

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="structured-conversation", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        long_text = "Visible beginning " + "message body " * 130 + "exact ending"
        output = {"approval_needed": False, "record_count": 0, "findings": [], "note": None}
        await runtime.session_store.append_transcript_messages(
            "structured-conversation",
            [
                Message.text("user", long_text),
                Message(
                    role="assistant", content=[StructuredOutputPart(name="Review", output=output)]
                ),
            ],
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/structured-conversation")
            await expect(page.get_by_text("exact ending", exact=False)).to_have_count(0)
            await page.get_by_role("button", name="Show all", exact=False).click()
            await expect(page.locator("p").filter(has_text=long_text)).to_be_visible()
            await page.get_by_role("button", name="Show less", exact=True).click()
            await expect(page.locator("p").filter(has_text=long_text)).to_have_count(0)
            await expect(page.get_by_text("Approval needed", exact=True)).to_be_visible()
            await expect(page.get_by_text("Record count", exact=True)).to_be_visible()
            await expect(page.get_by_text("No", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Show JSON", exact=True).click()
            assert json.loads(await page.locator("pre").inner_text()) == output
            await page.get_by_role("button", name="Show formatted", exact=True).click()
            await expect(page.get_by_text("Approval needed", exact=True)).to_be_visible()

    asyncio.run(run())


def test_detail_suppressed_arguments_and_structured_tool_result_remain_readable():
    from playwright.async_api import expect

    from cayu.messages import ToolCallPart, ToolResultPart

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="suppressed-tool", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await runtime.session_store.append_transcript_messages(
            "suppressed-tool",
            [
                Message(
                    role="assistant",
                    content=[
                        ToolCallPart(
                            tool_call_id="private-call",
                            tool_name="read",
                            arguments={},
                            arguments_state="unavailable",
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content=[
                        ToolResultPart(
                            tool_call_id="private-call",
                            tool_name="read",
                            content="Recovery guidance",
                            structured={"safe_findings": ["retained evidence"]},
                        )
                    ],
                ),
            ],
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/suppressed-tool")
            await page.get_by_role("button", name="Used read", exact=True).click()
            await page.get_by_role("button", name="read", exact=True).last.click()
            await expect(
                page.get_by_text("Original tool arguments are unavailable.", exact=False)
            ).to_be_visible()
            await expect(page.get_by_text("Recovery guidance", exact=True)).to_be_visible()
            await expect(page.get_by_text("Safe findings", exact=True)).to_be_visible()
            await expect(page.get_by_text("retained evidence", exact=True)).to_be_visible()

    asyncio.run(run())


def test_detail_direct_children_page_independently_of_unrelated_sessions():
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        for session_id, parent_id, title in [
            ("relation-parent", None, "Relation parent"),
            *[(f"child-{i:03}", "relation-parent", f"Child {i}") for i in range(101)],
            *[(f"unrelated-{i:03}", None, f"Unrelated {i}") for i in range(205)],
        ]:
            await runtime.session_store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="writer",
                    parent_session_id=parent_id,
                    title=title,
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/relation-parent")
            children = page.locator('a[href^="/cayu/sessions/child-"]')
            await expect(children).to_have_count(100)
            await expect(page.get_by_text("Unrelated 204", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Load more subagents", exact=True).click()
            await expect(children).to_have_count(101)
            await expect(
                page.get_by_role("button", name="Load more subagents", exact=True)
            ).to_have_count(0)
            await page.locator('a[href="/cayu/sessions/child-000"]').click()
            await page.get_by_role("button", name="Session details", exact=True).click()
            await expect(page.locator('a[href="/cayu/sessions/relation-parent"]')).to_be_visible()

    asyncio.run(run())


def test_detail_annotations_edit_user_fields_and_preserve_runtime_metadata():
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(
                session_id="annotated",
                agent_name="writer",
                title="Runtime title",
                labels={"team": "old"},
                metadata={"note": "old"},
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/annotated")
            await page.get_by_role("button", name="Session details", exact=True).click()
            await page.get_by_role("button", name="Edit labels", exact=True).click()
            await page.get_by_label("Session labels JSON", exact=True).fill('{"team":"new"}')
            await page.get_by_role("button", name="Save labels", exact=True).click()
            await expect(page.get_by_label("Session labels JSON", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Edit metadata", exact=True).click()
            editor = page.get_by_label("Session metadata JSON", exact=True)
            assert "cayu:title" not in await editor.input_value()
            await editor.fill('{"note":"updated"}')
            await page.get_by_role("button", name="Save metadata", exact=True).click()
            await expect(editor).to_have_count(0)
            response = await page.request.get(f"{base}/api/sessions/annotated")
            assert response.ok
            session = await response.json()
            assert session["labels"] == {"team": "new"}
            assert session["metadata"]["note"] == "updated"
            assert session["metadata"]["cayu:title"] == "Runtime title"

    asyncio.run(run())


@pytest.mark.parametrize("lost_ack", [False, True])
def test_detail_task_run_again_uses_exact_replay_and_receipt(lost_ack):
    from playwright.async_api import expect
    from tests.core.task_invocation_fixtures import stored_session_invocation
    from tests.core.test_cloud_reporting import PRICE, _Provider, _respond

    from cayu import AgentSpec, CayuApp
    from cayu.sessions.requests import run_request_with_task_invocation
    from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskTypeSpec

    async def run():
        tasks = InMemoryTaskStore()
        runtime = CayuApp(task_store=tasks, enable_logging=False)
        runtime.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        runtime.register_agent(AgentSpec(name="worker", model=PRICE.model))
        runtime.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
        await tasks.create_task(
            TaskCreate(task_id="source-task", type="report", input={"topic": "original"})
        )
        invocation = await tasks.load_invocation_snapshot("source-task")
        await runtime.session_store.create(
            run_request_with_task_invocation(
                RunRequest(
                    session_id="task-session",
                    task_id="source-task",
                    agent_name="worker",
                    messages=[Message.text("user", "Transcript differs from original task input")],
                ),
                invocation,
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await tasks.start_task(
            "source-task",
            session_id="task-session",
            session_invocation=await stored_session_invocation(
                runtime.session_store, "task-session"
            ),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            posts = []

            async def dispatch(route):
                posts.append(route.request.post_data_json)
                response = await route.fetch()
                if lost_ack and len(posts) == 1:
                    await route.abort("failed")
                else:
                    await route.fulfill(response=response)

            await page.route("**/api/tasks/source-task/replay", dispatch)
            await page.goto(f"{base}/cayu/sessions/task-session")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Run again", exact=True).click()
            if lost_ack:
                await expect(
                    page.get_by_text("Task creation is unconfirmed.", exact=False)
                ).to_be_visible()
                await page.get_by_role("button", name="Retry task replay", exact=True).click()
            await expect(page).to_have_url(__import__("re").compile(r"/cayu/tasks\?task_id=.+$"))
            from urllib.parse import parse_qs, urlsplit

            new_id = parse_qs(urlsplit(page.url).query)["task_id"][0]
            await expect(page.get_by_role("heading", name="Tasks", exact=True)).to_be_visible()
            await expect(page.get_by_text("Task Details", exact=True)).to_be_visible()
            await expect(page.get_by_text(new_id, exact=True).last).to_be_visible()
            assert new_id != "source-task"
            fresh = await tasks.load_task(new_id)
            assert fresh is not None
            assert fresh.input == {"topic": "original"}
            assert fresh.session_id is None
            assert len(posts) == (2 if lost_ack else 1)
            if lost_ack:
                assert posts[0] == posts[1]
            assert posts[0]["source"]["session_id"] == "task-session"
            assert posts[0]["source"]["session_instance_id"]

    asyncio.run(run())


@pytest.mark.parametrize("resolution", ["fail", "fallback_retry"])
def test_detail_provider_resolution_keeps_exact_stage_and_audit_reason(resolution):
    from playwright.async_api import expect
    from tests.core.test_provider_operation_offline_recovery import (
        _OfflineOperationProvider,
        _stage_offline_operation,
    )

    from cayu import AgentSpec, CayuApp
    from cayu.events import EventType
    from cayu.providers import ModelStreamEvent
    from cayu.providers.operations import ProviderOperationStatus
    from cayu.sessions import InMemorySessionStore
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

    async def run():
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.EXPIRED)
        await _stage_offline_operation(store, session_id="provider-resolution", provider=provider)
        runtime = CayuApp(session_store=store, enable_logging=False)
        runtime.register_provider(provider, default=True)
        runtime.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await runtime.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="provider-resolution",
                inactive_for_seconds=0,
            )
        )
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("Recovered provider response"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            state = await (
                await page.request.get(f"{base}/api/sessions/provider-resolution/state")
            ).json()
            authority = state["provider_operation"]
            posts = []
            page.on(
                "request",
                lambda request: (
                    posts.append(request.post_data_json)
                    if request.method == "POST"
                    and request.url.endswith("/provider-operations/resolve")
                    else None
                ),
            )
            await page.goto(f"{base}/cayu/sessions/provider-resolution")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            button = page.get_by_role(
                "button",
                name="Fail session" if resolution == "fail" else "Retry from Cayu boundary",
                exact=True,
            )
            await expect(button).to_be_disabled()
            await page.get_by_placeholder(
                "Required operator reason for the durable audit record"
            ).fill("Verified expired upstream request")
            await button.click()
            await expect(page.get_by_test_id("provider-operation-resolution")).to_have_count(0)
            assert len(posts) == 1
            assert posts[0]["stage_id"] == authority["stage_id"]
            assert posts[0]["expected_run_epoch"] == authority["run_epoch"]
            assert posts[0]["action"] == resolution
            assert posts[0]["reason"] == "Verified expired upstream request"
            assert provider.adapter.start_calls == (1 if resolution == "fallback_retry" else 0)
            events = await store.load_events("provider-resolution")
            assert sum(event.type is EventType.PROVIDER_OPERATION_RESOLVED for event in events) == 1
            assert (await store.load("provider-resolution")).status == (
                "completed" if resolution == "fallback_retry" else "failed"
            )

    asyncio.run(run())


def test_detail_direct_run_again_creates_one_new_session_from_original_prompt():
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import _drain

    async def run():
        runtime = _app()
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="direct-source",
                    agent_name="writer",
                    messages=[Message.text("user", "Original direct prompt")],
                )
            )
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            requests = []
            page.on(
                "request",
                lambda request: (
                    requests.append(request)
                    if request.method == "POST" and request.url.endswith("/api/run")
                    else None
                ),
            )
            await page.goto(f"{base}/cayu/sessions/direct-source")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Run again", exact=True).click()
            await page.wait_for_url(
                lambda url: "/sessions/" in str(url) and "direct-source" not in str(url)
            )
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            assert len(requests) == 1
            body = requests[0].post_data_json
            assert body["agent"] == "writer"
            assert body["prompt"] == "Original direct prompt"
            destination = page.url.split("/sessions/", 1)[1].split("?", 1)[0]
            response = await page.request.get(f"{base}/api/sessions/{destination}")
            assert response.ok
            assert (await response.json())["invocation"]["source"] != "task"

    asyncio.run(run())


def test_detail_member_protected_review_uses_exact_reference_and_reason():
    from playwright.async_api import expect
    from tests.server.test_dashboard_home_inbox import _home_browser

    async def run():
        async with _home_browser(role="member", decisions=True, protected=True) as (
            page,
            base,
            requests,
            responses,
        ):
            settlements = []
            page.on(
                "request",
                lambda request: (
                    settlements.append(request.post_data_json)
                    if request.method == "POST" and request.url.endswith("/settle")
                    else None
                ),
            )
            await page.goto(base + "/cayu/sessions/approval")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await expect(page.get_by_text("Protected human review", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            await page.get_by_role("button", name="Refresh review", exact=True).click()
            await expect(page.get_by_text("Review this exact proposal", exact=True)).to_be_visible()
            await page.get_by_label("Decision reason", exact=True).fill("Checked Session proposal")
            await page.get_by_role("button", name="Approve", exact=True).click()
            await expect(page.get_by_role("button", name="Approve", exact=True)).to_have_count(0)
            assert len(settlements) == 1
            assert settlements[0]["reason"] == "Checked Session proposal"
            assert settlements[0]["decision"] == "approve"
            assert settlements[0]["review_reference"]
            assert any("human-review" in request for request in requests)
            assert not any(response.status >= 400 for response in responses)

    asyncio.run(run())


def test_detail_interrupt_owns_separate_transport_while_followup_is_observed():
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import PRICE, _drain, _Provider, _respond

    from cayu import AgentSpec, CayuApp

    async def run():
        reached, release = asyncio.Event(), asyncio.Event()

        class BlockingFollowup(_Provider):
            calls = 0

            async def stream(self, request):
                self.calls += 1
                if self.calls > 1:
                    reached.set()
                    await release.wait()
                async for event in super().stream(request):
                    yield event

        runtime = CayuApp(enable_logging=False)
        runtime.register_provider(
            BlockingFollowup(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        runtime.register_agent(AgentSpec(name="writer", model=PRICE.model))
        await _drain(
            runtime.run(
                RunRequest(
                    session_id="overlap",
                    agent_name="writer",
                    messages=[Message.text("user", "original")],
                )
            )
        )
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            try:
                posts = []
                page.on(
                    "request",
                    lambda request: posts.append(request) if request.method == "POST" else None,
                )
                await page.goto(f"{base}/cayu/sessions/overlap")
                await expect(
                    page.get_by_role("tab", name="Conversation", exact=True)
                ).to_be_visible()
                await page.locator("#session-composer").fill("Block this follow-up")
                await page.get_by_role("button", name="Send", exact=True).click()
                await asyncio.wait_for(reached.wait(), 10)
                await page.get_by_role("button", name="Stop", exact=True).click()
                await page.locator("#stop-reason").fill("Stop during observation")
                await page.get_by_role("button", name="Stop session", exact=True).click()
                await expect(
                    page.get_by_text(
                        "This session was stopped. Send a message to continue it.", exact=True
                    )
                ).to_be_visible()
                assert len([request for request in posts if request.url.endswith("/resume")]) == 1
                interrupts = [request for request in posts if request.url.endswith("/interrupt")]
                assert len(interrupts) == 1
                assert interrupts[0].post_data_json == {"reason": "Stop during observation"}
                response = await page.request.get(f"{base}/api/sessions/overlap")
                assert (await response.json())["status"] == "interrupted"
            finally:
                release.set()

    asyncio.run(run())


def test_detail_existing_mutation_example_uses_conversation_shell(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import _exercise_existing_session_mutations, _seed_app

    async def run():
        runtime, provider, _store, _tasks, directory = await _seed_app()
        try:
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                await _exercise_existing_session_mutations(page, base, provider)
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


def test_detail_annotation_example_preserves_reserved_metadata(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import SESSION_ID, _exercise_session_annotations, _seed_app

    async def run():
        runtime, provider, _store, _tasks, directory = await _seed_app()
        try:
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                await page.goto(f"{base}/cayu/sessions/{SESSION_ID}")
                await _exercise_session_annotations(page)
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("state", ["terminal", "waiting"])
def test_detail_shared_progress_example_has_one_bounded_wait(monkeypatch, state):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import (
        _exercise_terminal_session_polling,
        _exercise_waiting_session_backoff,
        _seed_polling_budget_app,
    )

    async def run():
        runtime = await _seed_polling_budget_app()
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            exercise = (
                _exercise_terminal_session_polling
                if state == "terminal"
                else _exercise_waiting_session_backoff
            )
            await exercise(page.context.browser.new_context, base)

    asyncio.run(run())


def test_detail_embedded_workflow_example_keeps_native_controls(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from cayu.server import DashboardConfig, ServerConfig, create_server
    from dashboard_behavior_live import _dashboard_price_book, _exercise_workflow, _seed_app

    async def run():
        runtime, provider, store, tasks, directory = await _seed_app()
        try:

            def configured(_app):
                return create_server(
                    runtime,
                    config=ServerConfig.local_development(
                        dashboard=DashboardConfig(
                            directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY"),
                            runtime_config={
                                "priceBook": _dashboard_price_book().model_dump(mode="json")
                            },
                        )
                    ),
                )

            async with search_browser(runtime=runtime, wrapper=configured) as (page, base, _, _, _):
                aborts = []
                page.on(
                    "requestfailed",
                    lambda request: (
                        aborts.append(request.url)
                        if request.method == "POST"
                        and request.url.endswith("/topology")
                        and "ERR_ABORTED" in (request.failure or "")
                        else None
                    ),
                )
                await _exercise_workflow(page, base, store, tasks, aborts)
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("event_page", ["initial", "delayed", "older"])
def test_detail_observation_example_preserves_event_and_transcript_links(monkeypatch, event_page):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import SESSION_ID, _exercise_session_observation, _seed_app

    async def run():
        runtime, provider, store, _tasks, directory = await _seed_app()
        try:
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                if event_page == "older":
                    from cayu.events import Event, EventType

                    for index in range(150):
                        await store.append_event(
                            SESSION_ID,
                            Event(
                                type=EventType.KNOWLEDGE_LOOKUP,
                                session_id=SESSION_ID,
                                payload={"query": f"recent-{index}"},
                            ),
                        )
                entered, release = asyncio.Event(), asyncio.Event()
                held = False

                async def delay_initial(route):
                    nonlocal held
                    from urllib.parse import parse_qs, urlsplit

                    from playwright.async_api import Error as PlaywrightError

                    query = parse_qs(urlsplit(route.request.url).query)
                    if (
                        held
                        or query.get("limit") != ["100"]
                        or query.get("order_by") != ["sequence_desc"]
                        or "event_id" in query
                    ):
                        await route.continue_()
                        return
                    held = True
                    response = await route.fetch()
                    entered.set()
                    await release.wait()
                    with suppress(PlaywrightError):
                        await route.fulfill(response=response)

                if event_page == "delayed":
                    await page.route(f"**/api/sessions/{SESSION_ID}/events?*", delay_initial)
                await page.goto(f"{base}/cayu/sessions/{SESSION_ID}")
                exercise = asyncio.create_task(_exercise_session_observation(page, base, store))
                try:
                    if event_page == "delayed":
                        from playwright.async_api import expect

                        await asyncio.wait_for(entered.wait(), 10)
                        await expect(
                            page.get_by_role("button", name="Resume live tail", exact=True)
                        ).to_be_visible()
                        release.set()
                    await exercise
                finally:
                    release.set()
                    if not exercise.done():
                        exercise.cancel()
                    await asyncio.gather(exercise, return_exceptions=True)
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize(
    "identity", ["missing", "null", "partial", "complete-reversed", "window", "duplicate"]
)
def test_detail_incomplete_or_noncausal_tool_results_remain_independent(identity):
    from playwright.async_api import expect

    from cayu.messages import ToolCallPart, ToolResultPart

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="unpaired", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        fields = dict(
            model_step_id=f"mstep_{'1' * 32}",
            model_attempt_id=f"matt_{'2' * 32}",
            tool_round_id=f"tround_{'3' * 32}",
        )
        call = Message(
            role="assistant",
            content=[
                ToolCallPart(
                    tool_call_id="same", tool_name="read", arguments={"path": "new-call"}, **fields
                )
            ],
        )
        result = Message(
            role="tool",
            content=[
                ToolResultPart(
                    tool_call_id="same",
                    tool_name="read",
                    content="Earlier independent evidence",
                    **fields,
                )
            ],
        )
        messages = [result, call] if identity != "duplicate" else [call, call, result]
        if identity == "window":
            messages = [Message.text("assistant", f"filler {i}") for i in range(202)]
            messages[0], messages[2], messages[101], messages[201] = call, result, call, result
        await runtime.session_store.append_transcript_messages("unpaired", messages)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):

            async def project_optional_fields(route):
                response = await route.fetch()
                body = await response.json()
                for message in body["messages"]:
                    for part in message["content"]:
                        if part["type"] not in ("tool_call", "tool_result"):
                            continue
                        for key in fields:
                            if identity in ("missing", "window"):
                                part.pop(key, None)
                            elif identity == "null" or (
                                identity == "partial" and key != "tool_round_id"
                            ):
                                part[key] = None
                await route.fulfill(response=response, json=body)

            await page.route("**/api/sessions/unpaired/transcript?**", project_optional_fields)
            await page.goto(f"{base}/cayu/sessions/unpaired")
            if identity == "window":
                await page.get_by_role("button", name="Older messages", exact=True).click()
            await expect(
                page.get_by_text("result Earlier independent evidence", exact=True)
            ).to_be_visible()
            await page.get_by_role("button", name="read", exact=False).first.click()
            await expect(
                page.get_by_text("result Earlier independent evidence", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("result not loaded", exact=False).first).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("role", ["all", "user"])
def test_detail_user_file_parts_survive_paging_and_role_filter(role):
    from playwright.async_api import expect

    from cayu.messages import FilePart, TextPart

    async def run():
        runtime = _app()
        await runtime.session_store.create(
            RunRequest(session_id="files", agent_name="writer", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )

        def attachment(name):
            return FilePart(
                attachment={
                    "type": "cayu.file_attachment.v1",
                    "artifact_id": name,
                    "kind": "document",
                    "filename": f"{name}.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 10,
                }
            )

        messages = [
            Message(role="user", content=[attachment("first-file")]),
            Message(
                role="user",
                content=[
                    TextPart(text="Before attachment"),
                    attachment("mixed-file"),
                    TextPart(text="After attachment"),
                ],
            ),
        ]
        messages += [Message.text("user", f"padding {i}") for i in range(100)]
        await runtime.session_store.append_transcript_messages("files", messages)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            await page.goto(f"{base}/cayu/sessions/files")
            if role == "user":
                await page.get_by_label("Message role", exact=True).select_option("user")
            await page.get_by_role("button", name="Older messages", exact=True).click()
            await expect(page.locator("pre").filter(has_text="first-file.pdf")).to_be_visible()
            await expect(
                page.locator(
                    '[data-transcript-part="m1.1"][data-transcript-role="user"] pre'
                ).filter(has_text="mixed-file.pdf")
            ).to_be_visible()
            await expect(page.get_by_text("Before attachment", exact=True)).to_be_visible()
            await expect(page.get_by_text("After attachment", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Newer messages", exact=True).click()
            await expect(page.locator("pre").filter(has_text="first-file.pdf")).to_have_count(0)
            await page.get_by_role("button", name="Older messages", exact=True).click()
            await expect(
                page.locator(
                    '[data-transcript-part="m1.1"][data-transcript-role="user"] pre'
                ).filter(has_text="mixed-file.pdf")
            ).to_be_visible()

    asyncio.run(run())


@pytest.mark.parametrize("accepted", [False, True])
def test_detail_recovery_observation_loss_never_claims_decided(accepted):
    from playwright.async_api import expect
    from tests.server.test_pending_recovery_linkage import _app as recovery_app
    from tests.server.test_pending_recovery_linkage import _crash

    from cayu.sessions import InMemorySessionStore

    async def run():
        runtime, tool = recovery_app(InMemorySessionStore())
        await _crash(runtime)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            pending = await (await page.request.get(f"{base}/api/inbox?session_id=crashed")).json()
            posts = []

            async def lose_recovery(route):
                posts.append(route.request)
                headers = await route.request.all_headers()
                if "last-event-id" in headers:
                    if accepted:
                        await route.continue_()
                    else:
                        await route.fulfill(
                            status=422, json={"detail": "Observation cannot establish acceptance"}
                        )
                    return
                await page.route("**/api/inbox?**", lambda route: route.fulfill(json=pending))
                await page.route(
                    "**/api/sessions/*/events?**",
                    lambda route: route.fulfill(
                        status=422, json={"detail": "Injected observation loss"}
                    ),
                )
                if accepted:
                    response = await route.fetch()
                    assert response.ok, await response.text()
                error = {
                    "type": "stream.error",
                    "kind": "observer",
                    "code": "event_frame_too_large",
                    "error": "Injected observation loss",
                    "error_type": "ObserverFailure",
                    "retryable": False,
                    "session_id": "crashed",
                }
                await route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body=f"event: error\ndata: {json.dumps(error)}\n\n",
                )

            await page.route("**/api/user-input/recover", lose_recovery)
            await page.goto(f"{base}/cayu/sessions/crashed")
            await page.get_by_label("The person's answer", exact=True).fill("staging")
            await page.get_by_label("Verified result or failure reason", exact=True).fill(
                "Verified externally"
            )
            await page.get_by_role("button", name="Recover", exact=True).click()
            await expect(
                page.locator('[data-mutation-transport-phase="transport_failed"]')
            ).to_be_visible(timeout=25000)
            await expect(page.get_by_text("Decided", exact=True)).to_have_count(0)
            await expect(page.get_by_role("button", name="Recover", exact=True)).to_be_disabled()
            await expect(
                page.get_by_label("Verified result or failure reason", exact=True)
            ).to_have_value("Verified externally")
            await page.unroute("**/api/sessions/*/events?**")
            await page.unroute("**/api/inbox?**")
            async with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.endswith("/api/user-input/recover")
                )
            ):
                await page.get_by_role("button", name="Retry observation", exact=True).click()
            if accepted:
                await expect(page.get_by_role("button", name="Recover", exact=True)).to_have_count(
                    0
                )
                assert (await runtime.session_store.load("crashed")).status == "completed"
            else:
                await expect(
                    page.get_by_role("button", name="Recover", exact=True)
                ).to_be_disabled()
                await expect(page.get_by_text("Decided", exact=True)).to_have_count(0)
                assert (await runtime.session_store.load("crashed")).status != "completed"
            headers = [await request.all_headers() for request in posts]
            assert sum("last-event-id" not in header for header in headers) == 1
            assert tool.calls == 0

    asyncio.run(run())


def test_detail_definite_recovery_rejection_retains_form_for_corrected_attempt():
    from playwright.async_api import expect
    from tests.server.test_pending_recovery_linkage import _app as recovery_app
    from tests.server.test_pending_recovery_linkage import _crash

    from cayu.sessions import InMemorySessionStore

    async def run():
        runtime, tool = recovery_app(InMemorySessionStore())
        await _crash(runtime)
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            submissions = []

            async def reject_first(route):
                submissions.append(route.request.post_data_json)
                if len(submissions) == 1:
                    await route.fulfill(
                        status=422, json={"detail": "Verified result needs correction"}
                    )
                else:
                    await route.continue_()

            await page.route("**/api/user-input/recover", reject_first)
            await page.goto(f"{base}/cayu/sessions/crashed")
            await page.get_by_label("The person's answer", exact=True).fill("staging")
            result = page.get_by_label("Verified result or failure reason", exact=True)
            await result.fill("Initial evidence")
            await page.get_by_role("button", name="Recover", exact=True).click()
            await expect(
                page.locator('[data-mutation-transport-phase="request_failed"]')
            ).to_be_visible()
            await expect(page.get_by_text("Decided", exact=True)).to_have_count(0)
            await expect(result).to_have_value("Initial evidence")
            await expect(page.get_by_role("button", name="Recover", exact=True)).to_be_enabled()
            await result.fill("Corrected externally verified result")
            await page.get_by_role("button", name="Recover", exact=True).click()
            await expect(page.get_by_role("button", name="Recover", exact=True)).to_have_count(0)
            assert len(submissions) == 2
            assert submissions[1]["message"] == "Corrected externally verified result"
            assert (await runtime.session_store.load("crashed")).status == "completed"
            assert tool.calls == 0

    asyncio.run(run())


def test_detail_capability_example_preserves_current_shell_and_denial(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from cayu.server import DashboardConfig, ServerConfig, create_server
    from dashboard_behavior_live import (
        _dashboard_price_book,
        _exercise_capability_contract,
        _record_browser_failures,
        _require_no_browser_failures,
        _seed_app,
    )

    async def run():
        runtime, provider, _store, _tasks, directory = await _seed_app()
        try:

            def configured(_app):
                return create_server(
                    runtime,
                    config=ServerConfig.local_development(
                        dashboard=DashboardConfig(
                            directory=os.environ.get("CAYU_SESSION_SEARCH_DIRECTORY"),
                            runtime_config={
                                "priceBook": _dashboard_price_book().model_dump(mode="json")
                            },
                        ),
                    ),
                )

            async with search_browser(runtime=runtime, wrapper=configured) as (page, base, _, _, _):
                failures = {
                    key: []
                    for key in ("console_errors", "page_errors", "request_failures", "api_errors")
                }
                queue_denials, queue_console = [], []
                _record_browser_failures(
                    page, failures, [], set(), [], [], [], [], [], [], queue_denials, queue_console
                )
                await _exercise_capability_contract(page, base)
                _require_no_browser_failures(failures)
                assert queue_denials
                assert len(queue_denials) == len(queue_console)
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["workflow-reset", "event-clear", "workflow-namespace"])
def test_detail_trace_search_owners_are_independent(monkeypatch, mode):
    from pathlib import Path
    from urllib.parse import parse_qs, urlsplit

    from playwright.async_api import expect

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import SESSION_ID, _seed_app

    async def run():
        runtime, provider, _store, _tasks, directory = await _seed_app()
        try:
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                if mode == "workflow-reset":
                    await page.goto(f"{base}/cayu/sessions/{SESSION_ID}/workflow?status=unknown")
                    await page.get_by_role("button", name="Reset Workflow view", exact=True).click()
                    await expect(
                        page.get_by_role("button", name="Reset Workflow view", exact=True)
                    ).to_have_count(0)
                    assert "invalid" not in parse_qs(urlsplit(page.url).query)
                elif mode == "event-clear":
                    await page.goto(
                        f"{base}/cayu/sessions/{SESSION_ID}?view=trace&log=1&event_type=tool_call_started&transcript_role=user"
                    )
                    await page.get_by_role("button", name="Clear filters", exact=True).click()
                    await expect(page.get_by_label("Event type", exact=True)).to_have_value("")
                    assert "event_type" not in parse_qs(urlsplit(page.url).query)
                    assert parse_qs(urlsplit(page.url).query)["transcript_role"] == ["user"]
                else:
                    await page.goto(
                        f"{base}/cayu/sessions/{SESSION_ID}/workflow?agent_name=child-specialist&environment_name=child-env"
                    )
                    await expect(page).to_have_url(__import__("re").compile(r"view=trace"))
                    query = parse_qs(urlsplit(page.url).query)
                    assert query.get("workflow_agent_name") == ["child-specialist"]
                    assert query.get("workflow_environment_name") == ["child-env"]
                    assert "agent_name" not in query and "environment_name" not in query
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("selection", ["root", "tool", "event"])
def test_detail_hidden_thinking_keeps_native_tool_trace_positions(selection):
    from playwright.async_api import expect
    from tests.core.test_tool_completion import FinalTool, app_for
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    async def run():
        runtime = app_for(
            InMemorySessionStore(),
            _TwoCallProvider(
                [
                    [
                        ModelStreamEvent.thinking("private presentation thought"),
                        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c1"),
                        ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="c2"),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        _ = [
            event
            async for event in runtime.run(
                RunRequest(
                    session_id="hidden-thought",
                    agent_name="support",
                    messages=[Message.text("user", "Help")],
                )
            )
        ]
        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            response = await page.request.get(
                f"{base}/api/sessions/hidden-thought/trace?limit=1000"
            )
            tools = [span for span in (await response.json())["spans"] if span["kind"] == "tool"]
            target = tools[1]
            position = target["transcript_links"][0]
            assert position["part_index"] == 2
            await page.goto(f"{base}/cayu/sessions/hidden-thought?include_thinking=false")
            await page.get_by_role("button", name="Used ask_customer 2 times", exact=False).click()
            row = page.locator(
                f"#transcript-tool-{position['message_index']}-{position['part_index']}"
            )
            await expect(row).to_be_visible()
            await row.get_by_role("button").first.click()
            await expect(
                page.get_by_text("private presentation thought", exact=True)
            ).to_have_count(0)
            await row.get_by_role("button", name="View in trace", exact=True).click()
            await expect(page.get_by_role("complementary", name="Span detail")).to_be_visible()
            await (
                page.get_by_role("complementary", name="Span detail")
                .get_by_role("button", name="Show in conversation", exact=True)
                .click()
            )
            await expect(row.get_by_role("button").first).to_have_attribute("aria-expanded", "true")
            await row.get_by_role("button", name="View in trace", exact=True).click()
            tree = page.get_by_role("tree", name="Session trace")
            if selection == "event":
                await page.get_by_role("tab", name="Event log", exact=True).click()
                await page.get_by_label("Event ID", exact=True).fill(tools[0]["event_ids"][0])
                await page.get_by_role("button", name="Apply filters", exact=True).click()
                event_detail = page.get_by_role("complementary", name="Event detail")
                await event_detail.get_by_role(
                    "button", name="tool call ask_customer", exact=True
                ).click()
                selected = tree.get_by_role("treeitem").filter(has_text="ask_customer").first
            elif selection == "tool":
                selected = tree.get_by_role("treeitem").filter(has_text="ask_customer").first
                await selected.click()
            else:
                selected = tree.get_by_role("treeitem").first
                await selected.click()
            await expect(selected).to_have_attribute("aria-selected", "true")

    asyncio.run(run())


def test_detail_returning_to_bottom_reconciles_deferred_terminal_answer():
    from playwright.async_api import expect
    from tests.core.test_tool_completion import FinalTool, app_for
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

    from cayu.providers import ModelStreamEvent
    from cayu.sessions import InMemorySessionStore

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        class Provider(_TwoCallProvider):
            async def stream(self, request):
                entered.set()
                await release.wait()
                async for event in super().stream(request):
                    yield event

        runtime = app_for(
            InMemorySessionStore(),
            Provider(
                [
                    [
                        ModelStreamEvent.text_delta("Deferred final answer"),
                        ModelStreamEvent.completed(),
                    ]
                ]
            ),
            FinalTool(),
        )

        async def execute():
            return [
                event
                async for event in runtime.run(
                    RunRequest(
                        session_id="deferred-tail",
                        agent_name="support",
                        messages=[
                            Message.text("user", f"history-{i} " + "long message " * 50)
                            for i in range(35)
                        ],
                    )
                )
            ]

        async with search_browser(runtime=runtime) as (page, base, _, _, _):
            running = asyncio.create_task(execute())
            try:
                await asyncio.wait_for(entered.wait(), 10)
                await page.goto(f"{base}/cayu/sessions/deferred-tail")
                await expect(page.get_by_text("history-34", exact=False)).to_be_visible()
                scroller = page.locator("div.overflow-y-auto").filter(
                    has=page.get_by_label("Message role", exact=True)
                )
                await expect(
                    page.get_by_role("button", name="Latest messages", exact=True)
                ).to_be_disabled()
                await scroller.evaluate(
                    "node => { node.scrollTop = 0; node.dispatchEvent(new Event('scroll', { bubbles: true })); }"
                )
                await expect(
                    page.get_by_role("button", name="Latest messages", exact=True)
                ).to_be_enabled()
                release.set()
                await asyncio.wait_for(running, 15)
                await expect(
                    page.get_by_text("New activity is available.", exact=True)
                ).to_be_visible()
                await expect(page.get_by_text("Deferred final answer", exact=True)).to_have_count(0)
                await scroller.evaluate(
                    "node => { node.scrollTop = node.scrollHeight; node.dispatchEvent(new Event('scroll', { bubbles: true })); }"
                )
                await expect(page.get_by_text("Deferred final answer", exact=True)).to_be_visible()
                await expect(
                    page.get_by_text("New activity is available.", exact=True)
                ).to_have_count(0)
            finally:
                release.set()
                await asyncio.wait_for(running, 15)

    asyncio.run(run())


def test_detail_legacy_event_link_opens_exact_evidence_and_retains_explicit_tabs(monkeypatch):
    from pathlib import Path

    from playwright.async_api import expect

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    from dashboard_behavior_live import SESSION_ID, _seed_app

    async def run():
        runtime, provider, _store, _tasks, directory = await _seed_app()
        try:
            async with search_browser(runtime=runtime) as (page, base, _, _, _):
                response = await page.request.get(
                    f"{base}/api/sessions/{SESSION_ID}/events?event_type=model.completed&limit=100"
                )
                event = (await response.json())["events"][0]
                await page.goto(f"{base}/cayu/sessions/{SESSION_ID}?event_id={event['id']}")
                await expect(
                    page.get_by_role("tab", name="Event log", exact=True)
                ).to_have_attribute("aria-selected", "true")
                detail = page.get_by_role("complementary", name="Event detail")
                await expect(detail).to_contain_text("model.completed")
                await page.get_by_role("tab", name="Waterfall", exact=True).click()
                await expect(
                    page.get_by_role("tab", name="Waterfall", exact=True)
                ).to_have_attribute("aria-selected", "true")
                await page.reload()
                await expect(
                    page.get_by_role("tab", name="Waterfall", exact=True)
                ).to_have_attribute("aria-selected", "true")
                await page.get_by_role("tab", name="Conversation", exact=True).click()
                await page.reload()
                await expect(
                    page.get_by_role("tab", name="Conversation", exact=True)
                ).to_have_attribute("aria-selected", "true")
        finally:
            provider.release_direct()
            directory.cleanup()

    asyncio.run(run())
