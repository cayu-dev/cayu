"""Same-tab recovery through real dashboard entrances and durable server receipts."""

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
    reason="Opt-in packaged Chromium session recovery acceptance.",
)


@pytest.fixture(autouse=True)
def independent_embedded_server_shutdown(monkeypatch):
    from sse_starlette.sse import AppStatus

    # Each case starts a new Uvicorn server after the preceding one has stopped.
    # Its graceful-drain watcher leaves this process-global shutdown flag set.
    monkeypatch.setattr(AppStatus, "should_exit", False)


async def attempt_departure_and_stay(page):
    from playwright.async_api import expect

    before = page.url
    await page.get_by_role("link", name="Sessions", exact=True).first.click()
    await expect(page.get_by_role("dialog")).to_be_visible()
    await expect(page.get_by_text("An action is still unconfirmed", exact=True)).to_be_visible()
    assert page.url == before
    await page.get_by_role("button", name="Stay and recover", exact=True).click()
    await expect(page.get_by_role("dialog")).to_have_count(0)


@pytest.mark.parametrize("detail_failure", [None, "recover", "leave"])
def test_task_replay_guards_departure_and_recovers_exact_receipt(detail_failure):
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
            receipts = []

            async def dispatch(route):
                posts.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.ok
                receipts.append(await response.json())
                if len(posts) == 1:
                    await route.abort("failed")
                else:
                    await route.fulfill(response=response)

            await page.route("**/api/tasks/source-task/replay", dispatch)
            await page.goto(f"{base}/cayu/sessions/task-session")
            await expect(page.get_by_role("tab", name="Conversation", exact=True)).to_be_visible()
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Run again", exact=True).click()
            await expect(
                page.get_by_text("Task creation is unconfirmed.", exact=False)
            ).to_be_visible()
            if detail_failure:

                async def fail_detail(route):
                    await route.fulfill(
                        status=503, json={"detail": "Injected session read failure"}
                    )

                await page.route("**/api/sessions/task-session", fail_detail)
                # Real browser reconnection refetches the now-stale detail;
                # the accepted replay's mounted owner remains unresolved.
                await page.context.set_offline(True)
                await page.wait_for_timeout(10_100)
                await page.context.set_offline(False)
                await expect(page.get_by_role("alert")).to_contain_text(
                    "Injected session read failure", timeout=15_000
                )
                # A refetch failure retains the cached Session subtree: its
                # private Evals owners must coexist with task-replay recovery.
                await expect(
                    page.get_by_role("tab", name="Conversation", exact=True)
                ).to_be_visible()
            await attempt_departure_and_stay(page)
            if detail_failure == "leave":
                await page.get_by_role("link", name="Sessions", exact=True).first.click()
                await expect(page.get_by_role("dialog")).to_contain_text(
                    "does not cancel work already accepted"
                )
                await page.get_by_role(
                    "button", name="Leave without local recovery", exact=True
                ).click()
                await expect(page).to_have_url(f"{base}/cayu/sessions")
                assert len(posts) == 1
                accepted_task = await tasks.load_task(receipts[0]["task_id"])
                assert accepted_task is not None
                assert accepted_task.input == {"topic": "original"}
                return
            if detail_failure == "recover":
                await page.unroute("**/api/sessions/task-session", fail_detail)
                await page.get_by_role("button", name="Retry session details", exact=True).click()
                await expect(
                    page.get_by_role("tab", name="Conversation", exact=True)
                ).to_be_visible()
            menu = page.get_by_role("button", name="More actions", exact=True)
            if await menu.get_attribute("aria-expanded") != "true":
                await menu.click()
            await page.get_by_role("button", name="Retry task replay", exact=True).click()
            # Session replay keeps its existing destination contract; the Tasks
            # router resolves that old query link to the canonical detail page.
            await expect(page).to_have_url(__import__("re").compile(r"/cayu/tasks/[^?]+$"))
            from urllib.parse import unquote, urlsplit

            new_id = unquote(urlsplit(page.url).path.rsplit("/", 1)[1])
            await expect(page.get_by_role("heading", name="report", exact=True)).to_be_visible()
            await expect(page.get_by_text("Debug", exact=True)).to_be_visible()
            assert new_id != "source-task"
            assert new_id == receipts[0]["task_id"]
            fresh = await tasks.load_task(new_id)
            assert fresh is not None
            assert fresh.input == {"topic": "original"}
            assert fresh.session_id is None
            assert len(posts) == 2
            assert posts[0] == posts[1]
            assert posts[0]["source"]["session_id"] == "task-session"
            assert posts[0]["source"]["session_instance_id"]

    asyncio.run(run())


def test_followup_trace_recovery_clears_only_submitted_draft():
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
            await page.get_by_role("tab", name="Trace", exact=True).click()
            await page.get_by_text("Runtime internals", exact=True).click()
            await page.get_by_role("button", name="Retry observation", exact=True).click()
            await expect(page.locator('[data-mutation-transport-phase="terminal"]')).to_be_visible(
                timeout=15_000
            )
            await page.get_by_role("tab", name="Conversation", exact=True).click()
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


def test_queue_acceptance_guards_departure_and_recovers_delivered_receipt():
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
                    )
                )
            )
        )
        try:
            await asyncio.wait_for(reached.wait(), 10)
            session = await runtime.session_store.load("queue")
            assert session is not None
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
                mutations = []
                page.on(
                    "request",
                    lambda request: (
                        mutations.append(request)
                        if request.method == "POST" and "/messages" in request.url
                        else None
                    ),
                )

                async def lose_enqueue(route):
                    if route.request.method != "POST":
                        await route.continue_()
                        return
                    response = await route.fetch()
                    assert response.ok
                    await route.fulfill(status=200, content_type="text/event-stream", body="")

                await page.route("**/api/sessions/queue/messages", lose_enqueue)
                await page.goto(f"{base}/cayu/sessions/queue")
                composer = page.locator("#session-composer")
                await expect(composer).to_be_enabled()
                await composer.fill("Please verify the new entry")
                await page.get_by_role("button", name="Send", exact=True).click()
                await expect(
                    page.get_by_role("button", name="Retry message acceptance", exact=True)
                ).to_be_visible()
                await expect(composer).to_have_value("Please verify the new entry")
                await expect(composer).to_be_disabled()
                await page.unroute("**/api/sessions/queue/messages", lose_enqueue)
                await attempt_departure_and_stay(page)
                release.set()
                await asyncio.wait_for(execution, 10)
                await page.get_by_role(
                    "button", name="Retry message acceptance", exact=True
                ).click()
                await expect(
                    page.get_by_role("button", name="Retry message acceptance", exact=True)
                ).to_have_count(0)
                await expect(composer).to_have_value("")
                response = await page.request.get(f"{base}/api/sessions/queue/messages")
                records = (await response.json())["records"]
                assert len(records) == 1 and records[0]["status"] == "delivered"
                assert len(mutations) == 2
                assert mutations[0].post_data_json == mutations[1].post_data_json
                await page.get_by_role("link", name="Sessions", exact=True).first.click()
                await expect(page).to_have_url(f"{base}/cayu/sessions")
        finally:
            release.set()
            with suppress(BaseException):
                await asyncio.wait_for(execution, 10)

    asyncio.run(run())


@pytest.mark.parametrize("entrance", ["home", "sessions", "tasks", "rerun"])
def test_run_observation_guard_preserves_exact_destination(entrance):
    from playwright.async_api import expect
    from tests.core.test_cloud_reporting import PRICE, _drain, _Provider, _respond

    from cayu import AgentSpec, CayuApp
    from cayu.tasks import InMemoryTaskStore, TaskTypeSpec

    async def run():
        runtime = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        runtime.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        runtime.register_agent(AgentSpec(name="writer", model=PRICE.model))
        runtime.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
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
            submissions = []
            requests = []
            page.on(
                "request",
                lambda request: (
                    requests.append(request) if request.url.endswith("/api/run") else None
                ),
            )

            async def lost_observer(route):
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
                    "session_id": route.request.post_data_json["session_id"],
                }
                await route.fulfill(
                    status=200,
                    content_type="text/event-stream",
                    body=f"event: error\ndata: {json.dumps(error)}\n\n",
                )

            await page.route("**/api/run", lost_observer)
            await page.goto(f"{base}/cayu/sessions/direct-source")
            if entrance == "rerun":
                await page.get_by_role("button", name="More actions", exact=True).click()
                await page.get_by_role("button", name="Run again", exact=True).click()
            else:
                await page.get_by_role("link", name=entrance.title(), exact=True).first.click()
                await page.get_by_role(
                    "button",
                    name="Create a task" if entrance == "tasks" else "Start a session",
                    exact=True,
                ).click()
                if entrance == "tasks":
                    await page.get_by_role("tab", name="Run a specialist", exact=True).click()
                await page.locator("#start-specialist").select_option("writer")
                await page.get_by_label("Prompt", exact=True).fill("Original direct prompt")
                await page.get_by_role("button", name="Start", exact=True).click()
            await expect(
                page.locator('[data-mutation-transport-phase="transport_failed"]')
            ).to_be_visible(timeout=25000)
            before = page.url
            if entrance != "rerun":
                await expect(
                    page.get_by_role(
                        "button",
                        name="Create a task" if entrance == "tasks" else "Start a session",
                        exact=True,
                    )
                ).to_be_disabled()
                await page.get_by_role("button", name="Close", exact=True).click()
                await expect(page.get_by_role("dialog")).to_be_visible()
                await page.keyboard.press("Escape")
                await expect(page.get_by_role("dialog")).to_have_count(0)
                await expect(
                    page.locator('[data-mutation-transport-phase="transport_failed"]')
                ).to_be_visible()
                # SPA Back is a departure too; no reload and no replacement owner.
                await page.evaluate("history.back()")
                await expect(page.get_by_role("dialog")).to_be_visible()
                await page.get_by_role("button", name="Stay and recover", exact=True).click()
                await expect(page).to_have_url(before)
            else:
                await attempt_departure_and_stay(page)
                await page.get_by_role("tab", name="Trace", exact=True).click()
                await page.get_by_text("Runtime internals", exact=True).click()
            await page.unroute("**/api/sessions/*/events?**")
            await page.unroute("**/api/run", lost_observer)
            await page.get_by_role("button", name="Retry observation", exact=True).click()
            destination = submissions[0]["session_id"]
            await expect(page).to_have_url(f"{base}/cayu/sessions/{destination}", timeout=15000)
            assert len(submissions) == 1
            assert sum("last-event-id" not in request.headers for request in requests) == 1
            response = await page.request.get(
                f"{base}/api/sessions/{destination}/transcript?role=user"
            )
            messages = (await response.json())["messages"]
            assert [part["text"] for message in messages for part in message["content"]] == [
                "Original direct prompt"
            ]

    asyncio.run(run())


@pytest.mark.parametrize("departure", ["close", "route"])
def test_explicit_departure_warns_and_ignores_late_start_result(departure):
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import expect

    async def run():
        runtime = _app()
        accepted, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        submissions = []
        async with search_browser(runtime=runtime) as (page, base, _, _, _):

            async def hold_response(route):
                if "last-event-id" in route.request.headers:
                    await route.continue_()
                    return
                submissions.append(route.request.post_data_json)
                response = await route.fetch()
                assert response.ok
                accepted.set()
                await release.wait()
                try:
                    with suppress(PlaywrightError):
                        await route.fulfill(response=response)
                finally:
                    finished.set()

            await page.route("**/api/run", hold_response)
            try:
                await page.goto(f"{base}/cayu/?start=1")
                await page.locator("#start-specialist").select_option("writer")
                await page.get_by_label("Prompt", exact=True).fill("Original direct prompt")
                await page.get_by_role("button", name="Start", exact=True).click()
                await asyncio.wait_for(accepted.wait(), 10)
                if departure == "close":
                    await page.get_by_role("button", name="Close", exact=True).click()
                else:
                    await page.get_by_role("link", name="Sessions", exact=True).first.click()
                dialog = page.get_by_role("dialog")
                await expect(dialog).to_be_visible()
                await expect(dialog).to_contain_text("does not cancel work already accepted")
                await page.get_by_role(
                    "button", name="Leave without local recovery", exact=True
                ).click()
                if departure == "route":
                    await expect(page).to_have_url(f"{base}/cayu/sessions")
                else:
                    await expect(page.get_by_label("Prompt", exact=True)).to_have_count(0)
                destination = page.url
                release.set()
                await asyncio.wait_for(finished.wait(), 10)
                # Allow the old response continuation and passive unmount cleanup to run.
                await page.wait_for_timeout(150)
                assert page.url == destination
                assert len(submissions) == 1
                response = await page.request.get(
                    f"{base}/api/sessions/{submissions[0]['session_id']}"
                )
                assert response.ok
            finally:
                release.set()
                if accepted.is_set():
                    await asyncio.wait_for(finished.wait(), 10)

    asyncio.run(run())
