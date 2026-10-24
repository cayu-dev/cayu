"""Focused real-server Tasks page acceptance (explicit Chromium opt-in)."""

import asyncio
import os

import pytest

pytest.importorskip("playwright.async_api")
from playwright.async_api import expect
from tests.server.test_dashboard_home_inbox import _home_browser
from tests.server.test_server_inbox import _app

from cayu.tasks import InMemoryTaskStore, TaskCreate

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_DASHBOARD_TASKS_LIVE") != "1", reason="Opt-in Tasks browser acceptance."
)


def test_tasks_server_counts_deep_link_and_detail():
    async def scenario():
        store = InMemoryTaskStore()
        for index in range(205):
            await store.create_task(
                TaskCreate(
                    task_id=f"work-{index:03}",
                    type="report",
                    title=f"Report {index}",
                    assigned_agent_name="assistant",
                )
            )
        await store.create_task(TaskCreate(task_id="internal", type="cayu.housekeeping"))
        await store.block_task("work-000", reason="Awaiting approval")
        async with _home_browser(seeded_runtime=_app(task_store=store), nested=True) as (
            page,
            base,
            _requests,
            _responses,
        ):
            await page.goto(
                base + "/cayu/tasks?assigned_agent_name=assistant", wait_until="networkidle"
            )
            await expect(page.get_by_role("button", name="Waiting 204", exact=True)).to_be_visible()
            await expect(page.get_by_role("button", name="Held 1", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Held 1", exact=True).click()
            await page.get_by_role("link", name="Report 0", exact=True).click()
            await expect(page).to_have_url(__import__("re").compile(r"/tasks/work-000$"))
            await expect(page.get_by_text("Blocked: Awaiting approval", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Resume", exact=True).click()
            await expect(page.get_by_text("Queued", exact=True)).to_be_visible()
            await page.get_by_text("Debug", exact=True).click()
            await expect(page.get_by_text("Raw task", exact=True)).to_be_visible()
            assert any("include=counts" in response.url for response in _responses)

    asyncio.run(scenario())


def test_member_task_detail_has_no_mutation_controls():
    async def scenario():
        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(task_id="member-task", type="report", title="Member report")
        )
        await store.pause_task("member-task")
        async with _home_browser(
            role="member", mobile=True, seeded_runtime=_app(task_store=store)
        ) as (page, base, _requests, _responses):
            await page.goto(base + "/cayu/tasks/member-task", wait_until="networkidle")
            await expect(
                page.get_by_role("heading", name="Member report", exact=True)
            ).to_be_visible()
            for label in ["Resume", "Run again", "Cancel task", "Create a task", "Pause", "Block"]:
                await expect(page.get_by_role("button", name=label, exact=True)).to_have_count(0)
            assert not [
                response
                for response in _responses
                if response.request.method == "POST" and "/tasks" in response.url
            ]

    asyncio.run(scenario())


def test_declared_task_creation_recovers_lost_response_and_replays_exact_source():
    async def scenario():
        from cayu.tasks import TaskTypeSpec

        store = InMemoryTaskStore()
        runtime = _app(task_store=store)
        runtime.register_task_type(
            TaskTypeSpec(
                name="report",
                input_schema={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            )
        )
        async with _home_browser(seeded_runtime=runtime) as (page, base, _requests, _responses):
            submissions = []
            committed, release_response = asyncio.Event(), asyncio.Event()

            async def lose_first_response(route, request):
                if request.method != "POST":
                    return await route.continue_()
                submissions.append(request.post_data_json)
                response = await route.fetch()
                if len(submissions) == 1:
                    assert response.status == 200
                    committed.set()
                    await release_response.wait()
                    return await route.abort("connectionfailed")
                await route.fulfill(response=response)

            await page.route("**/api/tasks", lose_first_response)
            await page.goto(base + "/cayu/tasks", wait_until="networkidle")
            await page.get_by_role("button", name="Create a task", exact=True).click()
            await page.get_by_label("topic", exact=True).fill("Quarterly report")
            await page.get_by_role("button", name="Create task", exact=True).click()
            await asyncio.wait_for(committed.wait(), timeout=5)
            await page.get_by_role("button", name="Close", exact=True).click()
            release_response.set()
            await page.get_by_role("button", name="Create a task", exact=True).click()
            await expect(
                page.get_by_text("The creation outcome is unknown.", exact=False)
            ).to_be_visible()
            await page.get_by_role("button", name="Retry task creation", exact=True).click()
            await page.get_by_role("button", name="Open created task", exact=True).click()
            await expect(page.get_by_role("heading", name="report", exact=True)).to_be_visible()
            assert submissions[0] == submissions[1]
            created = await store.list_tasks(
                __import__("cayu.tasks", fromlist=["TaskQuery"]).TaskQuery()
            )
            assert len(created) == 1
            first_id = created[0].id
            replays = []

            async def lose_replay_response(route, request):
                if request.method != "POST":
                    return await route.continue_()
                replays.append(request.post_data_json)
                response = await route.fetch()
                if len(replays) == 1:
                    assert response.status == 200
                    return await route.abort("connectionfailed")
                await route.fulfill(response=response)

            await page.route("**/api/tasks/*/replay", lose_replay_response)
            await page.get_by_role("button", name="Run again", exact=True).click()
            await expect(
                page.get_by_text("The replay outcome is unknown.", exact=False)
            ).to_be_visible()
            await page.get_by_role("link", name="Tasks", exact=True).last.click()
            await page.locator(f'a[href$="/tasks/{first_id}"]').click()
            await page.get_by_role("button", name="Retry same replay", exact=True).click()
            await expect(page).not_to_have_url(__import__("re").compile(first_id + "$"))
            assert replays[0] == replays[1]
            tasks = await store.list_tasks(
                __import__("cayu.tasks", fromlist=["TaskQuery"]).TaskQuery()
            )
            assert len(tasks) == 2
            assert all(task.input == {"topic": "Quarterly report"} for task in tasks)
            await page.get_by_role("button", name="Cancel task", exact=True).click()
            await page.get_by_role("button", name="Confirm cancel task", exact=True).click()
            await expect(page.get_by_text("Cancelled", exact=True).first).to_be_visible()

    asyncio.run(scenario())


def test_task_graph_holds_schedule_history_and_debug():
    async def scenario():
        from datetime import UTC, datetime, timedelta

        from cayu.tasks import TaskGraphCreate, TaskGraphNode, TaskSchedulePolicy

        store = InMemoryTaskStore()
        await store.create_task_graph(
            TaskGraphCreate(
                graph_id="graph",
                nodes=(
                    TaskGraphNode(
                        task=TaskCreate(task_id="root", type="report", title="Root report")
                    ),
                    TaskGraphNode(
                        task=TaskCreate(task_id="child", type="report", title="Child report"),
                        prerequisite_task_ids=("root",),
                    ),
                ),
            )
        )
        await store.create_task(
            TaskCreate(
                task_id="schedule",
                type="report",
                title="Scheduled report",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _requests,
            _responses,
        ):
            await page.goto(base + "/cayu/tasks/root", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="Needed by", exact=True)).to_be_visible()
            await page.get_by_role("button", name="More actions", exact=True).click()
            await page.get_by_role("button", name="Block", exact=True).click()
            await page.get_by_label("Reason", exact=True).fill("Waiting on accounting")
            await page.get_by_role("button", name="Block task", exact=True).click()
            await expect(
                page.get_by_text("Blocked: Waiting on accounting", exact=True)
            ).to_be_visible()
            await page.get_by_role("button", name="Resume", exact=True).click()
            await expect(page.get_by_text("Queued", exact=True)).to_be_visible()
            await page.get_by_role("link", name=__import__("re").compile("Child report")).click()
            await expect(
                page.get_by_role("heading", name=__import__("re").compile("Waiting on"))
            ).to_be_visible()
            await page.goto(base + "/cayu/tasks/schedule", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="Schedule", exact=True)).to_be_visible()
            await expect(page.get_by_text("Scheduled", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Cancel task", exact=True).click()
            await page.get_by_role("button", name="Confirm cancel task", exact=True).click()
            await expect(page.get_by_text("Cancelled", exact=True).first).to_be_visible()
            await page.get_by_text("Debug", exact=True).click()
            await expect(page.get_by_text("Concurrency key", exact=True)).to_be_visible()

    asyncio.run(scenario())


def test_workflow_task_reference_journey_and_old_task_url():
    async def scenario():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
        from dashboard_behavior_live import (
            WORKFLOW_BLOCKED_TASK_ID,
            WORKFLOW_PARENT_TASK_ID,
            _exercise_workflow_task_pages,
        )

        store = InMemoryTaskStore()
        for task_id, title in [
            (WORKFLOW_PARENT_TASK_ID, "Coordinate workflow"),
            (WORKFLOW_BLOCKED_TASK_ID, "Linked Workflow step 0"),
        ]:
            await store.create_task(TaskCreate(task_id=task_id, type="workflow", title=title))
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            responses,
        ):
            await page.goto(
                base
                + "/cayu/tasks?q="
                + WORKFLOW_PARENT_TASK_ID
                + "&task_id="
                + WORKFLOW_PARENT_TASK_ID,
                wait_until="networkidle",
            )
            observed = [
                (
                    r.request.method,
                    __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(r.url).path,
                    None,
                )
                for r in responses
            ]
            await _exercise_workflow_task_pages(page, observed)

    asyncio.run(scenario())


def test_task_list_loading_cursor_refresh_deadline_and_empty_catalog():
    async def scenario():
        store = InMemoryTaskStore()
        for index in range(65):
            await store.create_task(
                TaskCreate(task_id=f"page-{index:03}", type="report", title=f"Page {index}")
            )
        await store.create_task(
            TaskCreate(task_id="internal", type="cayu.housekeeping", title="Housekeeping")
        )
        await store.create_task(
            TaskCreate(task_id="deadline", type="report", title="Deadline task")
        )
        await store.fail_task(
            "deadline", {"error": "ExecutionDeadlineExceeded", "scope": "task_handler"}
        )
        async with _home_browser(seeded_runtime=_app(task_store=store), mobile=True) as (
            page,
            base,
            _,
            responses,
        ):
            release = asyncio.Event()

            async def delay_index(route, request):
                if "include=counts" in request.url:
                    await release.wait()
                await route.continue_()

            await page.route("**/api/tasks?*", delay_index)
            await page.goto(base + "/cayu/tasks", wait_until="domcontentloaded")
            await expect(page.get_by_text("Loading tasks…", exact=True)).to_be_visible()
            await expect(
                page.get_by_text("No tasks match these filters", exact=True)
            ).to_have_count(0)
            release.set()
            await expect(page.get_by_role("button", name="All 66", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Load more tasks", exact=True).click()
            await expect(
                page.get_by_role("link", name=__import__("re").compile(r"^Page \d+$"))
            ).to_have_count(65)
            # Real interval delivery: a refreshed cursor chain must not be classified as a cycle.
            await page.wait_for_timeout(5500)
            await expect(
                page.get_by_role("link", name=__import__("re").compile(r"^Page \d+$"))
            ).to_have_count(65)
            await expect(
                page.get_by_text("Task handler ran out of time", exact=True)
            ).to_be_visible()
            await page.get_by_label("Show Cayu's own tasks", exact=True).check()
            await expect(page.get_by_role("button", name="All 67", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Create a task", exact=True).click()
            await expect(
                page.get_by_text("This agent does not declare task types.", exact=False)
            ).to_be_visible()
            await expect(
                page.get_by_role("button", name="Create task", exact=True)
            ).to_be_disabled()
            assert not [
                r for r in responses if r.request.method == "POST" and r.url.endswith("/api/tasks")
            ]

    asyncio.run(scenario())


def test_member_schedule_history_and_task_read_failure():
    async def scenario():
        from datetime import UTC, datetime, timedelta

        from cayu.tasks import TaskSchedulePolicy

        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="member-schedule",
                type="report",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        async with _home_browser(role="member", seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            responses,
        ):
            await page.goto(base + "/cayu/tasks/member-schedule", wait_until="networkidle")
            await expect(page.get_by_role("heading", name="Schedule", exact=True)).to_be_visible()
            await expect(page.get_by_text("Scheduled", exact=True)).to_be_visible()
            assert any(
                r.url.endswith("/schedule/events?limit=100") and r.status == 200 for r in responses
            )
            for label in ["Resume", "Run again", "Cancel task", "Pause", "Block"]:
                await expect(page.get_by_role("button", name=label, exact=True)).to_have_count(0)
            await page.goto(base + "/cayu/tasks/missing-task", wait_until="networkidle")
            await expect(page.get_by_text("Task not found", exact=False)).to_be_visible()
            await expect(page.get_by_role("heading", name="Schedule", exact=True)).to_have_count(0)

    asyncio.run(scenario())


def test_partial_prerequisites_never_report_unloaded_records_as_open():
    async def scenario():
        from cayu.tasks import TaskGraphCreate, TaskGraphNode

        store = InMemoryTaskStore()
        ids = [f"pre-{i:02}" for i in range(60)]
        await store.create_task_graph(
            TaskGraphCreate(
                graph_id="many-prerequisites",
                nodes=tuple(
                    [
                        TaskGraphNode(task=TaskCreate(task_id=identity, type="report"))
                        for identity in ids
                    ]
                    + [
                        TaskGraphNode(
                            task=TaskCreate(
                                task_id="dependent", type="report", title="Dependent report"
                            ),
                            prerequisite_task_ids=tuple(ids),
                        )
                    ]
                ),
            )
        )
        for identity in ids[:-1]:
            await store.complete_task(identity, {})
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            _responses,
        ):
            await page.goto(base + "/cayu/tasks?group=waiting", wait_until="networkidle")
            row = page.get_by_role("row").filter(has_text="Dependent report")
            await expect(row.get_by_text("Waiting on prerequisites", exact=True)).to_be_visible()
            await page.get_by_role("link", name="Dependent report", exact=True).click()
            await expect(page.get_by_text("Waiting on prerequisites", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Load more prerequisites", exact=True).click()
            await expect(page.get_by_text("Waiting on 1 task", exact=True)).to_be_visible()
            # The dependency ID list is unchanged; a worker's completion still refreshes linked state.
            await store.complete_task(ids[-1], {})
            await expect(page.get_by_text("Queued", exact=True)).to_be_visible(timeout=12000)
            last = page.locator('a[href$="/tasks/pre-59"]')
            await expect(last.get_by_text("Completed", exact=True)).to_be_visible(timeout=12000)

    asyncio.run(scenario())


def test_task_relations_and_retry_observe_native_worker_progress():
    async def scenario():
        from cayu.tasks import TaskQuery, TaskRetryPolicy
        from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest

        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="parent", type="parent", title="Parent"))
        await store.create_task(
            TaskCreate(
                task_id="retry-first",
                type="retry",
                retry_policy=TaskRetryPolicy(max_attempts=2, initial_backoff_seconds=0),
            )
        )
        claimed = await store.claim_task("browser-worker", TaskQuery(type="retry"))
        assert claimed and claimed.retry_series and claimed.lease_expires_at
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            _responses,
        ):
            await page.goto(base + "/cayu/tasks/parent", wait_until="networkidle")
            for index in range(51):
                await store.create_task(
                    TaskCreate(
                        task_id=f"child-{index:02}",
                        type="child",
                        parent_task_id="parent",
                        title=f"Child {index}",
                    )
                )
            await expect(
                page.get_by_role("button", name="Load more subtasks", exact=True)
            ).to_be_visible(timeout=12000)
            await page.get_by_role("button", name="Load more subtasks", exact=True).click()
            await expect(
                page.get_by_role("heading", name="Subtasks (51 loaded)", exact=True)
            ).to_be_visible()
            await store.complete_task("parent", {})
            await store.create_task(
                TaskCreate(
                    task_id="child-new", type="child", parent_task_id="parent", title="Final child"
                )
            )
            await expect(
                page.get_by_role("heading", name="Subtasks (52 loaded)", exact=True)
            ).to_be_visible(timeout=12000)
            await page.goto(base + "/cayu/tasks/retry-first", wait_until="networkidle")
            await expect(
                page.get_by_role("heading", name="Attempts (1 loaded)", exact=True)
            ).to_be_visible()
            receipt = await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="browser-worker",
                    lease_expires_at=claimed.lease_expires_at,
                    idempotency_key="browser-retry",
                    causal_budget_id=claimed.retry_series.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
                    error={"code": "temporary"},
                )
            )
            assert receipt.successor
            await expect(
                page.get_by_role("heading", name="Attempts (2 loaded)", exact=True)
            ).to_be_visible(timeout=12000)
            from urllib.parse import quote

            await expect(page.get_by_role("link", name="Attempt 2", exact=True)).to_have_attribute(
                "href", "/cayu/tasks/" + quote(receipt.successor.id, safe="")
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("uncertain", [False, True])
def test_cancel_rejection_distinguishes_initial_from_unknown_operation(uncertain):
    async def scenario():
        from datetime import UTC, datetime, timedelta

        from cayu.tasks import TaskSchedulePolicy
        from cayu.tasks.scheduling import TaskRescheduleRequest

        store = InMemoryTaskStore()
        await store.create_task(
            TaskCreate(
                task_id="cancel-fence",
                type="report",
                available_at=datetime.now(UTC) + timedelta(hours=1),
                schedule_policy=TaskSchedulePolicy(),
            )
        )
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            _responses,
        ):
            bodies = []

            async def intercept(route, request):
                bodies.append(request.post_data_json)
                if uncertain:
                    if len(bodies) == 1:
                        response = await route.fetch()
                        assert response.status == 200
                        return await route.abort("connectionfailed")
                    if len(bodies) == 2:
                        return await route.fulfill(
                            status=403,
                            json={"detail": "Recovery permission temporarily unavailable"},
                        )
                elif len(bodies) == 1:
                    await store.reschedule_task(
                        TaskRescheduleRequest(
                            task_id="cancel-fence",
                            operation_id="move",
                            expected_revision=1,
                            available_at=datetime.now(UTC) + timedelta(hours=2),
                        )
                    )
                await route.continue_()

            await page.route("**/api/tasks/schedule/cancel", intercept)
            await page.goto(base + "/cayu/tasks/cancel-fence", wait_until="networkidle")
            await page.get_by_role("button", name="Cancel task", exact=True).click()
            await page.get_by_role("button", name="Confirm cancel task", exact=True).click()
            if uncertain:
                await page.get_by_role("button", name="Retry same operation", exact=True).click()
                await expect(
                    page.get_by_text("Recovery permission temporarily unavailable", exact=False)
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Confirm cancel task", exact=True)
                ).to_be_disabled()
                await page.get_by_role("button", name="Retry same operation", exact=True).click()
                assert bodies[0] == bodies[1] == bodies[2]
            else:
                await expect(
                    page.get_by_text("Task schedule authority conflicts", exact=False)
                ).to_be_visible()
                await page.get_by_role("button", name="Confirm cancel task", exact=True).click()
                await expect(page.get_by_text("Cancelled", exact=True).first).to_be_visible()
                assert [body["expected_revision"] for body in bodies] == [1, 2]
                assert bodies[0]["operation_id"] != bodies[1]["operation_id"]
            assert (await store.load_task("cancel-fence")).status == "cancelled"

    asyncio.run(scenario())


@pytest.mark.parametrize("reject_recovery", [False, True])
def test_unknown_creation_does_not_block_specialist_and_recovery_rejection_stays_fenced(
    reject_recovery,
):
    async def scenario():
        from cayu.tasks import TaskTypeSpec

        store = InMemoryTaskStore()
        runtime = _app(task_store=store)
        runtime.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            bodies = []

            async def intercept(route, request):
                if request.method != "POST":
                    return await route.continue_()
                bodies.append(request.post_data_json)
                if len(bodies) == 1:
                    response = await route.fetch()
                    assert response.status == 200
                    return await route.abort("connectionfailed")
                if reject_recovery and len(bodies) == 2:
                    return await route.fulfill(
                        status=409, json={"detail": "Retained declaration changed"}
                    )
                await route.continue_()

            await page.route("**/api/tasks", intercept)
            await page.goto(base + "/cayu/tasks", wait_until="networkidle")
            await page.get_by_role("button", name="Create a task", exact=True).click()
            await page.get_by_role("button", name="Create task", exact=True).click()
            await expect(
                page.get_by_role("button", name="Retry task creation", exact=True)
            ).to_be_visible()
            if reject_recovery:
                await page.get_by_role("button", name="Retry task creation", exact=True).click()
                await expect(
                    page.get_by_text("Retained declaration changed", exact=False)
                ).to_be_visible()
                await expect(
                    page.get_by_role("button", name="Retry task creation", exact=True)
                ).to_be_visible()
            await page.get_by_role("button", name="Close", exact=True).click()
            await page.get_by_role("link", name="Sessions", exact=True).click()
            await page.get_by_role("button", name="Start a session", exact=True).click()
            await expect(page.get_by_label("Prompt", exact=True)).to_be_visible()
            await page.get_by_role("tab", name="Create a task", exact=True).click()
            await expect(
                page.get_by_role("button", name="Retry task creation", exact=True)
            ).to_be_visible()
            await page.get_by_role("button", name="Retry task creation", exact=True).click()
            await expect(page.get_by_text("Task created.", exact=True)).to_be_visible()
            await page.get_by_role("button", name="Dismiss task result", exact=True).click()
            await expect(page.get_by_role("button", name="Create task", exact=True)).to_be_enabled()
            assert all(body == bodies[0] for body in bodies)

    asyncio.run(scenario())


@pytest.mark.parametrize("held", [False, True])
def test_held_task_and_replay_authority_observe_external_changes(held):
    async def scenario():
        from tests.core.task_invocation_fixtures import stored_session_invocation

        from cayu import Message
        from cayu.sessions.records import SessionIdentity
        from cayu.sessions.requests import RunRequest, run_request_with_task_invocation
        from cayu.tasks import TaskTypeSpec

        store = InMemoryTaskStore()
        runtime = _app(task_store=store)
        runtime.register_task_type(TaskTypeSpec(name="report", input_schema={"type": "object"}))
        await store.create_task(TaskCreate(task_id="external", type="report"))
        if held:
            await store.pause_task("external")
        async with _home_browser(seeded_runtime=runtime) as (page, base, _, _responses):
            await page.clock.install()
            await page.goto(base + "/cayu/tasks/external", wait_until="networkidle")
            if held:
                await expect(page.get_by_text("Paused", exact=True)).to_be_visible()
                await store.resume_task("external")
                await page.clock.fast_forward(31000)
                await expect(page.get_by_text("Queued", exact=True)).to_be_visible()
            snapshot = await store.load_invocation_snapshot("external")
            run = run_request_with_task_invocation(
                RunRequest(
                    agent_name="worker",
                    session_id="attached",
                    task_id="external",
                    messages=[Message.text("user", "report")],
                ),
                snapshot,
            )
            await runtime.session_store.create(
                run, identity=SessionIdentity(provider_name="fake", model="fake")
            )
            binding = await stored_session_invocation(runtime.session_store, "attached")
            await store.start_task("external", session_id="attached", session_invocation=binding)
            sent = []

            async def capture(route, request):
                if request.method == "POST":
                    sent.append(request.post_data_json)
                await route.continue_()

            await page.route("**/api/tasks/external/replay", capture)
            await page.get_by_role("button", name="Run again", exact=True).click()
            await expect(page).not_to_have_url(__import__("re").compile(r"/tasks/external$"))
            assert sent[0]["source"]["session_id"] == "attached"

    asyncio.run(scenario())


def test_unavailable_related_task_preserves_readable_sibling():
    async def scenario():
        from cayu.tasks import TaskGraphCreate, TaskGraphNode

        store = InMemoryTaskStore()
        await store.create_task_graph(
            TaskGraphCreate(
                graph_id="partial",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="missing-related", type="report")),
                    TaskGraphNode(
                        task=TaskCreate(
                            task_id="readable-related", type="report", title="Readable prerequisite"
                        )
                    ),
                    TaskGraphNode(
                        task=TaskCreate(task_id="partial-root", type="report"),
                        prerequisite_task_ids=("missing-related", "readable-related"),
                    ),
                ),
            )
        )
        async with _home_browser(seeded_runtime=_app(task_store=store)) as (
            page,
            base,
            _,
            _responses,
        ):
            await page.route(
                "**/api/tasks/missing-related",
                lambda route: route.fulfill(status=404, json={"detail": "Task was pruned"}),
            )
            await page.goto(base + "/cayu/tasks/partial-root", wait_until="networkidle")
            await expect(
                page.get_by_role("link", name=__import__("re").compile("Readable prerequisite"))
            ).to_be_visible()
            await expect(
                page.get_by_text("Related task missing-related is unavailable.", exact=True)
            ).to_be_visible()
            await expect(page.get_by_text("Waiting on prerequisites", exact=True)).to_be_visible()

    asyncio.run(scenario())
