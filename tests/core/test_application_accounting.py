"""Independent reporting composition and application lifetime preservation."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

import cayu
from cayu import _application_accounting as accounting
from cayu.applications import CayuApp
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.events import Event, EventType
from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


def price_book():
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="test",
                model="priced",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        )
    )


async def prepare_reports(store):
    suffix = uuid4().hex
    budget = "budget-" + suffix
    sessions = ["charged-" + suffix, "empty-" + suffix, "outside-" + suffix]
    for index, session_id in enumerate(sessions):
        await store.create(
            RunRequest(
                session_id=session_id,
                causal_budget_id=budget if index < 2 else "other-" + suffix,
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="test", model="priced"),
        )
    for session_id, tokens in ((sessions[0], 1000), (sessions[2], 9000)):
        await store.append_event(
            session_id,
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={
                    "usage_metrics": {
                        "provider_name": "test",
                        "model": "priced",
                        "input_tokens": tokens,
                        "output_tokens": 100,
                        "total_tokens": tokens + 100,
                    }
                },
            ),
        )
    return sessions, budget


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_reports_compose_without_application_and_keep_native_reads(
    backend, request, sqlite_resources, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        async with sqlite_resources as resources:
            if backend == "memory":
                store = InMemorySessionStore()
            elif backend == "sqlite":
                store = resources.own(SQLiteSessionStore(resources.path()))
            else:
                store = resources.own(PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE))
            sessions, budget = await prepare_reports(store)
            reads = []
            usage_read, cost_read = store.read_usage_accounting, store.read_cost_accounting

            async def usage(query, **options):
                reads.append(("usage", query.session_id, query.causal_budget_id))
                return await usage_read(query, **options)

            async def cost(query, pricing, **options):
                reads.append(("cost", query.session_id, query.causal_budget_id))
                return await cost_read(query, pricing, **options)

            async def forbidden(*args, **kwargs):
                raise AssertionError("Reports must use native accounting, not load event histories")

            monkeypatch.setattr(store, "read_usage_accounting", usage)
            monkeypatch.setattr(store, "read_cost_accounting", cost)
            monkeypatch.setattr(store, "load_events", forbidden)
            monkeypatch.setattr(store, "query_events", forbidden)

            async def resolve(value):
                assert value in {"public-" + sessions[0], "public-" + budget}
                return value.removeprefix("public-")

            def project(value):
                return "public-" + value

            def project_budget(value, *, session_ids):
                assert list(session_ids) == sessions[:2]
                return project(value)

            common = dict(session_store=store, project_session=project)
            session_options = dict(common, resolve_session=resolve)
            budget_options = dict(common, resolve_budget=resolve, project_budget=project_budget)
            snapshot = await accounting.read_session_usage_snapshot(
                project(sessions[0]), **session_options
            )
            assert snapshot.summary.session_id == project(sessions[0])
            assert snapshot.summary.usage.total_tokens == 1100
            assert snapshot.through_sequence > 0
            repeated = await accounting.read_session_usage_snapshot(
                project(sessions[0]), **session_options
            )
            assert repeated == snapshot  # Includes the generation/watermark used by HTTP ETags.
            session_cost = await accounting.read_session_cost(
                project(sessions[0]), price_book(), **session_options
            )
            assert session_cost.session_id == project(sessions[0])
            assert session_cost.total_cost == Decimal("0.0028")
            budget_usage = await accounting.read_causal_budget_usage(
                project(budget), **budget_options
            )
            budget_cost = await accounting.read_causal_budget_cost(
                project(budget), price_book(), **budget_options
            )
            for summary in (budget_usage, budget_cost):
                assert summary.causal_budget_id == project(budget)
                assert summary.session_ids == [project(value) for value in sessions[:2]]
                assert summary.session_count == 2
            assert budget_usage.usage.total_tokens == 1100
            assert budget_usage.session_summaries[1].usage.total_tokens == 0
            assert [
                row.session_id for row in budget_usage.session_summaries
            ] == budget_usage.session_ids
            assert budget_cost.total_cost == session_cost.total_cost
            assert [row.session_id for row in budget_cost.session_costs] == budget_cost.session_ids
            assert reads == [
                ("usage", sessions[0], None),
                ("usage", sessions[0], None),
                ("cost", sessions[0], None),
                ("usage", None, budget),
                ("cost", None, budget),
            ]

    asyncio.run(run())


def test_reporting_imports_without_application_or_execution_controllers():
    script = """
import importlib.abc
import sys
blocked = {"cayu.applications", "cayu.runtime._session_engine", "cayu.runtime._model_step_executor",
           "cayu.runtime._tool_round_executor", "cayu.runtime._recovery_coordinator"}
class RejectControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectControllers())
from cayu import _application_accounting
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("assembled", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["get_session_usage", "get_session_cost", "get_causal_budget_usage", "get_causal_budget_cost"],
)
def test_rejected_reports_do_not_expose_dependency_representations(assembled, operation):
    secret = "private-reporting-dependency"

    class PrivateStore(InMemorySessionStore):
        def __repr__(self):
            return secret

    class RejectIdentity:
        def __repr__(self):
            return secret

        async def __call__(self, value):
            raise PermissionError("identity unavailable")

    async def run():
        store = PrivateStore()
        app = CayuApp(session_store=store, enable_logging=False)
        resolver = RejectIdentity()
        args = ["unknown", price_book()] if operation.endswith("cost") else ["unknown"]
        try:
            if assembled:
                app._resolve_public_session_id = resolver
                app._resolve_public_causal_budget_id = resolver
                target = getattr(app, operation)
                options = {}
            else:
                target = getattr(
                    accounting,
                    "read_session_usage_snapshot"
                    if operation == "get_session_usage"
                    else operation.replace("get_", "read_", 1),
                )
                options = dict(session_store=store, project_session=lambda value: value)
                if "causal" in operation:
                    options.update(
                        resolve_budget=resolver, project_budget=lambda value, **kwargs: value
                    )
                else:
                    options["resolve_session"] = resolver
            with pytest.raises(PermissionError, match="identity unavailable") as failure:
                await target(*args, **options)
            tb = failure.value.__traceback__
            while tb:
                if tb.tb_frame.f_globals.get("__name__", "").startswith("cayu."):
                    assert secret not in repr(tb.tb_frame.f_locals)
                tb = tb.tb_next
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "operation",
    ["get_session_usage", "get_session_cost", "get_causal_budget_usage", "get_causal_budget_cost"],
)
def test_application_tracks_cancelled_accounting_until_store_settles(operation, monkeypatch):
    async def run():
        store = InMemorySessionStore()
        sessions, budget = await prepare_reports(store)
        app = CayuApp(session_store=store, enable_logging=False)
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                raise
            raise AssertionError("caller cancellation was lost")

        monkeypatch.setattr(
            store,
            "read_cost_accounting" if operation.endswith("cost") else "read_usage_accounting",
            blocked,
        )
        args = [budget if "causal" in operation else sessions[0]]
        if operation.endswith("cost"):
            args.append(price_book())
        caller = asyncio.create_task(getattr(app, operation)(*args))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            caller.cancel()
            await asyncio.wait_for(cancelled.wait(), 5)
            pending = await app.aclose(timeout_s=0.01)
            assert not pending.settled and pending.open_operations > 0
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            settled = await app.aclose(timeout_s=1)
            assert settled.settled and settled.open_operations == 0
            with pytest.raises(ApplicationAdmissionsSealed):
                await getattr(app, operation)(*args)
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
            await app.aclose()

    asyncio.run(run())
