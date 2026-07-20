from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from cayu.core import Event, EventType, Message
from cayu.core.billing import BillingIdentity, PricingContext
from cayu.runtime.aggregates import (
    MAX_AGGREGATE_USAGE_COUNTER,
    AggregateAccuracy,
    AggregateAccuracyKind,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsagePricingInput,
    UsageRollupStoreResult,
    coalesce_usage_pricing_inputs,
    estimate_usage_rollup_cost,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import (
    InMemorySessionStore,
    RunRequest,
    SessionAggregateFilter,
    SessionIdentity,
    SessionStatus,
    UsageRollupQuery,
)
from cayu.runtime.tasks import InMemoryTaskStore, TaskAggregateFilter, TaskCreate
from cayu.runtime.usage import UsageMetrics
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="configured", model="configured-model")


def _request(session_id: str, *, labels: dict[str, str] | None = None) -> RunRequest:
    return RunRequest(
        session_id=session_id,
        agent_name="assistant",
        messages=[Message.text("user", "hello")],
        labels=labels or {},
    )


def _model_event(
    *,
    event_id: str,
    session_id: str,
    timestamp: datetime,
    provider_name: str | None,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> Event:
    usage_metrics: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if provider_name is not None:
        usage_metrics["provider_name"] = provider_name
    if model is not None:
        usage_metrics["model"] = model
    return Event(
        id=event_id,
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        timestamp=timestamp,
        payload={"usage_metrics": usage_metrics},
    )


def _billing_model_event(*, event_id: str, session_id: str, timestamp: datetime) -> Event:
    identity = BillingIdentity(
        provider_name="gateway",
        resource_id="priced-model",
        request_evidence={"customer_secret": "must-not-enter-the-rollup"},
        completion_evidence={"provider_trace": "also-not-needed-for-pricing"},
    )
    metrics = UsageMetrics(
        provider_name="gateway",
        model="priced-model",
        billing_identity=identity,
        input_tokens=7,
        total_tokens=7,
    )
    return Event(
        id=event_id,
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        timestamp=timestamp,
        payload={
            "usage_metrics": metrics.model_dump(mode="json"),
            "billing_identity": identity.model_dump(mode="json"),
        },
    )


def test_in_memory_operational_snapshots_are_exact_and_filtered() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        await sessions.create(_request("one", labels={"team": "red"}), identity=_identity())
        await sessions.create(_request("two", labels={"team": "red"}), identity=_identity())
        await sessions.create(_request("three", labels={"team": "blue"}), identity=_identity())
        await sessions.update_status("two", SessionStatus.RUNNING)
        await sessions.update_status("three", SessionStatus.FAILED)

        snapshot = await sessions.aggregate_operational_snapshot(
            SessionAggregateFilter(labels={"team": "red"})
        )
        assert snapshot.total_count == 2
        assert snapshot.counts_by_status.model_dump() == {
            "pending": 1,
            "running": 1,
            "interrupting": 0,
            "completed": 0,
            "failed": 0,
            "interrupted": 0,
        }
        assert snapshot.accuracy.kind == "exact"

        tasks = InMemoryTaskStore()
        await tasks.create_task(
            TaskCreate(task_id="red_pending", type="deploy", assigned_agent_name="red")
        )
        await tasks.create_task(
            TaskCreate(task_id="red_running", type="deploy", assigned_agent_name="red")
        )
        await tasks.start_task("red_running")
        await tasks.create_task(
            TaskCreate(task_id="blue_pending", type="deploy", assigned_agent_name="blue")
        )

        task_snapshot = await tasks.aggregate_operational_snapshot(
            TaskAggregateFilter(assigned_agent_name="red")
        )
        assert task_snapshot.total_count == 2
        assert task_snapshot.counts_by_status.pending == 1
        assert task_snapshot.counts_by_status.running == 1
        assert sum(task_snapshot.counts_by_status.model_dump().values()) == 2

    asyncio.run(run())


def test_aggregate_filters_cover_current_session_and_task_dimensions() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        await sessions.create(_request("parent"), identity=_identity())
        await sessions.create(
            RunRequest(
                session_id="target",
                agent_name="reviewer",
                messages=[Message.text("user", "review")],
                parent_session_id="parent",
                causal_budget_id="workflow-budget",
                environment_name="production",
                labels={"team": "red", "project": "launch"},
            ),
            identity=SessionIdentity(provider_name="gateway", model="routed-model"),
        )
        await sessions.create(
            RunRequest(
                session_id="decoy",
                agent_name="reviewer",
                messages=[Message.text("user", "review")],
                causal_budget_id="other-budget",
                environment_name="production",
                labels={"team": "red", "project": "launch"},
            ),
            identity=SessionIdentity(provider_name="gateway", model="routed-model"),
        )
        filters = SessionAggregateFilter(
            agent_name="reviewer",
            provider_name="gateway",
            model="routed-model",
            environment_name="production",
            parent_session_id="parent",
            causal_budget_id="workflow-budget",
            labels={"team": "red"},
            label_selectors=(
                {"key": "project", "operator": "in", "values": ["launch", "research"]},
            ),
        )
        snapshot = await sessions.aggregate_operational_snapshot(filters)
        usage = await sessions.aggregate_usage(
            UsageRollupQuery(
                start_at=datetime(2026, 7, 1, tzinfo=UTC),
                end_at=datetime(2026, 7, 2, tzinfo=UTC),
                sessions=filters,
            )
        )
        assert snapshot.total_count == 1
        assert usage.matching_session_count == 1

        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(task_id="parent-task", type="workflow"))
        await tasks.create_task(
            TaskCreate(
                task_id="target-task",
                type="deploy",
                session_id="target",
                parent_task_id="parent-task",
                assigned_agent_name="reviewer",
            )
        )
        task_snapshot = await tasks.aggregate_operational_snapshot(
            TaskAggregateFilter(
                type="deploy",
                session_id="target",
                parent_task_id="parent-task",
                assigned_agent_name="reviewer",
            )
        )
        assert task_snapshot.total_count == 1
        assert task_snapshot.counts_by_status.pending == 1

    asyncio.run(run())


def test_in_memory_usage_rollup_uses_half_open_utc_window_and_exact_remainder() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        for session_id in ("one", "two", "three"):
            await store.create(
                _request(session_id, labels={"scope": "included"}),
                identity=_identity(),
            )
        await store.update_status("one", SessionStatus.RUNNING)

        start = datetime(2026, 7, 1, tzinfo=UTC)
        end = start + timedelta(days=1)
        await store.append_events(
            "one",
            [
                _model_event(
                    event_id="at-start",
                    session_id="one",
                    timestamp=start,
                    provider_name="alpha",
                    model="large",
                    input_tokens=100,
                    output_tokens=20,
                ),
                Event(
                    id="tool",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="one",
                    timestamp=start + timedelta(hours=1),
                ),
                _model_event(
                    event_id="at-end",
                    session_id="one",
                    timestamp=end,
                    provider_name="ignored",
                    model="ignored",
                    input_tokens=999,
                    output_tokens=1,
                ),
            ],
        )
        await store.append_event(
            "two",
            _model_event(
                event_id="beta",
                session_id="two",
                timestamp=start + timedelta(hours=2),
                provider_name="beta",
                model="small",
                input_tokens=30,
                output_tokens=10,
            ),
        )
        await store.append_event(
            "three",
            Event(
                id="missing-usage",
                type=EventType.MODEL_COMPLETED,
                session_id="three",
                timestamp=start + timedelta(hours=3),
                payload={},
            ),
        )
        await store.update_status("two", SessionStatus.COMPLETED)
        await store.update_status("three", SessionStatus.FAILED)

        result = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=end,
                sessions=SessionAggregateFilter(labels={"scope": "included"}),
                group_limit=1,
                include_pricing_inputs=True,
                pricing_input_limit=2,
            )
        )

        assert result.start_at == start
        assert result.end_at == end
        assert result.matching_session_count == 3
        assert result.active_session_count == 1
        assert result.totals.session_count == 3
        assert result.totals.model_steps == 3
        assert result.totals.model_steps_with_usage == 2
        assert result.totals.tool_calls == 1
        assert result.totals.usage.total_tokens == 160
        assert [group.provider_name for group in result.provider_breakdown.groups] == ["alpha"]
        assert result.provider_breakdown.remainder is not None
        assert result.provider_breakdown.remainder.group_count == 2
        assert result.provider_breakdown.remainder.totals.model_steps == 2
        assert result.provider_breakdown.accuracy.kind == "truncated"
        assert result.pricing_input_group_count == 3
        assert result.pricing_inputs == ()
        assert result.pricing_inputs_accuracy.kind == "truncated"

    asyncio.run(run())


def test_usage_rollup_normalizes_offsets_and_allows_empty_future_windows() -> None:
    offset = timezone(timedelta(hours=6))
    query = UsageRollupQuery(
        start_at=datetime(2027, 7, 2, 6, tzinfo=offset),
        end_at=datetime(2027, 7, 2, 7, tzinfo=offset),
    )
    assert query.start_at == datetime(2027, 7, 2, tzinfo=UTC)
    assert query.end_at == datetime(2027, 7, 2, 1, tzinfo=UTC)

    async def run() -> None:
        result = await InMemorySessionStore().aggregate_usage(query)
        assert result.totals.model_steps == 0
        assert result.matching_session_count == 0
        assert result.as_of < result.start_at

    asyncio.run(run())


def test_usage_rollup_normalizes_a_daylight_saving_boundary() -> None:
    new_york = ZoneInfo("America/New_York")
    query = UsageRollupQuery(
        start_at=datetime(2026, 3, 8, 1, 30, tzinfo=new_york),
        end_at=datetime(2026, 3, 8, 3, 30, tzinfo=new_york),
    )

    assert query.start_at == datetime(2026, 3, 8, 6, 30, tzinfo=UTC)
    assert query.end_at == datetime(2026, 3, 8, 7, 30, tzinfo=UTC)
    assert query.end_at - query.start_at == timedelta(hours=1)


def test_usage_rollup_pricing_projection_redacts_billing_evidence() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        await store.create(_request("billing-evidence"), identity=_identity())
        await store.append_event(
            "billing-evidence",
            _billing_model_event(
                event_id="billing-evidence-model",
                session_id="billing-evidence",
                timestamp=start,
            ),
        )

        result = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                include_pricing_inputs=True,
            )
        )
        projected = result.pricing_inputs[0].metrics
        assert projected is not None
        assert projected.billing_identity is not None
        assert projected.billing_identity.request_evidence == {}
        assert projected.billing_identity.completion_evidence == {}

    asyncio.run(run())


def test_usage_rollup_treats_naive_event_timestamps_as_utc(tmp_path) -> None:
    async def run() -> None:
        stores = (
            InMemorySessionStore(),
            SQLiteSessionStore(tmp_path / "naive-event-time.sqlite"),
        )
        start = datetime(2026, 7, 1, tzinfo=UTC)
        for index, store in enumerate(stores):
            session_id = f"naive-event-time-{index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_event(
                session_id,
                _model_event(
                    event_id="naive-model",
                    session_id=session_id,
                    timestamp=datetime(2026, 7, 1),
                    provider_name="provider",
                    model="model",
                    input_tokens=3,
                    output_tokens=2,
                ),
            )
            result = await store.aggregate_usage(
                UsageRollupQuery(start_at=start, end_at=start + timedelta(days=1))
            )
            assert result.totals.model_steps == 1
            assert result.totals.usage.total_tokens == 5
        await stores[1].close()

    asyncio.run(run())


def test_usage_rollup_canonicalizes_pricing_groups_before_enforcing_limit(
    tmp_path,
) -> None:
    async def run() -> None:
        stores = (
            InMemorySessionStore(),
            SQLiteSessionStore(tmp_path / "canonical-pricing.sqlite"),
        )
        start = datetime(2026, 7, 1, tzinfo=UTC)
        variants = ({}, {"input_tokens": 0}, {"cache": {}})
        for store_index, store in enumerate(stores):
            session_id = f"canonical-pricing-{store_index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_events(
                session_id,
                [
                    Event(
                        id=f"variant-{index}",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(minutes=index),
                        payload={"usage_metrics": metrics},
                    )
                    for index, metrics in enumerate(variants)
                ],
            )
            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start,
                    end_at=start + timedelta(days=1),
                    include_pricing_inputs=True,
                    pricing_input_limit=2,
                )
            )
            assert result.pricing_inputs_accuracy.kind == "exact"
            assert result.pricing_input_group_count == 1
            assert result.pricing_inputs[0].occurrences == 3
        await stores[1].close()

    asyncio.run(run())


def test_usage_rollup_uses_python_unicode_whitespace_identity_rules(tmp_path) -> None:
    async def run() -> None:
        stores = (
            InMemorySessionStore(),
            SQLiteSessionStore(tmp_path / "unicode-identities.sqlite"),
        )
        start = datetime(2026, 7, 1, tzinfo=UTC)
        for store_index, store in enumerate(stores):
            session_id = f"unicode-identities-{store_index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_events(
                session_id,
                [
                    Event(
                        id="nonbreaking-space-provider",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start,
                        payload={"usage_metrics": {"provider_name": "\u00a0"}},
                    ),
                    Event(
                        id="missing-provider",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(minutes=1),
                        payload={"usage_metrics": {}},
                    ),
                ],
            )
            result = await store.aggregate_usage(
                UsageRollupQuery(start_at=start, end_at=start + timedelta(days=1))
            )
            assert len(result.provider_breakdown.groups) == 1
            assert result.provider_breakdown.groups[0].provider_name is None
            assert result.provider_breakdown.groups[0].totals.model_steps == 2
        await stores[1].close()

    asyncio.run(run())


def test_pricing_input_keys_ignore_nested_mapping_insertion_order() -> None:
    def metrics(dimensions: dict[str, str]) -> UsageMetrics:
        return UsageMetrics(
            billing_identity=BillingIdentity(
                provider_name="gateway",
                resource_id="resource",
                pricing_contexts=(PricingContext(dimensions=dimensions),),
            )
        )

    inputs = coalesce_usage_pricing_inputs(
        (
            UsagePricingInput(
                effective_on=datetime(2026, 7, 1).date(),
                occurrences=1,
                metrics=metrics({"region": "us", "tier": "standard"}),
            ),
            UsagePricingInput(
                effective_on=datetime(2026, 7, 1).date(),
                occurrences=1,
                metrics=metrics({"tier": "standard", "region": "us"}),
            ),
        )
    )
    assert len(inputs) == 1
    assert inputs[0].occurrences == 2


def test_sqlite_usage_rollup_sums_past_signed_64_bit_without_overflow(tmp_path) -> None:
    async def run() -> None:
        stores = (
            InMemorySessionStore(),
            SQLiteSessionStore(tmp_path / "exact-large-sum.sqlite"),
        )
        start = datetime(2026, 7, 1, tzinfo=UTC)
        maximum = 2**63 - 1
        for store_index, store in enumerate(stores):
            session_id = f"large-sum-{store_index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_events(
                session_id,
                [
                    _model_event(
                        event_id=f"maximum-{index}",
                        session_id=session_id,
                        timestamp=start + timedelta(minutes=index),
                        provider_name=f"provider-{index // 2}",
                        model="model",
                        input_tokens=maximum,
                        output_tokens=0,
                    )
                    for index in range(4)
                ],
            )
            result = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=start,
                    end_at=start + timedelta(days=1),
                    group_limit=1,
                )
            )
            assert result.totals.usage.input_tokens == 4 * maximum
            assert result.provider_breakdown.groups[0].totals.usage.input_tokens == (2 * maximum)
            assert result.provider_breakdown.remainder is not None
            assert result.provider_breakdown.remainder.totals.usage.input_tokens == 2 * maximum
        await stores[1].close()

    asyncio.run(run())


def test_aggregate_usage_rejects_per_event_counters_above_shared_limit(tmp_path) -> None:
    async def run() -> None:
        stores = (
            InMemorySessionStore(),
            SQLiteSessionStore(tmp_path / "oversized-counter.sqlite"),
        )
        start = datetime(2026, 7, 1, tzinfo=UTC)
        for store_index, store in enumerate(stores):
            session_id = f"oversized-counter-{store_index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_event(
                session_id,
                _model_event(
                    event_id="oversized-counter",
                    session_id=session_id,
                    timestamp=start,
                    provider_name="provider",
                    model="model",
                    input_tokens=MAX_AGGREGATE_USAGE_COUNTER + 1,
                    output_tokens=0,
                ),
            )

        query = UsageRollupQuery(
            start_at=start,
            end_at=start + timedelta(days=1),
            include_pricing_inputs=True,
        )
        results = [await store.aggregate_usage(query) for store in stores]
        assert results[1].model_dump(exclude={"as_of"}) == results[0].model_dump(exclude={"as_of"})
        assert results[0].totals.model_steps_with_usage == 1
        assert results[0].totals.usage.input_tokens == 0
        assert results[0].pricing_inputs[0].metrics is None
        await stores[1].close()

    asyncio.run(run())


def test_pricing_input_accumulator_enforces_serialized_byte_limit() -> None:
    accumulator = BoundedUsagePricingInputAccumulator(limit=10, max_bytes=32)
    accumulator.add(
        UsagePricingInput(
            effective_on=datetime(2026, 7, 1).date(),
            occurrences=1,
            metrics=UsageMetrics(provider_name="provider", model="model"),
        )
    )

    inputs, group_count, accuracy = accumulator.result()

    assert inputs == ()
    assert group_count == 1
    assert accuracy.kind == "truncated"
    assert accuracy.limit == 32
    assert "serialized-byte limit" in (accuracy.reason or "")


def test_pricing_input_accumulator_bounds_raw_payload_before_validation() -> None:
    accumulator = BoundedUsagePricingInputAccumulator(limit=10, max_bytes=64)
    accumulator.add_payload(
        effective_on=datetime(2026, 7, 1).date(),
        occurrences=1,
        payload={
            "usage_metrics": {
                "provider_name": "p" * 128,
                "model": "model",
                "input_tokens": 1,
                "total_tokens": 1,
            }
        },
    )

    inputs, group_count, accuracy = accumulator.result()

    assert inputs == ()
    assert group_count == 1
    assert accuracy.kind == "truncated"
    assert accuracy.limit == 64


def test_complete_sampled_usage_breakdown_is_representable() -> None:
    breakdown = UsageAggregateBreakdown(
        groups=(),
        remainder=None,
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.SAMPLED,
            reason="The store returned a representative sample.",
            limit=100,
        ),
    )

    assert breakdown.accuracy.kind == "sampled"


def test_sampled_rollup_requires_truthful_breakdown_accuracy() -> None:
    async def build_exact_result() -> UsageRollupStoreResult:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        return await store.aggregate_usage(
            UsageRollupQuery(start_at=start, end_at=start + timedelta(days=1))
        )

    payload = asyncio.run(build_exact_result()).model_dump(mode="json")
    sampled_accuracy = {
        "kind": "sampled",
        "reason": "The store returned a representative sample.",
        "limit": 100,
    }
    payload["totals_accuracy"] = sampled_accuracy

    with pytest.raises(ValidationError, match="cannot include an exact breakdown"):
        UsageRollupStoreResult.model_validate(payload)

    payload["provider_breakdown"]["accuracy"] = sampled_accuracy
    payload["model_breakdown"]["accuracy"] = sampled_accuracy
    result = UsageRollupStoreResult.model_validate(payload)
    assert result.provider_breakdown.accuracy.kind == "sampled"
    assert result.model_breakdown.accuracy.kind == "sampled"


def test_in_memory_usage_breakdown_remains_exact_within_candidate_bound() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        await store.create(_request("many-groups"), identity=_identity())
        await store.append_events(
            "many-groups",
            [
                _model_event(
                    event_id=f"group-{index}",
                    session_id="many-groups",
                    timestamp=start + timedelta(seconds=index),
                    provider_name=f"provider-{index:03d}",
                    model=f"model-{index:03d}",
                    input_tokens=index,
                    output_tokens=0,
                )
                for index in range(300)
            ],
        )
        result = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                group_limit=2,
            )
        )
        assert [group.provider_name for group in result.provider_breakdown.groups] == [
            "provider-299",
            "provider-298",
        ]
        assert result.provider_breakdown.remainder is not None
        assert result.provider_breakdown.remainder.group_count == 298
        assert result.provider_breakdown.remainder.totals.model_steps == 298
        assert result.totals.usage.input_tokens == sum(range(300))

    asyncio.run(run())


def test_in_memory_usage_breakdown_samples_bounded_heavy_hitters() -> None:
    from cayu.runtime import sessions as session_runtime

    async def run() -> None:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        await store.create(_request("high-cardinality"), identity=_identity())
        candidate_limit = session_runtime._IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT
        events = [
            _model_event(
                event_id=f"one-off-{index}",
                session_id="high-cardinality",
                timestamp=start + timedelta(seconds=index),
                provider_name=f"provider-{index:04d}",
                model=f"model-{index:04d}",
                input_tokens=1,
                output_tokens=0,
            )
            for index in range(candidate_limit + 8)
        ]
        events.extend(
            _model_event(
                event_id=f"heavy-{index}",
                session_id="high-cardinality",
                timestamp=start + timedelta(minutes=20, seconds=index),
                provider_name="heavy-provider",
                model="heavy-model",
                input_tokens=100,
                output_tokens=0,
            )
            for index in range(20)
        )
        await store.append_events("high-cardinality", events)

        result = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                group_limit=5,
            )
        )

        assert result.totals.model_steps == len(events)
        assert result.totals.usage.input_tokens == candidate_limit + 8 + 2_000
        assert result.provider_breakdown.accuracy.kind == "sampled"
        assert result.provider_breakdown.accuracy.limit == candidate_limit
        assert result.provider_breakdown.remainder is None
        assert result.provider_breakdown.groups[0].provider_name == "heavy-provider"
        assert result.model_breakdown.accuracy.kind == "sampled"
        assert result.model_breakdown.groups[0].model == "heavy-model"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("start_at", "end_at", "message"),
    [
        (
            datetime(2026, 7, 1),
            datetime(2026, 7, 2, tzinfo=UTC),
            "timezone-aware",
        ),
        (
            datetime(2026, 7, 2, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
            "before end_at",
        ),
        (
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
            "cannot exceed 366 days",
        ),
    ],
)
def test_usage_rollup_rejects_invalid_windows(
    start_at: datetime,
    end_at: datetime,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        UsageRollupQuery(start_at=start_at, end_at=end_at)


def test_session_aggregate_filters_reject_oversized_label_predicates() -> None:
    with pytest.raises(ValidationError, match="at most 50 items"):
        SessionAggregateFilter(labels={f"label-{index}": "value" for index in range(51)})
    with pytest.raises(ValidationError, match="more than 100 total values"):
        SessionAggregateFilter(
            label_selectors=(
                {
                    "key": "project",
                    "operator": "in",
                    "values": [f"project-{index}" for index in range(101)],
                },
            )
        )


def test_usage_rollup_cost_keeps_currencies_separate_and_missing_pricing_visible() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        for index, (provider, model, input_tokens, output_tokens) in enumerate(
            (
                ("alpha", "a", 1_000_000, 0),
                ("beta", "b", 0, 1_000_000),
                ("gamma", "missing", 1_000_000, 0),
            )
        ):
            session_id = f"cost-{index}"
            await store.create(_request(session_id), identity=_identity())
            await store.append_event(
                session_id,
                _model_event(
                    event_id=f"cost-event-{index}",
                    session_id=session_id,
                    timestamp=start + timedelta(minutes=index),
                    provider_name=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        pricing = PriceBook(
            price_book_version="mixed-currency-test",
            generated_at="2026-07-01",
            prices=(
                ModelPrice.fixed(
                    provider_name="alpha",
                    model="a",
                    match="exact",
                    currency="USD",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
                ModelPrice.fixed(
                    provider_name="beta",
                    model="b",
                    match="exact",
                    currency="EUR",
                    input_per_million=Decimal("2"),
                    output_per_million=Decimal("2"),
                ),
            ),
        )
        exact = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                include_pricing_inputs=True,
            )
        )
        cost = estimate_usage_rollup_cost(exact, pricing)
        assert [(item.currency, item.total_cost) for item in cost.currencies] == [
            ("EUR", Decimal("2")),
            ("USD", Decimal("1")),
        ]
        assert cost.priced_model_steps == 2
        assert cost.unpriced_model_steps == 1
        assert cost.unpriced_reasons[0].reason == "no matching model pricing"

        bounded = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                include_pricing_inputs=True,
                pricing_input_limit=1,
            )
        )
        bounded_cost = estimate_usage_rollup_cost(bounded, pricing)
        assert bounded_cost.accuracy.kind == "truncated"
        assert bounded_cost.evaluated_model_steps == 0
        assert bounded_cost.unevaluated_model_steps == 3
        assert bounded_cost.currencies == ()

    asyncio.run(run())


def test_usage_rollup_cost_never_overstates_store_accuracy() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        start = datetime(2026, 7, 1, tzinfo=UTC)
        await store.create(_request("sampled-cost"), identity=_identity())
        await store.append_event(
            "sampled-cost",
            _model_event(
                event_id="sampled-model",
                session_id="sampled-cost",
                timestamp=start,
                provider_name="provider",
                model="model",
                input_tokens=1_000_000,
                output_tokens=0,
            ),
        )
        result = await store.aggregate_usage(
            UsageRollupQuery(
                start_at=start,
                end_at=start + timedelta(days=1),
                include_pricing_inputs=True,
            )
        )
        pricing = PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="provider",
                    model="model",
                    match="exact",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        )

        sampled = result.model_copy(
            update={
                "totals_accuracy": AggregateAccuracy(
                    kind=AggregateAccuracyKind.SAMPLED,
                    reason="The store returned a representative sample.",
                    limit=100,
                )
            }
        )
        sampled_cost = estimate_usage_rollup_cost(sampled, pricing)
        assert sampled_cost.accuracy.kind == "sampled"
        assert sampled_cost.evaluated_model_steps == 1
        assert sampled_cost.currencies[0].total_cost == Decimal("1")

        truncated = result.model_copy(
            update={
                "totals_accuracy": AggregateAccuracy(
                    kind=AggregateAccuracyKind.TRUNCATED,
                    reason="The store omitted records beyond its bound.",
                    limit=100,
                )
            }
        )
        truncated_cost = estimate_usage_rollup_cost(truncated, pricing)
        assert truncated_cost.accuracy.kind == "truncated"
        assert truncated_cost.evaluated_model_steps == 0
        assert truncated_cost.unevaluated_model_steps == 1
        assert truncated_cost.currencies == ()

    asyncio.run(run())


def test_sqlite_aggregates_match_in_memory_reference(tmp_path) -> None:
    async def run() -> None:
        memory_sessions = InMemorySessionStore()
        sqlite_sessions = SQLiteSessionStore(tmp_path / "aggregate-sessions.sqlite")
        stores = (memory_sessions, sqlite_sessions)
        start = datetime(2026, 7, 1, tzinfo=UTC)
        end = start + timedelta(days=1)

        for prefix, store in zip(("memory", "sqlite"), stores, strict=True):
            one = f"{prefix}-one"
            two = f"{prefix}-two"
            await store.create(
                _request(one, labels={"team": "red"}),
                identity=_identity(),
            )
            await store.create(
                _request(two, labels={"team": "red"}),
                identity=_identity(),
            )
            await store.update_status(one, SessionStatus.RUNNING)
            await store.update_status(two, SessionStatus.COMPLETED)
            await store.append_events(
                one,
                [
                    _model_event(
                        event_id="one-model",
                        session_id=one,
                        timestamp=start,
                        provider_name="alpha",
                        model="large",
                        input_tokens=20,
                        output_tokens=5,
                    ),
                    Event(
                        id="one-tool",
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=one,
                        timestamp=start + timedelta(minutes=1),
                    ),
                ],
            )
            await store.append_event(
                two,
                _model_event(
                    event_id="two-model",
                    session_id=two,
                    timestamp=start + timedelta(hours=1),
                    provider_name="beta",
                    model="small",
                    input_tokens=20,
                    output_tokens=5,
                ),
            )
            await store.append_event(
                one,
                _billing_model_event(
                    event_id="billing-model",
                    session_id=one,
                    timestamp=start + timedelta(hours=2),
                ),
            )
            await store.append_event(
                one,
                Event(
                    id="malformed-billing-identity",
                    type=EventType.MODEL_COMPLETED,
                    session_id=one,
                    timestamp=start + timedelta(hours=3),
                    payload={
                        "usage_metrics": {
                            "provider_name": "invalid-pricing",
                            "model": "invalid-pricing",
                            "billing_identity": "not-an-object",
                            "input_tokens": 1,
                            "total_tokens": 1,
                        }
                    },
                ),
            )
            await store.append_event(
                one,
                Event(
                    id="raw-only-pricing",
                    type=EventType.MODEL_COMPLETED,
                    session_id=one,
                    timestamp=start + timedelta(hours=4),
                    payload={
                        "provider_name": "raw-provider",
                        "model": "raw-model",
                        "usage": {"input_tokens": 1},
                    },
                ),
            )

        filters = SessionAggregateFilter(
            agent_name="assistant",
            provider_name="configured",
            model="configured-model",
            labels={"team": "red"},
        )
        memory_snapshot = await memory_sessions.aggregate_operational_snapshot(filters)
        sqlite_snapshot = await sqlite_sessions.aggregate_operational_snapshot(filters)
        assert sqlite_snapshot.model_dump(exclude={"as_of"}) == memory_snapshot.model_dump(
            exclude={"as_of"}
        )

        query = UsageRollupQuery(
            start_at=start,
            end_at=end,
            sessions=filters,
            group_limit=1,
            include_pricing_inputs=True,
        )
        memory_rollup = await memory_sessions.aggregate_usage(query)
        sqlite_rollup = await sqlite_sessions.aggregate_usage(query)
        assert sqlite_rollup.model_dump(exclude={"as_of"}) == memory_rollup.model_dump(
            exclude={"as_of"}
        )

        memory_tasks = InMemoryTaskStore()
        sqlite_tasks = SQLiteTaskStore(tmp_path / "aggregate-tasks.sqlite")
        for store in (memory_tasks, sqlite_tasks):
            await store.create_task(
                TaskCreate(task_id="pending", type="deploy", assigned_agent_name="red")
            )
            await store.create_task(
                TaskCreate(task_id="running", type="deploy", assigned_agent_name="red")
            )
            await store.start_task("running")
        task_filters = TaskAggregateFilter(assigned_agent_name="red")
        memory_task_snapshot = await memory_tasks.aggregate_operational_snapshot(task_filters)
        sqlite_task_snapshot = await sqlite_tasks.aggregate_operational_snapshot(task_filters)
        assert sqlite_task_snapshot.model_dump(
            exclude={"as_of"}
        ) == memory_task_snapshot.model_dump(exclude={"as_of"})

        await sqlite_sessions.close()
        await sqlite_tasks.close()

    asyncio.run(run())


def test_sqlite_usage_rollup_handles_malformed_normalized_usage_like_memory(tmp_path) -> None:
    async def run() -> None:
        memory = InMemorySessionStore()
        sqlite = SQLiteSessionStore(tmp_path / "malformed-usage.sqlite")
        start = datetime(2026, 7, 1, tzinfo=UTC)
        for prefix, store in (("memory", memory), ("sqlite", sqlite)):
            session_id = f"{prefix}-malformed"
            await store.create(_request(session_id), identity=_identity())
            await store.append_event(
                session_id,
                Event(
                    id="malformed-normalized-usage",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=start,
                    payload={
                        "usage_metrics": {
                            "provider_name": " invalid ",
                            "model": [],
                            "input_tokens": "12",
                            "output_tokens": 3,
                            "total_tokens": 3,
                            "cache": {"read_tokens": -1, "write_tokens": 2},
                        }
                    },
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    id="raw-only-usage",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=start + timedelta(minutes=1),
                    payload={
                        "provider_name": "raw-provider",
                        "model": "raw-model",
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    },
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    id="malformed-billing-identity",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    timestamp=start + timedelta(minutes=2),
                    payload={
                        "usage_metrics": {
                            "provider_name": "invalid-pricing",
                            "model": "invalid-pricing",
                            "billing_identity": "not-an-object",
                            "input_tokens": 1,
                            "total_tokens": 1,
                        }
                    },
                ),
            )

        query = UsageRollupQuery(
            start_at=start,
            end_at=start + timedelta(days=1),
            include_pricing_inputs=True,
        )
        memory_result = await memory.aggregate_usage(query)
        sqlite_result = await sqlite.aggregate_usage(query)
        assert sqlite_result.model_dump(exclude={"as_of"}) == memory_result.model_dump(
            exclude={"as_of"}
        )
        assert sqlite_result.totals.model_steps == 3
        assert sqlite_result.totals.model_steps_with_usage == 2
        assert sqlite_result.totals.usage.output_tokens == 3
        assert sqlite_result.totals.usage.cache.write_tokens == 2
        assert sqlite_result.provider_breakdown.groups[0].provider_name is None
        assert sqlite_result.pricing_input_group_count == 1
        assert sqlite_result.pricing_inputs[0].metrics is None
        assert sqlite_result.pricing_inputs[0].occurrences == 3
        await sqlite.close()

    asyncio.run(run())


def test_sqlite_usage_rollup_plan_bounds_the_index_by_type_and_time(tmp_path) -> None:
    from cayu.storage import _session_store_sql, _sqlite_aggregates, _sqlite_support
    from cayu.storage.sqlite import _SQL_DIALECT

    database_path = tmp_path / "aggregate-plan.sqlite"
    store = SQLiteSessionStore(database_path)
    query = UsageRollupQuery(
        start_at=datetime(2026, 7, 1, tzinfo=UTC),
        end_at=datetime(2026, 7, 2, tzinfo=UTC),
    )
    plan = _session_store_sql.build_session_query_sql(None, dialect=_SQL_DIALECT)
    sql, params = _sqlite_aggregates.usage_rollup_statement(
        session_plan=plan,
        query=query,
    )
    pricing_sql, pricing_params = _sqlite_aggregates.pricing_input_statement(
        session_plan=plan,
        query=query,
    )
    connection = _sqlite_support.connect(database_path, read_only=True)
    try:
        details = [
            row[3] for row in connection.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
        ]
        pricing_details = [
            row[3]
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN {pricing_sql}",
                pricing_params,
            ).fetchall()
        ]
    finally:
        connection.close()
        asyncio.run(store.close())

    assert any(
        "idx_cayu_events_type_timestamp " in detail
        and "event_type=?" in detail
        and "timestamp>?" in detail
        and "timestamp<?" in detail
        for detail in details
    )
    assert any(
        "idx_cayu_events_type_timestamp " in detail
        and "event_type=?" in detail
        and "timestamp>?" in detail
        and "timestamp<?" in detail
        for detail in pricing_details
    )


def test_sqlite_pricing_projection_rejects_oversized_rows_before_transfer(tmp_path) -> None:
    from cayu.storage import _session_store_sql, _sqlite_aggregates, _sqlite_support
    from cayu.storage.sqlite import _SQL_DIALECT

    database_path = tmp_path / "aggregate-pricing-bound.sqlite"
    store = SQLiteSessionStore(database_path)
    start = datetime(2026, 7, 1, tzinfo=UTC)

    async def seed() -> None:
        await store.create(_request("oversized-pricing"), identity=_identity())
        await store.append_event(
            "oversized-pricing",
            _model_event(
                event_id="oversized-pricing-event",
                session_id="oversized-pricing",
                timestamp=start,
                provider_name="p" * 128,
                model="model",
                input_tokens=1,
                output_tokens=0,
            ),
        )

    asyncio.run(seed())
    plan = _session_store_sql.build_session_query_sql(None, dialect=_SQL_DIALECT)
    query = UsageRollupQuery(start_at=start, end_at=start + timedelta(days=1))
    sql, params = _sqlite_aggregates.pricing_input_statement(
        session_plan=plan,
        query=query,
        max_input_bytes=64,
    )
    connection = _sqlite_support.connect(database_path, read_only=True)
    try:
        rows = connection.execute(sql, params).fetchall()
    finally:
        connection.close()
        asyncio.run(store.close())

    assert len(rows) == 1
    row = rows[0]
    assert row["input_oversized"] == 1
    assert row["usage_metrics_json"] is None
    assert row["billing_identity_json"] is None
    accumulator = BoundedUsagePricingInputAccumulator(limit=10, max_bytes=64)
    _sqlite_aggregates._add_pricing_input_from_row(accumulator, row)
    assert accumulator.truncated is True
    assert accumulator.result()[0] == ()


def test_postgres_aggregates_match_in_memory_reference(postgres_dsn: str) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

    async def run() -> None:
        memory_sessions = InMemorySessionStore()
        postgres_sessions = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        stores = (memory_sessions, postgres_sessions)
        start = datetime(2026, 7, 1, tzinfo=UTC)
        end = start + timedelta(days=1)
        suffix = uuid4().hex
        label_value = f"aggregate-contract-postgres-{suffix}"

        for prefix, store in zip(("memory-pg", "postgres"), stores, strict=True):
            one = f"{prefix}-{suffix}-aggregate-one"
            two = f"{prefix}-{suffix}-aggregate-two"
            await store.create(
                _request(one, labels={"test-scope": label_value}),
                identity=_identity(),
            )
            await store.create(
                _request(two, labels={"test-scope": label_value}),
                identity=_identity(),
            )
            await store.update_status(one, SessionStatus.RUNNING)
            await store.update_status(two, SessionStatus.COMPLETED)
            await store.append_events(
                one,
                [
                    _model_event(
                        event_id="one-model",
                        session_id=one,
                        timestamp=start,
                        provider_name="alpha",
                        model="large",
                        input_tokens=20,
                        output_tokens=5,
                    ),
                    Event(
                        id="one-tool",
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=one,
                        timestamp=start + timedelta(minutes=1),
                    ),
                ],
            )
            await store.append_event(
                two,
                _model_event(
                    event_id="two-model",
                    session_id=two,
                    timestamp=start + timedelta(hours=1),
                    provider_name="beta",
                    model="small",
                    input_tokens=20,
                    output_tokens=5,
                ),
            )
            await store.append_event(
                one,
                _billing_model_event(
                    event_id="billing-model",
                    session_id=one,
                    timestamp=start + timedelta(hours=2),
                ),
            )
            await store.append_event(
                one,
                Event(
                    id="malformed-billing-identity",
                    type=EventType.MODEL_COMPLETED,
                    session_id=one,
                    timestamp=start + timedelta(hours=3),
                    payload={
                        "usage_metrics": {
                            "provider_name": "invalid-pricing",
                            "model": "invalid-pricing",
                            "billing_identity": "not-an-object",
                            "input_tokens": 1,
                            "total_tokens": 1,
                        }
                    },
                ),
            )
            await store.append_event(
                one,
                Event(
                    id="raw-only-pricing",
                    type=EventType.MODEL_COMPLETED,
                    session_id=one,
                    timestamp=start + timedelta(hours=4),
                    payload={
                        "provider_name": "raw-provider",
                        "model": "raw-model",
                        "usage": {"input_tokens": 1},
                    },
                ),
            )

        filters = SessionAggregateFilter(
            agent_name="assistant",
            provider_name="configured",
            model="configured-model",
            labels={"test-scope": label_value},
        )
        memory_snapshot = await memory_sessions.aggregate_operational_snapshot(filters)
        postgres_snapshot = await postgres_sessions.aggregate_operational_snapshot(filters)
        assert postgres_snapshot.model_dump(exclude={"as_of"}) == memory_snapshot.model_dump(
            exclude={"as_of"}
        )

        query = UsageRollupQuery(
            start_at=start,
            end_at=end,
            sessions=filters,
            group_limit=1,
            include_pricing_inputs=True,
        )
        memory_rollup = await memory_sessions.aggregate_usage(query)
        postgres_rollup = await postgres_sessions.aggregate_usage(query)
        assert postgres_rollup.model_dump(exclude={"as_of"}) == memory_rollup.model_dump(
            exclude={"as_of"}
        )

        edge_label = f"aggregate-edge-{suffix}"
        for prefix, store in zip(("memory-pg", "postgres"), stores, strict=True):
            session_id = f"{prefix}-{suffix}-aggregate-edge"
            await store.create(
                _request(session_id, labels={"test-scope": edge_label}),
                identity=_identity(),
            )
            await store.append_events(
                session_id,
                [
                    *(
                        Event(
                            id=f"canonical-{index}",
                            type=EventType.MODEL_COMPLETED,
                            session_id=session_id,
                            timestamp=start + timedelta(hours=10, minutes=index),
                            payload={"usage_metrics": metrics},
                        )
                        for index, metrics in enumerate(({}, {"input_tokens": 0}, {"cache": {}}))
                    ),
                    Event(
                        id="unicode-whitespace",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(hours=12),
                        payload={"usage_metrics": {"provider_name": "\u00a0"}},
                    ),
                    Event(
                        id="unknown-provider",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        timestamp=start + timedelta(hours=12, minutes=1),
                        payload={"usage_metrics": {}},
                    ),
                    _model_event(
                        event_id="oversized-counter",
                        session_id=session_id,
                        timestamp=start + timedelta(hours=14),
                        provider_name="provider",
                        model="model",
                        input_tokens=MAX_AGGREGATE_USAGE_COUNTER + 1,
                        output_tokens=0,
                    ),
                ],
            )
        edge_filter = SessionAggregateFilter(labels={"test-scope": edge_label})
        canonical_query = UsageRollupQuery(
            start_at=start + timedelta(hours=10),
            end_at=start + timedelta(hours=11),
            sessions=edge_filter,
            include_pricing_inputs=True,
            pricing_input_limit=2,
        )
        identity_query = UsageRollupQuery(
            start_at=start + timedelta(hours=12),
            end_at=start + timedelta(hours=13),
            sessions=edge_filter,
        )
        oversized_counter_query = UsageRollupQuery(
            start_at=start + timedelta(hours=14),
            end_at=start + timedelta(hours=15),
            sessions=edge_filter,
            include_pricing_inputs=True,
        )
        for edge_query in (canonical_query, identity_query, oversized_counter_query):
            memory_edge = await memory_sessions.aggregate_usage(edge_query)
            postgres_edge = await postgres_sessions.aggregate_usage(edge_query)
            assert postgres_edge.model_dump(exclude={"as_of"}) == memory_edge.model_dump(
                exclude={"as_of"}
            )
        canonical_edge = await postgres_sessions.aggregate_usage(canonical_query)
        assert canonical_edge.pricing_input_group_count == 1
        assert canonical_edge.pricing_inputs[0].occurrences == 3
        identity_edge = await postgres_sessions.aggregate_usage(identity_query)
        assert len(identity_edge.provider_breakdown.groups) == 1
        assert identity_edge.provider_breakdown.groups[0].provider_name is None
        oversized_counter_edge = await postgres_sessions.aggregate_usage(oversized_counter_query)
        assert oversized_counter_edge.totals.usage.input_tokens == 0
        assert oversized_counter_edge.pricing_inputs[0].metrics is None

        from psycopg import AsyncConnection

        from cayu.runtime.sessions import session_query_from_aggregate_filter
        from cayu.storage import _postgres_aggregates, _session_store_sql
        from cayu.storage.postgres import _SQL_DIALECT

        plan = _session_store_sql.build_session_query_sql(None, dialect=_SQL_DIALECT)
        sql, params = _postgres_aggregates.usage_rollup_statement(
            session_plan=plan,
            query=query.model_copy(update={"sessions": SessionAggregateFilter()}),
        )
        pricing_sql, pricing_params = _postgres_aggregates.pricing_input_statement(
            session_plan=plan,
            query=query.model_copy(update={"sessions": SessionAggregateFilter()}),
        )
        edge_plan = _session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(canonical_query.sessions),
            dialect=_SQL_DIALECT,
        )
        bounded_pricing_sql, bounded_pricing_params = _postgres_aggregates.pricing_input_statement(
            session_plan=edge_plan,
            query=canonical_query,
            max_input_bytes=1,
        )
        connection = await AsyncConnection.connect(postgres_dsn)
        try:
            async with connection.cursor() as cursor:
                await cursor.execute("SET enable_seqscan = off")
                await cursor.execute(f"EXPLAIN (COSTS OFF) {sql}", params)
                explain_rows = await cursor.fetchall()
                await cursor.execute(
                    f"EXPLAIN (COSTS OFF) {pricing_sql}",
                    pricing_params,
                )
                pricing_explain_rows = await cursor.fetchall()
                await cursor.execute(bounded_pricing_sql, bounded_pricing_params)
                bounded_pricing_row = await cursor.fetchone()
        finally:
            await connection.close()
        explanation = "\n".join(str(row[0]) for row in explain_rows)
        assert "idx_cayu_events_type_timestamp" in explanation
        assert "event_type = ANY" in explanation
        assert '"timestamp" >=' in explanation
        assert '"timestamp" <' in explanation
        assert 'COLLATE "C"' in sql
        pricing_explanation = "\n".join(str(row[0]) for row in pricing_explain_rows)
        assert "idx_cayu_events_type_timestamp" in pricing_explanation
        assert '"timestamp" >=' in pricing_explanation
        assert '"timestamp" <' in pricing_explanation
        assert bounded_pricing_row is not None
        assert bounded_pricing_row[1:4] == (None, None, True)

        memory_tasks = InMemoryTaskStore()
        postgres_tasks = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        assigned_agent = f"aggregate-contract-postgres-agent-{suffix}"
        for prefix, store in zip(
            ("memory-pg", "postgres"),
            (memory_tasks, postgres_tasks),
            strict=True,
        ):
            await store.create_task(
                TaskCreate(
                    task_id=f"{prefix}-{suffix}-aggregate-pending",
                    type="deploy",
                    assigned_agent_name=assigned_agent,
                )
            )
            await store.create_task(
                TaskCreate(
                    task_id=f"{prefix}-{suffix}-aggregate-running",
                    type="deploy",
                    assigned_agent_name=assigned_agent,
                )
            )
            await store.start_task(f"{prefix}-{suffix}-aggregate-running")
        task_filters = TaskAggregateFilter(assigned_agent_name=assigned_agent)
        memory_task_snapshot = await memory_tasks.aggregate_operational_snapshot(task_filters)
        postgres_task_snapshot = await postgres_tasks.aggregate_operational_snapshot(task_filters)
        assert postgres_task_snapshot.model_dump(
            exclude={"as_of"}
        ) == memory_task_snapshot.model_dump(exclude={"as_of"})

        await postgres_sessions.close()
        await postgres_tasks.close()

    asyncio.run(run())
