from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

import cayu._validation as validation
from cayu._validation import DurableValueError
from cayu.core.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_payload_authority,
)
from cayu.runtime._event_projection import prepare_new_runtime_event
from cayu.runtime.tool_terminal_publication import ToolTerminalPublicationGovernor
from cayu.vaults import SecretRedactor

_TIMING_FIELDS = (
    "tool_effect_completed_at",
    "tool_terminal_staged_at",
    "tool_terminal_publication_started_at",
)


def test_copy_event_reuses_validated_text_but_revalidates_mutation(monkeypatch) -> None:
    content = "large-validated-content-" + ("x" * 500_000)
    event = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="session-copy",
        payload={
            "tool_call_id": "call-copy",
            "result": {"content": content, "structured": {"nested": ["safe"]}},
        },
    )
    validated_lengths: list[int] = []
    original = validation._require_durable_text

    def counted(value: str, field_name: str, *, path: str) -> str:
        validated_lengths.append(len(value))
        return original(value, field_name, path=path)

    monkeypatch.setattr(validation, "_require_durable_text", counted)

    copied = copy_event(event)

    assert len(content) not in validated_lengths
    assert copied.payload is not event.payload
    assert copied.payload["result"] is not event.payload["result"]
    assert copied.payload["result"]["content"] is content

    event.payload["result"]["structured"]["nested"][0] = "changed"
    mutated = copy_event(event)
    assert len(content) in validated_lengths
    assert mutated.payload["result"]["structured"]["nested"] == ["changed"]

    event.payload["result"]["structured"]["nested"][0] = object()
    with pytest.raises(DurableValueError, match="invalid_json_type"):
        copy_event(event)


def test_large_publication_work_cannot_starve_unrelated_small_work() -> None:
    async def scenario() -> None:
        governor = ToolTerminalPublicationGovernor(
            slice_bytes=1024,
            capacity_bytes=4096,
            max_offloads=1,
        )
        worker_started = threading.Event()
        release_worker = threading.Event()

        def blocked_large_operation() -> str:
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("test worker was not released")
            return "large"

        large = asyncio.create_task(governor.run_cpu(10_000, blocked_large_operation))
        while not worker_started.is_set():
            await asyncio.sleep(0)

        started = asyncio.get_running_loop().time()
        assert (
            await asyncio.wait_for(
                governor.run_cpu(1, lambda: "small"),
                timeout=0.25,
            )
            == "small"
        )
        assert asyncio.get_running_loop().time() - started < 0.25

        large.cancel("operator interruption")
        await asyncio.sleep(0)
        assert not large.done()
        assert governor.snapshot().active_publication_bytes == 1024

        release_worker.set()
        with pytest.raises(asyncio.CancelledError, match="operator interruption"):
            await large
        metrics = governor.snapshot()
        assert metrics.active_publication_bytes == 0
        assert metrics.maximum_active_publication_bytes <= metrics.configured_capacity_bytes

    asyncio.run(scenario())


def test_hundred_session_publication_stress_has_bounded_active_work() -> None:
    async def scenario() -> tuple[float, int, object]:
        slice_bytes = 64 * 1024
        governor = ToolTerminalPublicationGovernor(
            slice_bytes=slice_bytes,
            capacity_bytes=4 * slice_bytes,
            max_offloads=4,
        )
        payload = "x" * (slice_bytes * 2)
        heartbeat_ticks = 0
        draining = True

        async def heartbeat() -> None:
            nonlocal heartbeat_ticks
            while draining:
                heartbeat_ticks += 1
                await asyncio.sleep(0)

        async def publish(session_index: int) -> None:
            effect_time = datetime.now(UTC) - timedelta(seconds=1)
            event_id = f"terminal-{session_index}"
            governor.stage(
                session_id=f"session-{session_index}",
                event_id=event_id,
                payload_bytes=len(payload),
                effect_completed_at=effect_time,
            )
            for _call_index in range(2):
                assert await governor.run_cpu(len(payload), lambda: payload.encode("utf-8"))
            governor.published(
                session_id=f"session-{session_index}",
                event_id=event_id,
                published_at=datetime.now(UTC),
            )

        heartbeat_task = asyncio.create_task(heartbeat())
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                asyncio.gather(*(publish(index) for index in range(100))),
                timeout=10,
            )
        finally:
            draining = False
            await heartbeat_task
        return time.monotonic() - started, heartbeat_ticks, governor.snapshot()

    elapsed, heartbeat_ticks, metrics = asyncio.run(scenario())

    # The stress ceiling is deliberately generous for shared CI machines. The
    # material invariant is the 256 KiB active publication working-set bound;
    # 100 sessions and 200 oversized validations must not suppress loop work.
    assert elapsed < 10
    assert heartbeat_ticks >= 2
    assert metrics.publication_count == 100
    assert metrics.staged_count == 0
    assert metrics.oversized_offloads == 200
    assert metrics.maximum_active_publication_bytes <= 256 * 1024


def test_round_reservation_covers_all_sibling_stages_without_post_effect_wait() -> None:
    async def scenario() -> None:
        governor = ToolTerminalPublicationGovernor(
            slice_bytes=16,
            capacity_bytes=64,
            staged_capacity_bytes=100,
        )
        effect_time = datetime.now(UTC)
        await governor.reserve_round(
            session_id="session-a",
            tool_round_id="round-a",
            maximum_bytes=80,
        )
        competing = asyncio.create_task(
            governor.reserve_round(
                session_id="session-b",
                tool_round_id="round-b",
                maximum_bytes=40,
            )
        )
        await asyncio.sleep(0)
        assert not competing.done()

        # Both terminals fit the lease acquired before effects. Staging the
        # second sibling is synchronous and therefore cannot wait behind the
        # first sibling that still needs whole-round publication.
        for index in range(2):
            governor.stage(
                session_id="session-a",
                tool_round_id="round-a",
                event_id=f"terminal-a-{index}",
                payload_bytes=40,
                effect_completed_at=effect_time,
            )
        assert governor.snapshot().staged_bytes == 80

        for index in range(2):
            governor.published(
                session_id="session-a",
                event_id=f"terminal-a-{index}",
                published_at=datetime.now(UTC),
            )
        governor.release_round(session_id="session-a", tool_round_id="round-a")
        await asyncio.wait_for(competing, timeout=0.25)
        governor.release_round(session_id="session-b", tool_round_id="round-b")

        metrics = governor.snapshot()
        assert metrics.staged_count == 0
        assert metrics.active_round_reservations == 0
        assert metrics.maximum_reserved_round_bytes <= 100

    asyncio.run(scenario())


def test_unbounded_round_is_exclusive_and_declared_round_rejects_overflow() -> None:
    async def scenario() -> None:
        governor = ToolTerminalPublicationGovernor(staged_capacity_bytes=100)
        await governor.reserve_round(
            session_id="session-unbounded",
            tool_round_id="round-unbounded",
            maximum_bytes=None,
        )
        bounded = asyncio.create_task(
            governor.reserve_round(
                session_id="session-bounded",
                tool_round_id="round-bounded",
                maximum_bytes=50,
            )
        )
        await asyncio.sleep(0)
        assert not bounded.done()
        assert governor.snapshot().active_exclusive_rounds == 1

        governor.release_round(
            session_id="session-unbounded",
            tool_round_id="round-unbounded",
        )
        await asyncio.wait_for(bounded, timeout=0.25)
        governor.stage(
            session_id="session-bounded",
            tool_round_id="round-bounded",
            event_id="terminal-in-bound",
            payload_bytes=40,
            effect_completed_at=datetime.now(UTC),
        )
        with pytest.raises(RuntimeError, match="exceeds its declared round"):
            governor.stage(
                session_id="session-bounded",
                tool_round_id="round-bounded",
                event_id="terminal-over-bound",
                payload_bytes=11,
                effect_completed_at=datetime.now(UTC),
            )
        governor.published(
            session_id="session-bounded",
            event_id="terminal-in-bound",
            published_at=datetime.now(UTC),
        )
        governor.release_round(
            session_id="session-bounded",
            tool_round_id="round-bounded",
        )

    asyncio.run(scenario())


def test_durable_stage_reconciliation_replaces_payload_accounting() -> None:
    async def scenario() -> None:
        governor = ToolTerminalPublicationGovernor(staged_capacity_bytes=100)
        effect_time = datetime.now(UTC)
        await governor.reserve_round(
            session_id="session-reconcile",
            tool_round_id="round-reconcile",
            maximum_bytes=80,
        )
        governor.stage(
            session_id="session-reconcile",
            tool_round_id="round-reconcile",
            event_id="terminal-reconcile",
            payload_bytes=20,
            effect_completed_at=effect_time,
        )

        governor.reconcile_stage(
            session_id="session-reconcile",
            tool_round_id="round-reconcile",
            event_id="terminal-reconcile",
            payload_bytes=60,
            effect_completed_at=effect_time,
        )

        assert governor.snapshot().staged_bytes == 60
        with pytest.raises(RuntimeError, match="exceeds its declared round"):
            governor.reconcile_stage(
                session_id="session-reconcile",
                tool_round_id="round-reconcile",
                event_id="terminal-reconcile",
                payload_bytes=81,
                effect_completed_at=effect_time,
            )
        governor.published(
            session_id="session-reconcile",
            event_id="terminal-reconcile",
            published_at=datetime.now(UTC),
        )
        governor.release_round(
            session_id="session-reconcile",
            tool_round_id="round-reconcile",
        )
        assert governor.snapshot().staged_bytes == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "event_type",
    (
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    ),
)
def test_terminal_publication_timing_requires_exact_runtime_provenance(
    event_type: EventType,
) -> None:
    effect = datetime(2026, 9, 3, 10, 0, tzinfo=UTC)
    staged = effect + timedelta(seconds=2)
    publication = staged + timedelta(seconds=3)
    payload = {
        "tool_call_id": "call-timing",
        "result": {"content": "ok", "is_error": event_type is not EventType.TOOL_CALL_COMPLETED},
        "arguments_state": "unavailable",
        "arguments_exact": False,
        _TIMING_FIELDS[0]: effect.isoformat(),
        _TIMING_FIELDS[1]: staged.isoformat(),
        _TIMING_FIELDS[2]: publication.isoformat(),
    }
    event = Event(
        type=event_type,
        session_id="session-timing",
        timestamp=publication,
        payload=payload,
    )
    untrusted = prepare_new_runtime_event(event, redactor=SecretRedactor())
    assert not set(_TIMING_FIELDS) & set(untrusted.payload)

    attested = event_with_runtime_payload_authority(event, *_TIMING_FIELDS)
    prepared = prepare_new_runtime_event(
        attested,
        redactor=SecretRedactor(["tool_effect_completed_at", publication.isoformat()]),
    )
    assert tuple(prepared.payload[field] for field in _TIMING_FIELDS) == (
        effect.isoformat(),
        staged.isoformat(),
        publication.isoformat(),
    )

    partial_payload = dict(payload)
    partial_payload.pop(_TIMING_FIELDS[2])
    partial = event_with_runtime_payload_authority(
        event.model_copy(update={"payload": partial_payload}),
        _TIMING_FIELDS[0],
        _TIMING_FIELDS[1],
    )
    with pytest.raises(ValueError, match="complete together"):
        prepare_new_runtime_event(partial, redactor=SecretRedactor())
