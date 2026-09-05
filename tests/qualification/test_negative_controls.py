"""Seed broken fixture behavior and require the same qualification assertions to reject it."""

import asyncio

import pytest

from tests.qualification import test_capacity


@pytest.mark.parametrize("fault", ["authority-loss", "blind-retry", "cleanup-leak", "hot-polling"])
def test_seeded_capacity_regression_is_rejected(tmp_path, monkeypatch, fault):
    monkeypatch.setenv("CAYU_QUALIFICATION_FAULT", fault)
    target = (
        test_capacity.test_empty_worker_pressure_preserves_control_latency
        if fault == "hot-polling"
        else test_capacity.test_durable_concurrent_long_trajectories
    )
    with pytest.raises(AssertionError):
        if fault == "hot-polling":
            target(tmp_path, record_property=lambda *args: None)
        else:
            target(tmp_path, record_property=lambda *args: None)


def test_seeded_semantic_stall_acceptance_is_rejected(monkeypatch):
    from cayu.providers.deadlines import ProviderDeadlineKind, ProviderStreamDeadlineController
    from tests.core.test_provider_stream_deadlines import (
        test_normalized_noop_events_do_not_refresh_semantic_progress,
    )

    original = ProviderStreamDeadlineController._deadline_at

    def ignore_semantic_stall(self, kind):
        if kind is ProviderDeadlineKind.SEMANTIC_IDLE:
            return float("inf")
        return original(self, kind)

    monkeypatch.setattr(ProviderStreamDeadlineController, "_deadline_at", ignore_semantic_stall)
    with pytest.raises(AssertionError):
        asyncio.run(test_normalized_noop_events_do_not_refresh_semantic_progress())
