"""Characterize exact maintenance readback; public journeys qualify the owners."""

from types import SimpleNamespace

import pytest
from tests.core.test_collaboration_host_output_selection import operation, snapshot

from cayu.collaboration import _host_producer_reconciliation as reconciliation
from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.collaboration._host_producer_maintenance import HostProducerMaintenance
from cayu.collaboration._producer_inspection import ProducerOutputInspection
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "action,values,updates,settled",
    [
        ("retain_completion", {}, {}, True),
        ("retain_completion", {}, {"completion": None}, False),
        ("export", {"export": "prepared"}, {}, False),
        ("export", {"export": "published"}, {}, True),
        ("export", {"export": "rejected"}, {}, True),
        ("publish_answer", {"request_state": "answered"}, {}, True),
        (
            "publish_answer",
            {"request_state": "answered"},
            {"answer_destination": operation("destination")},
            True,
        ),
        ("publish_failure", {"request_state": "failed"}, {}, True),
        ("publish_failure", {"request_state": "cancelled"}, {}, True),
        ("publish_answer", {"request_state": "open"}, {}, False),
        ("publish_failure", {"request_state": "open"}, {}, False),
        ("publish_answer", {"request_state": "expired"}, {}, True),
        ("publish_failure", {"request_state": "declined"}, {}, True),
        ("deliver", {"delivery": "pending"}, {}, False),
        ("deliver", {"delivery": "appended"}, {}, True),
        ("deliver", {"delivery": "excluded"}, {}, True),
        ("exclude", {"delivery": "appended"}, {}, True),
        ("deliver", {"delivery": None}, {}, False),
        ("exclude", {"delivery": "pending"}, {}, False),
        ("exclude", {"delivery": "excluded"}, {}, True),
        ("retire", {"cleanup": "released"}, {}, False),
        ("retire", {"cleanup": "retired"}, {}, True),
        ("retire", {"cleanup": "excluded"}, {}, True),
        ("release_export", {"cleanup": "retired"}, {}, False),
        ("release_export", {"cleanup": "released"}, {}, True),
        ("settle", {}, {"cleanup": operation("cleanup")}, False),
        ("settle", {}, {"cleanup_ack": operation("ack")}, True),
        ("service_disposition", {}, {"cleanup": operation("cleanup")}, False),
        ("service_disposition", {}, {"cleanup_ack": operation("ack")}, True),
    ],
)
async def test_reconciliation_requires_effect_specific_positive_evidence(
    monkeypatch, action, values, updates, settled
):
    from cayu.collaboration._host_producer_maintenance import _DESTINATION_ACTIONS

    observed = snapshot(**values).model_copy(update=updates)
    intent = HostProducerMaintenance(
        recovery=observed.recovery,
        action=action,
        destination=operation("destination") if action in _DESTINATION_ACTIONS else None,
    )

    async def inspect(app, recovery, *, context, wait_for_settlement):
        assert recovery == observed.recovery and wait_for_settlement is True
        return ExactMatch[ProducerOutputInspection](receipt=observed)

    monkeypatch.setattr(reconciliation, "inspect_producer_output", inspect)
    result = await reconciliation.reconcile_maintenance(
        SimpleNamespace(_secret_redactor=SecretRedactor()), intent, context=object()
    )
    assert (result is not None) is settled
    if settled:
        assert result == observed


@pytest.mark.parametrize("outcome", [ExactNotFound(), ExactConflict(), ExactUnavailable()])
async def test_missing_or_ambiguous_readback_never_releases_a_turn(monkeypatch, outcome):
    observed = snapshot(export="published")
    intent = HostProducerMaintenance(
        recovery=observed.recovery, action="export", destination=operation("destination")
    )

    async def inspect(*args, **kwargs):
        return outcome

    monkeypatch.setattr(reconciliation, "inspect_producer_output", inspect)
    assert await reconciliation.reconcile_maintenance(object(), intent, context=object()) is None
