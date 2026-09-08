"""Audit identity and safe-origin value contract, before native wiring."""

import pytest
from pydantic import ValidationError
from tests.core.test_browser_control import identity, request

from cayu.runtime.browser_control import (
    BrowserControlPageAudit,
    BrowserControlRecord,
    BrowserObservedPageLocation,
    request_browser_takeover,
)


def acquired_record():
    intent = request()
    audit = BrowserControlPageAudit(
        request_id=intent.request_id,
        phase="acquired",
        control_epoch=2,
        locations=(BrowserObservedPageLocation(page=intent.pages[0], origin="https://site.test"),),
    )
    return BrowserControlRecord(
        identity=identity(),
        request=intent,
        revision=3,
        control_epoch=2,
        state="operator_controlled",
        lease_until_ms=2000,
        acquisition_audit=audit,
    )


def test_audit_roundtrip_and_empty_handback_preserve_exact_interval():
    acquired = acquired_record()
    assert BrowserControlRecord.model_validate_json(acquired.model_dump_json()) == acquired
    after = BrowserControlPageAudit(
        request_id=request().request_id,
        phase="handed_back",
        control_epoch=3,
        locations=(),
    )
    closed_pages = acquired.model_copy(
        update={
            "state": "agent_controlled",
            "control_epoch": 3,
            "revision": 5,
            "lease_until_ms": None,
            "fresh_observation_required": True,
            "handback_audit": after,
        }
    )
    assert closed_pages.handback_audit is not None
    assert closed_pages.handback_audit.locations == ()
    assert BrowserControlRecord.model_validate_json(closed_pages.model_dump_json()) == closed_pages
    next_request = request().model_copy(
        update={
            "request_id": "bt_" + "2" * 32,
            "expected_record_revision": closed_pages.revision,
            "expected_control_epoch": closed_pages.control_epoch,
        }
    )
    next_interval = request_browser_takeover(closed_pages, next_request, now_ms=1000)
    assert next_interval.acquisition_audit is None and next_interval.handback_audit is None
    assert closed_pages.acquisition_audit is not None and closed_pages.handback_audit == after


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "bt_" + "2" * 32},
        {"phase": "handed_back"},
        {"control_epoch": 3},
    ],
)
def test_audit_cannot_be_rebound_under_same_record(change):
    original = acquired_record()
    assert original.acquisition_audit is not None
    altered = original.acquisition_audit.model_copy(update=change)
    with pytest.raises(ValidationError):
        original.model_copy(update={"acquisition_audit": altered})


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_id", "different"),
        ("revision", "different"),
        ("control_epoch", 2),
    ],
)
def test_acquisition_audit_authenticates_complete_page_tuple(field, value):
    original = acquired_record()
    assert original.acquisition_audit is not None
    location = original.acquisition_audit.locations[0]
    altered = location.model_copy(update={"page": location.page.model_copy(update={field: value})})
    audit = original.acquisition_audit.model_copy(update={"locations": (altered,)})
    with pytest.raises(ValidationError):
        original.model_copy(update={"acquisition_audit": audit})


@pytest.mark.parametrize(
    "origin", ["https://site.test/private-canary", "https://private-canary@site.test"]
)
def test_audit_rejects_non_origin_data_without_echo(origin, recwarn, caplog, capsys):
    with pytest.raises(ValidationError) as raised:
        BrowserControlPageAudit.model_validate(
            {
                "request_id": request().request_id,
                "phase": "acquired",
                "control_epoch": 2,
                "locations": ({"page": request().pages[0], "origin": origin},),
            }
        )
    captured = capsys.readouterr()
    assert "private-canary" not in str(raised.value) + repr(raised.value)
    assert "private-canary" not in captured.out + captured.err + caplog.text
    assert not recwarn
