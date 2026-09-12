"""Public API/subscription hosted-search controls for the observed reason classes."""

import json
from copy import deepcopy

import pytest
from tests.core import test_openai_search_ordering as search
from tests.providers._responses_ordering import run_ordering_attempts

from cayu import EventType


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
@pytest.mark.parametrize("case", ["orphan", "completed_added", "missing_status", "shifted"])
async def test_hosted_search_invalid_registration_keeps_typed_retry_bound(tmp_path, adapter, case):
    raw = [search.created(), search.lifecycle()]
    reason = "web_search_lifecycle_arrived_before_output_item_added"
    if case in {"completed_added", "missing_status"}:
        item = deepcopy(search.done()["item"])
        if case == "missing_status":
            del item["status"]
        raw = [search.created(), {**search.added(), "item": item}]
        reason = (
            "web_search_call_output_item_added_must_be_in_progress"
            if case == "completed_added"
            else "web_search_call_item_has_unsupported_status"
        )
    elif case == "shifted":
        raw = [
            search.created(),
            search.added(),
            {**search.lifecycle(1), "item_id": search.added()["item"]["id"]},
        ]
    events, durable, calls = await run_ordering_attempts(tmp_path, adapter, [raw, raw])
    assert not calls
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    starts = [e.payload for e in durable if e.type == EventType.MODEL_STARTED]
    assert len(errors) == len(starts) == 2
    assert len({e["model_attempt_id"] for e in starts}) == 2
    for error, start in zip(errors, starts, strict=True):
        assert error["provider_protocol_reason"] == reason
        assert error["model_attempt_id"] == start["model_attempt_id"]
        assert (
            error["provider_protocol_native_structure"]
            == error["provider_protocol_transport_structure"]
        )
        rows = json.loads(error["provider_protocol_native_structure"])
        assert rows[0][0] == 1
        assert error["provider_protocol_native_structure_truncated"] == 0
        if case == "shifted":
            assert rows[1][3] == 0 and rows[2][3] == 1
            assert rows[1][-1] == rows[2][-1] != 0
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"
    assert errors[-1]["effective_max_attempts"] == 2
    assert events[-1].type == EventType.SESSION_FAILED
