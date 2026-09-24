"""Private pruning cursors are bounded native evidence, not request authority."""

import pytest
from tests.core.test_clarification_contracts import OWNER, operation

from cayu.collaboration._clarification_records import clarification_record_projection
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._request_pruning import RequestPruningProgress
from cayu.collaboration.requests import RequestRef


@pytest.mark.parametrize("position", [1, 32, 127])
def test_pruning_cursor_reconstruction_binds_native_key_and_scope(position):
    cursor = RequestPruningProgress(
        operation=operation("request"),
        request=RequestRef(owner=OWNER, request_id="request", incarnation="one"),
        snapshot_sha256="a" * 64,
        clarification_sha256="b" * 64,
        clarification_events=(),
        next_event_index=position,
    )
    key = ("namespace", 1, "request")
    restored, projection = clarification_record_projection(
        "request_pruning", cursor.model_dump(mode="json"), scope="app", key=key
    )
    assert restored == cursor and projection == ()
    for wrong_key in (("namespace", True, "request"), ("namespace", 1, "other")):
        with pytest.raises(CollaborationContractError):
            clarification_record_projection("request_pruning", cursor, scope="app", key=wrong_key)
    with pytest.raises(CollaborationContractError):
        clarification_record_projection("request_pruning", cursor, scope="other", key=key)
    for changes in (
        {"schema_version": True},
        {"schema_version": 2},
        {"next_event_index": True},
        {"next_event_index": 0},
        {"next_event_index": 128},
        {"clarification_events": (True,)},
        {"clarification_events": (0,)},
        {"clarification_events": (2, 1)},
        {"clarification_events": (1, 1)},
        {"clarification_events": tuple(range(1, 66))},
        {"snapshot_sha256": "bad"},
        {"operation": operation("request").model_copy(update={"application_scope": "other"})},
    ):
        with pytest.raises(CollaborationContractError):
            clarification_record_projection(
                "request_pruning", cursor.model_copy(update=changes), scope="app", key=key
            )
