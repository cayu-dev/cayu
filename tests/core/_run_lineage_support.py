"""Exact ownership assertions for root metadata in session store regressions."""

from cayu.sessions._run_identity import RUN_LINEAGE_METADATA_KEY
from cayu.sessions.records import Session


def assert_root_run_lineage(session: Session) -> None:
    marker = session.metadata[RUN_LINEAGE_METADATA_KEY]
    assert marker == {
        "is_root": True,
        "root_run_id": None,
        "parent_run_id": None,
        "parent_session_id": None,
    }
    assert marker["is_root"] is True
