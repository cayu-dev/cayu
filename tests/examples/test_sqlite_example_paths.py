from pathlib import Path

import pytest

from examples import dashboard_knowledge_review, dashboard_pending_actions, server_example


@pytest.mark.parametrize(
    ("workspace", "data_dir"),
    [
        (server_example.WORKSPACE, server_example.DATA_DIR),
        (dashboard_knowledge_review.WORKSPACE, dashboard_knowledge_review.DATA_DIR),
        (dashboard_pending_actions.WORKSPACE, dashboard_pending_actions.DATA_DIR),
    ],
)
def test_sqlite_examples_use_canonical_data_directory(
    workspace: Path,
    data_dir: Path,
) -> None:
    assert data_dir == workspace / "data"
