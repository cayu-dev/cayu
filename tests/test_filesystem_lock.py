from pathlib import Path

import pytest

from cayu._filesystem_lock import cooperative_path_lock


def test_nonblocking_contention_preserves_owner_and_releases_descriptor(tmp_path: Path) -> None:
    def lock():
        return cooperative_path_lock(
            tmp_path,
            "allocation",
            lock_directory_name="cayu-test-allocation-locks",
            blocking=False,
        )

    with lock():
        for _ in range(3):
            with pytest.raises(BlockingIOError), lock():
                raise AssertionError("Contender entered another owner's lock")
    with lock():
        pass
