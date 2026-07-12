from __future__ import annotations

import json
import subprocess
import sys

from tests.egress_e2e_support import (
    bloom_maybe_contains,
    recursive_window_bloom_script,
)


def test_guest_bloom_scan_detects_secret_without_receiving_it(tmp_path) -> None:
    secret = b"sk_test_secret_only_known_to_host"
    absent = b"sk_test_different_secret_value"
    (tmp_path / "credentials.txt").write_bytes(b"prefix:" + secret + b":suffix")
    script = recursive_window_bloom_script(
        roots=(str(tmp_path),),
        window_size=len(secret),
    )

    assert secret.decode() not in script
    assert secret.hex() not in script

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["files_scanned"] == 1
    assert bloom_maybe_contains(result["bloom"], secret)
    assert not bloom_maybe_contains(result["bloom"], absent)
