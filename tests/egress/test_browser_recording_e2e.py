"""Real Docker lifecycle, restart and dashboard playback qualification."""

import asyncio
import os

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_RECORDING_BROWSER_ACCEPTANCE") != "1",
        reason="Requires recording demo/browser images, built dashboard, host FFmpeg and Chromium.",
    ),
]


@pytest.mark.parametrize(
    "explicit_close,restart,enabled",
    [
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, False, False),
    ],
)
def test_recording_lifecycle_and_playback(tmp_path, explicit_close, restart, enabled):
    from examples.browser_recording.run import run

    receipt = asyncio.run(
        run(tmp_path / "recording", explicit_close=explicit_close, restart=restart, enabled=enabled)
    )
    assert receipt["no_live_viewer"] and receipt["two_completed_invocations"]
    assert len(receipt["recording_ids"]) == (2 if enabled else 0)
    assert receipt["dashboard_playback"] is enabled
    assert receipt["worker_restart"] is restart
