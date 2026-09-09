# Record a public Docker browser run

This synthetic example records a changing public page without opening a live
viewer. It completes two invocations of the same session, retrieves and decodes
their distinct final videos, and preserves the first recording. No provider key
or external website is needed.

From the repository root, with Docker running and FFmpeg installed on the host:

```sh
uv sync --extra dev --extra server --extra browser --extra recording
docker build -f examples/browser_fetch/Dockerfile -t cayu-browser-fetch:13-playwright-1.62.0 .
docker build -f examples/browser_recording/Dockerfile -t cayu-browser-recording:local .
uv run python -m examples.browser_recording.run /tmp/cayu-recording-demo
```

Choose a new private state directory for each run. Add `--explicit-close` to test
browser-close finalization, `--restart` to verify a retained human-input pause
across application-worker process loss, or `--disabled` to verify that no media is published.
The example removes its Docker allocations and application container. Its private
state directory retains recordings and the qualification receipt for inspection;
delete that directory when finished.

`app.py` demonstrates application-owned consent, the private store, Docker
configuration, separate playback authorization, and the bundled dashboard player.
The recording policy cannot be set by the model. The example credential file is
application-private and must never be put in an agent workspace or published.

See the [recording contract](../../docs/browser-recording.md) for exclusions,
partial coverage, backend requirements, recovery, limits and retention duties.

The core consent setup is deliberately separate from model arguments:

```python
from cayu import BrowserRecordingPolicy, BrowserRecordingStore, BrowserSessionTool

store = BrowserRecordingStore("/private/application/recordings.sqlite")
recording = await store.authorize_capture(
    session_id="selected-session",
    policy=BrowserRecordingPolicy(
        scope="selected-run", allowed_origins=("https://public.example.test",),
        retention_seconds=3600,
    ),
    guest_endpoint="wss://cayu-control:8443/api/browser-recordings/guest",
)
tool = BrowserSessionTool(recording=recording, expected_runner_candidate="docker")
```

The application must also attach `BrowserRecordingServer` with its own operator
access policy, as shown in `app.py`.
