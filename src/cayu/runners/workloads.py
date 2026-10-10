"""Exact workload identities shipped for admitted runners."""

from cayu.runners.base import RunnerWorkloadAuthority

BROWSER_FETCH_WORKLOAD_NAME = "cayu.browser-fetch"
PINNED_BROWSER_FETCH_IMAGE = "cayu-browser-fetch:19-playwright-1.62.0"
PINNED_BROWSER_FETCH_WORKLOAD = RunnerWorkloadAuthority(
    name=BROWSER_FETCH_WORKLOAD_NAME,
    image=PINNED_BROWSER_FETCH_IMAGE,
    command=(
        "/usr/local/bin/python",
        "-I",
        "/opt/cayu-browser/worker.py",
    ),
    protocol_version="cayu.browser-fetch.v4",
    worker_version="4",
    component_versions=(("playwright", "1.62.0"),),
)

BROWSER_SESSION_WORKLOAD_NAME = "cayu.browser-session"
PINNED_BROWSER_SESSION_IMAGE = "cayu-browser-fetch:19-playwright-1.62.0"
PINNED_BROWSER_SESSION_WORKLOAD = RunnerWorkloadAuthority(
    name=BROWSER_SESSION_WORKLOAD_NAME,
    image=PINNED_BROWSER_SESSION_IMAGE,
    command=(
        "/usr/local/bin/python",
        "-I",
        "/opt/cayu-browser/worker.py",
    ),
    protocol_version="cayu.browser-session.v4",
    worker_version="19",
    component_versions=(
        ("playwright", "1.62.0"),
        ("browser", "chromium"),
    ),
)

BROWSER_WORKER_DIRECTORY = "/opt/cayu-browser"
# Guest file name -> packaged source module. Every browser image installs these
# exact sources, root-owned and read-only, under ``BROWSER_WORKER_DIRECTORY``.
BROWSER_WORKER_FILES = (
    ("worker.py", "_browser_guest.py"),
    ("_browser_visual_guest.py", "_browser_visual_guest.py"),
    ("_browser_control_guest.py", "_browser_control_guest.py"),
    ("_browser_recording_guest.py", "_browser_recording_guest.py"),
    ("_browser_control_transport.py", "_browser_control_transport.py"),
)
BROWSER_WORKER_PLAYWRIGHT_VERSION = "1.62.0"
BROWSER_WORKER_WEBSOCKETS_VERSION = "17.0.1"
BROWSER_WORKER_PLAYWRIGHT_BROWSERS_PATH = "/ms-playwright"


def browser_worker_sources() -> dict[str, bytes]:
    """Return the installed worker sources keyed by their guest file names."""

    from importlib.resources import files

    package = files("cayu.tools")
    return {
        guest_name: package.joinpath(source_name).read_bytes()
        for guest_name, source_name in BROWSER_WORKER_FILES
    }


def browser_worker_source_digests() -> dict[str, str]:
    """Return the sha256 of each installed worker source keyed by guest file name.

    Images are built from these same packaged sources, so a guest whose worker
    files hash differently was built for a different Cayu release.
    """

    from hashlib import sha256

    return {name: sha256(content).hexdigest() for name, content in browser_worker_sources().items()}
