"""Exact workload identities shipped for admitted runners."""

from cayu.runners.base import RunnerWorkloadAuthority

BROWSER_FETCH_WORKLOAD_NAME = "cayu.browser-fetch"
PINNED_BROWSER_FETCH_IMAGE = "cayu-browser-fetch:5-playwright-1.62.0"
PINNED_BROWSER_FETCH_WORKLOAD = RunnerWorkloadAuthority(
    name=BROWSER_FETCH_WORKLOAD_NAME,
    image=PINNED_BROWSER_FETCH_IMAGE,
    command=(
        "/usr/local/bin/python",
        "-I",
        "/opt/cayu-browser/worker.py",
    ),
    protocol_version="cayu.browser-fetch.v3",
    worker_version="3",
    component_versions=(("playwright", "1.62.0"),),
)

BROWSER_SESSION_WORKLOAD_NAME = "cayu.browser-session"
PINNED_BROWSER_SESSION_IMAGE = PINNED_BROWSER_FETCH_IMAGE
PINNED_BROWSER_SESSION_WORKLOAD = RunnerWorkloadAuthority(
    name=BROWSER_SESSION_WORKLOAD_NAME,
    image=PINNED_BROWSER_SESSION_IMAGE,
    command=(
        "/usr/local/bin/python",
        "-I",
        "/opt/cayu-browser/worker.py",
    ),
    protocol_version="cayu.browser-session.v1",
    worker_version="5",
    component_versions=(
        ("playwright", "1.62.0"),
        ("browser", "chromium"),
    ),
)
