"""agent-browser commands plus host fallback.

Browser work runs through the ToolContext runner when one is configured, isolating Chromium
from the host. Otherwise, including in the default scheduled refresh workflow, it runs as a
host subprocess. Agent-facing tools build commands here and execute them through the runner;
see ``agent_tools.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess

# `agent-browser open` already blocks until the page has loaded (~2s cold, <1s warm) and the page
# is fully readable immediately after — so NO extra wait is needed. A `wait --load load`/`networkidle`
# here is a ~25-30s no-op: it waits for the NEXT load/idle event, which never comes, so it blocks to
# the timeout. One short settle covers any late JS paint without risking that hang.
_PREP = "{browser} open {u} >/dev/null 2>&1 && {browser} wait 1200 >/dev/null 2>&1"
_HOST_BROWSER_ENV_KEYS = frozenset(
    {
        "CI",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_PATH",
        "PATH",
        "PLAYWRIGHT_BROWSERS_PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    }
)
_MAX_SUBPROCESS_ERROR_CHARS = 1_000


def _browser(*, session: str, allowed_hosts: frozenset[str]) -> str:
    if not session:
        raise ValueError("browser session must not be empty")
    if not allowed_hosts:
        raise ValueError("browser allowlist must not be empty")
    domains = ",".join(sorted(allowed_hosts))
    return (
        f"agent-browser --session {shlex.quote(session)} --allowed-domains {shlex.quote(domains)}"
    )


def open_page_command(url: str, *, session: str, allowed_hosts: frozenset[str]) -> str:
    """Open and settle a page without reading or capturing it yet."""

    executable = _browser(session=session, allowed_hosts=allowed_hosts)
    return _PREP.format(browser=executable, u=shlex.quote(url))


def snapshot_command(*, session: str, allowed_hosts: frozenset[str]) -> str:
    """Return the accessibility tree for the already-open page."""

    return f"{_browser(session=session, allowed_hosts=allowed_hosts)} snapshot"


def read_page_command(url: str, *, session: str, allowed_hosts: frozenset[str]) -> str:
    """Open the URL and return its ACCESSIBILITY TREE (snapshot), not flattened body text. The
    a11y tree preserves table rows/cells and which tab/radio is selected — flattened text loses
    that, which is what makes the agent cross columns on dense pricing tables (read Batch as a
    column, a tier threshold as the context window, etc.)."""
    return (
        open_page_command(url, session=session, allowed_hosts=allowed_hosts)
        + " && "
        + (snapshot_command(session=session, allowed_hosts=allowed_hosts))
    )


def click_command(ref: str, *, session: str, allowed_hosts: frozenset[str]) -> str:
    """Click a validated accessibility-tree reference and allow the page to repaint."""

    if not ref or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in ref
    ):
        raise ValueError("browser refs must contain only lowercase letters, digits, '_' or '-'")
    executable = _browser(session=session, allowed_hosts=allowed_hosts)
    return f"{executable} click @{ref} && {executable} wait 500 >/dev/null 2>&1"


def text_body_command(*, session: str, allowed_hosts: frozenset[str]) -> str:
    """Flattened visible text of the ALREADY-OPEN page (no re-open) — the fallback when the a11y
    snapshot came back empty."""
    return f"{_browser(session=session, allowed_hosts=allowed_hosts)} get text body"


def current_url_command(*, session: str, allowed_hosts: frozenset[str]) -> str:
    """Return the effective URL after redirects or interactive navigation."""

    return f"{_browser(session=session, allowed_hosts=allowed_hosts)} get url"


def screenshot_command(url: str, path: str, *, session: str, allowed_hosts: frozenset[str]) -> str:
    """Open the URL and capture a full-page screenshot to `path` (PNG)."""
    return (
        open_page_command(url, session=session, allowed_hosts=allowed_hosts)
        + " && "
        + (screenshot_current_page_command(path, session=session, allowed_hosts=allowed_hosts))
    )


def screenshot_current_page_command(
    path: str, *, session: str, allowed_hosts: frozenset[str]
) -> str:
    """Capture the already-open page without resetting interactive pricing controls."""

    return (
        f"{_browser(session=session, allowed_hosts=allowed_hosts)} screenshot --full "
        f"{shlex.quote(path)} >/dev/null 2>&1 && "
        f"echo {shlex.quote(path)}"
    )


def read_file_b64_command(path: str) -> str:
    """Base64-encode a file so its bytes can be pulled out of the sandbox over exec stdout."""
    return f"base64 -w0 {shlex.quote(path)}"


def run_bash_host(command: str, *, timeout: int = 120) -> str:
    environment = {key: value for key, value in os.environ.items() if key in _HOST_BROWSER_ENV_KEYS}
    completed = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if completed.returncode != 0:
        stderr = " ".join(completed.stderr.split())[:_MAX_SUBPROCESS_ERROR_CHARS]
        detail = stderr or "no stderr was produced"
        raise RuntimeError(f"browser command failed with exit {completed.returncode}: {detail}")
    return completed.stdout
