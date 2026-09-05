"""Explicit ownership for the standalone visual-guard test container."""

from __future__ import annotations

import secrets
import subprocess


def run_visual_container(arguments: list[str], *, timeout: float = 45) -> None:
    name = "cayu-visual-guard-" + secrets.token_hex(16)
    primary: BaseException | None = None
    try:
        subprocess.run(
            ["docker", "run", "--name", name, *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        # Killing the attached CLI does not stop Chromium in the daemon-owned
        # container. Address only this invocation's unguessable container name.
        try:
            subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
        except BaseException as cleanup_failure:
            if not isinstance(cleanup_failure, Exception):
                raise cleanup_failure from primary
            if primary is not None:
                raise primary from cleanup_failure
            raise
