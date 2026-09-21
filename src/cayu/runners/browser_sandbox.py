"""Versioned Docker prerequisite for Cayu's sandboxed Chromium workloads."""

from pathlib import Path


def browser_seccomp_profile() -> str:
    """Return the installed v1 Chromium sandbox seccomp profile's host path.

    Pass this to ``DockerEgressAdapter(seccomp_profile=...)`` for an explicitly
    configured compatible image. Pinned Cayu browser images select it by default.
    Docker consumes this file on the client host; it is not a guest mount.
    """
    return str(Path(__file__).resolve().parents[1] / "data" / "browser-seccomp-v1.json")
