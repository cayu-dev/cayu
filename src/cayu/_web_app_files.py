"""What a built web app served at ``/`` must hold, checked without the server extra.

``cayu.server.web_app`` serves it; ``cayu check`` checks it with the same rules.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

# Paths the web app never answers, whether or not a route is there: Cayu's API, the
# Developer dashboard, and what Cayu Cloud's Agent router serves itself.
RESERVED_PREFIXES = ("/api", "/cayu", "/.cayu")
# Where a build keeps its content-named files.
ASSETS = "assets"
PAGE = "index.html"


class WebAppError(ValueError):
    """The directory can't be served as a web app at ``/``: it is ``missing`` (or has no
    ``index.html``), or it ``shadows`` a reserved prefix."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def check_web_app_directory(directory: str | os.PathLike[str], reserved: Iterable[str]) -> Path:
    """``directory``, resolved, once it holds an ``index.html`` and nothing at its top
    level shadows a reserved prefix (a file or folder named ``api``, say, could never be
    served). Raises :class:`WebAppError` otherwise."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise WebAppError(f"The web app directory {directory} doesn't exist.", reason="missing")
    if not (root / PAGE).is_file():
        raise WebAppError(f"The web app directory {directory} has no {PAGE}.", reason="missing")
    prefixes = set(reserved)
    shadowed = sorted(f"/{entry.name}" for entry in root.iterdir() if f"/{entry.name}" in prefixes)
    if shadowed:
        raise WebAppError(
            f"The web app directory {directory} has {', '.join(shadowed)}, which "
            "the server's own routes answer; rename it.",
            reason="shadows",
        )
    return root
