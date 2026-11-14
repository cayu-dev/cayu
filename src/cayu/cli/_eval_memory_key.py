"""The memory evidence key `cayu eval run` runs a memory project with.

`cayu check` lends the factory a throwaway key in a fresh checkout because
describing the app stores nothing. An eval runs the Agent, and with automatic
memory it writes recall evidence fingerprinted with the key into the project's
store. A throwaway key is safe only when that store holds no such evidence yet:
no ``CAYU_DATABASE_URL``, and a local SQLite database that is missing or has
never stored a session (such as the empty one `cayu check` leaves). Otherwise
the run refuses, since records written with a throwaway key could never be
verified by a later run.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from cayu.cli._check_keys import (
    _LOCAL_MEMORY_KEY,
    MEMORY_EVIDENCE_KEY_ENV,
    _reads_memory_evidence_key,
    lent_memory_evidence_key,
)
from cayu.storage.targets import CANONICAL_SQLITE_PATH, DATABASE_URL_ENV, SessionStoreBackend

# The run metadata entry that says which key the run used.
MEMORY_EVIDENCE_KEY_METADATA = "cayu_memory_evidence_key"
MEMORY_EVIDENCE_KEY_REQUIRED = "MEMORY_EVIDENCE_KEY_REQUIRED"
_GUIDE = "cayu guide diagnostics#memory-evidence-key-required"
# Every record keyed with the memory evidence key belongs to a session, and
# recall receipts and context exposures are the keyed records themselves.
_EVIDENCE_TABLES = ("cayu_sessions", "cayu_recall_receipts", "cayu_context_exposures")


class MemoryEvidenceKeyRequired(RuntimeError):
    """A memory project has no key and its store may already hold keyed evidence."""

    code = MEMORY_EVIDENCE_KEY_REQUIRED

    def __init__(self, store: str) -> None:
        super().__init__(
            f"`cayu eval run` needs a memory evidence key: {MEMORY_EVIDENCE_KEY_ENV} is "
            f"unset and {_LOCAL_MEMORY_KEY.as_posix()} doesn't exist. With automatic memory "
            f"the run writes recall evidence fingerprinted with that key into {store}, which "
            "keeps it, so a throwaway key would leave records no later run can verify. Set "
            f"{MEMORY_EVIDENCE_KEY_ENV} to the key that store was written with; in CI, keep "
            "it as a secret and set it before `cayu eval run`. Only a run whose store holds "
            f"no sessions yet (no {DATABASE_URL_ENV}, and no {CANONICAL_SQLITE_PATH.as_posix()} "
            "or one without sessions) gets a throwaway key. "
            f"See `{_GUIDE}`."
        )


@contextmanager
def eval_memory_evidence_key(root: Path) -> Iterator[str | None]:
    """Make sure the eval's app can build with a memory evidence key, and say which.

    Yields ``None`` when the project doesn't read the key as `cayu new` generates,
    ``"environment"`` or ``"project_file"`` for a real key, and ``"throwaway"``
    when it lent one for this run only, which also reaches ``--processes``
    workers through the environment. Raises :class:`MemoryEvidenceKeyRequired`
    instead of lending when the run's store is external or already holds sessions.
    """

    if not _reads_memory_evidence_key(root):
        yield None
        return
    with lent_memory_evidence_key(root) as lent:
        if not lent:
            yield "environment" if MEMORY_EVIDENCE_KEY_ENV in os.environ else "project_file"
            return
        if DATABASE_URL_ENV in os.environ:
            raise MemoryEvidenceKeyRequired(f"the database {DATABASE_URL_ENV} names")
        databases = _local_databases(root)
        if databases is None:
            raise MemoryEvidenceKeyRequired("the project's configured session store")
        for path in databases:
            if _holds_sessions(path):
                raise MemoryEvidenceKeyRequired(_relative(root, path))
        _note(
            f"{MEMORY_EVIDENCE_KEY_ENV} is unset and {_LOCAL_MEMORY_KEY.as_posix()} doesn't "
            "exist, and this run's store holds no sessions yet, so `cayu eval run` runs with "
            "a throwaway memory evidence key that covers only this run."
        )
        try:
            yield "throwaway"
        finally:
            for path in databases:
                if _holds_sessions(path):
                    _note(
                        f"this run stored sessions in {_relative(root, path)} with a throwaway "
                        "memory evidence key, so later runs without a key refuse to use it. "
                        "Delete it if it holds nothing you need, or set a stable "
                        f"{MEMORY_EVIDENCE_KEY_ENV} from now on."
                    )


def _local_databases(root: Path) -> tuple[Path, ...] | None:
    """The generated storage's database and the CLI's configured SQLite path.

    ``None`` when the configured session store isn't a local SQLite file or
    can't be resolved, so its contents can't be checked here.
    """

    from cayu.cli.store_targets import resolve_project_session_store_target

    try:
        target = resolve_project_session_store_target(start=root, local_development=True)
    except Exception:
        return None
    if target is None or target.backend is not SessionStoreBackend.SQLITE:
        return None
    # configuration/storage.py of a project made by `cayu new` opens this one.
    generated = root / CANONICAL_SQLITE_PATH
    paths = [generated]
    if target.sqlite_path is not None:
        configured = Path(target.sqlite_path)
        if configured.resolve() != generated.resolve():
            paths.append(configured)
    return tuple(paths)


def _holds_sessions(path: Path) -> bool:
    """Whether the SQLite file at ``path`` has stored a session or keyed evidence.

    Read without writing. A file that can't be read as a database counts as
    holding them, so the caller refuses rather than guesses.
    """

    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
            tables = {
                str(row[0])
                for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            return any(
                db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
                for table in _EVIDENCE_TABLES
                if table in tables
            )
    except sqlite3.Error:
        return True


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _note(message: str) -> None:
    print(f"note: {message} See `{_GUIDE}`.", file=sys.stderr, flush=True)
