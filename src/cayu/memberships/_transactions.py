"""One transaction on a fresh DB-API connection, for the membership stores."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any


def run_transaction(
    connect: Callable[[], Any],
    work: Callable[[Any], Any],
    *,
    commit: bool = False,
    owner: str,
) -> Any:
    """Run ``work`` on a cursor of a new connection from ``connect``, then commit (or roll
    back when ``commit`` is false) and close the connection. ``owner`` names the store in
    the error for a connection with autocommit on."""

    connection = connect()
    if getattr(connection, "autocommit", False):
        connection.close()
        # Each statement would commit on its own: a refused change would stay made.
        raise ValueError(f"{owner} needs connections with autocommit off.")
    try:
        cursor = connection.cursor()
        try:
            result = work(cursor)
        finally:
            cursor.close()
        if commit:
            connection.commit()
        else:
            connection.rollback()
        return result
    except BaseException:
        # A broken connection can't roll back; closing it is all that's left to do.
        with contextlib.suppress(Exception):
            connection.rollback()
        raise
    finally:
        connection.close()
