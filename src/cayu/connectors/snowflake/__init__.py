"""Snowflake tools for an application's Snowflake connection slot, through its SQL API.

``tools(client=None, slot="snowflake", max_rows=200, timeout_seconds=120)`` answers
``snowflake_query``, one read statement (``SELECT``, ``SHOW``, ``DESCRIBE``, ``WITH ... SELECT``);
with ``include_writes=True`` also ``snowflake_execute``, one statement of any kind, on its own
``write_slot`` (default ``snowflake-write``). The read slot's Snowflake role should only read.
``approval_policy(inner)`` asks for approval before ``snowflake_execute`` runs;
``review_fields()`` shows the approver the SQL. See ``docs/connector-toolkits.md``.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.connectors.snowflake._exports import EXPORTS as _EXPORTS
from cayu.connectors.snowflake._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
