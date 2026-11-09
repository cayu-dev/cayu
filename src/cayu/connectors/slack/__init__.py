"""Slack tools for an application's ``slack`` connection slot.

``tools(client=None, slot="slack")`` answers ``slack_list_channels``, ``slack_read_channel``,
``slack_read_thread`` and ``slack_lookup_user``; with ``include_writes=True`` also
``slack_post_message`` and ``slack_update_message``. ``approval_policy(inner)`` asks for approval
before a post or an edit; ``review_fields()`` shows the approver the channel and the text. See
``docs/connector-toolkits.md``.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.connectors.slack._exports import EXPORTS as _EXPORTS
from cayu.connectors.slack._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
