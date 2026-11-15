"""Memberships and the access policy built on them, as a part an application assembles.

The application decides who belongs where; Cayu enforces what that allows. A
``MembershipDirectory`` reports a person's current memberships (``PostgresMemberships`` over
the application's own tables, ``InMemoryMembershipDirectory`` for tests, or one of your own),
``MembershipAccessPolicy`` turns them into Cayu resource-access grants, and
``conversation_labels`` gives a new conversation the labels those grants are about. See
``cayu guide b2b``.

Public import surface; implementations load on first access."""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.memberships._exports import EXPORTS as _EXPORTS
from cayu.memberships._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
