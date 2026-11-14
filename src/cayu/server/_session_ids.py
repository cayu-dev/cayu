"""Resolve the public session IDs of a ``sessions?ids=`` request with batched reads."""

from __future__ import annotations

from collections.abc import Sequence

from cayu.runtime.public_authority import (
    parse_public_authority_alias,
    public_authority_alias_is_reserved,
)
from cayu.sessions.store import SessionStore


async def resolve_public_session_ids(store: SessionStore, values: Sequence[str]) -> list[str]:
    """The private session ID of each value, in order, as ``CayuApp`` resolves one.

    One read finds which reserved-looking values are raw session IDs and one
    resolves the public aliases, however many values there are. A public alias
    that resolves to no session is left out, as a single resolution raises
    ``PublicSessionAliasNotFound`` for it. A malformed or ambiguous value raises
    ``ValueError``.
    """

    values = list(dict.fromkeys(values))
    reserved = [value for value in values if public_authority_alias_is_reserved(value)]
    raw = await store.load_session_instance_ids(reserved) if reserved else {}
    aliases = [
        value
        for value in reserved
        if (parsed := parse_public_authority_alias(value)) is not None
        and parsed.field_name == "session_id"
    ]
    resolved = (
        await store.resolve_public_authority_aliases(aliases, field_name="session_id")
        if aliases
        else {}
    )
    private: list[str] = []
    for value in values:
        if not public_authority_alias_is_reserved(value):
            private.append(value)
            continue
        exists = value in raw
        if value not in aliases:
            if not exists:
                raise ValueError("Public session_id alias is malformed or field-mismatched.")
            private.append(value)
        elif (target := resolved.get(value)) is None:
            if exists:
                private.append(value)
        elif exists and target != value:
            raise ValueError(
                "session_id is ambiguous between legacy private authority and a public event alias."
            )
        else:
            private.append(target)
    return private
