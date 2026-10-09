"""Derive a stable, non-secret prompt-cache affinity key for a session lineage.

Providers cache request prefixes. Sessions that share a transcript prefix should
be routed to the same cache, so the key identifies the fork lineage: the
original session whose transcript every fork in the lineage copied. Sibling
forks and forks of forks share one key. A delegated or child session that
starts its own transcript is its own lineage root.

The key is a domain-separated SHA-256 digest of the lineage root session id.
It is derived only from durable fork provenance, so it is the same across
process restarts, recovery and replay of the same session, and it never puts
a raw session identifier on the wire.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256

from cayu.sessions.base import SessionStore, session_fork_profile_relationship
from cayu.sessions.records import Session

CACHE_AFFINITY_KEY_PREFIX = "cayu-"
_CACHE_AFFINITY_DOMAIN = b"cayu.prompt-cache-affinity.v1\x00"
_CACHE_AFFINITY_DIGEST_CHARS = 32
# Bounds the provenance walk; real fork chains are far shorter.
_MAX_LINEAGE_DEPTH = 64
_MAX_CACHED_SESSIONS = 4096


def cache_affinity_key_for_lineage_root(root_session_id: str) -> str:
    """Return the wire key for one lineage root session id."""

    if type(root_session_id) is not str or not root_session_id:
        raise ValueError("A cache-affinity lineage root must be a nonblank session id.")
    digest = sha256(_CACHE_AFFINITY_DOMAIN + root_session_id.encode("utf-8")).hexdigest()
    return CACHE_AFFINITY_KEY_PREFIX + digest[:_CACHE_AFFINITY_DIGEST_CHARS]


def _fork_source_session_id(session: Session) -> str | None:
    try:
        relationship = session_fork_profile_relationship(session)
    except ValueError:
        # Malformed provenance fails closed elsewhere; a routing hint does not.
        return None
    return None if relationship is None else relationship.source_session_id


async def session_cache_lineage_root(session: Session, session_store: SessionStore) -> str:
    """Follow durable fork provenance to the session whose prefix the lineage shares.

    A missing or unreadable ancestor ends the walk at the last known source id,
    which every descendant through that ancestor still agrees on.
    """

    root = session.id
    source_id = _fork_source_session_id(session)
    seen = {session.id}
    while source_id is not None and source_id not in seen and len(seen) <= _MAX_LINEAGE_DEPTH:
        root = source_id
        seen.add(source_id)
        try:
            source = await session_store.load(source_id)
        except Exception:
            return root
        if source is None:
            return root
        source_id = _fork_source_session_id(source)
    return root


class SessionCacheAffinity:
    """Resolve and memoize each session's immutable cache-affinity key."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store
        self._keys: OrderedDict[str, str] = OrderedDict()

    async def key_for(self, session: Session) -> str:
        cached = self._keys.get(session.id)
        if cached is not None:
            self._keys.move_to_end(session.id)
            return cached
        key = cache_affinity_key_for_lineage_root(
            await session_cache_lineage_root(session, self._session_store)
        )
        self._keys[session.id] = key
        while len(self._keys) > _MAX_CACHED_SESSIONS:
            self._keys.popitem(last=False)
        return key
