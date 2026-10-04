"""A warm spare pool: resources created ahead of use, each handed out once.

:class:`WarmSparePool` owns the policy (size, single-flight refill, backoff,
handout and stale-spare reaping). A :class:`WarmSpareBackend` owns the
resource: it creates a spare through its ordinary cold-start path, claims one
for a consumer, and lists or removes spares by name. The Docker coding factory
composes the pool with a Docker backend through ``warm_spares=N``; any other
runner or factory can compose it with its own backend.

Spare names are ``<prefix><pid>-<nonce>-<id>``. The prefix should identify the
host, so a pool only reaps spares of dead processes (or closed pools of this
process) on the same host. A reused PID makes a dead owner look alive until the
PID exits again; such spares are reaped later, never handed out.
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import hashlib
import logging
import os
import time
import weakref
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from cayu.environments.admission import ExecutionRequirements

logger = logging.getLogger(__name__)

SpareT = TypeVar("SpareT")

MAX_WARM_SPARE_BACKOFF_S = 60.0
"""Longest wait between refill attempts after consecutive backend errors."""

DEFAULT_WARM_SPARE_REAP_INTERVAL_S = 300.0
"""How often a refill re-scans for spares left by dead processes."""

_MAX_UNSATISFIABLE_REQUIREMENTS = 64

# Nonces of warm pools alive in this process; spares of any other nonce with
# this process's PID belong to a closed pool and are safe to remove.
_LIVE_POOL_NONCES: set[str] = set()


class WarmSpareRequirementsUnsatisfied(RuntimeError):
    """Raised by ``claim_spare`` when a spare cannot meet the consumer's requirements.

    The pool then sends later claims with the same requirements straight to a
    cold start for a while, instead of spending a spare on each attempt.
    """


def _requirements_key(requirements: ExecutionRequirements) -> str:
    return hashlib.sha256(requirements.model_dump_json().encode("utf-8")).hexdigest()


class WarmSpareBackend(Protocol[SpareT]):
    """The resource side of a :class:`WarmSparePool`.

    Every method may raise. The pool treats any ``Exception`` as "no spare":
    it discards the spare it was handling and the caller falls back to its
    cold-start path.
    """

    async def create_spare(self, name: str) -> SpareT:
        """Create one spare named ``name`` through the backend's cold-start path.

        Run the same admission probes a cold start runs, on this exact resource,
        and remove it before raising if they fail.
        """
        ...

    async def spare_is_alive(self, spare: SpareT) -> bool:
        """Whether the idle spare can still be handed out."""
        ...

    async def claim_spare(
        self, spare: SpareT, name: str, requirements: ExecutionRequirements
    ) -> SpareT:
        """Make ``spare`` the consumer's resource ``name`` and return what it uses.

        Re-verify anything the consumer's ``requirements`` add beyond what the
        spare was created with. Raise to refuse; the pool then discards it and
        returns ``None``. Raise :class:`WarmSpareRequirementsUnsatisfied` when
        the requirements themselves cannot be met by a spare.
        """
        ...

    async def discard_spare(self, spare: SpareT) -> None:
        """Remove a spare that will never be handed out."""
        ...

    async def list_spare_names(self, prefix: str) -> Sequence[str]:
        """Names of every spare resource on this host that starts with ``prefix``."""
        ...

    async def remove_spare_named(self, name: str) -> None:
        """Remove the spare resource called ``name``; absence is not an error."""
        ...

    # Optional: ``remove_spares_at_exit(spares)``, a synchronous, bounded removal
    # the pool calls at normal interpreter exit for idle spares nobody drained
    # or closed. Without it those spares stay until another pool reaps them.


def _remove_idle_spares_at_exit(pool_ref: weakref.ref[WarmSparePool[Any]]) -> None:
    pool = pool_ref()
    # A forked child that exits normally must never remove the parent's spares,
    # one of which the parent may since have claimed for a live session.
    if pool is None or pool._owner_pid != os.getpid() or not pool._spares:
        return
    remove = getattr(pool._backend, "remove_spares_at_exit", None)
    if remove is None:
        return
    spares, pool._spares = list(pool._spares), []
    with suppress(Exception):  # Interpreter shutdown: never raise from an exit hook.
        remove(spares)


# Every pool in this process, so a forked child can drop inherited spares.
_POOLS: weakref.WeakSet[WarmSparePool[Any]] = weakref.WeakSet()


def _forget_inherited_spares() -> None:
    """In a forked child, drop the parent's spares and in-flight refill.

    The child never hands out or removes a spare the parent created; it builds
    its own, named with its own PID, on its next use.
    """

    for pool in list(_POOLS):
        pool._spares = []
        pool._refill_task = None
        pool._owner_pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_spares)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stale_spare_names(
    names: Sequence[str], *, prefix: str, live_nonces: frozenset[str]
) -> list[str]:
    """Spares whose owning pool is gone: a dead PID, or this PID with a closed pool.

    Names are ``<prefix><pid>-<nonce>-<id>``. Spares of other live processes,
    other prefixes and unrecognized names are left alone.
    """

    stale = []
    for name in names:
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].split("-")
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        pid, nonce = int(parts[0]), parts[1]
        if pid == os.getpid():
            if nonce not in live_nonces:
                stale.append(name)
        elif not _pid_is_alive(pid):
            stale.append(name)
    return stale


class WarmSparePool(Generic[SpareT]):
    """Keep up to ``size`` idle spares and hand each to exactly one consumer.

    ``take`` never raises for a backend failure: it returns ``None`` and the
    caller cold-starts. Refill is single-flight, capped at ``size`` and backs
    off on backend errors. ``release_idle`` trims idle spares and lets a later
    use refill; ``close`` trims them for good.
    """

    def __init__(
        self,
        backend: WarmSpareBackend[SpareT],
        *,
        size: int,
        name_prefix: str,
        reap_interval_s: float = DEFAULT_WARM_SPARE_REAP_INTERVAL_S,
    ) -> None:
        if type(size) is not int or size < 1:
            raise ValueError("size must be a positive integer.")
        if type(name_prefix) is not str or not name_prefix:
            raise ValueError("name_prefix must be a non-empty string.")
        if type(reap_interval_s) not in {int, float} or reap_interval_s <= 0:
            raise ValueError("reap_interval_s must be a positive number.")
        self._backend = backend
        self._size = size
        self._prefix = name_prefix
        self._reap_interval_s = float(reap_interval_s)
        self._spares: list[SpareT] = []
        self._unsatisfiable: dict[str, float] = {}
        self._refill_task: asyncio.Task[None] | None = None
        self._backoff_s = 0.0
        self._next_attempt_at = 0.0
        self._next_reap_at = 0.0
        self._idle_released = False
        self._closed = False
        self._nonce = uuid4().hex[:8]
        self._owner_pid = os.getpid()
        _POOLS.add(self)
        _LIVE_POOL_NONCES.add(self._nonce)
        self._nonce_finalizer = weakref.finalize(self, _LIVE_POOL_NONCES.discard, self._nonce)
        # Best effort for applications that never drain or close: at normal
        # interpreter exit, remove this pool's idle spares synchronously when
        # the backend can. A killed process leaves them to a later pool's reap.
        self._exit_hook = functools.partial(_remove_idle_spares_at_exit, weakref.ref(self))
        atexit.register(self._exit_hook)
        # A collected pool has nothing left to remove; drop its exit hook then.
        weakref.finalize(self, atexit.unregister, self._exit_hook).atexit = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def idle_count(self) -> int:
        return len(self._spares)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def name_prefix(self) -> str:
        return self._prefix

    def _spare_name(self) -> str:
        return f"{self._prefix}{os.getpid()}-{self._nonce}-{uuid4().hex[:8]}"

    def _back_off(self) -> None:
        self._backoff_s = min(MAX_WARM_SPARE_BACKOFF_S, max(1.0, self._backoff_s * 2))
        self._next_attempt_at = time.monotonic() + self._backoff_s

    async def reap_stale(self) -> list[str]:
        """Remove spares left by dead processes or closed pools on this host."""

        names = await self._backend.list_spare_names(self._prefix)
        stale = stale_spare_names(
            names, prefix=self._prefix, live_nonces=frozenset(_LIVE_POOL_NONCES)
        )
        for name in stale:
            await self._backend.remove_spare_named(name)
        return stale

    async def _reap_if_due(self) -> None:
        now = time.monotonic()
        if now < self._next_reap_at:
            return
        self._next_reap_at = now + self._reap_interval_s
        try:
            await self.reap_stale()
        except Exception:
            logger.warning("Warm spare reaping failed; retrying later.", exc_info=True)

    async def _discard(self, spare: SpareT) -> None:
        try:
            await self._backend.discard_spare(spare)
        except Exception:
            logger.warning("Discarding a warm spare failed.", exc_info=True)

    async def take(self, name: str, requirements: ExecutionRequirements) -> SpareT | None:
        """Hand one live spare to ``name``, or return ``None`` for a cold start.

        Only a spare that is no longer alive makes ``take`` try the next one. A
        failed claim discards that spare and stops, so one bad claim cannot
        burn the whole pool; requirements a spare cannot satisfy are
        remembered for a while and go straight to a cold start.
        """

        if self._closed:
            return None
        self._idle_released = False
        key = _requirements_key(requirements)
        unsatisfiable_until = self._unsatisfiable.get(key)
        if unsatisfiable_until is not None:
            if time.monotonic() < unsatisfiable_until:
                return None
            del self._unsatisfiable[key]
        while self._spares:
            spare = self._spares.pop(0)
            try:
                alive = await self._backend.spare_is_alive(spare)
            except BaseException as error:
                # Unknown state: keep the spare for a later take, unless the
                # pool was closed or trimmed meanwhile (nothing would discard it
                # then) or a concurrent refill already filled it.
                interrupted = None
                if self._closed or self._idle_released or len(self._spares) >= self._size:
                    interrupted = await self._discard_shielded(spare)
                else:
                    self._spares.insert(0, spare)
                if not isinstance(error, Exception):
                    raise
                if interrupted is not None:
                    raise interrupted from error
                logger.warning("Checking a warm spare failed.", exc_info=True)
                return None
            if not alive:
                interrupted = await self._discard_shielded(spare)
                if interrupted is not None:
                    raise interrupted
                continue
            try:
                claimed = await self._backend.claim_spare(spare, name, requirements)
            except BaseException as error:
                # The claim may have renamed the spare already: never keep it.
                interrupted = await self._discard_shielded(spare)
                if not isinstance(error, Exception):
                    raise
                if interrupted is not None:
                    # Cancelled while discarding: the caller must not cold-start.
                    raise interrupted from error
                if isinstance(error, WarmSpareRequirementsUnsatisfied):
                    self._remember_unsatisfiable(key)
                logger.warning("A warm spare could not be handed out.", exc_info=True)
                self.schedule_refill()
                return None
            self.schedule_refill()
            return claimed
        self.schedule_refill()
        return None

    async def _discard_shielded(self, spare: SpareT) -> BaseException | None:
        """Discard ``spare`` under a shield; report what interrupted the wait.

        If the caller is cancelled, the wait ends at once and the cancellation
        is returned so the caller re-raises it immediately; the discard keeps
        running detached. Should teardown cancel that detached discard too, the
        leftover is removed later by stale-spare reaping, or, for a claimed
        spare already renamed, through the session's reserved-name cleanup.
        """

        task = asyncio.ensure_future(self._discard(spare))
        try:
            await asyncio.shield(task)
        except BaseException as interrupted:
            return interrupted
        return None

    def _remember_unsatisfiable(self, key: str) -> None:
        if len(self._unsatisfiable) >= _MAX_UNSATISFIABLE_REQUIREMENTS:
            self._unsatisfiable.pop(next(iter(self._unsatisfiable)))
        self._unsatisfiable[key] = time.monotonic() + self._reap_interval_s

    def schedule_refill(self) -> None:
        """Start one background refill unless full, closed, released or backing off."""

        if self._closed or self._idle_released or len(self._spares) >= self._size:
            return
        if self._refill_task is not None and not self._refill_task.done():
            return
        if time.monotonic() < self._next_attempt_at:
            return
        self._refill_task = asyncio.get_running_loop().create_task(
            self._refill(), name="cayu-warm-spare-refill"
        )

    async def _refill(self) -> None:
        # Reaping runs here, off the materialization path, at most once per interval.
        await self._reap_if_due()
        while not self._closed and not self._idle_released and len(self._spares) < self._size:
            try:
                spare = await self._backend.create_spare(self._spare_name())
            except Exception:
                logger.warning("Warm spare creation failed; backing off.", exc_info=True)
                self._back_off()
                return
            if self._closed or self._idle_released:
                await self._discard(spare)
                return
            self._backoff_s = 0.0
            self._spares.append(spare)

    async def wait_for_refill(self) -> None:
        """Wait for the in-flight refill, if any, to settle."""

        task = self._refill_task
        if task is not None:
            await asyncio.shield(task)

    async def _settle_refill_and_trim(self) -> None:
        task, self._refill_task = self._refill_task, None
        if task is not None and not task.done():
            # Let an in-flight creation finish; the refill loop then sees the
            # release and removes that resource instead of keeping it.
            try:
                await asyncio.shield(task)
            except Exception:
                logger.warning("Warm spare refill failed while releasing.", exc_info=True)
        spares, self._spares = self._spares, []
        for spare in spares:
            await self._discard(spare)

    async def release_idle(self) -> None:
        """Remove every idle spare; the next ``take`` refills the pool."""

        self._idle_released = True
        await self._settle_refill_and_trim()
        self._next_attempt_at = 0.0

    async def close(self) -> None:
        """Remove every idle spare and create no more. Later takes cold-start."""

        self._closed = True
        await self._settle_refill_and_trim()
        self._nonce_finalizer()
        atexit.unregister(self._exit_hook)


__all__ = [
    "DEFAULT_WARM_SPARE_REAP_INTERVAL_S",
    "MAX_WARM_SPARE_BACKOFF_S",
    "WarmSpareBackend",
    "WarmSparePool",
    "WarmSpareRequirementsUnsatisfied",
    "stale_spare_names",
]
