"""Owned local publication of an application default and its pending report.

Any uncertain store outcome fences synchronous default capture until durable
readback. The owner never interprets cancellation as rollback of a dispatched
store operation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

from cayu.runtime._policy_contract import _integer
from cayu.runtime._policy_journal import (
    acknowledge_decision,
    append_decision,
    load_journal,
)
from cayu.runtime._policy_storage import (
    ModelPolicyStore,
    PolicyStorageAction,
    PolicyStorageCommand,
    PolicyStorageView,
    binding_bytes,
    state_bytes,
)
from cayu.runtime._policy_wire import canonical, decode, identifier, require


class PolicyInstallationOwner:
    """One process owner; callers run renewal and reporting under their lifecycle.

    This object does not own the supplied store. Release it before closing a
    shared application store. Installation and capture are ordered on the event
    loop; no await separates durable acknowledgement from local adoption.
    """

    def __init__(
        self,
        store: ModelPolicyStore,
        *,
        binding: dict[str, Any],
        incarnation: tuple[str, int],
    ) -> None:
        require(isinstance(store, ModelPolicyStore))
        require(type(incarnation) is tuple and len(incarnation) == 2)
        identifier(incarnation[0])
        _integer(incarnation[1])
        self._store = store
        self._binding = binding_bytes(binding)
        self._scope = decode(self._binding)["scope"]
        self._incarnation = incarnation
        self._owner = uuid4().hex
        self._lock = asyncio.Lock()
        self._view: PolicyStorageView | None = None
        self._deadline = 0.0
        self._started = False
        self._closed = False
        self._claim_attempted = False

    async def start(self) -> None:
        async with self._lock:
            require(not self._started and not self._closed)
            # A failed/lost claim may already own the row. Retrying start uses
            # the same identity; it does not manufacture a replacement owner.
            self._claim_attempted = True
            await self._execute("claim")
            await self._recover_publication()
            self._started = True

    async def reconcile(self) -> None:
        async with self._lock:
            require(self._started and not self._closed)
            # A claim is an exact-owner read while the lease remains live; after
            # expiry it atomically reacquires only an unowned/expired binding.
            await self._execute("claim", preserve_current=True)
            await self._recover_publication()

    async def renew(self) -> None:
        async with self._lock:
            require(self._started and not self._closed)
            await self._execute("claim", preserve_current=True)
            await self._recover_publication()
            await self._execute("renew")

    def current(self) -> bytes | None:
        state = self._state()
        publication = state["publication"]
        if publication is not None and publication["completed_ns"] is None:
            state = publication["prior"]
        current = state["current"]
        return None if current is None else canonical(current)

    def pending(self) -> tuple[bytes, ...]:
        state = self._state()
        publication = state["publication"]
        if publication is not None and publication["completed_ns"] is None:
            state = publication["prior"]
        return tuple(
            canonical(record["report"]) for record in state["reports"] if record["receipt"] is None
        )

    def state(self) -> dict[str, Any]:
        """Detached validated state; never expose a mutable hot default."""
        return self._state()

    async def install(self, report: bytes, *, deadline_ns: int | None = None) -> None:
        async with self._lock:
            self._state()
            assert self._view is not None
            candidate = append_decision(
                self._view.state, report, scope=self._scope, incarnation=self._incarnation
            )
            if deadline_ns is not None:
                # Retain the complete previous state until the native store has
                # positively acknowledged publication before the deadline. A
                # crash or lost first acknowledgement cannot manufacture proof.
                prior = load_journal(
                    self._view.state, scope=self._scope, incarnation=self._incarnation
                )
                prior["publication"] = None
                staged = load_journal(candidate, scope=self._scope, incarnation=self._incarnation)
                staged["publication"] = {
                    "prior": prior,
                    "operation_id": decode(report)["operation_id"],
                    "deadline_ns": deadline_ns,
                    "completed_ns": None,
                }
                candidate = state_bytes(staged)
            await self._execute("write", candidate=candidate, deadline_ns=deadline_ns)
            if deadline_ns is not None:
                completed = time.monotonic_ns()
                if completed >= deadline_ns:
                    await self._recover_publication()
                    require(False)
                # This second write records observed completion, not another
                # installation. Its acknowledgement can safely arrive late or
                # be reconstructed after restart.
                staged["publication"]["completed_ns"] = completed
                staged["publication"]["prior"] = None
                await self._execute("write", candidate=state_bytes(staged))

    async def _recover_publication(self) -> None:
        assert self._view is not None
        state = load_journal(self._view.state, scope=self._scope, incarnation=self._incarnation)
        publication = state["publication"]
        if publication is not None and publication["completed_ns"] is None:
            # Neither a matching row nor an expired deadline proves timely
            # completion. Conservatively restore the previous installation and
            # its exact pending reports, then obtain fresh policy authority.
            await self._execute("write", candidate=state_bytes(publication["prior"]))

    async def observe(
        self, wire: bytes, *, started_ns: int, received_ns: int, boot: str | None
    ) -> None:
        from cayu.runtime._policy_freshness import observe

        async with self._lock:
            state = self._state()
            state["observation"] = observe(
                state["observation"],
                wire,
                scope=self._scope,
                incarnation=self._incarnation,
                started_ns=started_ns,
                received_ns=received_ns,
                boot=boot,
            )
            await self._execute("write", candidate=state_bytes(state), preserve_current=True)

    async def resume_adoption(self) -> None:
        async with self._lock:
            state = self._state()
            state["adoption_enabled"] = True
            await self._execute("write", candidate=state_bytes(state), preserve_current=True)

    async def acknowledge(self, receipt: bytes, *, expected_report: bytes) -> None:
        async with self._lock:
            self._state()
            assert self._view is not None
            candidate = acknowledge_decision(
                self._view.state,
                receipt,
                expected_report=expected_report,
                scope=self._scope,
                incarnation=self._incarnation,
            )
            await self._execute("write", candidate=candidate, preserve_current=True)

    async def close(self) -> None:
        async with self._lock:
            if not self._closed:
                if self._claim_attempted:
                    await self._execute("release")
                self._closed = True

    def _state(self) -> dict[str, Any]:
        require(self._started and not self._closed)
        require(self._view is not None and time.monotonic() < self._deadline)
        assert self._view is not None
        return load_journal(self._view.state, scope=self._scope, incarnation=self._incarnation)

    async def _execute(
        self,
        action: PolicyStorageAction,
        *,
        candidate: bytes | None = None,
        deadline_ns: int | None = None,
        preserve_current: bool = False,
    ) -> None:
        expected = self._view if action == "write" else None
        if preserve_current and action == "write":
            require(expected is not None and candidate is not None)
            assert expected is not None and candidate is not None
            before = load_journal(expected.state, scope=self._scope, incarnation=self._incarnation)
            after = load_journal(candidate, scope=self._scope, incarnation=self._incarnation)
            require(before["current"] == after["current"])
        # Reads and renewals cannot replace the installed default. Keep the
        # previously proven lease usable while they wait; its original deadline
        # still applies. Mutations fence capture before dispatch.
        if not preserve_current and action not in ("read", "renew"):
            self._view = None
            self._deadline = 0.0
        started = time.monotonic()
        try:
            view = await self._store.execute(
                PolicyStorageCommand(
                    action, self._binding, self._owner, expected, candidate, deadline_ns
                )
            )
        except BaseException:
            self._view = None
            self._deadline = 0.0
            raise
        # Validation and adoption are synchronous. Invalid readback must not
        # leave the previous capture authority installed.
        self._view = None
        self._deadline = 0.0
        require(type(view) is PolicyStorageView)
        require(type(view.revision) is int and 0 <= view.revision < 2**53)
        if action == "write":
            require(expected is not None)
            assert expected is not None
            require(view.state == candidate and view.revision == expected.revision + 1)
        load_journal(view.state, scope=self._scope, incarnation=self._incarnation)
        if action != "release":
            require(type(view.lease_remaining_seconds) in (int, float))
            require(0 < view.lease_remaining_seconds <= 60)
            # Store time is authoritative. Time spent waiting never extends it.
            deadline = started + view.lease_remaining_seconds - 1
            require(time.monotonic() < deadline)
            self._view, self._deadline = view, deadline
