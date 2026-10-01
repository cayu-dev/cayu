"""Fenced application-policy state, independent of session and money stores."""

from __future__ import annotations

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.runtime._policy_contract import _scope
from cayu.runtime._policy_wire import canonical, decode, identifier, require

POLICY_STORAGE_REVISION = 114
MAX_POLICY_STATE_BYTES = 16 * 1024 * 1024
POLICY_LEASE_SECONDS = 60
PolicyStorageAction = Literal["claim", "renew", "read", "write", "release"]


def state_bytes(value: dict[str, Any]) -> bytes:
    require(type(value) is dict)
    return canonical_bounded_durable_json_bytes(
        value,
        "model_policy_state",
        max_bytes=MAX_POLICY_STATE_BYTES,
        max_nodes=250_000,
        max_nesting=24,
    )


def binding_bytes(binding: dict[str, Any]) -> bytes:
    require(type(binding) is dict and set(binding) == {"scope", "agent_name", "provider_name"})
    _scope(binding["scope"])
    identifier(binding["agent_name"])
    identifier(binding["provider_name"])
    return canonical(binding)


@dataclass(frozen=True)
class PolicyStorageView:
    revision: int
    state: bytes
    lease_remaining_seconds: float = 0.0


@dataclass(frozen=True)
class PolicyStorageCommand:
    action: PolicyStorageAction
    binding: bytes
    owner: str
    expected: PolicyStorageView | None = None
    state: bytes | None = None
    deadline_ns: int | None = None

    def validate(self) -> None:
        require(
            self.deadline_ns is None or (type(self.deadline_ns) is int and self.action == "write")
        )
        require(type(self.action) is str)
        require(self.action in ("claim", "renew", "read", "write", "release"))
        identifier(self.owner)
        require(type(self.binding) is bytes and 0 < len(self.binding) <= 65536)
        require(binding_bytes(decode(self.binding)) == self.binding)
        if self.action == "write":
            require(type(self.expected) is PolicyStorageView)
            assert self.expected is not None
            require(type(self.expected.revision) is int and self.expected.revision >= 0)
            require(type(self.state) is bytes)
            assert self.state is not None
            for body in (self.expected.state, self.state):
                parse_state(body)
        else:
            require(self.expected is None and self.state is None)

    @property
    def key(self) -> str:
        # One Cloud integration has one local report sequence. Changing the
        # application-owned agent/provider mapping must conflict with its full
        # retained binding, not silently create a second sequence owner.
        return sha256(canonical(decode(self.binding)["scope"])).hexdigest()


PolicyStorageRow = tuple[bytes, str | None, float, int, bytes]


def transition(
    row: PolicyStorageRow | None, command: PolicyStorageCommand, now: float
) -> tuple[PolicyStorageRow, PolicyStorageView]:
    """Pure transition; backend owns the binding lock and publication."""
    if row is None:
        require(command.action in ("claim", "release"))
        row = (command.binding, None, 0.0, 0, b"{}")
    binding, owner, expires, revision, body = row
    if owner is not None:
        identifier(owner)
    require(type(now) in (int, float) and math.isfinite(now))
    require(type(expires) in (int, float) and math.isfinite(expires))
    require(binding == command.binding)
    require(type(revision) is int and 0 <= revision < 2**53 - 1)
    parse_state(body)
    if command.action == "release":
        # Exact owner release is idempotent after acknowledgement loss. A
        # replacement's lease is never released by the retired owner.
        if owner == command.owner:
            owner, expires = None, now
    elif command.action == "claim":
        require(owner == command.owner or owner is None or expires <= now)
        if owner != command.owner or expires <= now:
            owner, expires = command.owner, now + POLICY_LEASE_SECONDS
    else:
        require(owner == command.owner and expires > now)
        if command.action == "renew":
            expires = now + POLICY_LEASE_SECONDS
        elif command.action == "write":
            require(command.deadline_ns is None or time.monotonic_ns() < command.deadline_ns)
            require(
                command.expected is not None
                and command.expected.revision == revision
                and command.expected.state == body
            )
            assert command.state is not None
            body, revision = command.state, revision + 1
    updated = binding, owner, expires, revision, body
    return updated, PolicyStorageView(revision, body, max(0.0, expires - now))


def parse_state(body: bytes) -> dict[str, Any]:
    require(type(body) is bytes and 0 < len(body) <= MAX_POLICY_STATE_BYTES)
    value = None
    with suppress(ValueError, UnicodeError, RecursionError):
        value = json.loads(body)
    require(type(value) is dict)
    assert isinstance(value, dict)
    require(state_bytes(value) == body)
    return value


class ModelPolicyStore(ABC):
    @abstractmethod
    async def execute(self, command: PolicyStorageCommand) -> PolicyStorageView:
        """One row-locked command; ambiguous publication requires readback."""

    async def close(self) -> None:
        """Close owned resources after policy workers have stopped."""
        return None


class InMemoryModelPolicyStore(ModelPolicyStore):
    def __init__(self) -> None:
        self._rows: dict[str, PolicyStorageRow] = {}
        self._lock = asyncio.Lock()

    async def execute(self, command: PolicyStorageCommand) -> PolicyStorageView:
        command.validate()
        async with self._lock:
            row, view = transition(self._rows.get(command.key), command, time.time())
            self._rows[command.key] = row
            return view
