"""Persisted same-boot monotonic freshness; replay never renews a deadline."""

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from uuid import uuid4

from cayu.runtime._policy_contract import _snapshot, _timestamp
from cayu.runtime._policy_wire import canonical, require
from cayu.sessions._process_liveness import current_process_identity


def boot_identity() -> str | None:
    identity = current_process_identity()
    return _process_clock_identity() if identity is None else identity.host_boot_id


@lru_cache(maxsize=1)
def _process_clock_identity() -> str:
    # Platforms without a verifiable boot identity can adopt fresh snapshots,
    # but cannot extend an old process's cached adoption authority after restart.
    return "process-" + uuid4().hex


def validate_observation(value, *, scope, incarnation):
    require(
        type(value) is dict and set(value) == {"snapshot", "boot", "received_ns", "deadline_ns"}
    )
    snapshot = _snapshot(canonical(value["snapshot"]), scope=scope)
    require((snapshot["incarnation_id"], snapshot["incarnation_epoch"]) == incarnation)
    require(value["boot"] is None or type(value["boot"]) is str)
    for key in ("received_ns", "deadline_ns"):
        require(type(value[key]) is int and 0 <= value[key] < 2**63)
    require(value["received_ns"] < value["deadline_ns"])
    return snapshot


def observe(previous, wire, *, scope, incarnation, started_ns, received_ns, boot):
    snapshot = _snapshot(wire, scope=scope)
    require((snapshot["incarnation_id"], snapshot["incarnation_epoch"]) == incarnation)
    require(0 <= received_ns - started_ns <= 10_000_000_000)
    if previous is not None:
        prior = validate_observation(previous, scope=scope, incarnation=incarnation)
        if prior["snapshot_id"] == snapshot["snapshot_id"]:
            require(prior == snapshot)
            return previous
        before, after = prior["effective"], snapshot["effective"]
        require(after["effective_revision"] >= before["effective_revision"])
        if after["effective_revision"] == before["effective_revision"]:
            require(after == before)
        require(snapshot["issued_at"] > prior["issued_at"])
    # Bound clock disagreement as well as monotonic elapsed time. A response
    # carrying an already-old snapshot cannot gain a fresh 55-second lease.
    age = datetime.now(UTC) - _timestamp(snapshot["issued_at"])
    require(-timedelta(seconds=5) <= age <= timedelta(seconds=10))
    remaining_ns = min(
        55_000_000_000, int((timedelta(seconds=55) - max(age, timedelta())).total_seconds() * 1e9)
    )
    return {
        "snapshot": snapshot,
        "boot": boot,
        "received_ns": received_ns,
        "deadline_ns": min(started_ns + 55_000_000_000, received_ns + remaining_ns),
    }


def require_fresh(observation, *, now_ns, boot):
    require(observation is not None and boot is not None and observation["boot"] == boot)
    require(observation["received_ns"] <= now_ns < observation["deadline_ns"])
