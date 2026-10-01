import copy
import time
from datetime import timedelta

import pytest
from tests.core.test_model_policy_runtime import Channel

from cayu.runtime._policy_contract import _timestamp
from cayu.runtime._policy_freshness import observe, require_fresh
from cayu.runtime._policy_wire import PolicyContractError, canonical, decode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_exact_replay_does_not_extend_deadline_and_reordered_or_changed_content_fails():
    channel = Channel()
    start = time.monotonic_ns()
    wire = await channel.read_snapshot()
    args = dict(scope=channel.scope, incarnation=channel.incarnation, boot="same-boot")
    first = observe(None, wire, started_ns=start, received_ns=start + 1, **args)
    replay = observe(
        first, wire, started_ns=start + 10_000_000_000, received_ns=start + 10_000_000_001, **args
    )
    assert replay == first
    require_fresh(replay, now_ns=first["deadline_ns"] - 1, boot="same-boot")
    with pytest.raises(PolicyContractError):
        require_fresh(replay, now_ns=first["deadline_ns"], boot="same-boot")
    with pytest.raises(PolicyContractError):
        require_fresh(replay, now_ns=start + 1, boot="different-boot")
    changed = decode(wire)
    for key in ("issued_at", "valid_until"):
        changed[key] = (_timestamp(changed[key]) + timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    with pytest.raises(PolicyContractError):
        observe(first, canonical(changed), started_ns=start, received_ns=start + 1, **args)
    wrong = copy.deepcopy(args)
    wrong["scope"]["application_id"] = "different"
    with pytest.raises(PolicyContractError):
        observe(None, wire, started_ns=start, received_ns=start + 1, **wrong)
