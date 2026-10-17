"""An app-scope dollar cap on a shared durable ledger, across processes.

Two real OS processes run the same application against one session store and
one budget ledger, start together, and keep spending until the cap refuses
them. The reservations are atomic in the shared ledger, so the combined spend
never passes the cap. The $0.75 cap is deliberately not a multiple of the two
processes' concurrent $0.25 calls: with a per-process in-memory ledger the same
run dispatches four calls and spends $1.00.
"""

from __future__ import annotations

import asyncio
import json
import sys
from decimal import Decimal

import pytest
from tests.recovery.spend_cap_process_worker import (
    DEPLOYMENT_CAP_USD,
    deployment_prices,
    open_stores,
)

from cayu.applications import CayuApp

_SESSIONS_PER_PROCESS = 4
_QUARTER = Decimal("0.25")


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_app_scope_cap_on_a_shared_ledger_holds_across_two_processes(
    backend: str, tmp_path, request
) -> None:
    address = str(tmp_path) if backend == "sqlite" else request.getfixturevalue("postgres_dsn")
    go = tmp_path / "go"

    async def run():
        # Create both schemas once so the workers never race a migration.
        sessions, ledger = open_stores(backend, address)
        await sessions.close()
        await ledger.close()

        children = []
        try:
            for name in ("process-a", "process-b"):
                child = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "tests.recovery.spend_cap_process_worker",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                children.append(child)
                assert child.stdin is not None
                child.stdin.write(
                    (
                        json.dumps(
                            {
                                "backend": backend,
                                "address": address,
                                "name": name,
                                "go": str(go),
                                "sessions": _SESSIONS_PER_PROCESS,
                            }
                        )
                        + "\n"
                    ).encode()
                )
                await child.stdin.drain()
            for child in children:
                assert child.stdout is not None
                ready = await asyncio.wait_for(child.stdout.readline(), 90)
                if ready.strip() != b"ready":
                    _, error = await asyncio.wait_for(child.communicate(), 30)
                    raise AssertionError(error.decode())
            go.touch()
            results = []
            for child in children:
                output, error = await asyncio.wait_for(child.communicate(), 120)
                assert child.returncode == 0, error.decode()
                results.append(json.loads(output))
        finally:
            for child in children:
                if child.returncode is None:
                    child.kill()
                    await child.wait()

        sessions, ledger = open_stores(backend, address)
        try:
            observer = CayuApp(session_store=sessions, budget_ledger=ledger, enable_logging=False)
            total = Decimal("0")
            for name in ("process-a", "process-b"):
                for index in range(_SESSIONS_PER_PROCESS):
                    cost = await observer.get_session_cost(f"{name}-{index}", deployment_prices())
                    total += cost.total_cost
        finally:
            await sessions.close()
            await ledger.close()
        return results, total

    results, total = asyncio.run(run())

    affordable = int(DEPLOYMENT_CAP_USD / _QUARTER)
    assert sum(result["provider_calls"] for result in results) == affordable
    # The session whose call brings spend to exactly the cap stops after that
    # call, so it may report a stop instead of completion.
    assert sum(result["completed"] + result["stopped"] for result in results) == (
        2 * _SESSIONS_PER_PROCESS
    )
    assert sum(result["stopped"] for result in results) >= 2 * _SESSIONS_PER_PROCESS - affordable
    assert total == DEPLOYMENT_CAP_USD
