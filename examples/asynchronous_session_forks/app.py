"""Run the deterministic asynchronous-session-forks tracer."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from examples.asynchronous_session_forks.scenario import run_asynchronous_fork_trace


async def main() -> None:
    result = await run_asynchronous_fork_trace()
    payload = asdict(result)
    payload["source"] = result.source.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
