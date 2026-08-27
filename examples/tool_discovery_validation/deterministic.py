from __future__ import annotations

import asyncio

from examples.tool_discovery_validation.scenario import run_scenario


def main() -> None:
    report = asyncio.run(run_scenario())
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
