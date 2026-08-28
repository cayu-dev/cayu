from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main(agent_path: Path, request_path: Path) -> None:
    specification = importlib.util.spec_from_file_location("candidate_agent", agent_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("candidate agent is not importable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = module.run(request)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "output": output,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
