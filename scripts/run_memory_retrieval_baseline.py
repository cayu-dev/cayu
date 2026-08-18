#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from cayu.evals.memory_baseline import (
    load_memory_retrieval_corpus,
    run_memory_retrieval_baseline,
)
from cayu.storage import InMemoryKnowledgeStore, SQLiteKnowledgeStore


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cayu's hermetic current-memory retrieval baseline."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/memory/corpus-v1.json"),
        help="Public hermetic or private external corpus JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this path instead of stdout.",
    )
    return parser.parse_args()


async def _run(corpus_path: Path) -> dict:
    corpus = load_memory_retrieval_corpus(corpus_path)
    memory = InMemoryKnowledgeStore()
    memory_result = await run_memory_retrieval_baseline(
        corpus,
        memory,
        backend="memory",
    )
    with tempfile.TemporaryDirectory(prefix="cayu-memory-baseline-") as directory:
        sqlite = SQLiteKnowledgeStore(Path(directory) / "baseline.sqlite")
        try:
            sqlite_result = await run_memory_retrieval_baseline(
                corpus,
                sqlite,
                backend="sqlite",
            )
        finally:
            await sqlite.close()
    return {
        "schema_version": "cayu.memory_retrieval_baseline_matrix.v1",
        "corpus_revision": corpus.corpus_revision,
        "results": [
            memory_result.model_dump(mode="json"),
            sqlite_result.model_dump(mode="json"),
        ],
    }


def main() -> None:
    arguments = _arguments()
    report = asyncio.run(_run(arguments.corpus))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
        return
    arguments.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
