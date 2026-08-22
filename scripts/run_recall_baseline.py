#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from cayu.evals.recall_baseline import (
    load_recall_baseline_corpus,
    run_recall_baseline,
)
from cayu.runtime import InMemorySessionStore
from cayu.storage import InMemoryKnowledgeStore, SQLiteKnowledgeStore, SQLiteSessionStore


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cayu's hermetic bounded recall and admission baseline."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/memory/recall-corpus-v2.json"),
        help="Public hermetic or private external corpus JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this path instead of stdout.",
    )
    return parser.parse_args()


async def _run(corpus_path: Path) -> dict:
    corpus = load_recall_baseline_corpus(corpus_path)
    memory_result = await run_recall_baseline(
        corpus,
        InMemoryKnowledgeStore(),
        InMemorySessionStore(),
        backend="memory",
    )
    with tempfile.TemporaryDirectory(prefix="cayu-recall-baseline-") as directory:
        database_path = Path(directory) / "baseline.sqlite"
        sqlite_knowledge = SQLiteKnowledgeStore(database_path)
        sqlite_sessions = SQLiteSessionStore(database_path)
        try:
            sqlite_result = await run_recall_baseline(
                corpus,
                sqlite_knowledge,
                sqlite_sessions,
                backend="sqlite",
            )
        finally:
            await sqlite_sessions.close()
            await sqlite_knowledge.close()
    return {
        "schema_version": "cayu.recall_baseline_matrix.v2",
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
