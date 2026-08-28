#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from cayu.evals.knowledge_maintenance import (
    load_knowledge_maintenance_evaluation_corpus,
    run_knowledge_maintenance_evaluation,
)
from cayu.storage import InMemoryKnowledgeStore, SQLiteKnowledgeStore


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Cayu's provider-free reviewed knowledge-maintenance evaluation."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/memory/knowledge-maintenance-corpus-v1.json"),
        help="Public hermetic or private external corpus JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the in-memory/SQLite report to this path instead of stdout.",
    )
    return parser.parse_args()


async def _run(corpus_path: Path) -> dict:
    corpus = load_knowledge_maintenance_evaluation_corpus(corpus_path)
    memory_result = await run_knowledge_maintenance_evaluation(
        corpus,
        InMemoryKnowledgeStore(),
        backend="memory",
    )
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-maintenance-eval-") as directory:
        sqlite = SQLiteKnowledgeStore(Path(directory) / "evaluation.sqlite")
        try:
            sqlite_result = await run_knowledge_maintenance_evaluation(
                corpus,
                sqlite,
                backend="sqlite",
            )
        finally:
            await sqlite.close()
    return {
        "schema_version": "cayu.knowledge_maintenance_evaluation_matrix.v1",
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
