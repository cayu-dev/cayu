"""Compare publication workloads using disposable databases and fresh processes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode


async def initialize(addresses):
    for backend, address in addresses.items():
        store = (
            SQLiteSessionStore(address)
            if backend == "sqlite"
            else PostgresSessionStore(address, schema_mode=SchemaMode.CREATE)
        )
        try:
            if backend == "postgres":
                await store._ensure_ready()
        finally:
            await store.close()


def compare(args, addresses):
    orders = [0, 1] * 3
    random.Random(2395).shuffle(orders)
    summary = {}
    for backend, address in addresses.items():
        samples = {"base": [], "head": []}
        for index, order in enumerate(orders):
            for version in ["base", "head"] if order == 0 else ["head", "base"]:
                root = getattr(args, version)
                env = {
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
                    "CAYU_PUBLICATION_BENCHMARK_ADDRESS": address,
                }
                with (args.output / f"benchmark-{backend}-{version}-{index}.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("benchmark_worker.py")),
                            backend,
                            version,
                            str(index),
                            str(args.output),
                        ],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                samples[version].append(
                    json.loads(
                        (args.output / f"benchmark-{backend}-{version}-{index}.json").read_text()
                    )
                )
            print(backend, "paired sample", index + 1, "complete", flush=True)
        summary[backend] = {}
        for name in samples["base"][0]:
            rows = {
                "pairs": len(orders),
                "calls_per_sample": len(samples["base"][0][name]),
                "warmups": 10,
                "results_and_sql_match": True,
            }
            reference = samples["base"][0][name][0]
            assert all(
                all(
                    row[field] == reference[field]
                    for field in (
                        "result",
                        "queries",
                        "postgres_executemany_calls",
                        "postgres_executemany_parameter_sets",
                    )
                )
                for version in samples.values()
                for sample in version
                for row in sample[name]
            ), (backend, name, "outcomes or SQL differ")
            rows["sql_execution_counts"] = reference["queries"]
            rows["postgres_executemany_calls"] = reference["postgres_executemany_calls"]
            rows["postgres_executemany_parameter_sets"] = reference[
                "postgres_executemany_parameter_sets"
            ]
            for field in ["elapsed_ms", "cpu_ms"]:
                base = [statistics.median(row[field] for row in s[name]) for s in samples["base"]]
                head = [statistics.median(row[field] for row in s[name]) for s in samples["head"]]
                a, b = statistics.median(base), statistics.median(head)
                rows[field] = {
                    "base": a,
                    "head": b,
                    "percent": 100 * (b / a - 1),
                    "base_samples": base,
                    "head_samples": head,
                    "paired_changes_percent": [
                        100 * (h / b - 1) for b, h in zip(base, head, strict=True)
                    ],
                }
            summary[backend][name] = rows
        (args.output / "performance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Parent checkout")
    parser.add_argument("--head", type=Path, required=True, help="Changed checkout")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    args = parser.parse_args()
    args.base, args.head, args.output = (
        path.resolve() for path in (args.base, args.head, args.output)
    )
    server = os.environ["CAYU_TEST_POSTGRES_DSN"]
    args.output.mkdir(parents=True, exist_ok=False)
    database = "cayu_publication_benchmark_" + uuid4().hex
    with psycopg.connect(server, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        with tempfile.TemporaryDirectory(prefix="cayu-publication-benchmark-") as temporary:
            addresses = {
                "sqlite": str(Path(temporary) / "benchmark.sqlite"),
                "postgres": make_conninfo(server, dbname=database),
            }
            asyncio.run(initialize(addresses))
            compare(args, addresses)
    finally:
        with psycopg.connect(server, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


if __name__ == "__main__":
    main()
