"""Time the dashboard's artifact reads over the artifact index at production size.

Run with ``uv run python scripts/benchmark_artifact_index_reads.py``. It saves
``--artifacts`` artifacts (100,000 by default) across ``--sessions`` sessions, half
to a local artifact store and half to an S3 store (the in-process fake S3 client
the tests use, unless ``--s3-bucket`` names a real bucket or ``--s3-endpoint`` a
MinIO), through the real ``put_bytes``, so the commit hook writes each index row.
It fills the index with one reindex walk of each store, then times every
artifact read the dashboard makes through the real server, each on a freshly
opened store and server (cold), and checks that none of them touched an
artifact store.

Pass ``--postgres DSN`` to use a Postgres session store instead of SQLite, and
``--keep DIR`` to keep the stores for another run with ``--reuse``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from cayu import AgentSpec, CayuApp, RunRequest
from cayu.artifacts._index import reindex_artifact_store
from cayu.artifacts.aws_s3 import S3ArtifactStore
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionIdentity

_READS = ("list", "scan_metadata", "count_session_artifacts", "read_bytes", "read_range")
_KINDS = (
    ("report-{index}.md", "text/markdown"),
    ("chart-{index}.png", "image/png"),
    ("data-{index}.csv", "text/csv"),
    ("bundle-{index}.zip", "application/zip"),
)


def _memory_s3() -> Any:
    """The in-process fake S3 client the artifact store tests use."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.artifacts.test_aws_s3 import _S3Client

    return _S3Client()


def _stores(root: Path, args: argparse.Namespace):
    local = LocalArtifactStore(root / "local", store_id="local")
    if args.s3_bucket:
        s3 = S3ArtifactStore(
            args.s3_bucket,
            prefix=f"bench-{root.name}",
            store_id="s3",
            endpoint_url=args.s3_endpoint,
        )
    else:
        s3 = S3ArtifactStore("bench", store_id="s3", client=_FAKE_S3)
    return local, s3


_FAKE_S3: Any = None


def _session_store(root: Path, args: argparse.Namespace):
    if args.postgres:
        from cayu.storage.postgres import PostgresSessionStore

        return PostgresSessionStore(args.postgres)
    from cayu.storage import SQLiteSessionStore

    return SQLiteSessionStore(root / "sessions.sqlite")


def _app(store, local, s3) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(AgentSpec(name="writer", model="configured"))
    for artifact_store in (local, s3):
        app.register_environment(
            Environment(EnvironmentSpec(name=artifact_store.id), artifact_store=artifact_store)
        )
    return app


async def _seed(root: Path, args: argparse.Namespace) -> dict[str, float]:
    store = _session_store(root, args)
    local, s3 = _stores(root, args)
    _app(store, local, s3)  # binds both stores to the session store's index
    try:
        started = time.perf_counter()
        for start in range(0, args.sessions, 200):
            await asyncio.gather(
                *(
                    store.create(
                        RunRequest(
                            session_id=f"bench-{index:05}",
                            agent_name="writer",
                            title=f"Quarterly review {index}",
                            messages=[],
                        ),
                        identity=SessionIdentity(provider_name="configured", model="configured"),
                    )
                    for index in range(start, min(args.sessions, start + 200))
                )
            )
        gate = asyncio.Semaphore(32)

        async def put(index: int) -> None:
            filename, content_type = _KINDS[index % len(_KINDS)]
            artifact_store = (local, s3)[index % 2]
            async with gate:
                await artifact_store.put_bytes(
                    b"x" * (index % 997),
                    filename=filename.format(index=index),
                    content_type=content_type,
                    session_id=f"bench-{index % args.sessions:05}",
                    agent_name="writer",
                    environment_name=artifact_store.id,
                )

        for start in range(0, args.artifacts, 2000):
            await asyncio.gather(*(put(i) for i in range(start, min(args.artifacts, start + 2000))))
            print(f"saved {min(args.artifacts, start + 2000)} artifacts", file=sys.stderr)
        saved = time.perf_counter() - started
        timings = {"seed_seconds": round(saved, 1)}
        for artifact_store in (local, s3):
            started = time.perf_counter()
            drift = await reindex_artifact_store(store, artifact_store.id, artifact_store)
            timings[f"fill_{artifact_store.id}_seconds"] = round(time.perf_counter() - started, 1)
            assert not drift.drift, drift.as_json()
        return timings
    finally:
        await store.close()


@asynccontextmanager
async def _server(root: Path, args: argparse.Namespace, calls: Counter):
    store = _session_store(root, args)
    local, s3 = _stores(root, args)
    for artifact_store in (local, s3):
        for name in _READS:
            original = getattr(artifact_store, name)

            def counted(*a, _original=original, _name=f"{artifact_store.id}.{name}", **k):
                calls[_name] += 1
                return _original(*a, **k)

            setattr(artifact_store, name, counted)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_server(_app(store, local, s3), config=ServerConfig.local_development())
            ),
            base_url="http://bench",
            timeout=600,
        ) as client:
            yield client
    finally:
        await store.close()


def _requests(args: argparse.Namespace) -> dict[str, tuple[str, dict[str, Any]]]:
    middle = args.artifacts // 2
    return {
        "artifacts page": ("/api/artifacts", {"limit": 100}),
        "artifacts page 2 (cursor)": ("/api/artifacts", {"limit": 100, "cursor": "__next__"}),
        "artifacts sorted by name": ("/api/artifacts", {"limit": 100, "order_by": "filename_asc"}),
        "artifacts sorted by size": (
            "/api/artifacts",
            {"limit": 100, "order_by": "size_bytes_desc"},
        ),
        "search: one file": (
            "/api/artifacts",
            {"limit": 100, "q": f"report-{middle - middle % 4}"},
        ),
        "search: a kind": ("/api/artifacts", {"limit": 100, "q": "png image"}),
        "search: a session title": ("/api/artifacts", {"limit": 100, "q": "quarterly review 77"}),
        "search: two letters": ("/api/artifacts", {"limit": 100, "q": "zi"}),
        "session Artifacts tab": ("/api/artifacts", {"limit": 6, "session_id": "bench-00077"}),
        "environment artifacts": ("/api/artifacts", {"limit": 5, "environment_name": "s3"}),
        "overview counts (100 sessions)": (
            "/api/sessions",
            {"include": "overview", "limit": 100},
        ),
    }


async def _measure(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, (path, params) in _requests(args).items():
        elapsed = []
        calls: Counter = Counter()
        for _ in range(args.samples):
            async with _server(root, args, calls) as client:
                query = dict(params)
                if query.get("cursor") == "__next__":
                    first = await client.get(path, params={"limit": 100})
                    query["cursor"] = first.json()["next_cursor"]
                started = time.perf_counter()
                response = await client.get(path, params=query)
                elapsed.append(time.perf_counter() - started)
                assert response.status_code == 200, response.text
        ordered = sorted(elapsed)
        results[name] = {
            "median_ms": round(statistics.median(ordered) * 1000, 1),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000, 1),
            "store_calls": dict(calls),
        }
    return results


async def main() -> None:
    global _FAKE_S3
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=int, default=100_000)
    parser.add_argument("--sessions", type=int, default=2_000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--postgres", metavar="DSN")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-endpoint")
    parser.add_argument("--keep", type=Path, help="Keep the stores in this directory.")
    parser.add_argument("--reuse", action="store_true", help="Reuse the stores in --keep.")
    args = parser.parse_args()
    if args.reuse and (args.keep is None or not args.s3_bucket):
        parser.error("--reuse needs --keep and a real --s3-bucket (the fake S3 is in memory).")
    _FAKE_S3 = None if args.s3_bucket else _memory_s3()
    with tempfile.TemporaryDirectory(prefix="cayu-artifact-bench-") as scratch:
        root = args.keep or Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        timings = {} if args.reuse else await _seed(root, args)
        results = await _measure(root, args)
    report = {
        "artifacts": args.artifacts,
        "sessions": args.sessions,
        "session_store": "postgres" if args.postgres else "sqlite",
        "s3": "fake" if not args.s3_bucket else "bucket",
        **timings,
        "requests": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
