"""One fresh-process sample for the paired native publication benchmark."""

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import cayu
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._checkpoint_mutations import runtime_publication_checkpoint_mutation
from cayu.sessions._runtime_publication import RuntimePublicationRequest
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage import PostgresSessionStore, SQLiteSessionStore

backend, version, index, output = sys.argv[1:]
p = Path(output)
assert Path(cayu.__file__).resolve().is_relative_to(Path.cwd() / "src")
config = {"address": os.environ["CAYU_PUBLICATION_BENCHMARK_ADDRESS"]}


async def run():
    queries = []
    postgres_batch_calls = Counter()
    postgres_batch_parameter_sets = Counter()
    original = sqlite3.connect
    if backend == "sqlite":

        def traced(*args, **kwargs):
            c = original(*args, **kwargs)
            if config["address"] in str(args[0]):
                c.set_trace_callback(
                    lambda q: (
                        None
                        if q.startswith("--")
                        else queries.append(re.sub(r"'(?:''|[^'])*'", "?", " ".join(q.split())))
                    )
                )
            return c

        sqlite3.connect = traced
    else:
        import psycopg

        original_execute = psycopg.AsyncCursor.execute
        original_executemany = psycopg.AsyncCursor.executemany

        async def execute(cur, query, params=None, **kwargs):
            queries.append(" ".join(str(query).split()))
            return await original_execute(cur, query, params, **kwargs)

        async def executemany(cur, query, params_seq, **kwargs):
            statement = " ".join(str(query).split())
            postgres_batch_calls[statement] += 1
            postgres_batch_parameter_sets[statement] += 0

            def counted_parameters():
                for params in params_seq:
                    queries.append(statement)
                    postgres_batch_parameter_sets[statement] += 1
                    yield params

            return await original_executemany(cur, query, counted_parameters(), **kwargs)

        psycopg.AsyncCursor.execute = execute
        psycopg.AsyncCursor.executemany = executemany
    store = (
        SQLiteSessionStore(config["address"])
        if backend == "sqlite"
        else PostgresSessionStore(config["address"])
    )
    serial = 0
    results = {}

    async def seed():
        nonlocal serial
        serial += 1
        sid = f"{backend}-{version}-{index}-{serial}"
        await store.create(
            RunRequest(agent_name="assistant", session_id=sid, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        session = await store.transition_status(
            sid, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
        )
        request = RuntimePublicationRequest(
            publication_id="publication",
            kind="model-step",
            interaction_id="interaction",
            intent={"step": 1},
            mutation=runtime_publication_checkpoint_mutation(None, {"phase": "published"}),
            transcript_messages=(Message.text("assistant", "answer"),),
            events=(
                Event(
                    id="completion-" + sid,
                    type=EventType.MODEL_COMPLETED,
                    session_id=sid,
                    interaction_id="interaction",
                    payload={"step": 1},
                ),
            ),
        )
        return session, request

    async def publish(session, request):
        return await store.publish_runtime_publication(
            session.id,
            request=request,
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=0,
        )

    try:
        if backend == "postgres":
            # Verify real batched writes outside the measured workloads. Count
            # parameters as the driver consumes them, including one-shot inputs.
            async with (
                await psycopg.AsyncConnection.connect(config["address"]) as connection,
                connection.cursor() as cursor,
            ):
                await cursor.execute(
                    "CREATE TEMP TABLE publication_count_probe (value integer) ON COMMIT DROP"
                )
                queries.clear()
                postgres_batch_calls.clear()
                postgres_batch_parameter_sets.clear()
                statement = "INSERT INTO publication_count_probe (value) VALUES (%s)"
                await cursor.executemany(statement, [(0,), (1,)])
                await cursor.executemany(statement, ((value,) for value in range(2, 5)))
                await cursor.executemany(statement, ())
                assert queries == [statement] * 5
                assert postgres_batch_calls == {statement: 3}
                assert postgres_batch_parameter_sets == {statement: 5}
                await cursor.execute("SELECT value FROM publication_count_probe ORDER BY value")
                assert await cursor.fetchall() == [(value,) for value in range(5)]
        committed, request = await seed()
        await publish(committed, request)
        for workload in ["receipt_read", "runtime_replay", "runtime_write", "checkpoint_events"]:
            rows = []
            count = 30 if workload in ["runtime_write", "checkpoint_events"] else 100
            for iteration in range(10 + count):
                if workload in ["runtime_write", "checkpoint_events"]:
                    session, publication = await seed()
                event = (
                    None
                    if workload != "checkpoint_events"
                    else Event(
                        id="checkpoint-" + session.id,
                        type="custom.publication",
                        session_id=session.id,
                        payload={"phase": "checkpoint"},
                    )
                )
                queries.clear()
                postgres_batch_calls.clear()
                postgres_batch_parameter_sets.clear()
                start = time.perf_counter_ns()
                cpu = time.process_time_ns()
                if workload == "receipt_read":
                    r = await store.load_runtime_publication_receipt(committed.id, "publication")
                    value = (
                        r.kind,
                        r.transcript_start_cursor,
                        r.transcript_end_cursor,
                        len(r.appended_event_ids),
                    )
                elif workload == "runtime_replay":
                    r = await publish(committed, request)
                    value = (r.replayed, r.receipt.kind, r.receipt.transcript_end_cursor)
                elif workload == "runtime_write":
                    r = await publish(session, publication)
                    value = (r.replayed, r.receipt.kind, r.receipt.transcript_end_cursor)
                else:
                    r = await store.publish_checkpoint_and_events(
                        session.id,
                        checkpoint_transform=lambda _s, c: {**(c or {}), "phase": "checkpoint"},
                        events=[event],
                        expected_statuses={SessionStatus.RUNNING},
                        expected_run_epoch=session.run_epoch,
                        expected_transcript_cursor=0,
                    )
                    value = (str(r.status), r.run_epoch)
                cpu_ms = (time.process_time_ns() - cpu) / 1e6
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                if iteration >= 10:
                    rows.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "cpu_ms": cpu_ms,
                            "result": hashlib.sha256(json.dumps(value).encode()).hexdigest(),
                            "queries": dict(Counter(queries)),
                            "postgres_executemany_calls": dict(postgres_batch_calls),
                            "postgres_executemany_parameter_sets": dict(
                                postgres_batch_parameter_sets
                            ),
                        }
                    )
            assert all(
                all(
                    x[field] == rows[0][field]
                    for field in (
                        "result",
                        "queries",
                        "postgres_executemany_calls",
                        "postgres_executemany_parameter_sets",
                    )
                )
                for x in rows
            ), workload
            results[workload] = rows
        (p / f"benchmark-{backend}-{version}-{index}.json").write_text(json.dumps(results) + "\n")
    finally:
        await store.close()
        sqlite3.connect = original
        if backend == "postgres":
            psycopg.AsyncCursor.execute = original_execute
            psycopg.AsyncCursor.executemany = original_executemany


asyncio.run(run())
