"""Routes that run a project's data moves in the server's process and read their records.

`data_move_router(moves, auth=..., store=...)` serves, relative to where it is mounted:

- `GET /data-moves`: the declared moves, whether each is running here, and recent runs;
- `POST /data-moves/{name}/runs`: start a run, or resume one (`"resume": true`), in a
  background thread, answering 202 with its record;
- `GET /data-moves/runs/{run_id}`: one run's record, saved after every page.

`cayu serve` mounts it under its API path when the project declares moves in
`[tool.cayu.data_moves]`, with the server's own auth dependency, and `cayu cloud data-move`
calls it on a deployed Agent. The router can also be included in any FastAPI app.

A run belongs to the process that started it. When that process stops (a deploy, a crash,
a host that sleeps when idle), the run's record keeps the cursor of its last page, and
`POST ... {"resume": true}` goes on from it (`cayu.data_move_runs`). One run of a move goes
at a time in a process; a second start answers 409 while one is running. A record left
`running` that no thread here runs reads as `stopped`, so serve the routes from one web
process: another process's live run would read as stopped too. A server without
authentication refuses every route: a data move writes the application's data.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path as PathType
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field

from cayu._data_move_declarations import DataMoveDeclarationError, load_data_move
from cayu.data_move_runs import (
    DataMoveRun,
    DataMoveRunStore,
    RunOptions,
    continue_run,
    default_run_store,
    open_run,
    reopen_run,
)
from cayu.data_moves import DataMove

#: The path the router's routes live under, relative to where it is mounted.
DATA_MOVES_PATH = "/data-moves"
_RECENT_RUNS = 20
_INTERRUPTED = "interrupted: the process that ran it stopped; resume it to go on"

MoveSource = DataMove[Any] | str | Callable[[], DataMove[Any]]


class DataMoveRunRequest(BaseModel):
    """How to start a run, or `resume` (with an optional `run_id`) to go on with one."""

    model_config = ConfigDict(extra="forbid")

    resume: bool = False
    run_id: str | None = Field(default=None, max_length=64)
    dry_run: bool = False
    after: str | None = Field(default=None, max_length=4096)
    page_size: int = Field(default=100, ge=1, le=10_000)
    max_items: int | None = Field(default=None, ge=1)
    stop_on_failure: bool = False


def declared_data_move_sources(start: PathType | None = None) -> dict[str, MoveSource]:
    """The project's `[tool.cayu.data_moves]`, each loaded from the project when first used.

    Pass it to `data_move_router`, or to `create_agent_service(data_moves=...)` in a
    maintained service's factory. Nothing is imported until a move runs.
    """

    from cayu._data_move_declarations import declared_data_move_project, load_declared_data_move

    project, declared = declared_data_move_project(start)
    return {
        name: partial(load_declared_data_move, name, target, project=project)
        for name, target in declared.items()
    }


def data_move_router(
    moves: Mapping[str, MoveSource],
    *,
    auth: Callable[..., Any] | None,
    store: DataMoveRunStore | None = None,
) -> APIRouter:
    """Routes over `moves` (name to a `DataMove`, its `module:name`, or a function)."""

    records = default_run_store() if store is None else store
    # The run each move's thread in this process runs.
    active: dict[str, tuple[str, threading.Thread]] = {}
    lock = threading.Lock()
    router = APIRouter(dependencies=[] if auth is None else [Depends(auth)])

    def require_auth() -> None:
        if auth is None:
            raise HTTPException(
                409,
                detail={
                    "code": "data_moves_require_authentication",
                    "message": "This server has no authentication, so it runs no data moves.",
                },
            )

    def move_named(name: str) -> DataMove[Any]:
        source = moves.get(name)
        if source is None:
            raise HTTPException(
                404,
                detail={
                    "code": "data_move_not_declared",
                    "message": f"No data move {name!r} is declared in [tool.cayu.data_moves].",
                },
            )
        try:
            move = (
                source
                if isinstance(source, DataMove)
                else (load_data_move(source) if isinstance(source, str) else source())
            )
        except DataMoveDeclarationError as exc:
            raise HTTPException(
                409, detail={"code": "data_move_unavailable", "message": str(exc)}
            ) from None
        if not isinstance(move, DataMove):
            raise HTTPException(
                409,
                detail={"code": "data_move_unavailable", "message": f"{name!r} is no DataMove."},
            )
        # Records and routes know a move by its declared name.
        return move if move.name == name else replace(move, name=name)

    def running(name: str) -> bool:
        entry = active.get(name)
        return entry is not None and entry[1].is_alive()

    def live_runs() -> frozenset[str]:
        """The runs a thread here is running, taken before their records are read.

        A thread saves its last record before it ends, so a record read after this snapshot
        is final for any run that isn't in it.
        """

        return frozenset(run_id for run_id, thread in active.values() if thread.is_alive())

    def answer(run: DataMoveRun, live: frozenset[str]) -> dict[str, Any]:
        """The record, with a `running` run that no thread here runs shown as stopped.

        A run is left `running` when the process that ran it stopped mid-way; nothing will
        finish it, so it reads as stopped, ready for `resume`. The record isn't changed.
        """

        record = run.to_json()
        if run.status == "running" and run.run_id not in live:
            record["status"] = "stopped"
            record["error"] = record["error"] or _INTERRUPTED
        return record

    @router.get(DATA_MOVES_PATH)
    def list_data_moves() -> dict[str, Any]:
        require_auth()
        live = live_runs()
        return {
            "moves": [{"name": name, "running": running(name)} for name in sorted(moves)],
            "runs": [answer(run, live) for run in records.list_runs(limit=_RECENT_RUNS)],
        }

    @router.get(DATA_MOVES_PATH + "/runs/{run_id}")
    def read_data_move_run(run_id: str = Path(max_length=64)) -> dict[str, Any]:
        require_auth()
        live = live_runs()
        try:
            run = records.get(run_id)
        except ValueError:
            run = None
        if run is None:
            raise HTTPException(
                404,
                detail={"code": "data_move_run_not_found", "message": f"No run {run_id!r}."},
            )
        return answer(run, live)

    @router.post(DATA_MOVES_PATH + "/{name}/runs", status_code=202)
    def start_data_move_run(
        body: DataMoveRunRequest, name: str = Path(max_length=128)
    ) -> dict[str, Any]:
        require_auth()
        move = move_named(name)
        with lock:
            if running(name):
                raise HTTPException(
                    409,
                    detail={
                        "code": "data_move_running",
                        "message": f"A run of {name!r} is already running in this process.",
                    },
                )
            try:
                if body.resume:
                    run = reopen_run(move, records, run_id=body.run_id)
                elif body.run_id is not None and records.get(body.run_id) is not None:
                    raise HTTPException(
                        409,
                        detail={
                            "code": "data_move_run_exists",
                            "message": f"A run {body.run_id!r} is already recorded.",
                        },
                    )
                else:
                    run = open_run(
                        move,
                        records,
                        dry_run=body.dry_run,
                        after=body.after,
                        options=RunOptions(
                            page_size=body.page_size,
                            max_items=body.max_items,
                            stop_on_failure=body.stop_on_failure,
                        ),
                        run_id=body.run_id,
                    )
            except LookupError as exc:
                raise HTTPException(
                    404, detail={"code": "data_move_run_not_found", "message": str(exc)}
                ) from None
            except ValueError as exc:
                raise HTTPException(
                    409, detail={"code": "data_move_run_refused", "message": str(exc)}
                ) from None
            thread = threading.Thread(
                target=_run_in_background,
                args=(move, records, run),
                name=f"cayu-data-move-{name}",
                daemon=True,
            )
            active[name] = (run.run_id, thread)
            thread.start()
        return run.to_json()

    return router


def _run_in_background(move: DataMove[Any], store: DataMoveRunStore, run: DataMoveRun) -> None:
    try:
        continue_run(move, store, run)
    except Exception as exc:  # The record says the run stopped, and why; log it as well.
        line = {
            "event": "cayu.data_move.stopped",
            "move": run.move,
            "run_id": run.run_id,
            "error": type(exc).__name__,
        }
        print(json.dumps(line), file=sys.stderr)
