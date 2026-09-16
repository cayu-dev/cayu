"""Durable, non-replayable case ownership for one native process launch."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu.evals._inspection_documents import ProcessDocuments, write_process_document
from cayu.evals.models import EvalRun, aggregate_eval_score, aggregate_eval_status


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class _Transition(_Receipt):
    state: Literal["claimed", "dispatching", "completed"]
    at: datetime


class _ResultReference(_Receipt):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Claim(_Receipt):
    case_id: str
    state: Literal["queued", "claimed", "dispatching", "completed"]
    worker: StrictInt | None
    slot: StrictInt | None
    transitions: list[_Transition]
    result: _ResultReference | None


class _Claims(_Receipt):
    schema_version: Literal[1]
    launch_id: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    processes: StrictInt
    max_concurrency: StrictInt
    cases: list[_Claim]


def initialize_claims(directory: Path, launch: dict, identity: dict) -> None:
    write_process_document(
        directory / "claims.json",
        {
            "schema_version": 1,
            "launch_id": launch["launch_id"],
            "fingerprint": identity["fingerprint"],
            "processes": launch["processes"],
            "max_concurrency": launch["max_concurrency"],
            "cases": [
                {
                    "case_id": case_id,
                    "state": "queued",
                    "worker": None,
                    "slot": None,
                    "transitions": [],
                    "result": None,
                }
                for case_id in identity["case_ids"]
            ],
        },
    )


def validate_claims(value: object, launch, identity) -> dict:
    value = _Claims.model_validate(value).model_dump(mode="json")
    if (
        value.get("schema_version") != 1
        or value.get("launch_id") != launch.launch_id
        or value.get("fingerprint") != identity.fingerprint
        or value.get("processes") != launch.processes
        or value.get("max_concurrency") != launch.max_concurrency
        or [row["case_id"] for row in value["cases"]] != list(identity.case_ids)
    ):
        raise ValueError("Case claims do not match their admitted launch")
    active = set()
    for position, row in enumerate(value["cases"]):
        state = row["state"]
        phases = [item["state"] for item in row["transitions"]]
        expected = {
            "queued": [],
            "claimed": ["claimed"],
            "dispatching": ["claimed", "dispatching"],
            "completed": ["claimed", "dispatching", "completed"],
        }
        if state not in expected or phases != expected[state]:
            raise ValueError("Invalid durable case transition history")
        if state == "queued":
            if row["worker"] is not None or row["slot"] is not None or row["result"] is not None:
                raise ValueError("Queued case has an owner or result")
            continue
        worker, slot = row["worker"], row["slot"]
        base, extra = divmod(launch.max_concurrency, launch.processes)
        if (
            type(worker) is not int
            or not 0 <= worker < launch.processes
            or type(slot) is not int
            or not 0 <= slot < base + (worker < extra)
        ):
            raise ValueError("Case claim exceeds admitted worker capacity")
        if state != "completed":
            if (worker, slot) in active or row["result"] is not None:
                raise ValueError("A worker slot has overlapping claims or an uncommitted result")
            active.add((worker, slot))
        elif (
            not isinstance(row["result"], dict)
            or row["result"].get("file") != f"case-result-{position}.json"
            or len(row["result"].get("sha256", "")) != 64
        ):
            raise ValueError("Completed claim lacks its immutable result reference")
    return value


class ProcessClaims:
    def __init__(self, directory, *, launch_id, fingerprint, worker):
        self.directory = directory
        self.launch_id = launch_id
        self.fingerprint = fingerprint
        self.worker = worker

    async def _change(self, operation):
        import fcntl

        while True:
            with (self.directory / "claims.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    try:
                        state = ProcessDocuments(self.directory).read("claims.json", required=True)
                        assert state is not None
                        if (
                            state["launch_id"] != self.launch_id
                            or state["fingerprint"] != self.fingerprint
                        ):
                            raise ValueError("Case ownership launch changed")
                        result = operation(state)
                        write_process_document(self.directory / "claims.json", state)
                        return result
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
            await asyncio.sleep(0.02)

    @staticmethod
    def _transition(row, state):
        row["state"] = state
        row["transitions"].append({"state": state, "at": datetime.now(UTC).isoformat()})

    async def claim(self, slot):
        def change(state):
            if any(
                row["worker"] == self.worker
                and row["slot"] == slot
                and row["state"] not in {"queued", "completed"}
                for row in state["cases"]
            ):
                raise ValueError("An unsettled slot cannot claim or replay another case")
            row = next((row for row in state["cases"] if row["state"] == "queued"), None)
            if row is None:
                return None
            row.update(worker=self.worker, slot=slot)
            self._transition(row, "claimed")
            return row["case_id"]

        return await self._change(change)

    async def dispatch(self, case_id, slot):
        def change(state):
            row = self._owned(state, case_id, slot, "claimed")
            self._transition(row, "dispatching")

        await self._change(change)

    def _owned(self, state, case_id, slot, phase):
        row = next(row for row in state["cases"] if row["case_id"] == case_id)
        if row["worker"] != self.worker or row["slot"] != slot or row["state"] != phase:
            raise ValueError("Case transition lacks its exact owner and prior phase")
        return row

    async def complete(self, case_id, slot, result):
        if [case.case_id for case in result.cases] != [case_id]:
            raise ValueError("Completion result does not match its owned case")

        def change(state):
            row = self._owned(state, case_id, slot, "dispatching")
            position = state["cases"].index(row)
            path = self.directory / f"case-result-{position}.json"
            if path.exists():
                raise ValueError("Existing completion bytes cannot be overwritten or replayed")
            write_process_document(path, result.model_dump(mode="json"))
            row["result"] = {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            self._transition(row, "completed")

        await self._change(change)


def read_claim_results(documents, rows, identity, max_concurrency):
    results = []
    for row in rows:
        if row["state"] != "completed":
            continue
        reference = row["result"]
        value = documents.read(reference["file"], required=True)
        if hashlib.sha256(documents.files[reference["file"]]).hexdigest() != reference["sha256"]:
            raise ValueError("Claim completion bytes changed")
        run = EvalRun.model_validate(value)
        if (
            run.suite_id != identity.suite_id
            or run.metadata != identity.metadata
            or run.run_contract is not None
            or [case.case_id for case in run.cases] != [row["case_id"]]
            or any(
                case.trial_policy.trial_count != 1
                or case.trial_policy.max_concurrency != max_concurrency
                for case in run.cases
            )
        ):
            raise ValueError("Claim result differs from its admitted case and policy")
        results.append(run)
    return results


def combine_claim_results(results, run_id):
    if not results:
        return None
    cases = tuple(case for run in results for case in run.cases)
    started = min(run.started_at for run in results)
    completed = max(run.completed_at for run in results)
    return EvalRun(
        run_id=run_id,
        suite_id=results[0].suite_id,
        cases=cases,
        status=aggregate_eval_status(case.status for case in cases),
        score=aggregate_eval_score(case.score for case in cases),
        started_at=started,
        completed_at=completed,
        duration_ms=int((completed - started).total_seconds() * 1000),
        metadata=results[0].metadata,
    )
