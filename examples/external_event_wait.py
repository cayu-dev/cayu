"""Credential-free external-job wait across fresh Python processes.

Run: python examples/external_event_wait.py demo /absolute/new/demo-directory
The directory is retained for inspection. No external providers, network, or paid calls.
Individual phases (start, deliver, resume) can also run in separate terminals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    ExternalCorrelationRequest,
    ExternalEventDelivery,
    ExternalEventWaits,
    ExternalWaitAccessPolicy,
    ExternalWaitContext,
    ExternalWaitHost,
    ExternalWaitRegistration,
    ExternalWaitScope,
    Message,
    RunRequest,
    SessionExternalWaitAdapter,
    SQLiteSessionStore,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent


class LocalJobPolicy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        # This is a local CLI identity. An HTTP host must authenticate its
        # webhook before constructing the context, not trust event JSON.
        return (
            context.principal == "local-example-host"
            and scope.application_scope == "external-job-example"
            and source == "simulated-renderer"
        )


CONTEXT = ExternalWaitContext(principal="local-example-host")


def retain_once(path: Path, value: dict) -> dict:
    """A tiny simulated external system's exact idempotency-key operation.

    This is job submission state, not a wait winner or continuation outbox.
    Production uses the external service's idempotency/reconciliation entrance.
    """
    encoded = json.dumps(value, sort_keys=True)
    try:
        with path.open("x") as stream:
            stream.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise ValueError("Simulated job idempotency key conflicts.") from None
    return value


async def phase(command: str, directory: Path, mode: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    request_file = directory / "correlation.json"
    if command == "start" and not request_file.exists():
        request = ExternalCorrelationRequest(
            scope=ExternalWaitScope(application_scope="external-job-example", generation=1),
            source="simulated-renderer",
            correlation_key="render-42",
            deadline=datetime.now(UTC) + timedelta(seconds=2) if mode == "timeout" else None,
        )
        retain_once(request_file, request.model_dump(mode="json"))
    request = ExternalCorrelationRequest.model_validate_json(request_file.read_text())
    store = SQLiteSessionStore(directory / "sessions.sqlite")
    waits = ExternalEventWaits(store=store, access_policy=LocalJobPolicy())
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("Done."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="job-agent", model="local-scripted-model"))
    adapter = SessionExternalWaitAdapter(app, waits)
    try:
        if command == "start":
            correlation = await waits.reserve_correlation(request, context=CONTEXT)
            # Reserve first. Lost job acknowledgement is reconciled by repeating
            # this exact external job operation, never by inventing another key.
            retain_once(
                directory / "external-job.json", {"job_id": "render-42", "result": "local-video"}
            )
        else:
            snapshot = await waits.lookup(request, context=CONTEXT)
            if snapshot is None:
                raise RuntimeError("Start the example before delivering or resuming.")
            correlation = snapshot.correlation
        registration = ExternalWaitRegistration(
            correlation=correlation,
            operation_key="wait-render-42",
            projector_id="json",
            projector_version=1,
        )
        if command == "deliver" or (command == "start" and mode == "early"):
            job = json.loads((directory / "external-job.json").read_text())
            delivery = ExternalEventDelivery(
                correlation=correlation, delivery_id="completion-42", payload_json=json.dumps(job)
            )
            first = await waits.deliver(delivery, context=CONTEXT)
            assert await waits.deliver(delivery, context=CONTEXT) == first
        if command == "start":
            await waits.register(registration, context=CONTEXT)
            await adapter.run_to_wait(
                RunRequest(
                    agent_name="job-agent",
                    session_id="job-session-42",
                    messages=[
                        Message.text("user", "The external job is submitted; await its result.")
                    ],
                ),
                registration,
                context=CONTEXT,
            )
        elif command == "resume":
            if request.deadline is not None:
                await asyncio.sleep(
                    max(0, (request.deadline - datetime.now(UTC)).total_seconds()) + 0.02
                )
            host = ExternalWaitHost(adapter, context=CONTEXT)
            await host.service_once(scope=request.scope, source=request.source)
        snapshot = await waits.inspect(correlation, context=CONTEXT)
        return {
            "phase": command,
            "mode": mode,
            "provider_dispatches_this_process": len(provider.requests),
            "outcome": None if snapshot.outcome is None else snapshot.outcome.kind,
            "handoff": snapshot.handoff,
            "pending_handoff": snapshot.pending_handoff,
        }
    finally:
        await app.aclose()
        await waits.aclose()
        await store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "start", "deliver", "resume"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mode", choices=("event", "early", "timeout"), default="event")
    args = parser.parse_args()
    if args.command != "demo":
        print(json.dumps(asyncio.run(phase(args.command, args.directory, args.mode))))
        return
    if args.directory.exists():
        parser.error("Use a new demo directory; existing evidence is never reset.")
    args.directory.mkdir(parents=True)
    for mode in ("event", "early", "timeout"):
        commands = (
            ["start", "deliver", "resume", "resume"]
            if mode == "event"
            else ["start", "resume", "resume"]
        )
        results = []
        for command in commands:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    command,
                    str(args.directory / mode),
                    "--mode",
                    mode,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(completed.stdout)
            results.append(result)
            print(json.dumps(result), flush=True)
        assert sum(result["provider_dispatches_this_process"] for result in results) == 2
        assert results[-1]["provider_dispatches_this_process"] == 0
        assert results[-1]["handoff"] == "settled" and not results[-1]["pending_handoff"]
        assert results[-1]["outcome"] == ("timeout" if mode == "timeout" else "event")
    print("Fresh-process event, early-event and timeout journeys passed; evidence retained.")


if __name__ == "__main__":
    main()
