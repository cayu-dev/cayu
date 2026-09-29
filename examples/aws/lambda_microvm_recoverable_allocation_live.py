"""Fresh-process live contract for recoverable Lambda MicroVM allocation.

Each phase runs in a new Python process so recovery can use only durable,
non-secret allocation metadata. The scenario proves that an acknowledgement
lost to worker death is adopted by client-token replay, concurrent recovery
converges on one MicroVM, changed parameters never create a replacement, a
suspended MicroVM is reattached by durable identity with a verified
unprivileged agent boundary, disposal is proven only from terminal readback,
and reaping leaves zero unaccounted MicroVMs.

Guest egress enforcement is outside this contract: it needs the integrated
private proxy and is covered by ``aws-lambda-microvm-metadata-isolation-live``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from cayu import ExecCommand, LambdaMicroVMRunner
from cayu.egress import VirtualEgressAllocationPreparation, VirtualEgressAllocationReap
from cayu.egress.aws_lambda_microvm_adapter import (
    LambdaMicroVMEgressAdapter,
    _client_token,
    _verify_agent_privilege_boundary,
)
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.runners import LambdaMicroVMClientTokenConflict, LambdaMicroVMOwnershipSuperseded
from cayu.runners.aws_lambda_microvm import _control_client, run_microvm_with_client_token

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_MODULE = "examples.aws.lambda_microvm_recoverable_allocation_live"
_DEATH_EXIT_CODE = 17
_MAXIMUM_DURATION_SECONDS = 900


def _configuration() -> dict[str, str]:
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    connector = os.environ.get("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    if not image.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE to a built MicroVM image ARN.")
    if not connector.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR to a network connector ARN.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")
    return {"image": image, "connector": connector, "region": region}


def _adapter(config: dict[str, str]) -> LambdaMicroVMEgressAdapter:
    return LambdaMicroVMEgressAdapter(
        region_name=config["region"],
        egress_network_connector_arn=config["connector"],
        # Allocation never contacts the proxy; the exposure only satisfies the
        # adapter's construction contract.
        exposure=VpcTaskProxyExposure("10.0.0.10"),
        runner_options={"maximum_duration_in_seconds": _MAXIMUM_DURATION_SECONDS},
    )


def _token(adapter: LambdaMicroVMEgressAdapter, allocation_id: str) -> str:
    generation = adapter.allocation_adapter_generation
    assert generation is not None
    return _client_token(generation, allocation_id)


async def _phase(name: str, state_path: Path) -> dict[str, Any]:
    config = _configuration()
    state = json.loads(state_path.read_text())
    adapter = _adapter(config)
    metadata = state["metadata"]
    client, _owned = _control_client(
        client=None, region_name=config["region"], profile_name=None, endpoint_url=None
    )
    if name == "submit-then-die":
        response = await run_microvm_with_client_token(
            client,
            adapter._run_options(metadata),
            client_token=_token(adapter, state["allocations"]["lost_ack"]),
        )
        # Observational evidence only; recovery never reads this value.
        print(json.dumps({"observed_microvm_id": response["microvmId"]}), flush=True)
        os._exit(_DEATH_EXIT_CODE)
    if name in {"recover", "replay-concurrent"}:
        key = "lost_ack" if name == "recover" else "concurrent"
        runner = await LambdaMicroVMRunner.create(
            metadata["image_arn"],
            region_name=config["region"],
            image_version=metadata["image_version"],
            ingress_network_connectors=adapter.ingress_network_connectors,
            egress_network_connectors=[config["connector"]],
            maximum_duration_in_seconds=metadata["maximum_duration_s"],
            client_token=_token(adapter, state["allocations"][key]),
            close_action="none",
        )
        try:
            result = await runner.exec(ExecCommand.process("python3", "-c", "print('adopted')"))
        except LambdaMicroVMOwnershipSuperseded:
            # A concurrent recovery worker claimed after this one; only the
            # latest claimant may execute.
            await runner.close()
            return {"microvm_id": runner.microvm_id, "exec": "fenced"}
        await runner.close()
        if result.exit_code != 0 or result.stdout != "adopted\n":
            raise RuntimeError("Recovered MicroVM did not execute a command.")
        return {"microvm_id": runner.microvm_id, "exec": "executed"}
    if name == "stale-owner":
        runner = await LambdaMicroVMRunner.from_existing(
            state["identity"]["microvm_id"], region_name=config["region"], close_action="none"
        )
        await runner.exec(ExecCommand.process("true"))
        go = state_path.with_suffix(".successor-attached")
        (state_path.with_suffix(".stale-ready")).write_text("ready")
        deadline = time.monotonic() + 120
        while not go.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("successor never attached")
            await asyncio.sleep(0.5)
        outcomes = {}
        # The stale owner has not seen a rejection yet, so only the sidecar's
        # lifecycle lease can stop this direct terminate.
        try:
            await runner.terminate()
            outcomes["terminate"] = "allowed"
        except LambdaMicroVMOwnershipSuperseded:
            outcomes["terminate"] = "fenced"
        try:
            await runner.exec(ExecCommand.process("true"))
            outcomes["exec"] = "allowed"
        except (LambdaMicroVMOwnershipSuperseded, RuntimeError):
            outcomes["exec"] = "fenced"
        runner.close_action = "none"
        with contextlib.suppress(Exception):
            await runner.close()
        return outcomes
    if name == "successor":
        runner = await LambdaMicroVMRunner.from_existing(
            state["identity"]["microvm_id"], region_name=config["region"], close_action="none"
        )
        state_path.with_suffix(".successor-attached").write_text("attached")
        await asyncio.sleep(8)  # let the stale owner try while this owner is current
        result = await runner.exec(ExecCommand.process("python3", "-c", "print('successor')"))
        await runner.close()
        return {"successor_exec": result.stdout.strip()}
    if name == "suspend":
        runner = await LambdaMicroVMRunner.from_existing(
            state["identity"]["microvm_id"], region_name=config["region"], close_action="suspend"
        )
        await runner.close()  # returns only after SUSPENDED readback
        return {"suspended": runner.microvm_id}
    if name == "reconnect":
        identity = adapter.validate_reconnect_metadata(state["identity"])
        if await adapter.is_allocation_disposed(identity):
            raise RuntimeError("A suspended MicroVM was reported as disposed.")
        runner = await LambdaMicroVMRunner.from_existing(
            identity["microvm_id"], region_name=identity["region"], close_action="none"
        )
        if (runner.endpoint, runner.image_identifier, runner.image_version) != (
            identity["endpoint"],
            identity["image_identifier"],
            identity["image_version"],
        ):
            raise RuntimeError("Reconnect resolved a different MicroVM identity.")
        await _verify_agent_privilege_boundary(runner, timeout_s=30)
        await runner.close()
        return {"reconnected": runner.microvm_id, "privilege_boundary": "verified"}
    if name == "disposal":
        return {"disposed": await adapter.is_allocation_disposed(state["identity"])}
    if name == "conflict":
        options = adapter._run_options(metadata)
        options["maximumDurationInSeconds"] = metadata["maximum_duration_s"] + 1
        try:
            await run_microvm_with_client_token(
                client, options, client_token=_token(adapter, state["allocations"]["lost_ack"])
            )
        except LambdaMicroVMClientTokenConflict:
            return {"conflict": "rejected"}
        raise RuntimeError("Changed client-token parameters were accepted.")
    if name == "reap":
        reaped = []
        for key, allocation_id in state["allocations"].items():
            await adapter.reap_allocation(
                VirtualEgressAllocationReap(
                    allocation_id=allocation_id,
                    session_id="live-recoverable-allocation",
                    environment_name="sandbox",
                    image=config["image"],
                    allocation_metadata=metadata,
                )
            )
            reaped.append(key)
        return {"reaped": reaped}
    raise SystemExit(f"Unknown phase {name!r}.")


def _run_phase(name: str, state_path: Path, *, expect_exit: int = 0) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", _MODULE, "--phase", name, "--state", str(state_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != expect_exit:
        raise RuntimeError(
            f"Phase {name} exited {completed.returncode}: {completed.stderr.strip()[-2000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def _census(client: Any, image: str, started: float) -> dict[str, str]:
    """Return every MicroVM of ``image`` started by this run, with its state."""

    states: dict[str, str] = {}
    next_token: str | None = None
    while True:
        page = client.list_microvms(
            imageIdentifier=image, **({"nextToken": next_token} if next_token else {})
        )
        for item in page["items"]:
            if item["startedAt"].timestamp() >= started - 5:
                states[item["microvmId"]] = client.get_microvm(microvmIdentifier=item["microvmId"])[
                    "state"
                ]
        next_token = page.get("nextToken")
        if not next_token:
            break
    return states


async def main() -> None:
    if os.environ.get("CAYU_LAMBDA_MICROVM_RECOVERABLE_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_RECOVERABLE_LIVE=1 to run this contract.")
    config = _configuration()
    adapter = _adapter(config)
    started = time.time()
    metadata = await adapter.prepare_allocation_metadata(
        VirtualEgressAllocationPreparation(
            allocation_id=f"ealloc_{uuid.uuid4().hex}",
            session_id="live-recoverable-allocation",
            environment_name="sandbox",
            image=config["image"],
        )
    )
    allocations = {
        key: f"ealloc_{uuid.uuid4().hex}" for key in ("lost_ack", "concurrent", "never_sent")
    }
    client, _owned = _control_client(
        client=None, region_name=config["region"], profile_name=None, endpoint_url=None
    )
    with tempfile.TemporaryDirectory(prefix="cayu-recoverable-live-") as directory:
        state_path = Path(directory) / "state.json"
        state_path.write_text(json.dumps({"metadata": metadata, "allocations": allocations}))
        observed: set[str] = set()
        try:
            died = _run_phase("submit-then-die", state_path, expect_exit=_DEATH_EXIT_CODE)
            observed.add(died["observed_microvm_id"])
            recovered = _run_phase("recover", state_path)
            observed.add(recovered["microvm_id"])
            if recovered["exec"] != "executed":
                raise RuntimeError("The only recovery worker was fenced.")
            if recovered["microvm_id"] != died["observed_microvm_id"]:
                raise RuntimeError("Recovery adopted a different MicroVM than the lost request.")
            concurrent = await asyncio.gather(
                asyncio.to_thread(_run_phase, "replay-concurrent", state_path),
                asyncio.to_thread(_run_phase, "replay-concurrent", state_path),
            )
            concurrent_ids = {item["microvm_id"] for item in concurrent}
            observed.update(concurrent_ids)
            if len(concurrent_ids) != 1:
                raise RuntimeError("Concurrent recovery workers created different MicroVMs.")
            if "executed" not in {item["exec"] for item in concurrent}:
                raise RuntimeError("No concurrent recovery worker could execute.")
            conflict = _run_phase("conflict", state_path)
            described = client.get_microvm(microvmIdentifier=recovered["microvm_id"])
            identity = {
                "microvm_id": described["microvmId"],
                "endpoint": described["endpoint"],
                "region": config["region"],
                "image_identifier": described["imageArn"],
                "image_version": described["imageVersion"],
                "session_id": "live-recoverable-allocation",
                "environment_name": "sandbox",
            }
            state = json.loads(state_path.read_text())
            state_path.write_text(json.dumps({**state, "identity": identity}))
            stale = asyncio.create_task(asyncio.to_thread(_run_phase, "stale-owner", state_path))
            ready = state_path.with_suffix(".stale-ready")
            deadline = time.monotonic() + 120
            while not ready.exists() and not stale.done():
                if time.monotonic() > deadline:
                    raise TimeoutError("stale owner never attached")
                await asyncio.sleep(0.5)
            successor = _run_phase("successor", state_path)
            stale_outcomes = await stale
            if stale_outcomes != {"exec": "fenced", "terminate": "fenced"}:
                raise RuntimeError(f"Stale owner was not fenced: {stale_outcomes}")
            if successor["successor_exec"] != "successor":
                raise RuntimeError("Successor owner could not execute.")
            if client.get_microvm(microvmIdentifier=recovered["microvm_id"])["state"] != "RUNNING":
                raise RuntimeError("A stale owner changed its successor's MicroVM lifecycle.")
            _run_phase("suspend", state_path)
            reconnect = _run_phase("reconnect", state_path)
            if reconnect["reconnected"] != recovered["microvm_id"]:
                raise RuntimeError("Reconnect attached a different MicroVM.")
            reap = _run_phase("reap", state_path)
            if not _run_phase("disposal", state_path)["disposed"]:
                raise RuntimeError("A reaped MicroVM was not proven disposed.")
        finally:
            # Reaping replays never_sent within its window and so may create it;
            # a second reap pass is idempotent and closes any partial failure.
            # It must never mask the scenario's own failure.
            primary = sys.exc_info()[1]
            try:
                _run_phase("reap", state_path)
            except Exception as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"cleanup reap also failed: {cleanup}")
        census = _census(client, config["image"], started)
    live = sorted(key for key, value in census.items() if value != "TERMINATED")
    if live:
        raise RuntimeError(f"Unaccounted Lambda MicroVMs remain: {live}")
    if not observed <= set(census) or len(census) > len(allocations):
        # One MicroVM per allocation intent; never_sent is created by its
        # in-window reap replay and terminated in the same operation.
        raise RuntimeError(f"Allocation census does not match intents: {sorted(census)}")
    print(
        EVIDENCE_PREFIX
        + json.dumps(
            {
                "adapter": "aws-lambda-microvm-recoverable-allocation",
                "region": config["region"],
                "lost_acknowledgement_recovery": "verified",
                "fresh_process_recovery": "verified",
                "concurrent_recovery_single_allocation": "verified",
                "changed_parameter_replay": conflict["conflict"],
                "suspended_reconnect_by_durable_identity": "verified",
                "cross_process_stale_owner_exec": stale_outcomes["exec"],
                "cross_process_stale_owner_terminate": stale_outcomes["terminate"],
                "agent_privilege_boundary": reconnect["privilege_boundary"],
                "disposal_proof_after_reap": "verified",
                "reaped_allocations": sorted(reap["reaped"]),
                "unaccounted_microvms": 0,
                "microvms_per_intent": len(census) / len(allocations),
                "pinned_image_version": metadata["image_version"],
                "replay_window_s": metadata["replay_window_s"],
                "unsupported_provider_guarantees": [
                    "client_token_retention_undocumented",
                    "throttling_not_induced_live",
                    "token_expiry_not_induced_live",
                ],
                "seconds": round(time.time() - started, 1),
            },
            sort_keys=True,
        )
    )


def _entrypoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    parser.add_argument("--state", type=Path)
    arguments = parser.parse_args()
    if arguments.phase is None:
        asyncio.run(main())
        return
    print(json.dumps(asyncio.run(_phase(arguments.phase, arguments.state))), flush=True)


if __name__ == "__main__":
    _entrypoint()
