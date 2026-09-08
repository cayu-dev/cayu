"""Cross-process browser receipt reconciliation on the real retained Docker boundary."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from tests.egress.docker_browser_reconnect_worker import Site, identity, persist

from cayu import ApprovedEgressDestination, BrowserSessionTool, LocalArtifactStore, ToolContext
from cayu.core.tools import _bind_runtime_tool_invocation_authority
from cayu.egress import HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD, ExecCommand
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import SecretRedactor


async def main(mode, root):
    store = LocalArtifactStore(root / "artifacts", store_id="receipt-fixture")
    adapter = DockerEgressAdapter(
        reconnect_state_dir=root / "ownership",
        seccomp_profile=str(
            Path(__file__).resolve().parents[2] / "examples/browser_fetch/seccomp_profile.json"
        ),
    )
    factory = VirtualEgressEnvironmentFactory(
        execution_profile_identity=identity("receipt-factory"),
        credentials=[],
        policies={
            "site": HttpEgressPolicy(
                name="site",
                allowed_hosts=("browser.test",),
                allowed_endpoints=(("GET", "/"), ("POST", "/effect"), ("GET", "/favicon.ico")),
            )
        },
        approved_destinations=(
            ApprovedEgressDestination(destination="browser.test", policy_name="site"),
        ),
        adapter=adapter,
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=store,
        upstream=Site(root),
    )
    metadata = json.loads((root / "allocation.json").read_text()) if mode == "resume" else {}
    result = await factory.create(
        EnvironmentFactoryRequest(
            session_id="parent-session",
            agent_name="assistant",
            environment_name="browser",
            operation=EnvironmentFactoryOperation.RECONNECT
            if metadata
            else EnvironmentFactoryOperation.CREATE,
            reconnect_metadata=metadata,
        )
    )
    if metadata:
        assert result.reconnect_metadata == metadata
    else:
        metadata = result.reconnect_metadata
        persist(root / "allocation.json", metadata)
    runner = result.environment.runner
    fingerprint = metadata["allocation_fingerprint"]
    records = json.loads((root / "records.json").read_text()) if mode == "resume" else {}
    stopping = False

    async def load(key):
        value = records.get(key)
        return None if value is None else json.loads(json.dumps(value))

    async def compare_and_set(key, expected, desired, secondary):
        assert records.get(key) == expected
        terminal = any(
            record.get("record_type") == "cayu.browser-operation"
            and record.get("state") == "terminal"
            for record in (desired, *secondary.values())
        )
        if stopping and terminal and mode == "crash_before_receipt":
            os._exit(0)
        records.update(json.loads(json.dumps({key: desired, **secondary})))
        persist(root / "records.json", records)
        if stopping and terminal and mode == "crash_after_receipt":
            os._exit(0)
        return json.loads(json.dumps(desired))

    def context(args, call):
        ctx = ToolContext(
            session_id="parent-session",
            agent_name="assistant",
            environment_name="browser",
            idempotency_key=f"tool-key-{call}",
            runner=InvocationRunnerHandle(
                runner,
                redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(0, SecretRedactor()),
            ),
            artifact_store=store,
            artifact_store_id=store.id,
        )
        _bind_runtime_tool_invocation_authority(
            ctx,
            parent_task_id=None,
            parent_run_epoch=1,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id=call,
            tool_name="browser_session",
            idempotency_key=f"tool-key-{call}",
            effective_arguments=args,
            execution_profile_fingerprint="b" * 64,
            environment_allocation_fingerprint=fingerprint,
            load_durable_operation=load,
            compare_and_set_durable_operation=compare_and_set,
            seal_durable_output=lambda value: json.loads(json.dumps(value)),
            secret_publication_sealer=lambda: None,
        )
        return ctx

    tool = BrowserSessionTool(expected_runner_candidate="docker", max_wait_ms=1000)
    if mode != "resume":
        written = await runner.exec(
            ExecCommand.process("sh", "-c", "printf continuity > /workspace/sentinel")
        )
        assert written.exit_code == 0
        args = {"operation": "navigate", "url": "https://browser.test/", "operation_id": "navigate"}
        nav = await tool.run(context(args, "navigate"), args)
        assert not nav.is_error, nav.structured
        value = nav.model_dump(mode="json")["structured"]
        persist(root / "navigation.json", value)
        args = {
            "operation": "click",
            "operation_id": "mutation",
            "session_id": value["session_id"],
            "page_id": value["page_id"],
            "ref": next(item["ref"] for item in value["refs"] if item["name"] == "Commit"),
            "expected_revision": value["revision"],
            "expected_control_epoch": value["control_epoch"],
        }
        persist(root / "mutation-arguments.json", args)
        stopping = True
        await tool.run(context(args, "mutation"), args)
        raise AssertionError("fault injection did not fire")
    try:
        read = await runner.exec(ExecCommand.process("cat", "/workspace/sentinel"))
        assert read.exit_code == 0 and read.stdout == "continuity"
        args = json.loads((root / "mutation-arguments.json").read_text())
        recovered = await tool.reconcile_durable_tool_call(
            parent_session_id="parent-session",
            parent_run_epoch=1,
            execution_profile_fingerprint="b" * 64,
            environment_name="browser",
            environment_allocation_fingerprint=fingerprint,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id="mutation",
            idempotency_key="tool-key-mutation",
            arguments=args,
            started=True,
            load_operation=load,
        )
        assert recovered is not None
        persist(
            root / "click-result.json",
            {"error": recovered.is_error, "value": recovered.model_dump(mode="json")["structured"]},
        )
        assert json.loads((root / "mutations.json").read_text()) == 1
        if not recovered.is_error:
            value = recovered.model_dump(mode="json")["structured"]
            nav = json.loads((root / "navigation.json").read_text())
            assert (value["session_id"], value["page_id"]) == (nav["session_id"], nav["page_id"])
            observe = {
                "operation": "observe",
                "operation_id": "observe",
                "session_id": value["session_id"],
                "page_id": value["page_id"],
            }
            observed = await tool.run(context(observe, "observe"), observe)
            assert not observed.is_error, observed.structured
            fresh = observed.model_dump(mode="json")["structured"]
            assert fresh["control_epoch"] > value["control_epoch"]
            assert fresh["revision"] != value["revision"]
            action = {
                "operation": "click",
                "operation_id": "fresh-tls",
                "session_id": fresh["session_id"],
                "page_id": fresh["page_id"],
                "ref": next(item["ref"] for item in fresh["refs"] if item["name"] == "Refresh"),
                "expected_revision": fresh["revision"],
                "expected_control_epoch": fresh["control_epoch"],
            }
            clicked = await tool.run(context(action, "fresh-tls"), action)
            assert not clicked.is_error, clicked.structured
        assert json.loads((root / "mutations.json").read_text()) == 1
    finally:
        await runner.finalize(outcome="completed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
