"""Real Docker / fresh-process fault control, using public Runtime APIs only.

The child dies after allocating a guest but before returning its binding. This
is not a live-model or complete Compound qualification receipt.
"""

import json
import os
import subprocess
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentAllocationContext,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    ImmutableInputStore,
    IncompleteSessionRecoveryRequest,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    inspect_local_immutable_input,
)
from cayu.environments import DockerCodingToolchainProfile
from cayu.runners import DockerWorkloadRestrictions


async def _child(
    root, image, crash, cleanup=False, fault="after_acknowledgement", with_inputs=True
):
    def record_allocation(metadata):
        with (root / "allocations.jsonl").open("a") as stream:
            stream.write(json.dumps({"container_id": metadata["container_id"]}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    digest = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        timeout=15,
    ).strip()
    architecture = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Architecture}}", image],
        text=True,
        timeout=15,
    ).strip()
    restrictions = DockerWorkloadRestrictions()
    profile = DockerCodingToolchainProfile(
        profile_id="qualification-prebinding",
        revision="1",
        image_identity=DockerImageIdentity(reference=image, content_digest=digest),
        platform_architecture=architecture,
        restrictions=restrictions,
        runtime_user=restrictions.user,
        command_authorities=(),
    )
    projection = inspect_local_immutable_input(
        root / "source",
        target_path="/evidence",
        policy_fingerprint="sha256:" + "1" * 64,
        runtime_compatibility_fingerprint="sha256:" + "2" * 64,
        authorization_scope_fingerprint="sha256:" + "3" * 64,
    )

    class FaultAllocation(EnvironmentAllocationContext):
        def __init__(self, inner):
            self.inner = inner

        @property
        def intent(self):
            return self.inner.intent

        @property
        def state(self):
            return self.inner.state

        @property
        def dispatch_precluded(self):
            return self.inner.dispatch_precluded

        @property
        def acknowledged_reconnect_metadata(self):
            return self.inner.acknowledged_reconnect_metadata

        async def prepare(self, metadata):
            result = await self.inner.prepare(metadata)
            if crash and fault == "after_intent":
                os._exit(73)
            return result

        async def mark_dispatched(self):
            await self.inner.mark_dispatched()

        async def acknowledge(self, metadata):
            if crash and fault == "after_remote_create":
                record_allocation(metadata)
                os._exit(73)
            await self.inner.acknowledge(metadata)

        async def mark_reaping(self):
            return await self.inner.mark_reaping()

        async def mark_reaped(self):
            await self.inner.mark_reaped()

    class FaultStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        async def transform_checkpoint(self, session_id, checkpoint_transform):
            await super().transform_checkpoint(session_id, checkpoint_transform)
            if crash and fault == "after_publication":
                checkpoint = await self.load_checkpoint(session_id)
                if (checkpoint or {}).get("environment_factory_allocation_receipts"):
                    os._exit(73)

    class FaultFactory(DockerCodingEnvironmentFactory):
        async def create_recoverable(self, request, allocation):
            if crash and fault == "before_intent":
                os._exit(73)
            result = await super().create_recoverable(request, FaultAllocation(allocation))
            record_allocation(result.reconnect_metadata)
            if crash and fault == "after_acknowledgement":
                os._exit(73)
            return result

    store = FaultStore(root / "sessions.db")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="probe", model="scripted-model"))
    app.register_environment_factory(
        EnvironmentSpec(
            name="coding",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="qualification:prebinding", behavior_version="1", implementation_version="1"
            ),
        ),
        FaultFactory(
            source_workspace=LocalWorkspace(root / "workspace", workspace_id="prebinding"),
            toolchain_profile=profile,
            immutable_inputs=(projection,) if with_inputs else (),
            immutable_input_store=ImmutableInputStore(root / "inputs") if with_inputs else None,
            immutable_input_runtime_compatibility_fingerprint="sha256:" + "2" * 64,
        ),
        default=True,
    )
    if cleanup:
        # Safety cleanup is explicitly operator intervention AFTER the census;
        # it cannot satisfy the assertion this test is trying to prove.
        checkpoint = await store.load_checkpoint("prebinding")
        allocation = json.loads((root / "allocations.jsonl").read_text())
        subprocess.run(["docker", "rm", "-f", allocation["container_id"]], check=True, timeout=15)
        record = (checkpoint or {}).get("environment_factory_allocation_intents", {}).get("coding")
        if record is None:
            record = checkpoint["environment_factory_allocation_receipts"]["coding"]
        ImmutableInputStore(root / "inputs").release_allocation_sync(
            record["intent"]["allocation_id"]
        )
        (root / "operator-cleanup.json").write_text(
            json.dumps(
                {
                    "container_id": allocation["container_id"],
                    "operator_cleanup": True,
                }
            )
        )
        await store.close()
        return
    if crash:
        stream = app.run(
            RunRequest(
                agent_name="probe",
                session_id="prebinding",
                messages=[Message.text("user", "Finish.")],
                max_steps=1,
            )
        )
    else:
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="prebinding", reason="fixture_process_loss")
        )
        (root / "recovery.json").write_text(recovery.model_dump_json())
        repeated = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="prebinding", reason="fixture_repeat")
        )
        (root / "repeated-recovery.json").write_text(repeated.model_dump_json())
        assert await app.drain_environment_cleanups(timeout_s=10)
    if crash:
        async for _ in stream:
            pass
    events = await store.load_events("prebinding")
    (root / "events.json").write_text(
        json.dumps([event.model_dump(mode="json") for event in events])
    )
    checkpoint = await store.load_checkpoint("prebinding")
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    await store.close()


@pytest.mark.skipif(
    not os.environ.get("CAYU_TEST_DOCKER_CODING_IMAGE"), reason="opt-in Docker proof"
)
@pytest.mark.parametrize("with_inputs", [True, False])
@pytest.mark.parametrize(
    "fault",
    [
        "before_intent",
        "after_intent",
        "after_remote_create",
        "after_acknowledgement",
        "after_publication",
    ],
)
def test_prebinding_crash_recovers_exact_allocation_without_leak(
    tmp_path,
    record_property,
    request,
    fault,
    with_inputs,
):
    image = os.environ["CAYU_TEST_DOCKER_CODING_IMAGE"]
    (tmp_path / "source").mkdir()
    (tmp_path / "source" / "evidence.txt").write_text("immutable evidence")
    (tmp_path / "workspace").mkdir()
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.environments.test_docker_prebinding_recovery_live import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],sys.argv[3]=='crash',"
        "sys.argv[3]=='cleanup',sys.argv[4],sys.argv[5]=='True'))"
    )
    first = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), image, "crash", fault, str(with_inputs)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert first.returncode == 73, first.stderr
    allocated = fault not in {"before_intent", "after_intent"}
    allocation = json.loads((tmp_path / "allocations.jsonl").read_text()) if allocated else {}
    container_id = allocation.get("container_id")
    if allocated:
        assert len(container_id) == 64

    def safety_cleanup():
        if container_id is None:
            return
        if not subprocess.check_output(
            ["docker", "ps", "-a", "--filter", f"id={container_id}", "--format", "{{.ID}}"],
            text=True,
            timeout=15,
        ).strip():
            return
        cleaned = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path),
                image,
                "cleanup",
                fault,
                str(with_inputs),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert cleaned.returncode == 0, cleaned.stderr

    request.addfinalizer(safety_cleanup)
    recovered = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), image, "recover", fault, str(with_inputs)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert recovered.returncode == 0, recovered.stderr
    events = json.loads((tmp_path / "events.json").read_text())
    bindings = [event for event in events if event["type"] == "environment.binding.completed"]
    assert not bindings
    assert sum(event["type"] == "session.interrupted" for event in events) == 1
    assert not any(event["type"] in {"model.requested", "session.completed"} for event in events)
    if allocated:
        assert len((tmp_path / "allocations.jsonl").read_text().splitlines()) == 1
    else:
        assert not (tmp_path / "allocations.jsonl").exists()
    remaining = (
        subprocess.check_output(
            [
                "docker",
                "ps",
                "-a",
                "--no-trunc",
                "--filter",
                f"id={container_id}",
                "--format",
                "{{.ID}}",
            ],
            text=True,
            timeout=15,
        ).strip()
        if allocated
        else ""
    )
    records = ImmutableInputStore(tmp_path / "inputs").inspect()
    (tmp_path / "resource-evidence.json").write_text(
        json.dumps(
            {
                "allocated_container": container_id,
                "remaining_container": remaining,
                "immutable_reference_counts": [record.reference_count for record in records],
                "operator_cleanup_before_census": False,
            }
        )
    )
    assert not remaining
    if allocated and with_inputs:
        assert records
    assert all(record.reference_count == 0 for record in records)
    assert not (tmp_path / "operator-cleanup.json").exists()
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    pending = checkpoint.get("environment_factory_allocation_intents", {}).get("coding")
    assert pending is None if fault == "before_intent" else pending["state"] == "reaped"
    record_property("prebinding_container_id", container_id)
    record_property("remaining_owned_containers", 0)
    record_property("immutable_input_references", 0)
