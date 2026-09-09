from __future__ import annotations

import asyncio
import base64
import secrets
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentSpec,
    EventQuery,
    EventType,
    ExecCommandTool,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ProcessCommandPolicy,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    RunRequest,
    RuntimeEvidenceRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SQLiteSessionStore,
    SQLiteTaskStore,
    runtime_evidence,
)
from cayu.runners import DockerRunner

pytestmark = pytest.mark.process


@pytest.mark.parametrize("explicit_redactor", [False, True])
def test_native_docker_final_revision_survives_closure_and_store_reopen(
    tmp_path: Path, explicit_redactor: bool
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('ok')\n")
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={
            "test": SecretStr(
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
            )
        },
    )
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(source),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
            platform_architecture=subprocess.check_output(
                [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
                text=True,
            ).strip(),
        ),
        docker_path=docker_path,
    )

    async def run():
        store = SQLiteSessionStore(
            tmp_path / "runtime.db", public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        tasks = SQLiteTaskStore(tmp_path / "runtime.db")
        app = CayuApp(
            session_store=store,
            task_store=tasks,
            public_authority_alias_keyring=keyring,
            enable_logging=False,
            **({"secret_redactor": SecretRedactor()} if explicit_redactor else {}),
        )
        app.register_environment_factory(
            EnvironmentSpec(
                name="coding", execution_profile_identity=factory.execution_profile_identity
            ),
            factory,
            default=True,
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="command",
                            name="exec_command",
                            arguments={
                                "argv": ["python3", "-B", "main.py"],
                                "cwd": "/workspace",
                                "timeout_s": 10,
                            },
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="probe", model="scripted"),
            tools=[
                ExecCommandTool(
                    policy=ProcessCommandPolicy(
                        allowed_executables=["python3"], allowed_cwds=["/workspace"]
                    )
                )
            ],
        )
        try:
            async with asyncio.timeout(90):
                events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="finalization",
                            agent_name="probe",
                            messages=[Message.text("user", "Run the command")],
                            max_steps=2,
                        )
                    )
                ]
            for name in (
                "drain_background_interruptions",
                "drain_provider_operation_cancellations",
                "drain_knowledge_publications",
                "drain_recovery_cleanups",
                "drain_environment_cleanups",
            ):
                assert await getattr(app, name)(timeout_s=30)
            terminal = next(e for e in events if e.type is EventType.TOOL_CALL_COMPLETED)
            assert terminal.payload["result"]["structured"]["stdout"] == "ok\n"
            assert terminal.payload["result"]["structured"]["exit_code"] == 0
            assert terminal.payload["result"]["is_error"] is False
            binding = next(e for e in events if e.type is EventType.ENVIRONMENT_BINDING_COMPLETED)
            container_id = binding.payload["bound_metadata"]["sync_binding"][
                "target_workspace_id"
            ].split(":")[1]
            assert not await DockerRunner.container_exists(container_id, docker_path=docker_path)
            final = next(
                e for e in events if e.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
            )
            revision = final.payload["final_revision"]
            assert revision["status"] == "supported", revision
            assert revision["revision"]
            observations = await store.query_events(
                EventQuery(
                    session_id="finalization", event_type=EventType.WORKSPACE_REVISION_OBSERVED
                )
            )
            after = next(
                record.event
                for record in reversed(observations)
                if record.event.payload["phase"] == "after"
            )
            assert after.payload["workspace_id"] == revision["workspace_id"]
            assert (
                after.payload["observer"] == revision["observer"] == "DockerCodingWorkspaceBinding"
            )
            delta = revision["finalization_delta"]
            assert delta["status"] == "no_change"
            assert delta["before_revision"] == delta["after_revision"] == revision["revision"]
            request = RuntimeEvidenceRequest(
                root_session_id="finalization", max_sessions=1, max_events=256
            )
            evidence = (await runtime_evidence(app, request)).model_dump(mode="json")
            assert evidence["sessions"][0]["status"] == "completed"
            final_evidence = evidence["sessions"][0]["workspace_finalization"]
            assert final_evidence["revision"] == revision["revision"]
        finally:
            await store.close()
            await tasks.close()
        reopened = SQLiteSessionStore(
            tmp_path / "runtime.db", public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        restarted = CayuApp(
            session_store=reopened, public_authority_alias_keyring=keyring, enable_logging=False
        )
        try:
            recovered = (await runtime_evidence(restarted, request)).model_dump(mode="json")
            assert recovered["sessions"][0]["workspace_finalization"] == final_evidence
        finally:
            await reopened.close()

    asyncio.run(run())
