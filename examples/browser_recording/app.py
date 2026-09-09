"""Synthetic public-page recording using only application-facing contracts."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path

import uvicorn

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserRecordingConfig,
    BrowserRecordingPolicy,
    BrowserRecordingStore,
    BrowserSessionTool,
    CayuApp,
    EnvironmentSpec,
    LocalArtifactStore,
    SQLiteSessionStore,
    ToolResultPart,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.egress import CapturedResponse, EgressUpstreamOperation, HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.runtime.human_review import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy
from cayu.server import BasicAuth, DashboardConfig, ServerConfig, create_server
from cayu.server.browser_recording import BrowserRecordingServer
from cayu.tools.user_input import UserInputTool

ROOT = Path(os.environ["CAYU_RECORDING_DEMO_STATE"])
REPO = Path(__file__).resolve().parents[2]
SESSION = "browser-recording-demo"
CONFIG = json.loads((ROOT / "configuration.json").read_text())


def identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class Site:
    def prepare(self, request, *, limits):
        async def send():
            return CapturedResponse(
                status_code=200,
                headers={"Content-Type": "text/html"},
                body=b"""<!doctype html>
              <body style="background:navy;color:white;font:32px sans-serif"><h1>Public recording demo</h1>
              <p id="clock">0</p><script>let n=0;setInterval(()=>{clock.textContent=++n;
              document.body.style.background=n%2?'navy':'teal'},250)</script></body>""",
            )

        return EgressUpstreamOperation(send)


class Review(HumanReviewPolicy):
    version = "recording-demo:v1"
    binding_key = bytes.fromhex(CONFIG["review_key"])

    def authorize(self, context, *, session_id, session_metadata, action):
        return (
            context.recipient == "operator"
            and context.tenant is None
            and context.purpose == "demo"
            and session_id == SESSION
        )

    def project(self, context, source):
        allowed = source.kind == "user_input" and source.question == "Continue this recording?"
        return HumanReviewDisclosure(
            status="permitted" if allowed else "redacted",
            fields=(HumanReviewField(label="Recording", text="Continue this recording?"),)
            if allowed
            else (),
            sensitive_content="application_attested",
        )


class Provider(ModelProvider):
    name = "recording-fixture"
    execution_profile_identity = identity("recording-fixture")

    async def stream(self, request):
        invocation = sum(message.role == "user" for message in request.messages)
        call = f"navigate-{invocation}"
        results = {
            part.tool_call_id: part
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        }
        if call not in results:
            yield ModelStreamEvent.tool_call(
                id=call,
                name="browser_session",
                arguments={
                    "operation": "navigate",
                    "operation_id": call,
                    "url": "https://public.example.test/",
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        result = results[call]
        if result.is_error:
            raise RuntimeError("The synthetic browser operation failed.")
        (ROOT / f"page-{invocation}.json").write_text(json.dumps(result.structured))
        if CONFIG.get("restart") and invocation == 1:
            if "pause" not in results:
                async with asyncio.timeout(120):
                    while not (ROOT / "pause").exists():
                        await asyncio.sleep(0.05)
                yield ModelStreamEvent.tool_call(
                    id="pause", name="ask_user", arguments={"question": "Continue this recording?"}
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if "observe-after-restart" not in results:
                yield ModelStreamEvent.tool_call(
                    id="observe-after-restart",
                    name="browser_session",
                    arguments={
                        "operation": "observe",
                        "operation_id": "observe-after-restart",
                        "session_id": result.structured["session_id"],
                        "page_id": result.structured["page_id"],
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            observed = results["observe-after-restart"]
            if observed.is_error:
                raise RuntimeError("Retained browser observation failed.")
            (ROOT / "resumed.json").write_text(json.dumps(observed.structured))
        async with asyncio.timeout(120):
            while not (ROOT / f"finish-{invocation}").exists():
                await asyncio.sleep(0.05)
        close_call = f"close-{invocation}"
        if CONFIG.get("explicit_close") and close_call not in results:
            yield ModelStreamEvent.tool_call(
                id=close_call,
                name="browser_session",
                arguments={
                    "operation": "close",
                    "operation_id": close_call,
                    "session_id": result.structured["session_id"],
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("The public-page demonstration is complete.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def build():
    recordings = BrowserRecordingStore(ROOT / "private-recordings.sqlite")
    recording_path = ROOT / "capture-configuration.json"
    if recording_path.exists():
        recording = BrowserRecordingConfig.model_validate_json(recording_path.read_text())
    else:
        recording = await recordings.authorize_capture(
            session_id=SESSION,
            policy=BrowserRecordingPolicy(
                scope="public-demo",
                allowed_origins=("https://public.example.test",),
                retention_seconds=3600,
                max_duration_seconds=120,
            ),
            guest_endpoint="wss://cayu-control:8443/api/browser-recordings/guest",
        )
        fd = os.open(recording_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(
                {
                    **recording.model_dump(mode="json"),
                    "credential": recording.credential.get_secret_value(),
                },
                stream,
            )
    app = CayuApp(
        session_store=SQLiteSessionStore(ROOT / "sessions.sqlite"),
        enable_logging=False,
        human_review_policy=Review(),
    )
    app.register_provider(Provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="recording-fixture"),
        tools=[
            BrowserSessionTool(
                expected_runner_candidate="docker",
                recording=recording if CONFIG.get("enabled", True) else None,
            ),
            UserInputTool(),
        ],
    )
    factory = VirtualEgressEnvironmentFactory(
        execution_profile_identity=identity("recording-factory"),
        credentials=[],
        policies={
            "public": HttpEgressPolicy(
                name="public",
                allowed_hosts=("public.example.test",),
                allowed_endpoints=(("GET", "/"), ("GET", "/favicon.ico")),
            )
        },
        approved_destinations=(
            ApprovedEgressDestination(destination="public.example.test", policy_name="public"),
        ),
        adapter=DockerEgressAdapter(
            reconnect_state_dir=ROOT / "ownership",
            control_server_container_id=CONFIG["control_server_container_id"],
            seccomp_profile=str(REPO / "examples/browser_fetch/seccomp_profile.json"),
        ),
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=LocalArtifactStore(ROOT / "ordinary-artifacts", store_id="demo-artifacts"),
        upstream=Site(),
        setup_commands=(
            f"printf %s {shlex.quote((ROOT / 'certificate.pem').read_text())} > /usr/local/share/ca-certificates/cayu-recording-demo.crt && update-ca-certificates",
        ),
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="browser", execution_profile_identity=identity("recording-environment")
        ),
        factory,
        default=True,
    )

    async def authorize(principal, manifest, purpose):
        return principal.subject == "operator" and manifest.identity.session_id == SESSION

    return create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password=CONFIG["password"]),
            dashboard=DashboardConfig(path="/operator"),
        ),
        browser_recordings=BrowserRecordingServer(store=recordings, authorize=authorize),
    )


async def main():
    (ROOT / "worker.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "start_time": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            }
        )
    )
    server = uvicorn.Server(
        uvicorn.Config(
            await build(),
            host="0.0.0.0",
            port=8443,
            ssl_certfile=str(ROOT / "certificate.pem"),
            ssl_keyfile=str(ROOT / "key.pem"),
            ws="websockets-sansio",
            ws_max_size=2 * 1024 * 1024 + 4096,
            ws_per_message_deflate=False,
            access_log=False,
            log_level="error",
            timeout_graceful_shutdown=5,
        )
    )
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
