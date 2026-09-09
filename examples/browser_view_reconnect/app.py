"""Protected, view-only synthetic portal journey with durable Docker recovery.

Run through run.py. Application code uses public Runtime configuration and tools;
coordination files belong only to this deterministic demonstration, never guests.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sqlite3
from pathlib import Path

import uvicorn
from pydantic import SecretBytes

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserSessionTool,
    CayuApp,
    EnvironmentSpec,
    LocalArtifactStore,
    SQLiteSessionStore,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyResult,
    ToolResult,
    ToolResultPart,
    ToolSpec,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import Tool, ToolEffect
from cayu.egress import CapturedResponse, EgressUpstreamOperation, HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD, ExecCommand
from cayu.runtime.browser_control import (
    BrowserControlPolicy,
    BrowserControlPolicyResult,
    BrowserOperatorPurpose,
)
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.runtime.human_review import HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy
from cayu.server import (
    BasicAuth,
    BrowserControlServerConfig,
    DashboardConfig,
    ServerConfig,
    create_server,
)
from cayu.tools.user_input import UserInputTool

SESSION = "viewer-reconnect-demo"
QUESTION = "Continue with the same browser?"
PROPOSAL = {"proposal_id": "demo-proposal", "value": "approved"}
ROOT = Path(os.environ["CAYU_DEMO_STATE"])
REPO = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configuration.json").read_text())


def identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


def persist(name, value):
    path = ROOT / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ViewOnly(BrowserControlPolicy):
    identity = "synthetic-view-only:v1"

    async def decide(self, request):
        return BrowserControlPolicyResult(
            allowed=(
                request.principal.subject == "operator"
                and request.identity.session_id == SESSION
                and request.action == "view"
                and not (ROOT / "revoke-view").exists()
            )
        )


class Review(HumanReviewPolicy):
    version = "synthetic-proposal:v1"
    binding_key = bytes.fromhex(CONFIG["review_key"])

    def authorize(self, context, *, session_id, session_metadata, action):
        return (
            context.recipient == "operator"
            and context.tenant is None
            and context.purpose == "demo"
            and session_id == SESSION
        )

    def project(self, context, source):
        permitted = (
            source.question == QUESTION
            if source.kind == "user_input"
            else (
                len(source.calls) == 1
                and source.calls[0].tool_name == "commit_proposal"
                and list(source.arguments_by_call.values()) == [PROPOSAL]
            )
        )
        return HumanReviewDisclosure(
            status="permitted" if permitted else "redacted",
            fields=(
                HumanReviewField(
                    label="Proposal",
                    text=QUESTION
                    if source.kind == "user_input"
                    else "Commit demo-proposal with value approved.",
                ),
            )
            if permitted
            else (),
            sensitive_content="application_attested",
        )


class Approval(ToolPolicy):
    execution_profile_identity = identity("synthetic-approval")

    async def authorize(self, request):
        return ToolPolicyResult(
            decision=(
                ToolPolicyDecision.REQUIRE_APPROVAL
                if request.tool_name == "commit_proposal"
                else ToolPolicyDecision.ALLOW
            )
        )


class Commit(Tool):
    execution_profile_identity = identity("synthetic-commit")
    spec = ToolSpec(
        name="commit_proposal",
        description="Commit the exact synthetic proposal after approval.",
        input_schema={
            "type": "object",
            "properties": {
                "proposal_id": {"const": "demo-proposal"},
                "value": {"const": "approved"},
            },
            "required": ["proposal_id", "value"],
            "additionalProperties": False,
        },
        effect=ToolEffect.EXTERNAL,
    )

    async def run(self, ctx, args):
        assert args == PROPOSAL
        # This database is the independent synthetic business system, not Runtime
        # state. A duplicate submission fails its unique proposal constraint.
        with sqlite3.connect(ROOT / "portal.sqlite") as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS receipts (proposal_id TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO receipts VALUES (?, ?)", (args["proposal_id"], args["value"])
            )
        return ToolResult(
            content="Synthetic proposal committed.",
            structured={"receipt_id": "demo-proposal", "committed": True},
        )


class Site:
    def prepare(self, request, *, limits):
        async def send():
            assert request.method == "GET"
            return CapturedResponse(
                status_code=200,
                headers={
                    "Content-Type": "text/html",
                    "Set-Cookie": "continuity=yes; Secure; HttpOnly; Path=/",
                },
                body=b"""<!doctype html><title>Synthetic review portal</title>
<body style="background:rgb(20,80,140);color:white;font:32px sans-serif"><h1>Same browser portal</h1>
<button onclick="document.body.style.backgroundColor = document.body.style.backgroundColor === 'rgb(20, 80, 140)' ? 'rgb(140, 60, 20)' : 'rgb(20, 80, 140)'">Change display</button><p>Viewing does not submit a proposal.</p></body>""",
            )

        return EgressUpstreamOperation(send)


class Factory(VirtualEgressEnvironmentFactory):
    async def create(self, request):
        result = await super().create(request)
        runner = result.environment.runner
        assert runner is not None
        path = ROOT / "allocation.json"
        if path.exists():
            assert json.loads(path.read_text()) == result.reconnect_metadata
            sentinel = await runner.exec(ExecCommand.process("cat", "/workspace/continuity"))
            assert sentinel.exit_code == 0 and sentinel.stdout == "viewer-reconnect"
        else:
            persist("allocation.json", result.reconnect_metadata)
            sentinel = await runner.exec(
                ExecCommand.process("sh", "-c", "printf viewer-reconnect > /workspace/continuity")
            )
            assert sentinel.exit_code == 0
        return result


class Provider(ModelProvider):
    name = "fixture"
    execution_profile_identity = identity("synthetic-model")

    async def gate(self, stage, result):
        persist(
            f"{stage}.json",
            {
                "session_id": result["session_id"],
                "page_id": result["page_id"],
                "control_epoch": result["control_epoch"],
                "revision": result["revision"],
            },
        )
        async with asyncio.timeout(120):
            while not (ROOT / f"{stage}.continue").exists():
                await asyncio.sleep(0.05)

    async def stream(self, request):
        results = {
            part.tool_call_id: part
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        }
        for result in results.values():
            assert not result.is_error, (
                "Synthetic tool failed; inspect private Runtime state locally."
            )
        name = "browser_session"
        if "navigate" not in results:
            call, args = (
                "navigate",
                {
                    "operation": "navigate",
                    "url": "https://portal.example.test/",
                    "operation_id": "navigate",
                },
            )
        elif "display-before" not in results:
            value = results["navigate"].structured
            await self.gate("before", value)
            call, args = "display-before", self.click(value, "display-before")
        elif "human" not in results:
            await self.gate("before-changed", results["display-before"].structured)
            call, name, args = "human", "ask_user", {"question": QUESTION}
        elif "observe" not in results:
            original = results["navigate"].structured
            call, args = (
                "observe",
                {
                    "operation": "observe",
                    "session_id": original["session_id"],
                    "page_id": original["page_id"],
                    "operation_id": "observe",
                },
            )
        elif "display-after" not in results:
            value = results["observe"].structured
            await self.gate("after", value)
            call, args = "display-after", self.click(value, "display-after")
        elif "commit" not in results:
            await self.gate("after-changed", results["display-after"].structured)
            call, name, args = "commit", "commit_proposal", PROPOSAL
        elif CONFIG.get("explicit_close") and "close" not in results:
            assert results["commit"].structured["committed"]
            await self.gate("committed", results["display-after"].structured)
            call, args = (
                "close",
                {
                    "operation": "close",
                    "operation_id": "close",
                    "session_id": results["navigate"].structured["session_id"],
                },
            )
        else:
            if CONFIG.get("explicit_close"):
                persist("close-receipt.json", results["close"].structured)
            else:
                await self.gate("committed", results["display-after"].structured)
            assert results["commit"].structured["committed"]
            persist("business-receipt.json", results["commit"].structured)
            yield ModelStreamEvent.text_delta("One approved synthetic proposal committed.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id=call, name=name, arguments=args)
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    @staticmethod
    def click(value, operation):
        return {
            "operation": "click",
            "operation_id": operation,
            "session_id": value["session_id"],
            "page_id": value["page_id"],
            "ref": next(item["ref"] for item in value["refs"] if item["name"] == "Change display"),
            "expected_revision": value["revision"],
            "expected_control_epoch": value["control_epoch"],
        }


def build():
    store = SQLiteSessionStore(ROOT / "sessions.sqlite")
    artifacts = LocalArtifactStore(ROOT / "artifacts", store_id="viewer-reconnect-demo")
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        human_review_policy=Review(),
        browser_control=BrowserControlConfig(
            policy=ViewOnly(),
            purpose=BrowserOperatorPurpose(
                code="demo", expected_origins=("https://portal.example.test",)
            ),
            guest_endpoint="wss://cayu-control:8443/api/browser-control/guest",
        ),
    )
    adapter = DockerEgressAdapter(
        reconnect_state_dir=ROOT / "ownership",
        control_server_container_id=CONFIG["control_server_container_id"],
        seccomp_profile=str(REPO / "examples/browser_fetch/seccomp_profile.json"),
    )
    public_ca = (ROOT / "certificate.pem").read_text()
    factory = Factory(
        execution_profile_identity=identity("synthetic-factory"),
        credentials=[],
        policies={
            "portal": HttpEgressPolicy(
                name="portal",
                allowed_hosts=("portal.example.test",),
                allowed_endpoints=(("GET", "/"), ("GET", "/favicon.ico")),
            )
        },
        approved_destinations=(
            ApprovedEgressDestination(destination="portal.example.test", policy_name="portal"),
        ),
        adapter=adapter,
        image=PINNED_BROWSER_SESSION_WORKLOAD.image,
        artifact_store=artifacts,
        upstream=Site(),
        setup_commands=(
            f"printf %s {shlex.quote(public_ca)} > /usr/local/share/ca-certificates/cayu-demo.crt && update-ca-certificates",
        ),
    )
    app.register_provider(Provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fixture"),
        tools=[
            BrowserSessionTool(expected_runner_candidate="docker", max_wait_ms=1000),
            UserInputTool(),
            Commit(),
        ],
        tool_policy=Approval(),
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="browser", execution_profile_identity=identity("synthetic-environment")
        ),
        factory,
        default=True,
    )
    return create_server(
        app,
        config=ServerConfig.protected(
            BasicAuth(username="operator", password=CONFIG["password"]),
            dashboard=DashboardConfig(path="/operator"),
            browser_control=BrowserControlServerConfig(
                operator_origin=CONFIG["origin"],
                signing_key=SecretBytes(bytes.fromhex(CONFIG["viewer_key"])),
            ),
        ),
    )


if __name__ == "__main__":
    persist(
        "worker.json",
        {
            "pid": os.getpid(),
            "start_time": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        },
    )
    uvicorn.run(
        build(),
        host="0.0.0.0",
        port=8443,
        ssl_certfile=str(ROOT / "certificate.pem"),
        ssl_keyfile=str(ROOT / "key.pem"),
        ws="websockets-sansio",
        ws_per_message_deflate=False,
        access_log=False,
        log_level="error",
        timeout_graceful_shutdown=5,
    )
