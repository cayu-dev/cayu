"""Real application/browser pause and process-loss fixture, using durable Runtime receipts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserSessionTool,
    CayuApp,
    EnvironmentSpec,
    EventType,
    LocalArtifactStore,
    Message,
    RunRequest,
    SQLiteSessionStore,
    ToolResultPart,
    UserInputResponse,
)
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.egress import CapturedResponse, EgressUpstreamOperation, HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD, ExecCommand
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.tools.user_input import UserInputTool


def identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


def persist(path, value):
    with path.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


class Site:
    def __init__(self, root):
        self.root = root

    def prepare(self, request, *, limits):
        async def send():
            cookie = next(
                (value for name, value in request.headers.items() if name.lower() == "cookie"), ""
            )
            if "fixture_session=retained" in cookie:
                path = self.root / "cookies.json"
                persist(path, (json.loads(path.read_text()) if path.exists() else 0) + 1)
            if request.method == "POST":
                path = self.root / "mutations.json"
                count = json.loads(path.read_text()) if path.exists() else 0
                persist(path, count + 1)
            return CapturedResponse(
                status_code=200,
                headers={
                    "Content-Type": "text/html",
                    "Set-Cookie": "fixture_session=retained; Path=/; Secure; HttpOnly",
                },
                body=(
                    b'<!doctype html><title>Continuity</title><form method="POST" action="/effect"><button>Commit</button></form><a href="/">Refresh</a>'
                ),
            )

        return EgressUpstreamOperation(send)


class Provider(ModelProvider):
    name = "fixture"
    execution_profile_identity = identity("docker-browser-provider")

    def __init__(self, root, store):
        self.root, self.store = root, store

    async def stream(self, request):
        results = {
            part.tool_call_id: part
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        }
        if "navigate" not in results:
            call, name, args = (
                "navigate",
                "browser_session",
                {
                    "operation": "navigate",
                    "url": "https://browser.test/",
                    "operation_id": "navigate",
                },
            )
        elif "click" not in results:
            nav = results["navigate"]
            assert not nav.is_error, nav.structured
            value = nav.structured
            persist(self.root / "navigation.json", value)
            ref = next(item["ref"] for item in value["refs"] if item["name"] == "Commit")
            call, name, args = (
                "click",
                "browser_session",
                {
                    "operation": "click",
                    "operation_id": "mutation",
                    "session_id": value["session_id"],
                    "page_id": value["page_id"],
                    "ref": ref,
                    "expected_revision": value["revision"],
                    "expected_control_epoch": value["control_epoch"],
                },
            )
        elif "human" not in results:
            clicked = results["click"]
            persist(
                self.root / "click-result.json",
                {"error": clicked.is_error, "value": clicked.structured},
            )
            if not clicked.is_error:
                persist(self.root / "terminal.json", clicked.structured)
            call, name, args = (
                "human",
                "ask_user",
                {"question": "Continue observing the retained browser?"},
            )
        elif "observe" not in results:
            nav = json.loads((self.root / "navigation.json").read_text())
            call, name, args = (
                "observe",
                "browser_session",
                {
                    "operation": "observe",
                    "operation_id": "observe",
                    "session_id": nav["session_id"],
                    "page_id": nav["page_id"],
                },
            )
        elif "fresh_tls" not in results:
            observed = results["observe"]
            assert not observed.is_error, observed.structured
            value = observed.structured
            persist(self.root / "observation.json", value)
            call, name, args = (
                "fresh_tls",
                "browser_session",
                {
                    "operation": "click",
                    "operation_id": "fresh-tls",
                    "session_id": value["session_id"],
                    "page_id": value["page_id"],
                    "ref": next(item["ref"] for item in value["refs"] if item["name"] == "Refresh"),
                    "expected_revision": value["revision"],
                    "expected_control_epoch": value["control_epoch"],
                },
            )
        else:
            assert not results["fresh_tls"].is_error, results["fresh_tls"].structured
            yield ModelStreamEvent.text_delta("continued without mutation replay")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id=call, name=name, arguments=args)
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class Factory(VirtualEgressEnvironmentFactory):
    async def create(self, request):
        result = await super().create(request)
        path = self.fixture_root / "allocation.json"
        if path.exists():
            assert json.loads(path.read_text()) == result.reconnect_metadata
        else:
            persist(path, result.reconnect_metadata)
        runner = result.environment.runner
        command = (
            ExecCommand.process("cat", "/workspace/sentinel")
            if self.resuming
            else ExecCommand.process("sh", "-c", "printf continuity > /workspace/sentinel")
        )
        value = await runner.exec(command)
        assert value.exit_code == 0
        if self.resuming:
            assert value.stdout == "continuity"
        return result


async def main(mode, root):
    store = SQLiteSessionStore(root / "sessions.sqlite")
    artifacts = LocalArtifactStore(root / "artifacts", store_id="browser-fixture")
    adapter = DockerEgressAdapter(
        reconnect_state_dir=root / "ownership",
        seccomp_profile=str(
            Path(__file__).resolve().parents[2] / "examples/browser_fetch/seccomp_profile.json"
        ),
    )
    factory = Factory(
        execution_profile_identity=identity("docker-browser-factory"),
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
        artifact_store=artifacts,
        upstream=Site(root),
    )
    factory.fixture_root, factory.resuming = root, mode == "resume"
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(Provider(root, store), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fixture"),
        tools=[
            BrowserSessionTool(expected_runner_candidate="docker", max_wait_ms=1000),
            UserInputTool(),
        ],
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="browser", execution_profile_identity=identity("docker-browser-environment")
        ),
        factory,
        default=True,
    )

    async def drain(stream):
        events = []
        async for event in stream:
            events.append(event)
            if event.type == EventType.SESSION_AWAITING_USER_INPUT:
                persist(root / "pause.json", event.payload)
        errors = [
            event.model_dump(mode="json")
            for event in events
            if event.type == EventType.SESSION_FAILED
        ]
        assert not errors, errors
        return events

    if mode != "resume":
        await drain(
            app.run(
                RunRequest(
                    session_id="browser-fixture",
                    agent_name="assistant",
                    messages=[Message.text("user", "navigate, submit once, then ask me")],
                )
            )
        )
        assert (root / "pause.json").exists()
    else:
        pause = json.loads((root / "pause.json").read_text())
        await drain(
            app.resolve_user_input(
                UserInputResponse(
                    session_id="browser-fixture", input_id=pause["input_id"], answer="yes"
                )
            )
        )
        assert json.loads((root / "mutations.json").read_text()) == 1
        before = json.loads((root / "navigation.json").read_text())
        after = json.loads((root / "observation.json").read_text())
        assert (before["session_id"], before["page_id"]) == (after["session_id"], after["page_id"])
        assert after["control_epoch"] > before["control_epoch"]
        assert before["refs"][0]["ref"] != after["refs"][0]["ref"]
    await store.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
