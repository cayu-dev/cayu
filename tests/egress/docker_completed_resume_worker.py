"""Public-API Docker completion/resume proof across independent worker versions."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from docker_browser_reconnect_worker import Site, identity, persist

from cayu import (
    AgentSpec,
    ApprovedEgressDestination,
    BrowserSessionTool,
    CayuApp,
    EnvironmentSpec,
    EventType,
    ExecutionProfileAdoptionIntent,
    LocalArtifactStore,
    Message,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    ToolResultPart,
)
from cayu.egress import HttpEgressPolicy
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.runtime.egress import VirtualEgressEnvironmentFactory
from cayu.runtime.execution_profiles import (
    ExecutionProfileAuthorityDecision,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
)

SESSION = "completed-browser"


class Policy(ExecutionProfilePolicy):
    identity = "docker-completed-resume-exact-pair-v1"

    def __init__(self):
        self.pair = None

    async def decide(self, request):
        assert request.session_id == SESSION
        if self.pair is None:
            assert request.intent is None
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.REJECT,
                reason="A version upgrade requires an exact authorized pair and caller intent.",
            )
        assert self.pair == (
            request.expected_profile.fingerprint,
            request.candidate_profile.fingerprint,
        )
        assert request.intent is not None
        assert request.intent.requested_by.subject == "fixture-operator"
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Authorize the exact completed-session Runtime/browser upgrade.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class Provider(ModelProvider):
    name = "fixture"
    execution_profile_identity = identity("completed-browser-provider")

    def __init__(self, root):
        self.root = root
        self.turn = "initial"
        self.calls = 0

    async def stream(self, request):
        self.calls += 1
        call_id = f"{self.turn}-navigation"
        result = next(
            (
                part
                for message in request.messages
                for part in message.content
                if isinstance(part, ToolResultPart) and part.tool_call_id == call_id
            ),
            None,
        )
        if result is None:
            yield ModelStreamEvent.tool_call(
                id=call_id,
                name="browser_session",
                arguments={
                    "operation": "navigate",
                    "operation_id": call_id,
                    "url": "https://browser.test/",
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            assert not result.is_error, result.structured
            persist(self.root / f"{self.turn}-browser.json", result.structured)
            yield ModelStreamEvent.text_delta(f"{self.turn} business draft remains available")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


class Factory(VirtualEgressEnvironmentFactory):
    async def create(self, request):
        result = await super().create(request)
        self.allocations += 1
        persist(self.root / f"{self.provider.turn}-allocation.json", result.reconnect_metadata)
        return result


async def main(mode, root):
    config = json.loads((root / "configuration.json").read_text())
    store = SQLiteSessionStore(root / "sessions.sqlite")
    artifacts = LocalArtifactStore(root / "artifacts", store_id="completed-browser-artifacts")
    adapter = DockerEgressAdapter(
        reconnect_state_dir=root / "ownership",
        control_server_container_id=config["control_server_container_id"],
        seccomp_profile=str(
            Path(__file__).resolve().parents[2] / "examples/browser_fetch/seccomp_profile.json"
        ),
    )
    provider = Provider(root)
    factory = Factory(
        execution_profile_identity=identity("completed-browser-factory"),
        credentials=[],
        policies={
            "site": HttpEgressPolicy(
                name="site",
                allowed_hosts=("browser.test",),
                allowed_endpoints=(("GET", "/"), ("GET", "/favicon.ico")),
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
    factory.root, factory.provider, factory.allocations = root, provider, 0
    policy = Policy()
    app = CayuApp(session_store=store, enable_logging=False, execution_profile_policy=policy)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fixture"),
        tools=[BrowserSessionTool(expected_runner_candidate="docker", max_wait_ms=1000)],
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="browser", execution_profile_identity=identity("completed-browser-environment")
        ),
        factory,
        default=True,
    )

    async def drain(stream):
        events = [event async for event in stream]
        failures = [
            event.model_dump(mode="json")
            for event in events
            if event.type == EventType.SESSION_FAILED
        ]
        assert not failures, failures
        assert events[-1].type == EventType.SESSION_COMPLETED
        return events

    def disposed(turn):
        metadata = json.loads((root / f"{turn}-allocation.json").read_text())
        journal = json.loads(
            (root / "ownership" / f"{metadata['identity']['allocation_id']}.json").read_text()
        )
        assert journal["state"] == "disposed", journal
        assert journal["identity"] == metadata["identity"]
        return metadata

    try:
        if mode in {"complete", "same-worker"}:
            await drain(
                app.run(
                    RunRequest(
                        session_id=SESSION,
                        agent_name="assistant",
                        messages=[Message.text("user", "prepare a browser draft")],
                    )
                )
            )
            source = await store.load(SESSION)
            persist(
                root / "source-session.json",
                {"instance_id": source.instance_id, "run_epoch": source.run_epoch},
            )
            persist(
                root / "source-transcript.json",
                [
                    message.model_dump(mode="json")
                    for message in await store.load_transcript(SESSION)
                ],
            )
            disposed("initial")
            if mode == "complete":
                return
        provider.turn = "followup"
        request = ResumeRequest(
            session_id=SESSION, messages=[Message.text("user", "continue with a fresh browser")]
        )
        if mode == "refused":
            events = [event async for event in app.resume(request)]
            assert events[-1].type == EventType.SESSION_FAILED
            assert "disposed" in events[-1].payload["error"]
            assert provider.calls == 0 and factory.allocations == 0
            disposed("initial")
            return
        if mode == "adopt":
            try:
                await drain(app.resume(request))
            except ExecutionProfileMismatchError as exc:
                assert {"runtime", "tool_implementations"} <= {
                    part.value for part in exc.changed_component_classes
                }
                policy.pair = (exc.expected_profile_fingerprint, exc.candidate_profile_fingerprint)
                persist(root / "adoption-pair.json", policy.pair)
            else:
                raise AssertionError("Version change did not require explicit adoption")
            assert provider.calls == 0 and factory.allocations == 0
            request = request.model_copy(
                update={
                    "profile_adoption": ExecutionProfileAdoptionIntent(
                        idempotency_key="completed-browser-upgrade",
                        reason="Continue the exact completed draft with the new Runtime and browser.",
                        requested_by=ResolutionActor(
                            subject="fixture-operator", source=ResolutionActorSource.REQUEST
                        ),
                    )
                }
            )
        events = await drain(app.resume(request))
        if mode == "adopt":
            decisions = [
                event
                for event in events
                if event.type == EventType.SESSION_EXECUTION_PROFILE_DECIDED
            ]
            assert decisions and decisions[-1].payload["decision"] == "adopted"
        before, after = disposed("initial"), disposed("followup")
        assert before["identity"]["allocation_id"] != after["identity"]["allocation_id"]
        assert before["identity"]["container_id"] != after["identity"]["container_id"]
        initial_browser = json.loads((root / "initial-browser.json").read_text())
        followup_browser = json.loads((root / "followup-browser.json").read_text())
        assert initial_browser["session_id"] != followup_browser["session_id"]
        transcript = [
            message.model_dump(mode="json") for message in await store.load_transcript(SESSION)
        ]
        original = json.loads((root / "source-transcript.json").read_text())
        assert transcript[: len(original)] == original
        source = json.loads((root / "source-session.json").read_text())
        session = await store.load(SESSION)
        assert (
            session.instance_id == source["instance_id"] and session.run_epoch > source["run_epoch"]
        )
        persist(
            root / "completed-resume-evidence.json",
            {
                "fresh_allocation": True,
                "fresh_browser": True,
                "transcript_preserved": True,
                "same_session": True,
                "both_disposals_proven": True,
                "mode": mode,
                "browser_worker": PINNED_BROWSER_SESSION_WORKLOAD.worker_version,
            },
        )
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
