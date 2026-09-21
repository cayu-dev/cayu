"""Scripted normal admission with the supported Docker browser defaults."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from cayu.egress.runtime import VirtualEgressEnvironmentFactory
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(
        os.environ.get("CAYU_RUN_BROWSER_STARTUP_ACCEPTANCE") != "1",
        reason="Requires installed pinned Docker browser image.",
    ),
]


@pytest.mark.parametrize("sandbox_denied", [False, True])
def test_native_docker_default_setup_navigates(tmp_path, sandbox_denied):
    """Exercise public profile admission and inspect actual native tool output."""
    from cayu import (
        AgentSpec,
        ApprovedEgressDestination,
        BrowserEgressPolicy,
        CayuApp,
        EnvironmentSpec,
        ExecutionProfileBehaviorIdentity,
        InMemorySessionStore,
        LocalArtifactStore,
        Message,
        ModelStreamEvent,
        RunRequest,
        ScriptedModelProvider,
        WebBridge,
        run_to_completion,
    )

    async def scenario():
        artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="public-web-run")
        selection = {"runner_kind": "docker"}
        if sandbox_denied:
            from cayu.egress.docker_adapter import DockerEgressAdapter
            from cayu.runners.browser_sandbox import browser_seccomp_profile

            profile = json.loads(Path(browser_seccomp_profile()).read_text())
            # Remove only the Chromium namespace extension from the maintained
            # Docker profile; ordinary Python/worker execution remains permitted.
            profile["syscalls"] = [
                entry
                for entry in profile["syscalls"]
                if entry.get("comment") != "Allow create user namespaces"
            ]
            path = tmp_path / "namespace-denied.json"
            path.write_text(json.dumps(profile))
            selection = {"adapter": DockerEgressAdapter(seccomp_profile=str(path))}
        factory = VirtualEgressEnvironmentFactory(
            policies={
                "research": BrowserEgressPolicy(
                    name="research", allowed_hosts=("example.com",), allowed_path_prefixes=("/",)
                )
            },
            approved_destinations=(
                ApprovedEgressDestination(destination="example.com", policy_name="research"),
            ),
            **selection,
            image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            artifact_store=artifacts,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="public-web-test",
                behavior_version="1",
                implementation_version="1",
            ),
            egress_authority_source="public-web-test",
            egress_policy_version="1",
        )
        bridge = WebBridge.sandboxed_browser(
            environment=factory,
            browser_image=PINNED_BROWSER_SESSION_WORKLOAD.image,
            interactive=True,
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="browser_session",
                        arguments={
                            "operation": "navigate",
                            "operation_id": "public-web-open",
                            "url": "https://example.com",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="research"), factory, default=True)
        app.register_agent(
            AgentSpec(name="researcher", model="scripted"),
            tools=bridge.tools,
            execution_requirements=bridge.execution_requirements,
        )
        outcome = await run_to_completion(
            app,
            RunRequest(
                agent_name="researcher",
                messages=[Message.text("user", "Open example.com")],
            ),
        )
        assert outcome.error is None, outcome.error
        transcript = await store.load_transcript(outcome.session_id)
        results = [p for m in transcript for p in m.content if p.type == "tool_result"]
        assert len(results) == 1
        if sandbox_denied:
            assert results[0].is_error
            assert results[0].structured["error"] == "browser_sandbox_unavailable"
            assert results[0].structured["allocation_disposition"] == "retired"
            assert results[0].structured["execution"] == {
                "admission": "admitted",
                "dispatch": "completed",
                "observation": "not_published",
                "terminal": "settled",
            }
            assert "Chromium sandbox startup was denied" in results[0].content
            assert "browser_seccomp_profile" in results[0].content
            assert "--" not in results[0].content
        else:
            assert not results[0].is_error, results[0].content
            assert "Example Domain" in results[0].content

    asyncio.run(scenario())
