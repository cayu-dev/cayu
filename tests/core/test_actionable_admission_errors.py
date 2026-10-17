from __future__ import annotations

import asyncio
import re

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.build_provenance import current_runtime_build_provenance
from cayu.cli import main
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionRejected,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentityStrength,
    ExecutionProfileMigrationRequired,
    ExecutionProfileMismatchError,
    build_execution_profile_identity,
    changed_execution_profile_components,
)
from cayu.storage import SQLiteSessionStore


def test_duplicate_run_explains_continuation_without_admitting_work() -> None:
    async def scenario() -> None:
        provider = ScriptedModelProvider([[ModelStreamEvent.completed()]])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted"))
        request = RunRequest(
            agent_name="assistant", session_id="existing", messages=[Message.text("user", "hello")]
        )
        _ = [event async for event in app.run(request)]
        with pytest.raises(ValueError) as caught:
            _ = [event async for event in app.run(request)]
        message = str(caught.value)
        assert "app.run creates a new session" in message
        assert "app.resume(ResumeRequest(" in message
        assert "pending approval/input" in message
        assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error_type,meaning",
    [
        (ExecutionProfileMismatchError, "Start a new session"),
        (ExecutionProfileAdoptionRejected, "adoption was rejected"),
        (ExecutionProfileMigrationRequired, "requires an execution-profile migration"),
    ],
)
def test_unknown_evidence_and_hostile_objects_are_not_inspected(error_type, meaning) -> None:
    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError("secret-bearing object inspected")

        def __repr__(self):
            raise AssertionError("secret-bearing object rendered")

    error = error_type(
        session_id="session",
        expected_profile_fingerprint="a" * 64,
        candidate_profile_fingerprint="b" * 64,
        changed_component_classes=(ExecutionProfileComponentClass.EXECUTION_POLICIES,),
        expected_profile=Hostile(),
        candidate_profile=Hostile(),
    )
    assert error.differences[0].category == "other_or_unknown"
    assert error.differences[0].component_class is ExecutionProfileComponentClass.EXECUTION_POLICIES
    assert meaning in str(error)
    assert "cayu guide durable-service-tools" in str(error)
    assert "Process-local" not in str(error)


_REVISION = f"sha256:{'c' * 64}"


def _profile(**overrides):
    values = {
        "runtime_name": "test",
        "runtime_version": "1",
        "provider_name": "scripted",
        "model": "model-a",
        "durable_system_prompt": None,
        "direct_tools": (),
        "tool_catalogue_revision": _REVISION,
        # Without build provenance the runtime class is unavailable and always differs.
        "runtime_build_provenance": current_runtime_build_provenance(),
    }
    values.update(overrides)
    return build_execution_profile_identity(**values)


def _mismatch(stored, current, error_type=ExecutionProfileMismatchError, **kwargs):
    return error_type(
        session_id="paused",
        expected_profile_fingerprint=stored.fingerprint,
        candidate_profile_fingerprint=current.fingerprint,
        changed_component_classes=changed_execution_profile_components(stored, current),
        expected_profile=stored,
        candidate_profile=current,
        **kwargs,
    )


_OPAQUE = {"kind": "process_local_unverifiable", "slot": "first-process"}
_OPAQUE_AGAIN = {"kind": "process_local_unverifiable", "slot": "second-process"}

# One case per component class family: stored profile, current profile, the
# class-level line the message must contain, and the class-specific fix.
_CLASS_CASES = {
    "invocation_policies": (
        {"invocation_policies": [_OPAQUE], "invocation_policies_process_local": True},
        {"invocation_policies": []},
        "invocation_policies (request loop_policies): opaque -> structural",
        "pass the same policies on every RunRequest and ResumeRequest",
    ),
    "execution_policies": (
        {
            "execution_policies": {"policy": "v1"},
            "execution_policies_application_versioned": True,
        },
        {
            "execution_policies": {"policy": "v2"},
            "execution_policies_application_versioned": True,
        },
        "execution_policies (tool, command, exposure and app or agent loop policies): "
        "declared -> declared",
        "Override its `execution_profile_identity` property",
    ),
    "provider_adapter": (
        {"provider_adapter": _OPAQUE, "provider_adapter_process_local": True},
        {"provider_adapter": _OPAQUE_AGAIN, "provider_adapter_process_local": True},
        "provider_adapter (model provider adapter): opaque -> opaque",
        "build the provider the same way in each one",
    ),
    "tool_implementations": (
        {
            "tool_implementations": [{"implementation": "v1"}],
            "tool_implementations_application_versioned": True,
        },
        {
            "tool_implementations": [{"implementation": "v2"}],
            "tool_implementations_application_versioned": True,
        },
        "tool_implementations (tool implementations): declared -> declared",
        "execution_profile_identity=ExecutionProfileBehaviorIdentity(",
    ),
    "provider_target": (
        {},
        {"model": "model-b"},
        "provider_target (provider name and model): structural -> structural",
        "target=ModelTarget(provider_name=..., model=...)",
    ),
    "execution_environment": (
        {
            "execution_environment": _OPAQUE,
            "execution_environment_process_local": True,
        },
        {
            "execution_environment": _OPAQUE_AGAIN,
            "execution_environment_process_local": True,
        },
        "execution_environment (environment and runner): opaque -> opaque",
        "custom environments, runners, environment factories and context policies",
    ),
}


@pytest.mark.parametrize("case", sorted(_CLASS_CASES))
def test_mismatch_names_each_changed_class_and_gives_runnable_next_steps(case: str) -> None:
    stored_overrides, current_overrides, change_line, fix = _CLASS_CASES[case]
    error = _mismatch(
        _profile(**stored_overrides),
        _profile(**current_overrides),
        boundary="resume",
        application_policy_configured=False,
    )
    message = str(error)

    assert error.changed_component_classes == (ExecutionProfileComponentClass(case),)
    assert f"execution profile changed in: {case}." in message
    assert change_line in message
    assert fix in message
    # Both next steps are spelled out as code against the public API.
    assert 'app.run(RunRequest(agent_name=..., session_id="<new id>"' in message
    assert 'app.resume(ResumeRequest(session_id="paused"' in message
    assert "profile_adoption=ExecutionProfileAdoptionIntent(idempotency_key=" in message
    assert "requested_by=ResolutionActor(" in message
    assert "CayuApp(execution_profile_policy=...)" in message
    assert "has no ExecutionProfilePolicy" in message
    assert "cayu guide diagnostics#execution-profile-mismatch" in message
    assert len(message) < 3_000


def test_opaque_both_sides_is_reported_as_identity_not_code_change() -> None:
    error = _mismatch(
        _profile(provider_adapter=_OPAQUE, provider_adapter_process_local=True),
        _profile(provider_adapter=_OPAQUE_AGAIN, provider_adapter_process_local=True),
    )
    (difference,) = error.differences
    assert difference.category == "opaque_identity"
    assert difference.expected_strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert difference.candidate_strength is ExecutionProfileIdentityStrength.PROCESS_LOCAL
    assert "differs even when your code did not change" in str(error)
    assert "only lasts until the next process" in str(error)


def test_continuation_boundary_withholds_adoption_advice() -> None:
    error = _mismatch(
        _profile(),
        _profile(
            tool_implementations=[{"implementation": "v2"}],
            tool_implementations_application_versioned=True,
        ),
        boundary="continuation",
    )
    message = str(error)
    assert error.boundary == "continuation"
    assert "adoption is not available here" in message
    assert "Restore the previous code or identities" in message
    assert "profile_adoption=" not in message
    assert "Start a new session" in message


def test_unknown_boundary_qualifies_adoption_advice() -> None:
    error = _mismatch(_profile(), _profile(model="model-b"))
    message = str(error)
    assert error.boundary is None
    assert error.application_policy_configured is None
    assert "profile_adoption=ExecutionProfileAdoptionIntent(" in message
    assert "pending recovery requires a restored compatible registration" in message
    assert "has no ExecutionProfilePolicy" not in message


@pytest.mark.parametrize(
    "configured,expected",
    [
        (False, "without one Cayu rejects every adoption request"),
        (True, "The configured ExecutionProfilePolicy rejected it"),
        (None, "Adoption needs an ExecutionProfilePolicy that returns adopt"),
    ],
)
def test_adoption_rejection_explains_the_policy_requirement(configured, expected) -> None:
    error = _mismatch(
        _profile(),
        _profile(model="model-b"),
        error_type=ExecutionProfileAdoptionRejected,
        boundary="resume",
        application_policy_configured=configured,
    )
    message = str(error)
    assert "execution-profile adoption was rejected for changes in: provider_target" in message
    assert expected in message
    assert "Start a new session" in message


def test_every_class_changed_keeps_the_message_bounded() -> None:
    changed = tuple(ExecutionProfileComponentClass)
    error = ExecutionProfileMismatchError(
        session_id="s" * 10_000,
        expected_profile_fingerprint="a" * 64,
        candidate_profile_fingerprint="b" * 64,
        changed_component_classes=changed,
        boundary="resume",
    )
    message = str(error)
    assert f"and {len(changed) - 8} more component classes" in message
    # A long session id is named once in the headline, never copied into snippets.
    assert "ResumeRequest(session_id=...," in message
    assert len(message) - 10_000 < 6_000


def _guide_namespace(capsys) -> dict:
    assert main(["guide", "diagnostics#execution-profile-mismatch"]) == 0
    guide = capsys.readouterr().out
    namespace: dict = {}
    for snippet in re.findall(r"```python\n(.*?)```", guide, re.DOTALL):
        exec(compile(snippet, "diagnostics#execution-profile-mismatch", "exec"), namespace)
    return namespace


def _lookup_tool(identity):
    class Lookup(Tool):
        spec = ToolSpec(
            name="lookup",
            description="Look something up.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.NONE,
            execution_profile_identity=identity,
        )

        async def run(self, ctx, args):
            return ToolResult(content="found")

    return Lookup()


def test_guide_snippets_recover_an_edited_session_across_app_instances(
    capsys, sqlite_resources
) -> None:
    guide = _guide_namespace(capsys)

    def build(path, *, tool_version: str, policy=None) -> CayuApp:
        # One store, a fresh CayuApp per simulated process.
        app = CayuApp(
            session_store=SQLiteSessionStore(path),
            execution_profile_policy=policy,
            enable_logging=False,
        )
        app.register_provider(guide["scripted_provider"](), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted"),
            tools=[
                _lookup_tool(
                    ExecutionProfileBehaviorIdentity(
                        name="lookup",
                        behavior_version=tool_version,
                        implementation_version=tool_version,
                    )
                )
            ],
        )
        return app

    def resume(app: CayuApp, text: str, *loop_policies):
        return app.resume(
            ResumeRequest(
                session_id="edited",
                messages=[Message.text("user", text)],
                loop_policies=loop_policies,
            )
        )

    async def scenario() -> None:
        async with sqlite_resources as resources:
            path = resources.path()
            first = build(path, tool_version="1")
            _ = [
                event
                async for event in first.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="edited",
                        messages=[Message.text("user", "hello")],
                        loop_policies=(guide["CompletionCheck"](),),
                    )
                )
            ]
            await first.aclose()

            # Unchanged code with declared identities resumes in a new app instance.
            same = build(path, tool_version="1")
            _ = [event async for event in resume(same, "again", guide["CompletionCheck"]())]
            await same.aclose()

            edited = build(path, tool_version="2")
            with pytest.raises(ExecutionProfileMismatchError) as caught:
                _ = [
                    event
                    async for event in resume(edited, "after edit", guide["CompletionCheck"]())
                ]
            await edited.aclose()
            error = caught.value
            assert error.boundary == "resume"
            assert error.application_policy_configured is False
            assert "tool_implementations (tool implementations): declared -> declared" in str(error)

            # The guide's policy and helper adopt the edited profile explicitly. The
            # helper passes no request loop policies, so their removal is adopted too.
            adopting = build(path, tool_version="2", policy=guide["AdoptExplicitRequests"]())
            await guide["adopt_current_profile"](adopting, "edited", "adopt-1")
            await adopting.aclose()

            adopted = build(path, tool_version="2")
            _ = [event async for event in resume(adopted, "after adoption")]
            await adopted.aclose()

    asyncio.run(scenario())
