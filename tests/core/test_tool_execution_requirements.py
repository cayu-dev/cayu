from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentFactoryRequest,
    ExecutionRequirements,
    ExecutionToolRequirement,
    Tool,
    ToolDescriptor,
    ToolEffect,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolRunnerCapabilityRequirement,
    ToolSpec,
)


def _search_requirement() -> ToolExecutionRequirement:
    return ToolExecutionRequirement(
        name="workspace_search",
        alternatives=(
            ToolRunnerCapabilityRequirement(capability="workspace_text_search_v1"),
            ToolExecutableRequirement(executable="rg"),
        ),
    )


class _RequirementTool(Tool):
    async def run(self, ctx, args):
        raise AssertionError("Registration must not dispatch the tool.")


def test_registration_owns_requirements_and_binds_catalogue_and_exposure_identity() -> None:
    requirement = _search_requirement()
    tool = _RequirementTool(
        ToolSpec(name="search", effect=ToolEffect.NONE, execution_requirements=(requirement,))
    )
    app = CayuApp()
    app.register_agent(AgentSpec(name="agent", model="fake"), tools=[tool])
    registered = app._agents["agent"]
    descriptor = registered.tool_catalogue.descriptors[0]
    capability = registered.tool_capabilities[0]
    assert descriptor.execution_requirements == (requirement,)
    assert capability.execution_requirements == (requirement,)
    assert registered.tools["search"].execution_requirements == (requirement,)
    assert ToolDescriptor.model_validate_json(descriptor.model_dump_json()) == descriptor

    # Replacing the public spec after registration cannot change frozen authority.
    tool.spec = ToolSpec(name="search", effect=ToolEffect.NONE)
    assert registered.tools["search"].execution_requirements == (requirement,)
    other = CayuApp()
    other.register_agent(AgentSpec(name="agent", model="fake"), tools=[tool])
    assert other._agents["agent"].tool_catalogue.revision != registered.tool_catalogue.revision
    assert (
        other._agents["agent"].tool_capabilities[0].definition_fingerprint
        != capability.definition_fingerprint
    )


def test_requirement_spec_copy_revalidates_and_detaches_nested_models() -> None:
    requirement = _search_requirement()
    spec = ToolSpec(name="search", execution_requirements=(requirement,))
    object.__setattr__(requirement.alternatives[1], "executable", "changed")
    assert spec.execution_requirements[0].alternatives[1].executable == "rg"
    assert ToolSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.model_copy() == spec
    with pytest.raises(ValidationError):
        spec.model_copy(update={"execution_requirements": [{"name": "bad", "alternatives": []}]})


@pytest.mark.parametrize(
    "executable",
    ["", "../rg", "bin/rg", "/bin/../rg", "/bin//rg", "//bin/rg", "/", "rg;echo", "rg\n", True],
)
def test_executable_declaration_rejects_ambiguous_paths_and_shell_authority(executable) -> None:
    with pytest.raises(ValidationError):
        ToolExecutableRequirement(executable=executable)


@pytest.mark.parametrize(
    "codes", [(True,), (), (0, 0), (1, 0), (-1,), (126,), (127,), (130,), (256,)]
)
def test_probe_exit_codes_are_bounded_unique_strict_integers(codes) -> None:
    with pytest.raises((TypeError, ValueError)):
        ToolExecutableRequirement(
            executable="rg", probe_arguments=("--version",), accepted_exit_codes=codes
        )


def test_availability_and_explicit_process_probes_have_distinct_exact_identity() -> None:
    available = ToolExecutableRequirement(executable="/opt/tools/rg")
    version = ToolExecutableRequirement(executable="/opt/tools/rg", probe_arguments=("--version",))
    assert available.probe_arguments is None
    assert available.fingerprint != version.fingerprint
    assert available.fingerprint != ToolExecutableRequirement(executable="rg").fingerprint
    assert ToolExecutableRequirement.model_validate_json(available.model_dump_json()) == available
    with pytest.raises(ValidationError, match="successful executable lookup"):
        ToolExecutableRequirement(executable="rg", accepted_exit_codes=(1,))


def test_requirement_iterables_stop_at_the_bound() -> None:
    consumed = 0

    def requirements():
        nonlocal consumed
        while True:
            consumed += 1
            yield _search_requirement()

    with pytest.raises(ValueError, match="more than 32"):
        ToolSpec(name="search", execution_requirements=requirements())
    assert consumed == 33


def test_duplicate_alternatives_and_requirement_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="alternatives must be unique"):
        ToolExecutionRequirement(
            name="search",
            alternatives=(ToolExecutableRequirement(executable="rg"),) * 2,
        )
    with pytest.raises(ValueError, match="unique names"):
        ToolSpec(name="search", execution_requirements=(_search_requirement(),) * 2)


@pytest.mark.parametrize(
    "entrance", ["agent", "factory_request", "manifest", "microsandbox_declaration"]
)
@pytest.mark.parametrize(
    "field",
    ["accepted_exit_codes", "probe_arguments", "executable", "requirement_name", "invalid_name"],
)
def test_agent_requirement_revalidation_does_not_publish_rejected_values(
    capsys, caplog, entrance, field
):
    canary = "private-rejected-probe-value-860"
    sibling_canary = "private-valid-probe-argument-860"

    class RejectedValue:
        def __repr__(self):
            return canary

        __str__ = __repr__

    requirements = ExecutionRequirements(
        tool_requirements=(
            ExecutionToolRequirement(
                tool_name="search",
                requirement=ToolExecutionRequirement(
                    name="search",
                    alternatives=(
                        ToolExecutableRequirement(
                            executable="rg", probe_arguments=("--version", sibling_canary)
                        ),
                    ),
                ),
            ),
        )
    )
    app = CayuApp(enable_logging=False)
    if entrance == "manifest":
        app.register_agent(
            AgentSpec(name="agent", model="fake"), execution_requirements=requirements
        )
        requirements = app._agents["agent"].execution_requirements
    if field in {"requirement_name", "invalid_name"}:
        object.__setattr__(
            requirements.tool_requirements[0].requirement,
            "name",
            canary if field == "invalid_name" else RejectedValue(),
        )
    else:
        object.__setattr__(
            requirements.tool_requirements[0].requirement.alternatives[0],
            field,
            RejectedValue() if field == "executable" else (RejectedValue(),),
        )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises((TypeError, ValueError)) as failure:
            if entrance == "agent":
                app.register_agent(
                    AgentSpec(name="agent", model="fake"), execution_requirements=requirements
                )
            elif entrance == "factory_request":
                EnvironmentFactoryRequest(
                    session_id="session",
                    agent_name="agent",
                    environment_name="environment",
                    execution_requirements=requirements,
                )
            elif entrance == "microsandbox_declaration":
                from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter

                MicrosandboxEgressAdapter().execution_admission_evidence_for(requirements)
            else:
                app.describe()
    output = capsys.readouterr()
    diagnostics = "\n".join(
        [str(item.message) for item in captured]
        + [str(failure.value), repr(failure.value), output.out, output.err, caplog.text]
    )
    assert canary not in diagnostics
    assert sibling_canary not in diagnostics
    if entrance != "manifest":
        assert app._agents == {}
