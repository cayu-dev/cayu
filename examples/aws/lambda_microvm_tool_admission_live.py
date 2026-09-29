"""Live executable admission evidence for tools on an AWS Lambda MicroVM.

One MicroVM is allocated and terminated. Admission probes run through the
agent execution lane of that exact MicroVM, bound to its control-plane image,
sidecar protocol, and guest boot identity. The scenario proves that an
executable the image contains is ``live_verified`` and admits a tool, that a
missing executable (``node``, which the first-party image does not ship)
refuses the tool before any model request, that explicit probe arguments invoke the program,
and that renewal re-observes the same identity with a new validity window.
The image under test must contain ``python3`` and ``bash`` and must not
contain ``node``, as the first-party sidecar image does.
"""

from __future__ import annotations

import asyncio
import json
import os

from examples._live_checks import require

from cayu import AgentSpec, CayuApp, LambdaMicroVMRunner, Message, RunRequest
from cayu.environments import (
    EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
    ExecutionRequirements,
    ExecutionToolRequirement,
    evaluate_execution_admission,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import (
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolResult,
    ToolSpec,
)

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_MAXIMUM_DURATION_SECONDS = 600
_PRESENT = "python3"


class _PythonTool(Tool):
    spec = ToolSpec(
        name="python_program",
        execution_requirements=(
            ToolExecutionRequirement(
                name="python",
                alternatives=(ToolExecutableRequirement(executable=_PRESENT),),
            ),
        ),
    )

    async def run(self, ctx, args):
        return ToolResult(content="not dispatched by this contract")


class _NodeTool(Tool):
    spec = ToolSpec(
        name="node_program",
        execution_requirements=(
            ToolExecutionRequirement(
                name="node",
                alternatives=(ToolExecutableRequirement(executable="node"),),
            ),
        ),
    )

    async def run(self, ctx, args):
        return ToolResult(content="must be refused before dispatch")


def _requirements(*alternatives: ToolExecutableRequirement) -> ExecutionRequirements:
    return ExecutionRequirements(
        tool_requirements=tuple(
            ExecutionToolRequirement(
                tool_name=f"tool-{index}",
                requirement=ToolExecutionRequirement(name="program", alternatives=(alternative,)),
            )
            for index, alternative in enumerate(alternatives)
        )
    )


async def _session(runner: LambdaMicroVMRunner, tool: Tool, session_id: str):
    provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="lambda"), runner=runner), default=True
    )
    app.register_agent(AgentSpec(name="agent", model="scripted"), tools=[tool])
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="agent",
                session_id=session_id,
                messages=[Message.text("user", "admit")],
            )
        )
    ]
    return events, provider


async def main() -> None:
    if os.environ.get("CAYU_LAMBDA_MICROVM_TOOL_ADMISSION_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_TOOL_ADMISSION_LIVE=1 to run this contract.")
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not image.startswith("arn:"):
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE to a built MicroVM image ARN.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")
    missing = "node"
    ingress = os.environ.get(
        "CAYU_LAMBDA_MICROVM_INGRESS_CONNECTOR",
        f"arn:aws:lambda:{region}:aws:network-connector:aws-network-connector:ALL_INGRESS",
    )

    runner: LambdaMicroVMRunner | None = None
    try:
        runner = await LambdaMicroVMRunner.create(
            image,
            region_name=region,
            ingress_network_connectors=[ingress],
            maximum_duration_in_seconds=_MAXIMUM_DURATION_SECONDS,
            close_action="terminate",
        )

        requirements = _requirements(
            ToolExecutableRequirement(executable=_PRESENT),
            ToolExecutableRequirement(executable=missing),
            ToolExecutableRequirement(
                executable="bash", probe_arguments=("-c", "exit 0"), accepted_exit_codes=(0,)
            ),
        )
        observer = runner.execution_admission_observer(requirements)
        candidate = await observer.collect()
        evidence = candidate.evidence
        require(evidence is not None and evidence.tool_requirements is not None, "no evidence")
        claims = {claim.executable: claim for claim in evidence.tool_requirements.executables}
        present = claims[_PRESENT]
        require(present.state == "live_verified", f"{_PRESENT} was not live_verified")
        require(
            present.valid_until is not None
            and present.observed_at is not None
            and (present.valid_until - present.observed_at).total_seconds()
            == EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
            "live evidence validity window is not the bounded TTL",
        )
        require(claims[missing].state == "unavailable", f"{missing} was not refused")
        require(claims["bash"].state == "live_verified", "probe arguments were not accepted")
        refused = evaluate_execution_admission(
            candidate="lambda-microvm", requirements=requirements, evidence=evidence
        )
        require(
            refused.status == "refused"
            and {refusal.executable for refusal in refused.refusals} == {missing},
            "admission did not refuse exactly the missing executable",
        )
        require(observer.snapshot() == candidate, "snapshot did not return the exact evidence")

        await observer.refresh()
        renewed = observer.snapshot()
        require(
            renewed.evidence.environment_fingerprint == evidence.environment_fingerprint,
            "renewal observed a different MicroVM identity",
        )
        renewed_present = renewed.evidence.tool_requirements.executable_for(_PRESENT)
        require(
            renewed_present is not None
            and renewed_present.state == "live_verified"
            and renewed_present.observed_at > present.observed_at,
            "renewal did not re-observe the executable",
        )

        admitted_events, admitted_provider = await _session(
            runner, _PythonTool(), "lambda-tool-admission-present"
        )
        require(
            EventType.SESSION_COMPLETED in {event.type for event in admitted_events}
            and len(admitted_provider.requests) == 1,
            f"a tool requiring {_PRESENT} was not admitted",
        )
        refused_events, refused_provider = await _session(
            runner, _NodeTool(), "lambda-tool-admission-missing"
        )
        failed = next(
            (event for event in refused_events if event.type is EventType.SESSION_FAILED), None
        )
        require(
            failed is not None
            and not refused_provider.requests
            and any(
                refusal.get("tool_name") == "node_program" and refusal.get("executable") == missing
                for refusal in failed.payload["execution_admission"]["refusals"]
            ),
            "node_program was not refused before any model request",
        )

        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "lambda-microvm",
                    "microvm_id": runner.microvm_id,
                    "region": region,
                    "identity": "microvm+endpoint+image+sidecar-protocol+boot-id",
                    "environment_fingerprint": evidence.environment_fingerprint,
                    "present_executable": {_PRESENT: "live_verified"},
                    "missing_executable": {missing: "unavailable"},
                    "probe_arguments": "live_verified",
                    "ttl_seconds": EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
                    "renewal": "same_identity",
                    "tool_admitted": "verified",
                    "tool_refused_before_model": "verified",
                },
                sort_keys=True,
            )
        )
    finally:
        if runner is not None:
            await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
