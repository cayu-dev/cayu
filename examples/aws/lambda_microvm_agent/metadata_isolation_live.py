"""Launch the Lambda MicroVM metadata-isolation contract from outside the VPC."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from typing import Any

from examples.aws.lambda_microvm_agent.metadata_isolation_task import (
    EVIDENCE_PREFIX,
    REQUIRED_EVIDENCE_POSITIVE_INTEGERS,
    REQUIRED_EVIDENCE_VALUES,
)

_CHECK_FLAG = "CAYU_AWS_METADATA_ISOLATION_LIVE"
_RUN_ID_ENV = "CAYU_AWS_METADATA_ISOLATION_RUN_ID"
_STACK_ENV = "CAYU_AWS_METADATA_ISOLATION_STACK"
_TIMEOUT_SECONDS = 20 * 60
_MAX_LOG_PAGES = 100


async def main() -> None:
    if os.environ.get(_CHECK_FLAG) != "1":
        raise SystemExit(f"Set {_CHECK_FLAG}=1 to run this contract.")
    await asyncio.to_thread(_run_via_ecs)


def _run_via_ecs() -> None:
    import boto3  # ty: ignore[unresolved-import]

    region = _required_region()
    stack_name = _required_env(_STACK_ENV)
    run_id = f"metadata-isolation-{uuid.uuid4().hex[:12]}"
    session = boto3.Session(region_name=region)
    cloudformation = session.client("cloudformation")
    ecs = session.client("ecs")
    logs = session.client("logs")
    stack = cloudformation.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack.get("Outputs", [])}
    required_outputs = {
        "ClusterArn",
        "ControlTaskDefinitionArn",
        "ControlSecurityGroupId",
        "PrivateSubnetId",
        "ControlLogGroupName",
    }
    missing = sorted(required_outputs.difference(outputs))
    if missing:
        raise RuntimeError("Deployed stack is missing live-check outputs: " + ", ".join(missing))

    started_ms = int(time.time() * 1000)
    response = ecs.run_task(
        cluster=outputs["ClusterArn"],
        taskDefinition=outputs["ControlTaskDefinitionArn"],
        launchType="FARGATE",
        count=1,
        networkConfiguration={
            "awsvpcConfiguration": {
                "subnets": [outputs["PrivateSubnetId"]],
                "securityGroups": [outputs["ControlSecurityGroupId"]],
                "assignPublicIp": "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": "control",
                    "command": [
                        "python",
                        "-m",
                        "examples.aws.lambda_microvm_agent.metadata_isolation_task",
                    ],
                    "environment": [
                        {"name": _CHECK_FLAG, "value": "1"},
                        {"name": _RUN_ID_ENV, "value": run_id},
                    ],
                }
            ]
        },
    )
    failures = response.get("failures", [])
    if failures:
        raise RuntimeError("ECS rejected metadata-isolation task: " + json.dumps(failures))
    task_arn = response["tasks"][0]["taskArn"]
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    task: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        described = ecs.describe_tasks(cluster=outputs["ClusterArn"], tasks=[task_arn])
        task = described["tasks"][0]
        if task.get("lastStatus") == "STOPPED":
            break
        time.sleep(5)
    else:
        with contextlib.suppress(Exception):
            ecs.stop_task(
                cluster=outputs["ClusterArn"],
                task=task_arn,
                reason="Cayu metadata-isolation live check timed out",
            )
        raise TimeoutError("ECS metadata-isolation task did not stop before its deadline.")

    if task is None:
        raise RuntimeError("ECS metadata-isolation task disappeared before inspection.")
    container = task.get("containers", [{}])[0]
    exit_code = container.get("exitCode")
    messages = _evidence_messages(
        logs,
        log_group=outputs["ControlLogGroupName"],
        run_id=run_id,
        start_time_ms=started_ms,
    )
    if exit_code != 0:
        task_messages = _task_log_messages(
            logs,
            log_group=outputs["ControlLogGroupName"],
            task_arn=task_arn,
        )
        detail = "\n".join(task_messages or messages)[-4000:]
        raise RuntimeError(
            f"ECS metadata-isolation task exited with {exit_code}: "
            f"{detail or task.get('stoppedReason')}"
        )
    evidence_lines = [message for message in messages if EVIDENCE_PREFIX in message]
    if len(evidence_lines) != 1:
        raise RuntimeError("ECS metadata-isolation task did not emit exactly one evidence record.")
    encoded = evidence_lines[0].split(EVIDENCE_PREFIX, 1)[1].strip()
    evidence = _validated_evidence(encoded, run_id=run_id)
    print(EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True))


def _validated_evidence(encoded: str, *, run_id: str) -> dict[str, Any]:
    try:
        evidence = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ECS metadata-isolation evidence is not valid JSON.") from exc
    if type(evidence) is not dict:
        raise RuntimeError("ECS metadata-isolation evidence must be an object.")
    expected_keys = (
        set(REQUIRED_EVIDENCE_VALUES) | set(REQUIRED_EVIDENCE_POSITIVE_INTEGERS) | {"run_id"}
    )
    actual_keys = set(evidence)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            "ECS metadata-isolation evidence has an invalid schema "
            f"(missing={missing}, unexpected={unexpected})."
        )
    mismatches = {
        key: {"expected": expected, "actual": evidence.get(key)}
        for key, expected in REQUIRED_EVIDENCE_VALUES.items()
        if evidence.get(key) != expected
    }
    if evidence.get("run_id") != run_id:
        mismatches["run_id"] = {"expected": run_id, "actual": evidence.get("run_id")}
    for key in REQUIRED_EVIDENCE_POSITIVE_INTEGERS:
        value = evidence.get(key)
        if type(value) is not int or value <= 0:
            mismatches[key] = {"expected": "positive integer", "actual": value}
    if mismatches:
        raise RuntimeError(
            "ECS metadata-isolation evidence did not prove the required boundary: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return evidence


def _evidence_messages(
    logs: Any,
    *,
    log_group: str,
    run_id: str,
    start_time_ms: int,
) -> list[str]:
    deadline = time.monotonic() + 60
    messages: list[str] = []
    while time.monotonic() < deadline:
        messages = []
        next_token: str | None = None
        for _page in range(_MAX_LOG_PAGES):
            request: dict[str, Any] = {
                "logGroupName": log_group,
                "startTime": start_time_ms,
                "filterPattern": f'"{run_id}"',
            }
            if next_token is not None:
                request["nextToken"] = next_token
            response = logs.filter_log_events(**request)
            messages.extend(str(event.get("message", "")) for event in response.get("events", []))
            returned_token = response.get("nextToken")
            if type(returned_token) is not str or returned_token == next_token:
                break
            next_token = returned_token
        else:
            raise RuntimeError("CloudWatch evidence exceeded the bounded pagination limit.")
        if any(EVIDENCE_PREFIX in message for message in messages):
            return messages
        time.sleep(2)
    return messages


def _task_log_messages(logs: Any, *, log_group: str, task_arn: str) -> list[str]:
    task_id = task_arn.rsplit("/", 1)[-1]
    stream = f"control/control/{task_id}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            response = logs.get_log_events(
                logGroupName=log_group,
                logStreamName=stream,
                startFromHead=True,
            )
        except logs.exceptions.ResourceNotFoundException:
            time.sleep(2)
            continue
        return [str(event.get("message", "")) for event in response.get("events", [])]
    return []


def _required_region() -> str:
    value = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if value is None or not value.strip():
        raise RuntimeError("Set AWS_REGION or AWS_DEFAULT_REGION.")
    return value.strip()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Set {name}.")
    return value.strip()


if __name__ == "__main__":
    asyncio.run(main())
