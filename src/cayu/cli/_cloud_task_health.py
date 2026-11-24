"""What `cayu cloud service status` says about tasks held by a stopped worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def fenced_tasks_warning(service: Mapping[str, Any], application: str = "AGENT") -> str | None:
    """A warning naming the Agent's fenced tasks and how to settle them, or None.

    Cayu Cloud returns ``task_health`` in the service status: the latest count the Agent's
    Runtime reported of tasks still claimed after their worker was lost. It is ``null`` until
    Cloud has read one, and an older Cloud doesn't send it.
    """

    health = service.get("task_health")
    if not isinstance(health, Mapping):
        return None
    count = health.get("fenced_after_worker_loss")
    if type(count) is not int or count < 1:
        return None
    # The Runtime stops counting at 100.
    tasks = f"{count} task" if count == 1 else f"{count} tasks"
    if health.get("complete") is False:
        tasks = f"{tasks} or more"
    return (
        f"{tasks} stayed claimed after the worker running them stopped, as when a deploy "
        "replaces the worker during a task that can't be restarted. No worker runs them again "
        "until the Agent settles them. "
        f"`cayu task fenced --application {application}` lists them with whether Cayu Cloud "
        "recorded that their worker's run stopped, and `--settle` settles those (a task "
        "worker on Cloud also does this itself for workers named with `cloud_worker_id`). "
        "For the rest, a single worker run with `cayu worker "
        "--reconcile-abandoned-after-seconds` settles them itself, and `cayu inspect "
        "--fenced-tasks` gives the step for each."
    )
