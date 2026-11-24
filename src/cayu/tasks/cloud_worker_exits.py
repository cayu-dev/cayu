"""Settle tasks fenced after worker loss once Cayu Cloud has recorded that worker's exit.

Cayu never settles a task on lease expiry alone: an expired lease doesn't prove the
old handler stopped. On Cayu Cloud the platform does know. Cloud records each worker
instance and how it stopped, and the Agent reads that record at
``CAYU_WORKER_EXITS_URL`` with its attention credential
(``CAYU_ATTENTION_HANDOFF_TOKEN``). Each part here works on its own:

- :func:`worker_run_reference` names the run this process belongs to, the way Cloud's
  record names it: the ECS task ARN from the task metadata endpoint, or on Kubernetes
  a random run marker in its container's termination-message file, shared by every
  process of that container run, which the kubelet hands to Cloud only after the
  container exited. Outside Cayu Cloud it reads and writes nothing.
- :func:`cloud_worker_id` builds a task worker id that carries that reference (as a
  digest), so a task's ``worker_id`` says which run held it.
- :func:`read_cloud_worker_exits` reads Cloud's record.
- :func:`reconcile_tasks_of_exited_workers` settles fenced tasks whose worker id names a
  run the record shows stopped, through the same paths as
  :func:`~cayu.tasks.reconcile_abandoned_tasks`, recording the exit as the evidence.

A worker id without a run reference (built outside Cayu Cloud, or by hand) is never
matched, so those tasks are left for the other settlement paths.
"""

from __future__ import annotations

import importlib
import logging
import os
import secrets
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import httpx

from cayu._validation import require_clean_nonblank
from cayu.approvals.tools import ResolutionActor
from cayu.human_attention_handoff import TOKEN_ENVIRONMENT
from cayu.tasks.abandoned import (
    CLOUD_EXIT_SETTLEMENT_ENV,
    AbandonedTaskSettlement,
    RestartLimitHandler,
    _list_fenced,
    _settle_abandoned,
    _settleable,
    abandoned_task_settlement_choice,
    cloud_exit_settlement_enabled,
    validate_max_restarts,
)
from cayu.tasks.queries import TaskQuery, copy_task_query
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.restart import DEFAULT_MAX_ABANDONED_RESTARTS
from cayu.tasks.store import TaskStore

try:
    _FCNTL = importlib.import_module("fcntl")
except ImportError:  # pragma: no cover - Windows
    _FCNTL = None

#: Where Cayu Cloud binds the Agent's worker exit record.
WORKER_EXITS_URL_ENVIRONMENT = "CAYU_WORKER_EXITS_URL"
#: The validator :func:`reconcile_tasks_of_exited_workers` records.
CLOUD_WORKER_EXIT_VALIDATOR_ID = "cayu.cloud.worker-exit"
CLOUD_WORKER_EXIT_VALIDATOR_VERSION = "1"
#: What a Kubernetes worker writes to its termination-message file.
RUN_MARKER_PREFIX = "cayu-worker-run:"
#: Kubernetes' default termination-message file.
TERMINATION_LOG = Path("/dev/termination-log")
_ECS_METADATA_ENVIRONMENT = "ECS_CONTAINER_METADATA_URI_V4"
_KUBERNETES_ENVIRONMENT = "KUBERNETES_SERVICE_HOST"
_WORKER_ID_MARK = "@cloud-run:"
_DIGEST_CHARS = 24
logger = logging.getLogger(__name__)
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_EXITS = 1_000

# The run reference is read once per process: writing a second marker would make the
# first one unprovable.
_process_reference: str | None = None
_process_reference_read = False


class WorkerExitsUnsupported(Exception):
    """This process can't read a worker exit record: outside Cayu Cloud, or a Cloud
    that doesn't serve it."""


class WorkerExitsUnavailable(Exception):
    """The worker exit record could not be read now."""


@dataclass(frozen=True)
class WorkerExit:
    """One worker instance as Cloud last recorded it.

    ``state`` is ``stopping``, ``stopped`` or ``gone``. ``run_marker`` is the marker a
    Kubernetes run wrote, once Cloud saw that run exit; it needs a Cayu Cloud that
    records worker run markers, and is ``None`` on ECS and on older Clouds.
    """

    instance: str
    state: str
    stopped_at: datetime | None = None
    reason: str | None = None
    exit_code: int | None = None
    run_marker: str | None = None

    def run_references(self) -> frozenset[str]:
        """The run references this entry names."""

        references = {self.instance}
        if self.run_marker is not None:
            references.add(f"{RUN_MARKER_PREFIX}{self.run_marker}")
        return frozenset(references)


@dataclass(frozen=True)
class WorkerExitRecord:
    """Cloud's record: ``looked_at`` is when it last asked the platform (``None``: never)."""

    exits: tuple[WorkerExit, ...]
    looked_at: datetime | None = None


async def worker_run_reference(
    environ: Mapping[str, str] = os.environ,
    *,
    termination_log: Path = TERMINATION_LOG,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str | None:
    """This process's run, as Cayu Cloud's worker exit record names it, or ``None``.

    It is ``None`` unless Cayu Cloud bound ``CAYU_WORKER_EXITS_URL``, so nothing is
    read or written outside Cayu Cloud. On ECS it is the task ARN from the task
    metadata endpoint. On Kubernetes it is ``cayu-worker-run:<marker>``, the marker in
    the container's ``termination_log``: the first process of a container run writes
    it, under a file lock, and every later process of the same run reuses it, since
    the kubelet creates the file afresh for each container start. Cloud learns the
    marker only once the container exited, and only a Cayu Cloud that records worker
    run markers reports it. When neither can be read it is ``None``. The reference is
    read once per process and then reused.
    """

    global _process_reference, _process_reference_read
    if _process_reference_read:
        return _process_reference
    reference = None
    metadata_uri = environ.get(_ECS_METADATA_ENVIRONMENT, "").strip()
    on_cloud = bool(environ.get(WORKER_EXITS_URL_ENVIRONMENT, "").strip())
    if on_cloud and metadata_uri:
        reference = await _ecs_task_arn(metadata_uri, transport)
    elif on_cloud and environ.get(_KUBERNETES_ENVIRONMENT, "").strip():
        reference = _write_run_marker(termination_log)
    _process_reference, _process_reference_read = reference, True
    return reference


def worker_id_for_run(name: str, run_reference: str | None) -> str:
    """``name`` plus a digest of ``run_reference``; ``name`` alone without one."""

    name = require_clean_nonblank(name, "name")
    if _WORKER_ID_MARK in name:
        raise ValueError(f"name must not contain {_WORKER_ID_MARK!r}.")
    if run_reference is None:
        return name
    return f"{name}{_WORKER_ID_MARK}{_run_digest(run_reference)}"


async def cloud_worker_id(
    name: str,
    environ: Mapping[str, str] = os.environ,
    *,
    termination_log: Path = TERMINATION_LOG,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """A task worker id for this process that names its Cloud run.

    Pass it as ``run_task_worker(worker_id=...)``. ``name`` must be unique among the
    workers of one process. Outside Cayu Cloud it is ``name`` unchanged.
    """

    reference = await worker_run_reference(
        environ, termination_log=termination_log, transport=transport
    )
    return worker_id_for_run(name, reference)


async def read_cloud_worker_exits(
    environ: Mapping[str, str] = os.environ,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> WorkerExitRecord:
    """Read the Agent's worker exit record from Cayu Cloud.

    Raises :class:`WorkerExitsUnsupported` when ``CAYU_WORKER_EXITS_URL`` or the
    attention credential isn't bound or Cloud answers 404, and
    :class:`WorkerExitsUnavailable` for any other failure. The credential never
    appears in an error.
    """

    url = environ.get(WORKER_EXITS_URL_ENVIRONMENT, "").strip()
    token = environ.get(TOKEN_ENVIRONMENT, "").strip()
    if not url or not token:
        missing = WORKER_EXITS_URL_ENVIRONMENT if not url else TOKEN_ENVIRONMENT
        raise WorkerExitsUnsupported(
            f"{missing} is not set: reading worker exits needs a Cayu Cloud that records them."
        )
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS, transport=transport, follow_redirects=False
    ) as client:
        try:
            response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as exc:
            raise WorkerExitsUnavailable(
                f"Cayu Cloud's worker exit record could not be reached ({type(exc).__name__})."
            ) from None
    if response.status_code == 404:
        raise WorkerExitsUnsupported("This Cayu Cloud doesn't serve worker exits (HTTP 404).")
    if response.status_code != 200:
        raise WorkerExitsUnavailable(
            f"Cayu Cloud's worker exit record answered HTTP {response.status_code}."
        )
    return _parse_record(response)


def exited_run_references(record: WorkerExitRecord, *, accept_gone: bool = False) -> frozenset[str]:
    """The run references the record shows exited.

    Only ``stopped`` entries count: the platform reported the exit. ``gone`` (the
    instance left the platform's view before Cloud saw it stop) proves nothing on its
    own; ``accept_gone=True`` counts it too, for a deployment whose platform removes an
    instance only after it stopped.
    """

    references: set[str] = set()
    for entry in _exited_entries(record, accept_gone=accept_gone):
        references |= entry.run_references()
    return frozenset(references)


def _exited_entries(record: WorkerExitRecord, *, accept_gone: bool) -> Iterator[WorkerExit]:
    states = {"stopped", "gone"} if accept_gone else {"stopped"}
    return (entry for entry in record.exits if entry.state in states)


async def reconcile_tasks_of_exited_workers(
    task_store: TaskStore,
    record: WorkerExitRecord,
    *,
    accept_gone: bool = False,
    query: TaskQuery | None = None,
    settlement: AbandonedTaskSettlement | str = AbandonedTaskSettlement.CANCEL,
    max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
    reconciled_by: ResolutionActor | None = None,
    on_restart_limit_reached: RestartLimitHandler | None = None,
) -> list[Task]:
    """Settle fenced tasks whose worker's run the record shows exited; return them.

    A task counts only when its ``worker_id`` came from :func:`cloud_worker_id` (or
    :func:`worker_id_for_run`) and names a run in :func:`exited_run_references`. It is
    settled like :func:`~cayu.tasks.reconcile_abandoned_tasks` settles it: cancelled,
    or with ``settlement="restart"`` a restartable task is requeued. The validator is
    ``cayu.cloud.worker-exit`` and the evidence id names the exit. Session-attached and
    retry-series tasks are left alone, as there. ``max_restarts`` and
    ``on_restart_limit_reached`` work as they do there.
    """

    settlement = abandoned_task_settlement_choice(settlement)
    max_restarts = validate_max_restarts(max_restarts, "max_restarts")
    exited = _exit_evidence(record, accept_gone=accept_gone)
    if not exited:
        return []
    selected = copy_task_query(query)
    if selected.fenced_after_worker_loss is False:
        raise ValueError("reconcile_tasks_of_exited_workers only considers fenced tasks.")
    selected = selected.model_copy(update={"fenced_after_worker_loss": True})
    tasks, _ = await _list_fenced(task_store, selected)
    due = [
        task
        for task in tasks
        if _settleable(task, live_worker_ids=frozenset())
        and _worker_digest(task.worker_id) in exited
    ]
    return await _settle_abandoned(
        task_store,
        due,
        validator_id=CLOUD_WORKER_EXIT_VALIDATOR_ID,
        validator_version=CLOUD_WORKER_EXIT_VALIDATOR_VERSION,
        reconciled_by=reconciled_by,
        settlement=settlement,
        max_restarts=max_restarts,
        on_restart_limit_reached=on_restart_limit_reached,
        evidence_for=lambda task: exited[_worker_digest(task.worker_id) or ""],
    )


class CloudWorkerExitSettler:
    """What a task worker runs on its fenced-task cadence: read Cloud's record, settle.

    Built by :func:`cloud_worker_exit_settler`. A Cloud that doesn't serve the record
    (or a process outside Cayu Cloud) turns it off after one log line; a failed read is
    logged and retried at the next run.
    """

    def __init__(
        self,
        task_store: TaskStore,
        *,
        query: TaskQuery | None = None,
        settlement: AbandonedTaskSettlement | str = AbandonedTaskSettlement.CANCEL,
        max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
        on_restart_limit_reached: RestartLimitHandler | None = None,
        environ: Mapping[str, str] = os.environ,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._task_store = task_store
        self._on_restart_limit_reached = on_restart_limit_reached
        #: Whether the latest run couldn't read the record; it is retried next run.
        self.last_run_failed = False
        self._query = query
        self._settlement = abandoned_task_settlement_choice(settlement)
        self._max_restarts = validate_max_restarts(max_restarts, "max_restarts")
        self._environ = environ
        self._transport = transport
        self._off = False

    @property
    def active(self) -> bool:
        return not self._off

    async def run(self) -> list[Task]:
        self.last_run_failed = False
        if self._off:
            return []
        try:
            record = await read_cloud_worker_exits(self._environ, transport=self._transport)
        except WorkerExitsUnsupported as exc:
            self._off = True
            logger.info("Not settling fenced tasks from Cloud's worker exits: %s", exc)
            return []
        except WorkerExitsUnavailable as exc:
            self.last_run_failed = True
            logger.warning("Reading Cloud's worker exits failed; retrying later: %s", exc)
            return []
        settled = await reconcile_tasks_of_exited_workers(
            self._task_store,
            record,
            query=self._query,
            settlement=self._settlement,
            max_restarts=self._max_restarts,
            on_restart_limit_reached=self._on_restart_limit_reached,
        )
        for task in settled:
            logger.warning(
                "%s task %s (type %s): Cayu Cloud recorded that the worker run holding it stopped.",
                "Restarted" if task.status is TaskStatus.PENDING else "Settled as cancelled",
                task.id,
                task.type,
            )
        return settled


def cloud_worker_exit_settler(
    task_store: TaskStore,
    *,
    enabled: bool | None = None,
    query: TaskQuery | None = None,
    max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
    on_restart_limit_reached: RestartLimitHandler | None = None,
    environ: Mapping[str, str] = os.environ,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CloudWorkerExitSettler | None:
    """The settler a task worker runs, or ``None``.

    ``enabled=None`` turns it on when Cayu Cloud bound ``CAYU_WORKER_EXITS_URL``, unless
    ``CAYU_WORKER_CLOUD_EXIT_SETTLEMENT`` is ``0``; ``False`` turns it off. It restarts
    tasks created with ``restartable=True`` when the store can, and cancels the rest.
    ``run_task_worker`` passes its own ``max_abandoned_restarts`` and
    ``on_restart_limit_reached``, so the cap and the callback are the same on either path.
    """

    if not cloud_exit_settlement_enabled(enabled, environ):
        return None
    restarts = type(task_store).restart_abandoned_task is not TaskStore.restart_abandoned_task
    return CloudWorkerExitSettler(
        task_store,
        query=query,
        settlement=AbandonedTaskSettlement.RESTART if restarts else AbandonedTaskSettlement.CANCEL,
        max_restarts=max_restarts,
        on_restart_limit_reached=on_restart_limit_reached,
        environ=environ,
        transport=transport,
    )


def _exit_evidence(record: WorkerExitRecord, *, accept_gone: bool) -> dict[str, str]:
    # The evidence id per run digest: which state Cloud recorded, and when it stopped.
    evidence: dict[str, str] = {}
    for entry in _exited_entries(record, accept_gone=accept_gone):
        when = "unknown" if entry.stopped_at is None else entry.stopped_at.isoformat()
        for reference in entry.run_references():
            evidence.setdefault(_run_digest(reference), f"cloud-worker-{entry.state}-at:{when}")
    return evidence


def _run_digest(reference: str) -> str:
    return sha256(reference.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


def _worker_digest(worker_id: str | None) -> str | None:
    if worker_id is None:
        return None
    _, mark, digest = worker_id.rpartition(_WORKER_ID_MARK)
    return digest if mark and len(digest) == _DIGEST_CHARS else None


async def _ecs_task_arn(
    metadata_uri: str, transport: httpx.AsyncBaseTransport | None
) -> str | None:
    # The metadata endpoint is link-local: never send it through a proxy.
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS, transport=transport, trust_env=False
    ) as client:
        try:
            response = await client.get(f"{metadata_uri.rstrip('/')}/task")
            body = response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
    arn = body.get("TaskARN") if isinstance(body, dict) else None
    return arn if isinstance(arn, str) and arn.startswith("arn:") else None


def _write_run_marker(termination_log: Path) -> str | None:
    """The container run's marker: the one already in the file, or a new one written now."""

    try:
        # Only an existing file: the kubelet mounts it, and nothing else may stand in for it.
        with termination_log.open("r+") as file:
            if _FCNTL is not None:
                # Processes of one container starting together agree on one marker.
                _FCNTL.flock(file.fileno(), _FCNTL.LOCK_EX)
            existing = file.read(len(RUN_MARKER_PREFIX) + 33).strip()
            if existing.startswith(RUN_MARKER_PREFIX) and _is_marker(
                existing[len(RUN_MARKER_PREFIX) :]
            ):
                return existing
            marker = f"{RUN_MARKER_PREFIX}{secrets.token_hex(16)}"
            file.seek(0)
            file.truncate()
            file.write(marker)
            file.flush()
            return marker
    except OSError:
        return None


def _parse_record(response: httpx.Response) -> WorkerExitRecord:
    def malformed() -> WorkerExitsUnavailable:
        return WorkerExitsUnavailable(
            "Cayu Cloud's worker exit record has a body this Runtime can't read."
        )

    try:
        body = response.json()
    except ValueError:
        raise malformed() from None
    items = body.get("exits") if isinstance(body, dict) else None
    if not isinstance(items, list) or len(items) > _MAX_EXITS:
        raise malformed()
    exits: list[WorkerExit] = []
    for item in items:
        if not isinstance(item, dict):
            raise malformed()
        instance, state = item.get("instance"), item.get("state")
        if not isinstance(instance, str) or not instance or not isinstance(state, str):
            raise malformed()
        marker = item.get("run_marker")
        exit_code = item.get("exit_code")
        reason = item.get("reason")
        exits.append(
            WorkerExit(
                instance=instance,
                state=state,
                stopped_at=_instant(item.get("stopped_at")),
                reason=reason if isinstance(reason, str) else None,
                exit_code=exit_code if type(exit_code) is int else None,
                run_marker=marker if isinstance(marker, str) and _is_marker(marker) else None,
            )
        )
    return WorkerExitRecord(exits=tuple(exits), looked_at=_instant(body.get("looked_at")))


def _is_marker(value: str) -> bool:
    return len(value) == 32 and all(char in "0123456789abcdef" for char in value)


def _instant(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = [
    "CLOUD_EXIT_SETTLEMENT_ENV",
    "CLOUD_WORKER_EXIT_VALIDATOR_ID",
    "CLOUD_WORKER_EXIT_VALIDATOR_VERSION",
    "RUN_MARKER_PREFIX",
    "WORKER_EXITS_URL_ENVIRONMENT",
    "CloudWorkerExitSettler",
    "WorkerExit",
    "WorkerExitRecord",
    "WorkerExitsUnavailable",
    "WorkerExitsUnsupported",
    "cloud_worker_exit_settler",
    "cloud_worker_id",
    "exited_run_references",
    "read_cloud_worker_exits",
    "reconcile_tasks_of_exited_workers",
    "worker_id_for_run",
    "worker_run_reference",
]
