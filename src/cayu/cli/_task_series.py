"""`cayu task series`: list, show, pause, resume and delete recurring task series.

Series live in the task store (``TaskStore.save_task_series``). These commands work
on a local store (``--sqlite``, ``--postgres`` or the project's) directly, and on a
running server (``--server-url``) or a Cayu Cloud Agent (``--application``) through
its ``/task-series`` routes; an Agent whose Cayu predates them gets a clear error. Each
occurrence is an ordinary task whose metadata names its series, so
``cayu task list --metadata cayu_series_id=ID`` lists a series' occurrences anywhere task
reads work.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from cayu.cli._output import add_output_options, print_table
from cayu.cli._read_targets import ReadTarget, add_read_target_options
from cayu.sessions._call_inspection import redact_inspection_value
from cayu.tasks.series import TaskSeriesState

CLI_SCHEMA_VERSION = "1"
_DEFAULT_LIMIT = 100


def add_series_parser(commands: Any) -> None:
    series = commands.add_parser(
        "series",
        help="List, show, pause, resume or delete recurring task series.",
        description=(
            "Recurring task series kept in the task store: a local one (--sqlite, --postgres "
            "or the project's), a running server (--server-url) or a Cloud Agent "
            "(--application). List a series' occurrences with "
            "`cayu task list --metadata cayu_series_id=ID`."
        ),
    )
    actions = series.add_subparsers(dest="series_command", required=True)
    listing = actions.add_parser(
        "list",
        help="List the stored series.",
        description=(
            "List the stored series by id: whether each is paused, its rule and the "
            "occurrence queued last. Inspect one with `cayu task series show ID`."
        ),
    )
    listing.add_argument("--limit", type=int, default=_DEFAULT_LIMIT, help="At most this many.")
    listing.add_argument("--offset", type=int, default=0, help="Skip this many first.")
    show = actions.add_parser(
        "show",
        help="Show one series.",
        description=(
            "Show one stored series. List its occurrences with "
            "`cayu task list --metadata cayu_series_id=ID`."
        ),
    )
    pause = actions.add_parser(
        "pause",
        help="Pause a series: an occurrence that ends queues nothing more.",
        description=(
            "Pause a series: an occurrence that ends queues nothing more, and the one "
            "already queued stays. Continue it with `cayu task series resume ID`."
        ),
    )
    resume = actions.add_parser(
        "resume",
        help="Resume a series; queue its next occurrence if nothing is queued.",
        description=(
            "Resume a paused series and queue its next occurrence if nothing is queued. "
            "Check it with `cayu task series show ID`."
        ),
    )
    delete = actions.add_parser(
        "delete",
        help="Delete a series; its tasks stay.",
        description=(
            "Delete a stored series; its tasks stay, and an occurrence that ends queues "
            "nothing more. Find what is left with "
            "`cayu task list --metadata cayu_series_id=ID`."
        ),
    )
    for parser in (show, pause, resume, delete):
        parser.add_argument("series_id")
    for parser in (listing, show, pause, resume, delete):
        add_read_target_options(parser)
        add_output_options(parser, formats=("table", "json"), default="table")


def _open_store(target: ReadTarget) -> tuple[Any, dict[str, Any]]:
    """The series store for ``target``: its local database's task store, opened for writing
    (never migrated), or an agent's API for ``--server-url`` and ``--application``."""

    from cayu.cli._task_readers import open_writable_task_store

    if target.store is None:
        return _open_api(target)
    return open_writable_task_store(target.store)


_UNSUPPORTED_HINT = (
    "List a series' occurrences with `cayu task list --metadata cayu_series_id=ID` instead."
)


def _state(answer: Any) -> TaskSeriesState:
    if not isinstance(answer, dict) or not isinstance(answer.get("state"), dict):
        raise ValueError("The agent returned an unusable task series.")
    return TaskSeriesState.model_validate(answer["state"])


class _ApiSeriesStore:
    """The task-series store methods these commands use, through an agent's API."""

    def __init__(self, api: Any) -> None:
        self._api = api
        self._discovered = False

    def _call(self, call: Any) -> Any:
        from cayu.cli._cloud_agent_api import CayuApiError

        try:
            if not self._discovered:
                # Find the API first, so a missing series route reads as an older Cayu.
                self._api.get("/app", resource="agent")
                self._discovered = True
            return call()
        except CayuApiError as exc:
            if exc.category == "unsupported":
                raise ValueError(f"{exc} {_UNSUPPORTED_HINT}") from None
            raise

    @staticmethod
    def _path(series_id: str, action: str = "") -> str:
        from urllib.parse import quote

        return f"/task-series/{quote(series_id, safe='')}{action}"

    async def list_task_series(self, *, limit: int, offset: int) -> list[TaskSeriesState]:
        page = self._call(
            lambda: self._api.get(
                "/task-series",
                query={"limit": str(limit), "offset": str(offset)},
                resource="task series",
            )
        )
        if not isinstance(page, dict) or not isinstance(page.get("series"), list):
            raise ValueError("The agent returned an unusable task series page.")
        return [_state(item) for item in page["series"]]

    async def load_task_series(self, series_id: str) -> TaskSeriesState | None:
        from cayu.cli._cloud_agent_api import CayuApiNotFound

        try:
            state = self._call(lambda: self._api.get(self._path(series_id), resource="task series"))
        except CayuApiNotFound:
            return None
        return _state(state)

    async def set_task_series_paused(self, series_id: str, paused: bool) -> TaskSeriesState:
        from cayu.cli._cloud_agent_api import CayuApiNotFound

        action = "/pause" if paused else "/resume"
        try:
            state = self._call(
                lambda: self._api.mutate(self._path(series_id, action), {}, resource="task series")
            )
        except CayuApiNotFound:
            raise KeyError(series_id) from None
        return _state(state)

    async def delete_task_series(self, series_id: str) -> bool:
        from cayu.cli._cloud_agent_api import CayuApiNotFound

        try:
            self._call(
                lambda: self._api.mutate(
                    self._path(series_id, "/delete"), {}, resource="task series"
                )
            )
        except CayuApiNotFound:
            return False
        return True

    async def close(self) -> None:
        self._api.close()


def _open_api(target: ReadTarget) -> tuple[Any, dict[str, Any]]:
    if target.server_url is not None:
        from cayu.cli._cloud_agent_api import CayuApi
        from cayu.cli.diagnostics import read_authorization

        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
        return _ApiSeriesStore(api), {"kind": "server", "url": target.server_url}
    from cayu.cli._cloud_agent_api import open_cloud_agent_api

    assert target.application is not None
    api = open_cloud_agent_api(
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
    )
    return _ApiSeriesStore(api), {"kind": "cayu_cloud", "application": target.application}


def series_row(state: TaskSeriesState) -> dict[str, Any]:
    recurrence = state.series.recurrence
    return {
        "series_id": state.series.series_id,
        "paused": state.paused,
        "type": state.series.task.type,
        "rule": (
            f"cron {recurrence.cron}"
            if recurrence.cron is not None
            else f"every {recurrence.every_seconds}s"
        ),
        "timezone": recurrence.timezone,
        "next_task_id": state.next_task_id,
        "revision": state.revision,
        "updated_at": state.updated_at.isoformat(),
    }


_COLUMNS = ("series_id", "paused", "type", "rule", "timezone", "next_task_id", "updated_at")


async def run_series(args: argparse.Namespace, target: ReadTarget) -> int:
    store, source = _open_store(target)
    try:
        command = args.series_command
        if command == "list":
            states = await store.list_task_series(limit=args.limit, offset=args.offset)
            if args.output_format == "json":
                payload = {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "source": source,
                    "series": [state.model_dump(mode="json") for state in states],
                }
                print(
                    json.dumps(redact_inspection_value(payload), ensure_ascii=False, sort_keys=True)
                )
            elif not states:
                print("No task series.")
            else:
                print_table(
                    _COLUMNS, [redact_inspection_value(series_row(state)) for state in states]
                )
            return 0
        if command == "delete":
            deleted = await store.delete_task_series(args.series_id)
            if args.output_format == "json":
                print(
                    json.dumps(
                        {
                            "schema_version": CLI_SCHEMA_VERSION,
                            "source": source,
                            "series_id": args.series_id,
                            "deleted": deleted,
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(
                    f"Deleted series {args.series_id}; its tasks stay."
                    if deleted
                    else f"No task series {args.series_id}."
                )
            return 0 if deleted else 1
        if command == "show":
            state = await store.load_task_series(args.series_id)
            if state is None:
                raise ValueError(f"No task series {args.series_id}.")
        else:
            try:
                state = await store.set_task_series_paused(args.series_id, command == "pause")
            except KeyError:
                raise ValueError(f"No task series {args.series_id}.") from None
        if args.output_format == "json":
            print(
                json.dumps(
                    redact_inspection_value(
                        {
                            "schema_version": CLI_SCHEMA_VERSION,
                            "source": source,
                            "series": state.model_dump(mode="json"),
                        }
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print_table(
                ("field", "value"),
                [
                    {"field": k, "value": v}
                    for k, v in redact_inspection_value(series_row(state)).items()
                ],
            )
            print(
                f"Its occurrences: cayu task list --metadata cayu_series_id={state.series.series_id}"
            )
        return 0
    finally:
        await store.close()
