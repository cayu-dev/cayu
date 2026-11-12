"""Snowflake's SQL API as tools, through a connection slot.

A statement is submitted asynchronously with a request ID derived from the call's idempotency
key and ``retry=true``, so Snowflake can answer a retried call with the first submission's
result instead of running the statement again. The tool reads the statement's handle until it
finishes or the tool's time runs out, and cancels it then.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import re
import uuid
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from cayu.connections import (
    ConnectionCallError,
    ConnectionClient,
    ConnectionResponse,
    retry_after_seconds,
)
from cayu.connectors import _common
from cayu.connectors.snowflake.sql import (
    StatementShapeError,
    first_keyword,
    is_read_statement,
    statement_tokens,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy

DEFAULT_SLOT = "snowflake"
#: The slot ``snowflake_execute`` uses unless the toolkit is given another.
DEFAULT_WRITE_SLOT = "snowflake-write"
#: Rows one answer carries unless the toolkit is given another number.
DEFAULT_MAX_ROWS = 200
MAX_ROWS = 1_000
#: How long a statement may run before the tool cancels it, unless the toolkit is given another.
DEFAULT_TIMEOUT_SECONDS = 120.0
#: The longest statement a call sends.
MAX_STATEMENT_CHARACTERS = 1024 * 1024
#: Time kept back from the run's deadline to cancel a statement that is still running.
_CANCEL_SECONDS = 10.0
#: How long a cancelled tool call waits for its statement's cancel to be sent.
_CANCELLED_CALL_CANCEL_SECONDS = 5.0
_FIRST_POLL_SECONDS = 0.25
_LONGEST_POLL_SECONDS = 2.0
#: The largest result partition read once decompressed.
_MAX_PARTITION_BYTES = 64 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
#: The longest value one cell carries; a longer one is shortened.
_CELL_CHARACTERS = 4_000
_STATEMENTS = "/api/v2/statements"
_HANDLE = re.compile(r"[0-9A-Za-z][0-9A-Za-z_-]{0,127}")
_HEADERS = {"content-type": "application/json", "accept": "application/json"}
_REQUEST_ID_NAMESPACE = b"cayu.connectors.snowflake.request-id.v1\x00"

#: Broker refusals a read of a running statement tries again: the call didn't get through.
_TRANSIENT_CODES = frozenset({None, "upstream_failed"})

QUERY = "snowflake_query"
EXECUTE = "snowflake_execute"


def tools(
    client: ConnectionClient | None = None,
    slot: str = DEFAULT_SLOT,
    *,
    write_slot: str = DEFAULT_WRITE_SLOT,
    include_writes: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    warehouse: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    role: str | None = None,
) -> list[Tool]:
    """``snowflake_query`` on ``slot``, and with ``include_writes=True`` ``snowflake_execute`` on
    ``write_slot`` (pair it with ``approval_policy()``), calling through ``client`` (default:
    Cayu Cloud's broker).

    Bind ``slot`` to a connection whose Snowflake role can only read: that role, not the
    toolkit's check of the statement's text, is what keeps ``snowflake_query`` to reads.
    ``write_slot`` may name the same slot when one role should do both.

    ``max_rows`` (at most 1,000) bounds the rows one answer carries; ``timeout_seconds`` bounds
    how long a statement runs before it is cancelled, within the run's own deadline.
    ``warehouse``, ``database``, ``schema`` and ``role`` set the statement's context where the
    connection's user's defaults shouldn't apply; a call may name its own database and schema.
    """

    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError(f"max_rows must be an integer from 1 to {MAX_ROWS}.")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number.")
    context = {
        name: value
        for name, value in (
            ("warehouse", warehouse),
            ("database", database),
            ("schema", schema),
            ("role", role),
        )
        if value is not None
    }
    settings = _Settings(
        client=_common.resolved_client(client),
        slot=slot,
        max_rows=max_rows,
        timeout_seconds=float(timeout_seconds),
        context=context,
    )
    if include_writes is not True:
        return [_Query(settings)]
    return [_Query(settings), _Execute(replace(settings, slot=write_slot))]


def approval_policy(inner: ToolPolicy | None = None) -> ToolPolicy:
    """Ask for approval before ``snowflake_execute`` runs a statement on its write slot; every
    other decision is ``inner``'s (default: allow)."""

    return _common.approval_policy(inner, _REASONS)


def review_fields() -> dict[str, tuple[str, ...]]:
    """The arguments an approver sees, for ``ToolArgumentReview``."""

    return {EXECUTE: ("statement", "database", "schema")}


def _execute_reason(arguments: Mapping[str, Any]) -> str:
    statement = arguments.get("statement")
    try:
        keyword = first_keyword(statement_tokens(statement)) if isinstance(statement, str) else None
    except StatementShapeError:
        keyword = None
    if not keyword:
        return "Run a statement in Snowflake."
    article = "an" if keyword[0] in "AEIOU" else "a"
    return f"Run {article} {_common.reason_argument(keyword)} statement in Snowflake."


_REASONS: dict[str, Callable[[Mapping[str, Any]], str]] = {EXECUTE: _execute_reason}


@dataclass(frozen=True)
class _Settings:
    client: ConnectionClient | None
    slot: str
    max_rows: int
    timeout_seconds: float
    context: dict[str, str]


def _request_id(idempotency_key: str | None) -> str:
    """Snowflake's request ID for a call: the same for every retry of one call, and a random
    one for a call without an idempotency key."""

    if idempotency_key is None:
        return str(uuid.uuid4())
    digest = hashlib.sha256(_REQUEST_ID_NAMESPACE + idempotency_key.encode()).digest()
    return str(uuid.UUID(bytes=digest[:16], version=4))


class _Statement:
    """One call's statement, from submission to its bounded answer."""

    def __init__(self, settings: _Settings, client: ConnectionClient, budget: float) -> None:
        self._settings = settings
        self._client = client
        self._budget = budget
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._deadline = loop.time() + budget

    def _remaining(self) -> float:
        return self._deadline - self._loop.time()

    async def _send(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> ConnectionResponse:
        """One request, within the tool's remaining time (``TimeoutError`` past it)."""

        async with asyncio.timeout(max(self._remaining(), 0.0)):
            return await self._client.request(
                self._settings.slot, method, path, query=query, headers=_HEADERS, body=body
            )

    async def run(
        self, statement: str, request: str, context: Mapping[str, str], *, write: bool
    ) -> ToolResult:
        """The statement's bounded answer. A ``write`` whose submission may have reached
        Snowflake fails ``ambiguous``: a retry of the same call is deduplicated by its request
        ID, but a new call would run it again."""

        body = {
            "statement": statement,
            # Snowflake stops the statement itself shortly after the tool would cancel it.
            "timeout": math.ceil(self._budget + _CANCEL_SECONDS),
            "parameters": {"MULTI_STATEMENT_COUNT": "1"},
            **context,
        }
        try:
            response = await self._send(
                "POST",
                _STATEMENTS,
                query={"requestId": request, "async": "true", "retry": "true"},
                body=json.dumps(body, ensure_ascii=False).encode(),
            )
        except TimeoutError:
            return _timed_out(self._budget, None, cancelled=False)
        except ConnectionCallError as error:
            return _common.call_error_result(
                error,
                self._settings.slot,
                write=write,
                check="query what the statement changes, or the query history",
            )
        if response.status != 202:
            return await self._result(response, None)
        handle = _handle(_common.json_body(response) or {})
        if handle is None:
            return _no_handle()
        return await self._poll(handle)

    async def _poll(self, handle: str) -> ToolResult:
        """Read the running statement until it settles. A rate limit waits its delay and a
        temporary failure is read again; past the tool's time, on a final failure, or when the
        call itself is cancelled, the statement is cancelled."""

        try:
            return await self._settle(handle)
        except asyncio.CancelledError:
            # The call was cancelled: cancel the statement too, within a bound, so it can't run
            # on (and commit) after the tool stopped. The request is shielded from a second
            # cancellation and never delays the caller past the bound.
            request = asyncio.ensure_future(self._cancel_request(handle))
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(asyncio.shield(request), _CANCELLED_CALL_CANCEL_SECONDS)
            raise

    async def _settle(self, handle: str) -> ToolResult:
        interval = _FIRST_POLL_SECONDS
        wait = interval
        while True:
            if wait >= self._remaining():
                return await self._cancel(handle, None)
            await asyncio.sleep(wait)
            interval = min(interval * 2, _LONGEST_POLL_SECONDS)
            wait = interval
            try:
                response = await self._send("GET", f"{_STATEMENTS}/{handle}")
            except TimeoutError:
                return await self._cancel(handle, None)
            except ConnectionCallError as error:
                if error.code == "too_many_calls":
                    wait = error.retry_after if error.retry_after is not None else interval
                    continue
                if error.code in _TRANSIENT_CODES:
                    continue
                failure = _common.call_error_result(error, self._settings.slot)
                if error.code == "upstream_response_too_large":
                    # The statement finished; only its answer is too large to carry.
                    return _with(failure, statement_handle=handle)
                return await self._cancel(handle, failure)
            if response.status == 429:
                delay = retry_after_seconds(response.header("retry-after"))
                wait = delay if delay is not None else interval
            elif response.status in (200, 408, 422):
                return await self._result(response, handle)
            elif response.status != 202 and response.status < 500:
                return await self._cancel(handle, _status_error(response))

    async def _cancel(self, handle: str, failure: ToolResult | None) -> ToolResult:
        """Cancel the statement; answer ``failure`` (else a timeout) with its handle."""

        try:
            async with asyncio.timeout(_CANCEL_SECONDS):
                cancelled = await self._cancel_request(handle)
        except TimeoutError:
            cancelled = False
        if failure is None:
            return _timed_out(self._budget, handle, cancelled=cancelled)
        return _with(failure, statement_handle=handle, cancelled=cancelled)

    async def _cancel_request(self, handle: str) -> bool:
        """Ask Snowflake to cancel the statement; whether it confirmed."""

        try:
            response = await self._client.request(
                self._settings.slot, "POST", f"{_STATEMENTS}/{handle}/cancel", headers=_HEADERS
            )
        except ConnectionCallError:
            return False
        return response.status == 200

    async def _result(self, response: ConnectionResponse, handle: str | None) -> ToolResult:
        answer = _common.json_body(response)
        if response.status == 429:
            return _common.connector_rate_limited(response, "Snowflake")
        if response.status == 422 and answer is not None:
            return _common.error_result(
                "sql_error",
                str(answer.get("message") or "Snowflake refused the statement."),
                snowflake_code=answer.get("code"),
                sql_state=answer.get("sqlState"),
                statement_handle=answer.get("statementHandle") or handle,
            )
        if response.status == 408:
            return _timed_out(
                self._budget, (answer or {}).get("statementHandle") or handle, cancelled=True
            )
        if response.status != 200 or answer is None:
            return _with(_status_error(response), statement_handle=handle)
        answered = _handle(answer) or handle
        if answered is None:
            return _no_handle()
        return await self._rows(answer, answered)

    async def _rows(self, answer: Mapping[str, Any], handle: str) -> ToolResult:
        metadata = answer.get("resultSetMetaData")
        metadata = metadata if isinstance(metadata, dict) else {}
        columns = [
            {"name": column.get("name"), "type": column.get("type")}
            for column in metadata.get("rowType") or []
            if isinstance(column, dict)
        ]
        partitions = metadata.get("partitionInfo")
        partition_count = len(partitions) if isinstance(partitions, list) else 1
        cap = self._settings.max_rows
        rows = _partition_rows(answer)
        reasons: list[str] = []
        partition = 1
        while len(rows) < cap and partition < partition_count:
            page, reason = await self._partition(handle, partition)
            if page is None:
                reasons.append(reason)
                break
            rows.extend(page)
            partition += 1
        total = metadata.get("numRows")
        kept = [_cells(row) for row in rows[:cap]]
        if len(rows) > cap or (partition < partition_count and not reasons):
            reasons.append("max_rows")
        if any(cell != row for cell, row in zip(kept, rows, strict=False)):
            reasons.append("cells_shortened")
        if not reasons and type(total) is int and total > len(kept):
            reasons.append("rows_missing")
        return _fitted(
            {
                "statement_handle": handle,
                "columns": columns,
                "row_count": total if type(total) is int else None,
                **({"stats": answer["stats"]} if isinstance(answer.get("stats"), dict) else {}),
            },
            kept,
            reasons,
        )

    async def _partition(self, handle: str, partition: int) -> tuple[list[Any] | None, str]:
        """One result partition's rows, or None and why it couldn't be read."""

        try:
            response = await self._send(
                "GET", f"{_STATEMENTS}/{handle}", query={"partition": str(partition)}
            )
        except TimeoutError:
            return None, "deadline"
        except ConnectionCallError as error:
            if error.code == "upstream_response_too_large":
                return None, "partition_too_large"
            return None, "partition_unreadable"
        if response.status != 200:
            return None, "partition_unreadable"
        body = response.body
        if body.startswith(_GZIP_MAGIC):
            # The broker asks for an uncompressed answer; a partition gzipped regardless is
            # decompressed here, within a bound.
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                body = decompressor.decompress(body, _MAX_PARTITION_BYTES)
            except zlib.error:
                return None, "partition_unreadable"
            if decompressor.unconsumed_tail:
                return None, "partition_too_large"
        try:
            page = json.loads(body)
        except ValueError:
            return None, "partition_unreadable"
        if isinstance(page, list):
            return page, ""
        if isinstance(page, dict):
            return _partition_rows(page), ""
        return None, "partition_unreadable"


def _fitted(answer: dict[str, Any], rows: list[Any], reasons: list[str]) -> ToolResult:
    """The answer with as many of ``rows`` as fit in the tool's budget; an error when even its
    columns don't."""

    def built(count: int) -> ToolResult:
        dropped = ["answer_size"] if count < len(rows) else []
        return _common.answer_result(
            {
                **answer,
                "rows": rows[:count],
                "returned_rows": count,
                "truncated": bool(reasons or dropped),
                "truncation_reasons": [*reasons, *dropped],
            }
        )

    result = built(len(rows))
    if _common.answer_fits(result):
        return result
    if not _common.answer_fits(built(0)):
        return _common.error_result(
            "result_too_wide",
            f"The result's {len(answer['columns'])} columns alone are too large to answer.",
            "Select fewer columns, or give them shorter names.",
            statement_handle=answer["statement_handle"],
            row_count=answer["row_count"],
        )
    low, high = 0, len(rows) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _common.answer_fits(built(middle)):
            low = middle
        else:
            high = middle - 1
    return built(low)


def _partition_rows(answer: Mapping[str, Any]) -> list[Any]:
    data = answer.get("data")
    return list(data) if isinstance(data, list) else []


def _cells(row: Any) -> Any:
    if not isinstance(row, list):
        return row
    return [
        _common.shortened(cell, _CELL_CHARACTERS)[0] if isinstance(cell, str) else cell
        for cell in row
    ]


def _handle(answer: Mapping[str, Any]) -> str | None:
    handle = answer.get("statementHandle")
    return handle if isinstance(handle, str) and _HANDLE.fullmatch(handle) else None


def _no_handle() -> ToolResult:
    return _common.error_result(
        "upstream_invalid_response",
        "Snowflake's answer has no usable statement handle.",
        _common.TEMPORARY,
    )


def _status_error(response: ConnectionResponse) -> ToolResult:
    message = (_common.json_body(response) or {}).get("message")
    return _common.error_result(
        "upstream_status",
        f"Snowflake answered HTTP {response.status}"
        + (f": {message}" if isinstance(message, str) and message else "."),
        _common.TEMPORARY if response.status >= 500 else None,
        status=response.status,
    )


def _with(result: ToolResult, **details: Any) -> ToolResult:
    """``result`` with more structured details; a None detail is left out."""

    structured = dict(result.structured or {})
    structured.update({name: value for name, value in details.items() if value is not None})
    return ToolResult(content=result.content, structured=structured, is_error=result.is_error)


def _timed_out(budget: float, handle: Any, *, cancelled: bool) -> ToolResult:
    outcome = "it was cancelled" if cancelled else "cancelling it wasn't confirmed"
    return _common.error_result(
        "statement_timed_out",
        f"The statement didn't finish within {math.ceil(budget)} seconds; {outcome}.",
        "Narrow the statement, or ask for a longer timeout.",
        statement_handle=handle if isinstance(handle, str) else None,
        cancelled=cancelled,
    )


_STATEMENT_SCHEMA = {
    "type": "string",
    "maxLength": MAX_STATEMENT_CHARACTERS,
    "description": "One SQL statement.",
}
_CONTEXT_SCHEMA = {
    "database": {"type": "string", "description": "The database to run in, if not the default."},
    "schema": {"type": "string", "description": "The schema to run in, if not the default."},
}


class _SnowflakeTool(Tool):
    reads_only = False

    def __init__(self, settings: _Settings) -> None:
        super().__init__()
        self._settings = settings

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        statement = args.get("statement")
        if not isinstance(statement, str) or len(statement) > MAX_STATEMENT_CHARACTERS:
            return _common.invalid_arguments(
                f"statement must be a string of at most {MAX_STATEMENT_CHARACTERS} characters."
            )
        try:
            reads = is_read_statement(statement)
        except StatementShapeError as error:
            return _common.invalid_arguments(str(error))
        if self.reads_only and not reads:
            return _common.error_result(
                "not_a_read",
                f"{QUERY} runs only read statements: SELECT, SHOW, DESCRIBE, or WITH ... "
                "SELECT, without SYSTEM$ functions.",
                f"Use {EXECUTE} for any other statement.",
            )
        context = dict(self._settings.context)
        for name in ("database", "schema"):
            value = args.get(name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                return _common.invalid_arguments(f"{name} must be a nonblank string.")
            context[name] = value
        client = self._settings.client
        if client is None:
            return _common.no_client_result()
        budget = self._settings.timeout_seconds
        remaining = ctx.execution_deadline.remaining_seconds()
        if remaining is not None:
            budget = min(budget, remaining - _CANCEL_SECONDS)
        if budget <= 0:
            return _common.error_result(
                "deadline_too_close",
                "The run's deadline leaves no time to run a statement.",
            )
        return await _Statement(self._settings, client, budget).run(
            statement, _request_id(ctx.idempotency_key), context, write=not reads
        )


def _spec(name: str, description: str, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"statement": _STATEMENT_SCHEMA, **_CONTEXT_SCHEMA},
            "required": ["statement"],
            "additionalProperties": False,
        },
        max_terminal_payload_bytes=_common.MAX_TERMINAL_PAYLOAD_BYTES,
        **kwargs,
    )


class _Query(_SnowflakeTool):
    reads_only = True
    spec = _spec(
        QUERY,
        "Run one read statement in Snowflake (SELECT, SHOW, DESCRIBE, or WITH ... SELECT) and "
        "answer its columns and rows. Add a LIMIT: only the first rows are answered, and a "
        f"very large result can fail. Any other statement goes to {EXECUTE}.",
        effect=ToolEffect.NONE,
    )


class _Execute(_SnowflakeTool):
    spec = _spec(
        EXECUTE,
        "Run one SQL statement of any kind in Snowflake and answer its result. It may need a "
        f"person's approval first. Use {QUERY} for reads.",
        effect=ToolEffect.EXTERNAL,
        parallel_safe=False,
    )
