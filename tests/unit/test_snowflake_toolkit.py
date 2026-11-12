"""The Snowflake toolkit: one statement per call, run to the end within the tool's time."""

from __future__ import annotations

import asyncio
import gzip
import json
import time
from typing import Any

import httpx
import pytest
from tests.connector_broker_support import (
    Answer,
    FakeBroker,
    Refusal,
    Sent,
    assert_within_budgets,
    context,
    policy_request,
)

from cayu.approvals import ToolArgumentReview
from cayu.connections import LocalConnections, LocalSlot
from cayu.connectors import slack, snowflake
from cayu.connectors.snowflake.toolkit import _request_id
from cayu.deadlines import ExecutionDeadline
from cayu.tools import (
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolResult,
)

HANDLE = "01b2c3d4-0000-1111-0000-000000000001"
COLUMNS = [{"name": "ID", "type": "fixed", "nullable": False}, {"name": "NAME", "type": "text"}]


def _tools(broker: FakeBroker, **kwargs: Any) -> dict[str, Tool]:
    kwargs.setdefault("slot", "warehouse")
    kwargs.setdefault("include_writes", True)
    return {tool.name: tool for tool in snowflake.tools(broker.client, **kwargs)}


def _run(tool: Tool, key: str | None = "call-key-1", **args: Any) -> ToolResult:
    return asyncio.run(tool.run(context(idempotency_key=key), args))


def _result_set(rows: list[list[Any]], *, partitions: list[int] | None = None, **extra: Any):
    partitions = partitions or [len(rows)]
    return {
        "statementHandle": HANDLE,
        "resultSetMetaData": {
            "numRows": sum(partitions),
            "rowType": COLUMNS,
            "partitionInfo": [{"rowCount": count} for count in partitions],
        },
        "data": rows,
        **extra,
    }


def _running() -> Answer:
    return Answer(
        status=202,
        body={
            "code": "333334",
            "message": "Asynchronous execution in progress.",
            "statementHandle": HANDLE,
            "statementStatusUrl": f"/api/v2/statements/{HANDLE}",
        },
    )


def test_the_toolkit_has_query_and_execute_only_when_asked() -> None:
    assert [tool.name for tool in snowflake.tools(LocalConnections({}))] == ["snowflake_query"]
    tools = snowflake.tools(LocalConnections({}), include_writes=True)
    assert {tool.name: tool.spec.effect for tool in tools} == {
        "snowflake_query": ToolEffect.NONE,
        "snowflake_execute": ToolEffect.EXTERNAL,
    }
    assert "LIMIT" in tools[0].description
    assert "may need a person's approval" in tools[1].description
    for bad in ({"max_rows": 0}, {"max_rows": 1001}, {"timeout_seconds": 0}):
        with pytest.raises(ValueError):
            snowflake.tools(LocalConnections({}), **bad)


def test_a_query_is_submitted_asynchronously_once_and_answers_columns_and_rows() -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([["1", "ada"], ["2", None]])))
    result = _run(
        _tools(broker, warehouse="WH", role="ANALYST")["snowflake_query"],
        statement="select id, name from people",
        database="SALES",
    )

    [sent] = broker.sent
    assert (sent.slot, sent.method, sent.path) == ("warehouse", "POST", "/api/v2/statements")
    assert sent.query == {"requestId": _request_id("call-key-1"), "async": "true", "retry": "true"}
    assert sent.headers == {"content-type": "application/json", "accept": "application/json"}
    body = sent.json()
    assert body["statement"] == "select id, name from people"
    assert body["parameters"] == {"MULTI_STATEMENT_COUNT": "1"}
    assert (body["warehouse"], body["role"], body["database"]) == ("WH", "ANALYST", "SALES")
    assert "schema" not in body
    assert body["timeout"] == 130
    assert not result.is_error
    assert result.structured == {
        "statement_handle": HANDLE,
        "columns": [{"name": "ID", "type": "fixed"}, {"name": "NAME", "type": "text"}],
        "row_count": 2,
        "rows": [["1", "ada"], ["2", None]],
        "returned_rows": 2,
        "truncated": False,
        "truncation_reasons": [],
    }
    assert json.loads(result.content) == result.structured


def test_execute_runs_on_its_own_write_slot() -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([])))
    tools = _tools(broker, slot="warehouse-read", write_slot="warehouse-write")
    _run(tools["snowflake_query"], statement="select 1")
    _run(tools["snowflake_execute"], statement="delete from t")
    assert [sent.slot for sent in broker.sent] == ["warehouse-read", "warehouse-write"]

    shared = FakeBroker(lambda _: Answer(body=_result_set([])))
    tools = _tools(shared, slot="warehouse", write_slot="warehouse")
    _run(tools["snowflake_execute"], statement="delete from t")
    assert [sent.slot for sent in shared.sent] == ["warehouse"]
    assert snowflake.DEFAULT_WRITE_SLOT == "snowflake-write"


def test_the_request_id_is_derived_from_the_calls_idempotency_key() -> None:
    first, again, other = (_request_id(key) for key in ("key-a", "key-a", "key-b"))
    assert first == again != other
    assert len(first) == 36 and first[14] == "4"
    assert _request_id(None) != _request_id(None)


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 1",
        "select 1;",
        "  -- a note\n  /* and\n another */ WITH t AS (SELECT 1) SELECT * FROM t",
        "// note\nshow tables",
        "describe table people",
        "desc table people",
        "(select 1) union (select 2)",
        "select ';' as semicolon, 'it''s' as quote -- ; drop table x",
        "select $$a; b$$",
        "select 1; -- trailing note",
        "with recursive r (n) as (select 1 union all select n + 1 from r) select n from r",
        'with "a b" as (select 1), c as (select 2) select * from "a b", c',
        "select a$$b from t; -- a$$b is an identifier",
        "select f.value from t, table(flatten(input => t.items)) f",
        "select * from table(result_scan(last_query_id()))",
        "select * from table(information_schema.query_history())",
        'select * from table(sales."INFORMATION_SCHEMA".query_history_by_user())',
        "select seq4() from table(generator(rowcount => 10))",
        'select "system" from t',
        "select filter([1, 2, 3], x -> x > 1)",
        "select '->>' as arrow",
    ],
)
def test_read_statements_are_recognized_by_their_first_keyword(statement: str) -> None:
    assert snowflake.is_read_statement(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "insert into t values (1)",
        "-- select\ndelete from t",
        "/* select */ update t set x = 1",
        "create table t (x int)",
        "call proc()",
        "'select'",
        "with p as procedure () returns varchar language sql as $$ begin delete from t; "
        "return 'x'; end $$ call p()",
        "with f as function (x int) returns int as $$ x $$ select f(1)",
        "with s as (select 1) insert into t select * from s",
        "with s as (select 1) merge into t using s on t.id = s.id when matched then delete",
        "with s as (select 1) delete from t using s where t.id = s.id",
        "with s as (select 1) update t set x = 1 from s",
        "with s as (select 1) (select * from s)",
        "(with s as (select 1) select * from s)",
        "with s as (select 1",
        "select system$cancel_all_queries(current_session())",
        "SELECT SYSTEM$CANCEL_ALL_QUERIES()",
        "with s as (select system$abort_session(1)) select * from s",
        "select * from table(my_proc())",
        "select * from table(db.schema.my_proc())",
        "select * from table(other.flatten(input => 1))",
        "select * from table(information_schema)",
        'select "SYSTEM$CANCEL_ALL_QUERIES"()',
        'select "system$cancel_all_queries"()',
        "select identifier('SYSTEM$CANCEL_ALL_QUERIES')()",
        "select * from identifier('t')",
        "select 1 ->> delete from t",
        "show tables ->> drop table t",
        "describe table t ->> insert into t select 1",
        "select 1->>call p()",
        "with s as (select 1) select * from s ->> update t set x = 1",
    ],
)
def test_other_statements_are_not_reads(statement: str) -> None:
    assert not snowflake.is_read_statement(statement)


@pytest.mark.parametrize("statement", ["", "  -- only a note", ";"])
def test_an_empty_statement_is_refused(statement: str) -> None:
    with pytest.raises(snowflake.StatementShapeError, match="empty"):
        snowflake.is_read_statement(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "select 1; select 2",
        "select 1;drop table t",
        "select 'a;'; delete from t",
        "select $$a$$; delete from t",
        "select a$$b; delete from t",
    ],
)
def test_several_statements_are_refused_before_sending(statement: str) -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([])))
    with pytest.raises(snowflake.StatementShapeError, match="one statement"):
        snowflake.is_read_statement(statement)
    for name in ("snowflake_query", "snowflake_execute"):
        result = _run(_tools(broker)[name], statement=statement)
        assert result.structured is not None
        assert result.structured["error"] == "invalid_arguments"
    assert broker.sent == []


def test_query_refuses_a_write_and_points_at_execute() -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([])))
    result = _run(_tools(broker)["snowflake_query"], statement="delete from people")

    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == "not_a_read"
    assert "snowflake_execute" in result.structured["advice"]
    assert broker.sent == []


def test_execute_runs_any_statement_and_answers_its_stats() -> None:
    broker = FakeBroker(
        lambda _: Answer(
            body=_result_set([["3"]], stats={"numRowsDeleted": 3}),
        )
    )
    result = _run(_tools(broker)["snowflake_execute"], statement="delete from people")
    assert result.structured is not None
    assert result.structured["stats"] == {"numRowsDeleted": 3}
    assert result.structured["rows"] == [["3"]]


def test_a_running_statement_is_polled_until_it_finishes() -> None:
    answers = iter([_running(), _running(), Answer(body=_result_set([["1", "a"]]))])
    broker = FakeBroker(lambda _: next(answers))
    result = _run(_tools(broker)["snowflake_query"], statement="select * from big")

    assert [(sent.method, sent.path) for sent in broker.sent] == [
        ("POST", "/api/v2/statements"),
        ("GET", f"/api/v2/statements/{HANDLE}"),
        ("GET", f"/api/v2/statements/{HANDLE}"),
    ]
    assert broker.sent[1].query == {}
    assert result.structured is not None
    assert result.structured["rows"] == [["1", "a"]]


def test_a_sql_error_is_passed_through_as_a_tool_error() -> None:
    answers = iter(
        [
            _running(),
            Answer(
                status=422,
                body={
                    "code": "002003",
                    "sqlState": "42S02",
                    "message": "SQL compilation error: Object 'NOPE' does not exist.",
                    "statementHandle": HANDLE,
                },
            ),
        ]
    )
    broker = FakeBroker(lambda _: next(answers))
    result = _run(_tools(broker)["snowflake_query"], statement="select * from nope")

    assert result.is_error
    assert result.structured == {
        "error": "sql_error",
        "message": "SQL compilation error: Object 'NOPE' does not exist.",
        "snowflake_code": "002003",
        "sql_state": "42S02",
        "statement_handle": HANDLE,
    }


def test_a_statement_still_running_at_the_tools_time_is_cancelled() -> None:
    def connector(sent: Sent) -> Answer:
        if sent.path.endswith("/cancel"):
            return Answer(body={"code": "000604", "message": "Statement cancelled."})
        return _running()

    broker = FakeBroker(connector)
    started = time.monotonic()
    result = _run(_tools(broker, timeout_seconds=0.6)["snowflake_execute"], statement="call slow()")

    assert time.monotonic() - started < 5
    assert broker.sent[-1].method == "POST"
    assert broker.sent[-1].path == f"/api/v2/statements/{HANDLE}/cancel"
    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == "statement_timed_out"
    assert result.structured["cancelled"] is True
    assert result.structured["statement_handle"] == HANDLE


def _polled(*answers: Answer | Refusal, finish: Answer | None = None) -> FakeBroker:
    """A statement that answers 202, then ``answers`` to its reads, then ``finish``."""

    pending = iter(answers)

    def connector(sent: Sent) -> Answer | Refusal:
        if sent.method == "POST" and sent.path.endswith("/cancel"):
            return Answer(body={"message": "Statement cancelled."})
        if sent.method == "POST":
            return _running()
        return next(pending, finish or _running())

    return FakeBroker(connector)


def _reads(broker: FakeBroker) -> list[str]:
    return [f"{sent.method} {sent.path.removeprefix('/api/v2/statements')}" for sent in broker.sent]


def test_a_rate_limited_read_waits_its_delay_and_reads_again() -> None:
    broker = _polled(
        Refusal(429, "too_many_calls", headers={"Retry-After": "1"}),
        Answer(status=429, body=b"", headers={"Retry-After": "0"}),
        finish=Answer(body=_result_set([["1", "a"]])),
    )
    started = time.monotonic()
    result = _run(_tools(broker)["snowflake_query"], statement="select 1")

    assert time.monotonic() - started >= 1
    assert _reads(broker) == ["POST ", f"GET /{HANDLE}", f"GET /{HANDLE}", f"GET /{HANDLE}"]
    assert result.structured is not None
    assert result.structured["rows"] == [["1", "a"]]


def test_a_temporary_failure_while_reading_is_read_again() -> None:
    broker = _polled(
        Refusal(502, "upstream_failed"),
        Answer(status=503, body=b"busy"),
        finish=Answer(body=_result_set([["1", "a"]])),
    )
    result = _run(_tools(broker)["snowflake_query"], statement="select 1")
    assert not result.is_error
    assert len(broker.sent) == 4


def test_a_rate_limit_past_the_tools_time_cancels_with_the_handle() -> None:
    broker = _polled(Refusal(429, "too_many_calls", headers={"Retry-After": "60"}))
    result = _run(_tools(broker, timeout_seconds=2)["snowflake_query"], statement="select 1")

    assert _reads(broker)[-1] == f"POST /{HANDLE}/cancel"
    assert result.structured is not None
    assert result.structured["error"] == "statement_timed_out"
    assert result.structured["statement_handle"] == HANDLE


@pytest.mark.parametrize(
    "failure",
    [Refusal(409, "connection_not_ready"), Answer(status=403, body={"message": "Forbidden."})],
)
def test_a_final_failure_while_reading_cancels_and_names_the_handle(
    failure: Answer | Refusal,
) -> None:
    broker = _polled(failure)
    result = _run(_tools(broker)["snowflake_execute"], statement="call slow()")

    assert _reads(broker) == ["POST ", f"GET /{HANDLE}", f"POST /{HANDLE}/cancel"]
    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] in {"connection_not_ready", "upstream_status"}
    assert result.structured["statement_handle"] == HANDLE
    assert result.structured["cancelled"] is True


def test_an_answer_too_large_while_reading_is_answered_at_once_with_the_handle() -> None:
    broker = _polled(Refusal(502, "upstream_response_too_large"))
    result = _run(_tools(broker)["snowflake_query"], statement="select * from big")

    assert _reads(broker) == ["POST ", f"GET /{HANDLE}"]
    assert result.structured is not None
    assert result.structured["error"] == "upstream_response_too_large"
    assert "ask for less" in result.structured["advice"]
    assert result.structured["statement_handle"] == HANDLE


def test_a_cancelled_call_cancels_its_running_statement() -> None:
    broker = _polled()
    tool = _tools(broker)["snowflake_execute"]

    async def cancelled() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                tool.run(context(idempotency_key="k"), {"statement": "call slow()"}), 0.6
            )

    asyncio.run(cancelled())
    assert _reads(broker)[0] == "POST "
    assert _reads(broker)[-1] == f"POST /{HANDLE}/cancel"


def test_a_call_without_an_idempotency_key_runs_with_a_random_request_id() -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([])))
    tool = _tools(broker)["snowflake_execute"]
    _run(tool, key=None, statement="delete from t")
    _run(tool, key=None, statement="delete from t")
    first, second = (sent.query["requestId"] for sent in broker.sent)
    assert first != second


def test_the_runs_deadline_bounds_the_tools_time() -> None:
    broker = FakeBroker(lambda _: Answer(body=_result_set([])))
    tool = _tools(broker)["snowflake_query"]
    soon = context(idempotency_key="k", execution_deadline=ExecutionDeadline.after(5))
    result = asyncio.run(tool.run(soon, {"statement": "select 1"}))

    assert result.structured is not None
    assert result.structured["error"] == "deadline_too_close"
    assert broker.sent == []

    later = context(idempotency_key="k", execution_deadline=ExecutionDeadline.after(40))
    asyncio.run(tool.run(later, {"statement": "select 1"}))
    assert broker.sent[0].json()["timeout"] <= 40


def test_rows_are_read_across_partitions_up_to_the_cap() -> None:
    first = [[str(index)] for index in range(3)]

    def connector(sent: Sent) -> Answer:
        if sent.method == "POST":
            return Answer(body=_result_set(first, partitions=[3, 3, 3]))
        partition = int(sent.query["partition"])
        return Answer(body={"data": [[f"{partition}-{index}"] for index in range(3)]})

    broker = FakeBroker(connector)
    result = _run(_tools(broker, max_rows=5)["snowflake_query"], statement="select x from t")

    assert [sent.query for sent in broker.sent[1:]] == [{"partition": "1"}]
    assert result.structured is not None
    assert result.structured["rows"] == [["0"], ["1"], ["2"], ["1-0"], ["1-1"]]
    assert (result.structured["row_count"], result.structured["returned_rows"]) == (9, 5)
    assert result.structured["truncated"] is True

    whole = FakeBroker(connector)
    complete = _run(_tools(whole, max_rows=9)["snowflake_query"], statement="select x from t")
    assert [sent.query for sent in whole.sent[1:]] == [{"partition": "1"}, {"partition": "2"}]
    assert complete.structured is not None
    assert complete.structured["returned_rows"] == 9
    assert complete.structured["truncated"] is False


def test_a_gzipped_partition_is_read_and_an_unreadable_one_is_reported() -> None:
    def connector(sent: Sent) -> Answer:
        if sent.method == "POST":
            return Answer(body=_result_set([["0"]], partitions=[1, 1, 1]))
        if sent.query["partition"] == "1":
            return Answer(body=gzip.compress(json.dumps([["1"]]).encode()))
        return Answer(status=500, body=b"")

    broker = FakeBroker(connector)
    result = _run(_tools(broker)["snowflake_query"], statement="select x from t")

    assert result.structured is not None
    assert result.structured["rows"] == [["0"], ["1"]]
    assert result.structured["truncated"] is True
    assert result.structured["truncation_reasons"] == ["partition_unreadable"]


def test_columns_too_wide_to_answer_are_an_error_naming_the_handle() -> None:
    columns = [{"name": f"{index:03d}_" + "a" * 196, "type": "text"} for index in range(400)]
    answer = _result_set([])
    answer["resultSetMetaData"]["rowType"] = columns
    broker = FakeBroker(lambda _: Answer(body=answer))
    result = _run(_tools(broker)["snowflake_query"], statement="select * from wide limit 0")

    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == "result_too_wide"
    assert result.structured["statement_handle"] == HANDLE
    assert "fewer columns" in result.structured["advice"]
    assert_within_budgets(result)


def test_an_answer_too_large_keeps_the_rows_that_fit() -> None:
    rows = [[str(index), "v" * 3_000] for index in range(200)]
    broker = FakeBroker(lambda _: Answer(body=_result_set(rows)))
    result = _run(_tools(broker)["snowflake_query"], statement="select * from wide")

    assert not result.is_error
    assert result.structured is not None
    assert 0 < result.structured["returned_rows"] < 200
    assert result.structured["rows"] == rows[: result.structured["returned_rows"]]
    assert result.structured["truncated"] is True
    assert_within_budgets(result)


@pytest.mark.parametrize(
    ("refusal", "code", "retry_after"),
    [
        (Refusal(429, "too_many_calls", headers={"Retry-After": "2"}), "too_many_calls", 2),
        (Refusal(409, "connection_not_ready"), "connection_not_ready", None),
        (Refusal(422, "path_not_allowed"), "path_not_allowed", None),
        (Refusal(502, "upstream_response_too_large"), "upstream_response_too_large", None),
    ],
)
def test_broker_refusals_map_to_tool_errors(
    refusal: Refusal, code: str, retry_after: int | None
) -> None:
    broker = FakeBroker(lambda _: refusal)
    result = _run(_tools(broker)["snowflake_query"], statement="select 1")

    assert result.is_error
    assert result.structured is not None
    assert result.structured["error"] == code
    assert result.structured.get("retry_after_seconds") == retry_after


def test_snowflakes_own_rate_limit_carries_its_delay() -> None:
    broker = FakeBroker(lambda _: Answer(status=429, body=b"", headers={"Retry-After": "7"}))
    result = _run(_tools(broker)["snowflake_query"], statement="select 1")
    assert result.structured is not None
    assert result.structured["error"] == "too_many_calls"
    assert result.structured["retry_after_seconds"] == 7


def test_local_connections_reach_the_accounts_sql_api() -> None:
    sent: list[httpx.Request] = []

    def sql_api(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=_result_set([["1"]]))

    local = LocalConnections(
        {"snowflake": LocalSlot("https://acme.snowflakecomputing.com", {"authorization": "x"})},
        transport=httpx.MockTransport(sql_api),
    )
    [query] = snowflake.tools(local)
    result = _run(query, statement="select 1")

    [request] = sent
    assert request.url.path == "/api/v2/statements"
    assert request.url.params["async"] == "true"
    assert request.headers["authorization"] == "x"
    assert result.structured is not None
    assert result.structured["rows"] == [["1"]]


def test_approval_policy_asks_before_execute_and_leaves_the_rest_to_inner() -> None:
    def decide(policy, tool_name: str, statement: str = "select 1"):
        return asyncio.run(policy.authorize(policy_request(tool_name, statement=statement)))

    policy = snowflake.approval_policy()
    execute = decide(policy, "snowflake_execute", "-- note\ninsert into t values (1)")
    assert execute.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert execute.reason == "Run an INSERT statement in Snowflake."
    assert decide(policy, "snowflake_execute", "delete from t").reason == (
        "Run a DELETE statement in Snowflake."
    )
    assert decide(policy, "snowflake_query").decision == ToolPolicyDecision.ALLOW

    denying = snowflake.approval_policy(StaticToolPolicy(deny=["snowflake_execute"]))
    assert decide(denying, "snowflake_execute").decision == ToolPolicyDecision.DENY

    both = snowflake.approval_policy(slack.approval_policy())
    assert decide(both, "snowflake_execute").decision == ToolPolicyDecision.REQUIRE_APPROVAL


def test_review_fields_show_the_sql() -> None:
    assert snowflake.review_fields() == {"snowflake_execute": ("statement", "database", "schema")}
    ToolArgumentReview(snowflake.review_fields(), binding_key=b"k" * 32)


def test_a_write_statement_that_may_have_run_is_ambiguous_and_a_query_isnt() -> None:
    broker = FakeBroker(lambda _: Refusal(502, "upstream_failed"))
    tools = _tools(broker)
    written = _run(tools["snowflake_execute"], statement="delete from t")

    assert written.structured is not None
    assert written.structured["ambiguous"] is True
    assert "may have taken effect" in written.structured["advice"]

    read = _run(tools["snowflake_query"], statement="select 1")
    assert read.structured is not None and "ambiguous" not in read.structured
