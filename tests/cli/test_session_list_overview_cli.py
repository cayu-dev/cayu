"""`cayu session list --overview` shows each session's overview, locally and on an Agent."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
from tests.cli.test_session_cli_cloud import _SESSION, _both, agent

from cayu.cli import main
from cayu.cli._agent_session_reader import AgentSessionReader
from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.sessions.queries import SessionQuery

__all__ = ["agent"]

# What a bare local store can't know, or prices with the CLI's book instead of the
# Agent's: the application's artifact stores and its price book.
_STORE_DEPENDENT = {"cost", "uses"}


def _comparable(row: dict) -> dict:
    overview = row["overview"]
    return {
        **{key: value for key, value in overview.items() if key not in _STORE_DEPENDENT},
        "uses": {key: value for key, value in overview["uses"].items() if key != "artifacts"},
    }


def test_overview_rows_match_between_a_local_store_and_the_agent(agent, capsys) -> None:
    database, _, seen = agent

    local, remote = _both(capsys, database, "list", "--overview")

    assert [row["id"] for row in remote["sessions"]] == [row["id"] for row in local["sessions"]]
    assert [_comparable(row) for row in remote["sessions"]] == [
        _comparable(row) for row in local["sessions"]
    ]
    rows = {row["id"]: row["overview"] for row in local["sessions"]}
    assert rows[_SESSION]["first_message"] == "Help"
    assert rows[_SESSION]["first_message_unavailable"] is False
    assert rows["sess_paused"]["state"] == "waiting_approval"
    # A bare store doesn't know the application's artifact stores; the Agent does.
    assert all(row["overview"]["uses"]["artifacts"] is None for row in local["sessions"])
    assert all(row["overview"]["uses"]["artifacts"] == "0" for row in remote["sessions"])
    # Locally the bundled price book prices the overview.
    assert rows[_SESSION]["cost"]["currency"] == "USD"
    # The Agent served the overview and the previews in the list request itself.
    listed = [request for request in seen if request.url.path.endswith("/api/sessions")]
    assert listed and all(
        request.url.params.get_list("include") == ["overview", "first_message"]
        for request in listed
    )


def test_overview_table_shows_one_line_per_session(agent, capsys) -> None:
    database, _, _ = agent

    assert main(["session", "list", "--overview", "--table", "--sqlite", str(database)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:5] == ["id", "state", "agent", "title", "first_message"]
    (help_line,) = [line for line in lines if line.startswith(_SESSION + " ")]
    assert "Help" in help_line
    assert len(lines) == 1 + len({line.split()[0] for line in lines[1:]})


def test_overview_jsonl_prints_one_row_per_session(agent, capsys) -> None:
    database, _, _ = agent

    assert main(["session", "list", "--overview", "--jsonl", "--sqlite", str(database)]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line]
    assert rows and all('"overview": {' in row for row in rows)


def test_price_book_applies_only_to_local_overviews(agent, capsys, tmp_path) -> None:
    database, _, _ = agent
    book = tmp_path / "prices.json"

    assert main(["session", "list", "--price-book", str(book), "--sqlite", str(database)]) == 1
    assert "--price-book prices --overview costs" in capsys.readouterr().out
    assert (
        main(
            [
                "session",
                "list",
                "--overview",
                "--price-book",
                str(book),
                "--application",
                "bug-hunter-agent",
            ]
        )
        == 1
    )
    assert "--price-book prices --overview costs" in capsys.readouterr().out


class _OlderAgent:
    """An Agent whose Cayu serves overviews but rejects first-message previews."""

    def __init__(self) -> None:
        self.includes: list[list[str]] = []

    def get(self, path: str, *, query: dict, resource: str) -> dict:
        include = query.get("include", [])
        self.includes.append(include)
        if "first_message" in include:
            raise CayuApiError("request_rejected", "rejected", status_code=422)
        return {
            "sessions": [{"id": "s1", "status": "completed"}],
            "overviews": [{"session_id": "s1", "state": "completed", "first_message": None}],
            "next_cursor": None,
            "total_count": 1,
        }


def test_an_agent_without_previews_lists_overviews_with_previews_unavailable() -> None:
    api = _OlderAgent()
    reader = AgentSessionReader(cast("CayuApi", api))

    page = asyncio.run(reader.list_sessions(SessionQuery(limit=10), overview=True))

    assert api.includes == [["overview", "first_message"], ["overview"]]
    (row,) = page.rows
    assert row["overview"]["first_message"] is None
    assert row["overview"]["first_message_unavailable"] is True


def test_an_agent_answering_without_overviews_is_refused() -> None:
    class _NoOverviews(_OlderAgent):
        def get(self, path: str, *, query: dict, resource: str) -> dict:
            return {"sessions": [{"id": "s1"}], "next_cursor": None}

    reader = AgentSessionReader(cast("CayuApi", _NoOverviews()))
    with pytest.raises(CayuApiError, match="no overview for every session"):
        asyncio.run(reader.list_sessions(SessionQuery(limit=10), overview=True))
