"""Saved-file CLI routes use the same protected global-query contract."""

import json

import pytest
from tests.cli.test_knowledge_cli import agent as _agent

from cayu.cli import main

agent = _agent


def test_artifact_cli_global_query_cloud_transport(agent, capsys):
    agent["routes"]["/cayu/api/contract"] = (
        200,
        {"capabilities": {"surfaces": {"artifact_search": {"read": {"enabled": True}}}}},
    )
    body = {
        "artifacts": [
            {
                "id": "same",
                "artifact_store_id": "one",
                "filename": "report.csv",
                "session_title": "Quarterly review",
                "session_id": "producer",
                "size_bytes": 10,
            }
        ],
        "total_count": 11,
        "next_offset": 5,
    }
    agent["routes"]["/cayu/api/artifacts"] = (200, body)
    code = main(
        [
            "artifact",
            "list",
            "--application",
            "bug-hunter",
            "--query",
            "Quarterly λ",
            "--order-by",
            "filename_asc",
            "--artifact-store-id",
            "one",
            "--environment",
            "Review / λ",
            "--limit",
            "3",
            "--offset",
            "2",
            "--json",
        ]
    )
    assert code == 0, capsys.readouterr().err
    assert json.loads(capsys.readouterr().out) == body
    query = next(query for path, query in agent["requests"] if path.endswith("/artifacts"))
    assert query == {
        "q": ["Quarterly λ"],
        "order_by": ["filename_asc"],
        "artifact_store_id": ["one"],
        "environment_name": ["Review / λ"],
        "limit": ["3"],
        "offset": ["2"],
    }


@pytest.mark.parametrize("enabled", [None, False, "true"])
def test_artifact_cli_requires_positive_search_support(agent, capsys, enabled):
    agent["routes"]["/cayu/api/contract"] = (
        200,
        {"capabilities": {"surfaces": {"artifact_search": {"read": {"enabled": enabled}}}}},
    )
    assert (
        main(["artifact", "list", "--application", "bug-hunter", "--query", "report", "--json"])
        != 0
    )
    assert not any(path.endswith("/artifacts") for path, _ in agent["requests"])
    assert "does not advertise" in str(capsys.readouterr())


def test_artifact_cli_table_titles_unknown_count_and_invalid_bounds(agent, capsys):
    agent["routes"]["/cayu/api/artifacts"] = (
        200,
        {
            "artifacts": [
                {
                    "id": "same",
                    "artifact_store_id": "one",
                    "filename": "report.csv",
                    "session_title": "Quarterly review",
                    "session_id": "producer",
                    "size_bytes": 10,
                }
            ],
            "total_count": None,
            "next_offset": 1,
        },
    )
    assert main(["artifact", "list", "--application", "bug-hunter"]) == 0
    text = capsys.readouterr().out
    assert "Quarterly review" in text and "total unavailable" in text and "--offset 1" in text
    count = len(agent["requests"])
    assert main(["artifact", "list", "--application", "bug-hunter", "--limit", "501"]) != 0
    assert len(agent["requests"]) == count


def test_artifact_cli_selected_store_defers_capability_check_to_server(agent, capsys):
    agent["routes"]["/cayu/api/artifacts"] = (
        200,
        {"artifacts": [], "total_count": 0, "next_offset": None},
    )
    assert (
        main(
            [
                "artifact",
                "list",
                "--application",
                "bug-hunter",
                "--query",
                "report",
                "--artifact-store-id",
                "one",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["total_count"] == 0
    assert not any(path.endswith("/contract") for path, _ in agent["requests"])


def test_artifact_cli_escapes_pagination_metadata(agent, capsys):
    agent["routes"]["/cayu/api/artifacts"] = (
        200,
        {"artifacts": [], "total_count": "\x1b[31m42", "next_offset": "\x1b[2J5"},
    )
    assert main(["artifact", "list", "--application", "bug-hunter"]) == 0
    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "42" in output
