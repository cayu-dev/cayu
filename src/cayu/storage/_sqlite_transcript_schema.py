"""Validate SQLite transcript-search schema against its supplied tokenizer identity."""

from __future__ import annotations

import json
import sqlite3


def _validate_revision_46_transcript_search_schema(
    connection: sqlite3.Connection,
    *,
    expected_tokenizer_version: str,
) -> None:
    transcript_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_transcript_messages)")
    }
    if transcript_columns.get("transcript_search_document") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite transcript search document column is missing or nullable. "
            "Recreate or restore a known-good revision-46 Cayu database."
        )
    configuration_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_transcript_search_configuration)")
    )
    if configuration_columns != (
        ("singleton", "INTEGER", 0, 1),
        ("tokenizer_version", "TEXT", 1, 0),
    ):
        raise RuntimeError(
            "SQLite transcript search tokenizer configuration is missing or malformed. "
            "Recreate a revision-46 Cayu database with this runtime."
        )
    configuration_table = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'cayu_transcript_search_configuration'"
    ).fetchone()
    configuration_sql = (
        ""
        if configuration_table is None
        else "".join(str(configuration_table[0] or "").lower().split())
    )
    if "check(singleton=1)" not in configuration_sql:
        raise RuntimeError(
            "SQLite transcript search tokenizer configuration lacks its singleton "
            "constraint. Recreate a revision-46 Cayu database."
        )
    configuration = connection.execute(
        "SELECT singleton, tokenizer_version "
        "FROM cayu_transcript_search_configuration ORDER BY singleton"
    ).fetchall()
    if len(configuration) != 1 or tuple(configuration[0]) != (
        1,
        expected_tokenizer_version,
    ):
        raise RuntimeError(
            "SQLite transcript search tokenizer identity conflicts with this runtime. "
            "Recreate a revision-46 Cayu database with this runtime."
        )
    expected = {
        "cayu_transcript_messages_fts": "table",
        "cayu_transcript_messages_fts_insert": "trigger",
        "cayu_transcript_messages_fts_delete": "trigger",
        "cayu_transcript_messages_fts_update": "trigger",
        "cayu_transcript_messages_search_document_insert": "trigger",
        "cayu_transcript_messages_search_document_update": "trigger",
    }
    rows = connection.execute(
        "SELECT name, type, sql FROM sqlite_master WHERE name IN (?, ?, ?, ?, ?, ?)",
        tuple(expected),
    ).fetchall()
    found = {str(row[0]): (str(row[1]), str(row[2] or "")) for row in rows}
    if set(found) != set(expected) or any(
        found[name][0] != object_type for name, object_type in expected.items()
    ):
        raise RuntimeError(
            "SQLite transcript search schema is incomplete. Recreate or restore "
            "a known-good revision-46 Cayu database."
        )
    fts_sql = "".join(found["cayu_transcript_messages_fts"][1].lower().split())
    if "usingfts5(session_token,message_text,content='')" not in fts_sql:
        raise RuntimeError(
            "SQLite transcript search index conflicts with Cayu's contentless FTS contract."
        )
    insert_sql = "".join(found["cayu_transcript_messages_fts_insert"][1].lower().split())
    delete_sql = "".join(found["cayu_transcript_messages_fts_delete"][1].lower().split())
    update_sql = "".join(found["cayu_transcript_messages_fts_update"][1].lower().split())
    document_insert_sql = "".join(
        found["cayu_transcript_messages_search_document_insert"][1].lower().split()
    )
    document_update_sql = "".join(
        found["cayu_transcript_messages_search_document_update"][1].lower().split()
    )
    if (
        not all(
            fragment in insert_sql
            for fragment in (
                "afterinsertoncayu_transcript_messages",
                "whennew.rolein('user','assistant')",
                "cayu_transcript_session_token(new.session_id)",
                "new.transcript_search_document",
            )
        )
        or not all(
            fragment in delete_sql
            for fragment in (
                "afterdeleteoncayu_transcript_messages",
                "whenold.rolein('user','assistant')",
                "'delete',old.sequence",
                "cayu_transcript_session_token(old.session_id)",
                "old.transcript_search_document",
            )
        )
        or not all(
            fragment in update_sql
            for fragment in (
                "afterupdateofsession_id,role,message_json",
                "'delete',old.sequence",
                "cayu_transcript_search_document(new.message_json)",
                "wherenew.rolein('user','assistant')",
            )
        )
        or not all(
            fragment in document_insert_sql
            for fragment in (
                "beforeinsertoncayu_transcript_messages",
                "new.transcript_search_documentisnullor",
                "cayu_transcript_search_document(new.message_json)",
                "raise(abort,'invalidtranscriptsearchdocument')",
            )
        )
        or not all(
            fragment in document_update_sql
            for fragment in (
                "beforeupdateoftranscript_search_document",
                "new.transcript_search_documentisnullor",
                "cayu_transcript_search_document(new.message_json)",
                "raise(abort,'invalidtranscriptsearchdocument')",
            )
        )
    ):
        raise RuntimeError(
            "SQLite transcript search maintenance triggers conflict with Cayu's contract."
        )
    fixture = json.dumps(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "visible"},
                {"type": "thinking", "text": "hidden"},
            ],
        }
    )
    projected = connection.execute(
        "SELECT cayu_transcript_search_document(?)",
        (fixture,),
    ).fetchone()
    if projected is None or projected[0] != "x76697369626c65":
        raise RuntimeError(
            "SQLite transcript search projection does not preserve the narrative-only boundary."
        )
