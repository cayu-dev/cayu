"""SQLite schema checks for session identity, grants, messages and child lifecycle.

Reconciliation and migration callers retain revision gates, validation order and
native transaction ownership. These checks inspect existing schema and session
identity rows without importing migration or store implementations.
"""

from __future__ import annotations

import re
import sqlite3
from hashlib import sha256
from typing import NoReturn

from cayu.storage import _sqlite_catalog as sqlite_catalog

_REVISION_17_INDEX_NAMES = frozenset(
    {
        "idx_cayu_checkpoints_pending_control_action",
        "idx_cayu_events_pending_action_barrier",
        "idx_cayu_events_pending_action_lookup",
    }
)


_PENDING_ACTION_SCOPE_INDEX_NAMES = frozenset(
    {
        "idx_cayu_events_pending_action_round_scope",
        "idx_cayu_events_pending_action_attempt_scope",
    }
)


_WORKFLOW_REPLAY_INDEX_NAMES = frozenset(
    {
        "idx_cayu_events_workflow_step_replay",
        "idx_cayu_events_workflow_step_attempt",
        "idx_cayu_events_workflow_attempt_marker",
    }
)


def _revision_17_index_definitions(revision_sql: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in sqlite_catalog._iter_statements(revision_sql):
        match = re.match(
            r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
            statement,
            flags=re.IGNORECASE,
        )
        if match is not None and match.group(1) in _REVISION_17_INDEX_NAMES:
            definitions[match.group(1)] = statement
    if definitions.keys() != _REVISION_17_INDEX_NAMES:
        raise RuntimeError("Cayu revision 17 index definitions are incomplete.")
    return definitions


def _validate_revision_17_indexes(
    connection: sqlite3.Connection,
    *,
    revision_sql: str,
    require_all: bool,
) -> None:
    """Reject same-name SQLite indexes whose structure is not Cayu's contract."""
    for index_name, expected in _revision_17_index_definitions(revision_sql=revision_sql).items():
        row = connection.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if row is None:
            if require_all:
                raise RuntimeError(
                    f"Required Cayu SQLite index is missing: {index_name}. "
                    "Run with schema_mode='migrate' to repair the schema."
                )
            continue
        actual_type, _table_name, actual_definition = row
        if (
            actual_type != "index"
            or actual_definition is None
            or (
                sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
                != sqlite_catalog._normalize_sqlite_schema_definition(expected)
            )
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu revision 17. "
                "Rename or remove the conflicting object, then run with "
                "schema_mode='migrate' to create the required index."
            )


def _workflow_replay_index_definitions(revision_sql: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in sqlite_catalog._iter_statements(revision_sql):
        match = re.match(
            r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
            statement,
            flags=re.IGNORECASE,
        )
        if match is not None and match.group(1) in _WORKFLOW_REPLAY_INDEX_NAMES:
            definitions[match.group(1)] = statement
    if definitions.keys() != _WORKFLOW_REPLAY_INDEX_NAMES:
        raise RuntimeError("Cayu workflow replay index definitions are incomplete.")
    return definitions


def _validate_workflow_replay_indexes(
    connection: sqlite3.Connection,
    *,
    revision_sql: str,
    require_all: bool,
) -> None:
    for index_name, expected in _workflow_replay_index_definitions(
        revision_sql=revision_sql
    ).items():
        row = connection.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if row is None:
            if require_all:
                raise RuntimeError(
                    f"Required Cayu SQLite index is missing: {index_name}. "
                    "Run with schema_mode='migrate' to repair the schema."
                )
            continue
        actual_type, table_name, actual_definition = row
        if (
            actual_type != "index"
            or table_name != "cayu_events"
            or actual_definition is None
            or sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
            != sqlite_catalog._normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "workflow replay contract. Rename or remove the conflicting "
                "object, then run with schema_mode='migrate'."
            )


def _pending_action_scope_index_definitions(revision_sql: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in sqlite_catalog._iter_statements(revision_sql):
        for index_name in _PENDING_ACTION_SCOPE_INDEX_NAMES:
            if index_name in statement:
                definitions[index_name] = statement
    if definitions.keys() != _PENDING_ACTION_SCOPE_INDEX_NAMES:
        raise RuntimeError("Cayu pending-action scope index definitions are incomplete.")
    return definitions


def _validate_pending_action_scope_indexes(
    connection: sqlite3.Connection,
    *,
    revision_sql: str,
    require_all: bool,
) -> None:
    for index_name, expected in _pending_action_scope_index_definitions(
        revision_sql=revision_sql
    ).items():
        row = connection.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if row is None:
            if require_all:
                raise RuntimeError(
                    f"Required Cayu SQLite index is missing: {index_name}. "
                    "Run with schema_mode='migrate' to repair the schema."
                )
            continue
        actual_type, table_name, actual_definition = row
        if (
            actual_type != "index"
            or table_name != "cayu_events"
            or actual_definition is None
            or sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
            != sqlite_catalog._normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "pending-action scope contract. Rename or remove the conflicting "
                "object, then run with schema_mode='migrate'."
            )


def _validate_session_instance_schema(connection: sqlite3.Connection) -> None:
    session_columns = {row[1] for row in connection.execute("PRAGMA table_info(cayu_sessions)")}
    task_columns = {row[1] for row in connection.execute("PRAGMA table_info(cayu_tasks)")}
    if "instance_id" not in session_columns or "session_instance_id" not in task_columns:
        raise RuntimeError("SQLite session-instance authority columns are missing.")
    invalid = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM cayu_sessions WHERE instance_id IS NULL)"
    ).fetchone()
    if invalid is None or invalid[0]:
        raise RuntimeError("SQLite session-instance authority is incomplete.")


def _validate_session_invocation_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_sessions)")
    }
    if columns.get("invocation_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object 'cayu_sessions.invocation_json' conflicts with "
            "Cayu's required invocation-provenance contract. Recreate the Cayu "
            "database from a known-good revision-36 schema."
        )


def _validate_targeted_tool_grant_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_targeted_tool_grants": (
            ("grant_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("request_id", "TEXT", 1, 0),
            ("tool_ref", "TEXT", 1, 0),
            ("generation_id", "TEXT", 1, 0),
            ("tool_id", "TEXT", 1, 0),
            ("tool_name", "TEXT", 1, 0),
            ("catalogue_revision", "TEXT", 1, 0),
            ("descriptor_version", "TEXT", 1, 0),
            ("issued_at", "TEXT", 1, 0),
            ("expires_at", "TEXT", 1, 0),
            ("max_calls", "INTEGER", 1, 0),
            ("used_calls", "INTEGER", 1, 0),
            ("revoked_at", "TEXT", 0, 0),
            ("record_json", "TEXT", 1, 0),
        ),
        "cayu_targeted_tool_grant_uses": (
            ("use_id", "TEXT", 0, 1),
            ("grant_id", "TEXT", 1, 0),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("outer_tool_call_id", "TEXT", 1, 0),
            ("arguments_sha256", "TEXT", 1, 0),
            ("invocation_id", "TEXT", 1, 0),
            ("bound_at", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's revision-52 "
                "targeted-grant durability contract. Run `cayu storage migrate` or "
                "restore the database from a known-good backup."
            )

    required_indexes = {
        "cayu_public_authority_aliases": {
            (False, ("field_name", "public_alias")),
        },
        "cayu_targeted_tool_grants": {
            (False, ("session_id", "interaction_id", "issued_at", "grant_id")),
            (True, ("session_id", "interaction_id", "request_id")),
            (True, ("session_id", "interaction_id", "tool_id")),
        },
        "cayu_targeted_tool_grant_uses": {
            (False, ("grant_id", "bound_at", "use_id")),
            (True, ("session_id", "interaction_id", "invocation_id")),
            (True, ("session_id", "interaction_id", "outer_tool_call_id")),
        },
    }
    for table, required in required_indexes.items():
        actual: set[tuple[bool, tuple[str, ...]]] = set()
        for index_row in connection.execute(f"PRAGMA index_list({table})"):
            columns = tuple(
                str(column_row[0])
                for column_row in connection.execute(
                    "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                    (str(index_row[1]),),
                )
            )
            actual.add((bool(index_row[2]), columns))
        if not required <= actual:
            raise RuntimeError(
                f"SQLite indexes for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant contention contract."
            )

    expected_foreign_keys = {
        "cayu_targeted_tool_grants": {
            ("cayu_sessions", "session_id", "id", "CASCADE"),
        },
        "cayu_targeted_tool_grant_uses": {
            ("cayu_targeted_tool_grants", "grant_id", "grant_id", "CASCADE"),
            ("cayu_sessions", "session_id", "id", "CASCADE"),
        },
    }
    for table, expected in expected_foreign_keys.items():
        actual_foreign_keys: set[tuple[str, str, str, str]] = {
            (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        if actual_foreign_keys != expected:
            raise RuntimeError(
                f"SQLite foreign keys for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant scope contract."
            )

    required_table_fragments = {
        "cayu_targeted_tool_grants": {
            "check(max_calls>=1andmax_calls<=32)",
            "check(used_calls>=0andused_calls<=max_calls)",
            "check(json_valid(record_json))",
        },
        "cayu_targeted_tool_grant_uses": {"check(json_valid(record_json))"},
    }
    for table, required in required_table_fragments.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
        if any(fragment not in normalized for fragment in required):
            raise RuntimeError(
                f"SQLite checks for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant budget contract."
            )


def _validate_revision_sixty_two_payload_schema(connection: sqlite3.Connection) -> None:
    """Validate revision-62 storage shape without scanning durable history."""

    deferred_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_deferred_interaction_inputs)")
    )
    if deferred_columns != (
        ("session_id", "TEXT", 0, 1),
        ("interaction_id", "TEXT", 1, 0),
        ("source_messages_json", "TEXT", 1, 0),
    ):
        raise RuntimeError(
            "SQLite deferred interaction input schema conflicts with Cayu's "
            "revision-62 durable payload contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )


def _validate_session_message_queue_typed_message_column(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_session_message_queue)")
    }
    if columns.get("message_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite schema object 'cayu_session_message_queue.message_json' conflicts "
            "with Cayu's revision-57 typed queued-message contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_session_message_lifecycle_columns(connection: sqlite3.Connection) -> None:
    index = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        ("idx_cayu_events_queue_acceptance",),
    ).fetchone()
    definition = sqlite_catalog._normalize_sqlite_schema_sql(
        None if index is None else index[0]
    ).replace(" ", "")
    if (
        "oncayu_events(session_id,json_extract(payload_json,'$.queue_id'))" not in definition
        or "whereevent_type='session.message.queued'" not in definition
    ):
        raise RuntimeError("SQLite queue acceptance lookup index conflicts with revision 83.")
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), row[4])
        for row in connection.execute("PRAGMA table_info(cayu_session_message_queue)")
    }
    if any(columns.get(name) != ("TEXT", 0, None) for name in ("conditions_json", "terminal_json")):
        raise RuntimeError("SQLite session-message lifecycle columns conflict with revision 83.")
    receipts = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), row[4])
        for row in connection.execute("PRAGMA table_info(cayu_session_message_deliveries)")
    }
    if receipts.get("reject_only") != ("INTEGER", 1, "0"):
        raise RuntimeError("SQLite queue rejection receipt conflicts with revision 83.")


def _validate_revision_79_child_lifecycle_schema(connection: sqlite3.Connection) -> None:
    table = "cayu_child_session_lifecycle_candidates"
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    expected_foreign_keys = {
        ("cayu_sessions", ("child_session_id",), ("id",), "CASCADE"),
        ("cayu_sessions", ("parent_session_id",), ("id",), "CASCADE"),
    }
    if (
        columns
        != (
            ("child_session_id", "TEXT", 0, 1),
            ("parent_session_id", "TEXT", 1, 0),
            ("priority", "INTEGER", 1, 0),
            ("sort_at", "TEXT", 1, 0),
        )
        or sqlite_catalog._sqlite_foreign_key_groups(connection, table) != expected_foreign_keys
    ):
        _raise_revision_79_child_lifecycle_schema_error(table)
    # These hashes bind the normalized SQL installed by revision 79. Checking
    # names and columns alone is unsafe here: IF NOT EXISTS would otherwise
    # accept a same-named view, trigger, or index with different semantics.
    expected_objects = {
        table: (
            "table",
            table,
            "72ae90b389f12b0ef9e50d17589552bfc638ea17238c668049d77ab980c6bb62",
        ),
        "cayu_child_session_lifecycle_canonical": (
            "view",
            "cayu_child_session_lifecycle_canonical",
            "a8e1d479280bd9c5396a2098a242410a76c0ce767d737287c41d4c12b3964ddf",
        ),
        "cayu_index_child_lifecycle_session_insert": (
            "trigger",
            "cayu_sessions",
            "36c3d0aa69d7598b2bf6b8bc5f0b8421a105803176ad540ce8438f30ced6b077",
        ),
        "cayu_index_child_lifecycle_session_update": (
            "trigger",
            "cayu_sessions",
            "8a35cda77227a94beb8fb8116044b133bcb7272461ebb88fb5960375b8afd9e2",
        ),
        "cayu_index_child_lifecycle_event_insert": (
            "trigger",
            "cayu_events",
            "6694b0f3885709f2204bb42e81051b9abafd86931d93e19b64ab7534e0887c23",
        ),
        "cayu_index_child_lifecycle_event_delete": (
            "trigger",
            "cayu_events",
            "eceae14afdb0aebdbceff8ea3007fa2bb65af73a53370dae33b5193e4e3cd00e",
        ),
        "cayu_index_child_lifecycle_event_update": (
            "trigger",
            "cayu_events",
            "71643cc73a086c8f2787428ccbf4b40acfaa73b376c4af7b83889ca3537709b3",
        ),
        "cayu_index_child_lifecycle_consumption": (
            "trigger",
            "cayu_session_operations",
            "8162e6e0f5fab3987cb574fae8f21e14ffa3f1971c8d1e1c047aabb4c7aa5aff",
        ),
        "cayu_index_child_lifecycle_consumption_delete": (
            "trigger",
            "cayu_session_operations",
            "1ae3c9c751ebda820bb0ec04ab508ed74cde0756375fcda80e92c00863a8b6e3",
        ),
        "cayu_index_child_lifecycle_consumption_update": (
            "trigger",
            "cayu_session_operations",
            "85547f6faa7802c004277b0f36dc5ec5c1895f219c4f5fb1239aead9e211a85d",
        ),
    }
    for name, expected in expected_objects.items():
        row = connection.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (name,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[2])
        actual = (
            None
            if row is None
            else (str(row[0]), str(row[1]), sha256(normalized.encode("utf-8")).hexdigest())
        )
        if actual != expected:
            _raise_revision_79_child_lifecycle_schema_error(name)
    expected_indexes = {
        "idx_cayu_child_lifecycle_candidates_page": (
            "cayu_child_session_lifecycle_candidates",
            (
                ("parent_session_id", 0, "BINARY"),
                ("priority", 0, "BINARY"),
                ("sort_at", 0, "BINARY"),
                ("child_session_id", 0, "BINARY"),
            ),
            "37fb6e7aea8b228a2427048a2e38cab0e7f7d108a3745fb125b829456be980d6",
        ),
        "idx_cayu_events_child_lifecycle": (
            "cayu_events",
            (
                ("session_id", 0, "BINARY"),
                ("event_type", 0, "BINARY"),
                ("sequence", 1, "BINARY"),
            ),
            "b7efb7c9b58d8bd832053808aee296c2c8ed3bde173fb3b1f8575f98393d92a6",
        ),
        "idx_cayu_transcript_messages_session_role_order": (
            "cayu_transcript_messages",
            (
                ("session_id", 0, "BINARY"),
                ("role", 0, "BINARY"),
                ("session_order", 1, "BINARY"),
            ),
            "6ccc9d35738eac56087c1b9a34816a1796f24417260ae6bcb005e2b744b945e3",
        ),
    }
    for index, (expected_table, expected_keys, expected_digest) in expected_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        keys = tuple(
            (str(index_row[2]), int(index_row[3]), str(index_row[4]).upper())
            for index_row in connection.execute(f"PRAGMA index_xinfo({index})")
            if int(index_row[5]) == 1
        )
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[1])
        if (
            row is None
            or str(row[0]) != expected_table
            or keys != expected_keys
            or sha256(normalized.encode("utf-8")).hexdigest() != expected_digest
        ):
            _raise_revision_79_child_lifecycle_schema_error(index)


def _raise_revision_79_child_lifecycle_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's bounded child-lifecycle projection. "
        "Run schema_mode=MIGRATE to install revision 79 or recreate the database."
    )
